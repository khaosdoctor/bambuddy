"""Scale reader wrapper with stability detection, auto-zero, and periodic recalibration."""

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

MOVING_AVG_SIZE = 20

# ponytail: auto-zero corrects thermal/mechanical tare drift when scale is empty
AUTO_ZERO_THRESHOLD_G = 5.0
AUTO_ZERO_SETTLE_S = 10.0
AUTO_ZERO_COOLDOWN_S = 60.0

# ponytail: 6h default, raise if recal proves disruptive
AFE_RECAL_INTERVAL_S = 6 * 3600


class ScaleReader:
    def __init__(self, tare_offset: int = 0, calibration_factor: float = 1.0):
        self._scale = None
        self._tare_offset = tare_offset
        self._calibration_factor = calibration_factor
        self._samples: deque[float] = deque(maxlen=MOVING_AVG_SIZE)
        self._stability_history: deque[tuple[float, float]] = deque(maxlen=20)
        self._ok = False
        self._last_raw = 0

        self._auto_zero_enter: float | None = None
        self._last_auto_zero: float = -AUTO_ZERO_COOLDOWN_S
        self._auto_zero_pending = False
        self._pending_zero_avg: float | None = None
        self._last_afe_recal: float = time.monotonic()

        try:
            from .nau7802 import NAU7802

            self._scale = NAU7802()
            self._scale.init()
            self._ok = True
            bus_num = getattr(self._scale, "_bus_num", "?")
            logger.info(
                "Scale initialized on I2C bus %s (tare=%d, cal=%.6f)",
                bus_num,
                tare_offset,
                calibration_factor,
            )
        except Exception as e:
            logger.info("Scale not available: %s", e)

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def last_raw(self) -> int:
        return self._last_raw

    def close(self):
        try:
            if self._scale:
                self._scale.close()
        except Exception:
            pass

    def update_calibration(self, tare_offset: int, calibration_factor: float):
        self._tare_offset = tare_offset
        self._calibration_factor = calibration_factor
        logger.info("Calibration updated: tare=%d, factor=%.6f", tare_offset, calibration_factor)

    def tare(self):
        """Set current raw reading as tare offset."""
        if self._last_raw:
            self._tare_offset = self._last_raw
            self._samples.clear()
            self._stability_history.clear()
            logger.info("Tared at raw=%d", self._tare_offset)
        return self._tare_offset

    @property
    def auto_zero_pending(self) -> bool:
        return self._auto_zero_pending

    @property
    def afe_recal_due(self) -> bool:
        if not self._scale:
            return False
        return time.monotonic() - self._last_afe_recal >= AFE_RECAL_INTERVAL_S

    def recalibrate_afe(self) -> bool:
        """Re-run NAU7802 internal offset calibration (mode=0). Blocks ~400ms.

        Safe to run any time (spool on scale or not) because internal cal
        disconnects inputs and shorts to internal reference.
        Per datasheet section 8.6: removes PGA gain and offset errors.
        """
        if not self._scale:
            return False
        try:
            self._scale.calibrate_afe(timeout_ms=1000, mode=0)
            self._scale.flush_readings(count=2, timeout_s=1.0)
            self._samples.clear()
            self._stability_history.clear()
            self._last_afe_recal = time.monotonic()
            logger.info("Periodic internal offset cal complete")
            return True
        except Exception as e:
            logger.warning("Internal offset cal failed: %s", e)
            return False

    def calibrate_zero(self) -> int | None:
        """Run zero-point calibration if auto-zero detected empty scale.

        Per datasheet section 8.6.1: system offset calibration (mode=2) uses
        the actual inputs to set the hardware zero point, removing both internal
        PGA errors and external DC errors (load cell drift, thermal offset).
        More precise than software tare because it corrects at 24-bit ADC level
        before gain multiplication.

        Falls back to software tare adjustment when hardware is unavailable.
        Returns new tare offset, or None if no calibration was pending.
        """
        if not self._auto_zero_pending:
            return None
        self._auto_zero_pending = False

        if self._scale:
            try:
                # ponytail: system offset cal (mode=2), upgrade to temp-triggered if 6h proves too coarse
                self._scale.calibrate_afe(timeout_ms=1000, mode=2)
                self._scale.flush_readings(count=4, timeout_s=2.0)
                readings = []
                for _ in range(5):
                    if self._scale.wait_data_ready(timeout_s=0.5):
                        readings.append(self._scale.read_raw())
                if readings:
                    self._tare_offset = sum(readings) // len(readings)
                self._samples.clear()
                self._stability_history.clear()
                logger.info("Auto-zero: system offset cal, tare=%d", self._tare_offset)
                return self._tare_offset
            except Exception as e:
                logger.warning("System offset cal failed, using software tare: %s", e)

        # Fallback: software-only tare adjustment
        if self._calibration_factor != 0 and self._pending_zero_avg is not None:
            adjustment = int(round(self._pending_zero_avg / self._calibration_factor))
            if adjustment != 0:
                self._tare_offset += adjustment
        self._samples.clear()
        self._stability_history.clear()
        logger.info("Auto-zero: software tare=%d", self._tare_offset)
        return self._tare_offset

    def _check_auto_zero(self, avg_grams: float, stable: bool, now: float):
        """Detect empty-scale condition for calibrate_zero()."""
        if stable and abs(avg_grams) <= AUTO_ZERO_THRESHOLD_G:
            if self._auto_zero_enter is None:
                self._auto_zero_enter = now
            elif (
                now - self._auto_zero_enter >= AUTO_ZERO_SETTLE_S
                and now - self._last_auto_zero >= AUTO_ZERO_COOLDOWN_S
            ):
                self._auto_zero_pending = True
                self._pending_zero_avg = avg_grams
                self._auto_zero_enter = None
                self._last_auto_zero = now
        else:
            self._auto_zero_enter = None

    def read(self) -> tuple[float, bool, int] | None:
        """Read current weight. Returns (grams, stable, raw_adc) or None."""
        try:
            if not self._scale.data_ready():
                return None

            raw = self._scale.read_raw()
            self._last_raw = raw
            self._ok = True

            grams = (raw - self._tare_offset) * self._calibration_factor
            self._samples.append(grams)

            # Moving average
            avg_grams = sum(self._samples) / len(self._samples)

            # Stability: track readings over time
            now = time.monotonic()
            self._stability_history.append((now, avg_grams))

            # Stable if all readings within 1s window are within 2g of each other
            stable = False
            if len(self._stability_history) >= 5:
                cutoff = now - 1.0
                recent = [g for t, g in self._stability_history if t >= cutoff]
                if len(recent) >= 3:
                    spread = max(recent) - min(recent)
                    stable = spread < 2.0

            self._check_auto_zero(avg_grams, stable, now)

            return round(avg_grams, 1), stable, raw

        except Exception as e:
            logger.debug("Scale read error: %s", e)
            self._ok = False
            return None
