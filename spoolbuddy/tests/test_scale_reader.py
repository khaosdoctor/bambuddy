"""Tests for ScaleReader auto-zero, system offset cal, and periodic AFE recalibration."""

from unittest.mock import MagicMock, patch

from daemon.scale_reader import (
    AUTO_ZERO_COOLDOWN_S,
    AUTO_ZERO_SETTLE_S,
    AUTO_ZERO_THRESHOLD_G,
    AFE_RECAL_INTERVAL_S,
    ScaleReader,
)


def _make_reader(tare=1000, cal=1.0):
    # Init catches hardware errors gracefully; we attach a mock scale after
    reader = ScaleReader(tare_offset=tare, calibration_factor=cal)
    reader._scale = MagicMock()
    reader._ok = True
    return reader


class TestAutoZeroDetection:
    """_check_auto_zero sets the pending flag; calibrate_zero() acts on it."""

    def test_sets_pending_after_stable_settle(self):
        reader = _make_reader()
        reader._check_auto_zero(avg_grams=3.0, stable=True, now=0.0)
        assert not reader.auto_zero_pending

        reader._check_auto_zero(avg_grams=3.0, stable=True, now=AUTO_ZERO_SETTLE_S + 1)
        assert reader.auto_zero_pending

    def test_not_pending_when_unstable(self):
        reader = _make_reader()
        reader._check_auto_zero(avg_grams=3.0, stable=False, now=0.0)
        reader._check_auto_zero(avg_grams=3.0, stable=False, now=AUTO_ZERO_SETTLE_S + 1)
        assert not reader.auto_zero_pending

    def test_not_pending_above_threshold(self):
        reader = _make_reader()
        reader._check_auto_zero(avg_grams=AUTO_ZERO_THRESHOLD_G + 1, stable=True, now=0.0)
        reader._check_auto_zero(avg_grams=AUTO_ZERO_THRESHOLD_G + 1, stable=True, now=AUTO_ZERO_SETTLE_S + 1)
        assert not reader.auto_zero_pending

    def test_cooldown_respected(self):
        reader = _make_reader()

        reader._check_auto_zero(avg_grams=3.0, stable=True, now=0.0)
        reader._check_auto_zero(avg_grams=3.0, stable=True, now=AUTO_ZERO_SETTLE_S + 1)
        assert reader.auto_zero_pending
        reader._auto_zero_pending = False  # consumed

        # Within cooldown
        t = AUTO_ZERO_SETTLE_S + 2
        reader._check_auto_zero(avg_grams=2.0, stable=True, now=t)
        reader._check_auto_zero(avg_grams=2.0, stable=True, now=t + AUTO_ZERO_SETTLE_S + 1)
        assert not reader.auto_zero_pending

        # After cooldown
        t2 = AUTO_ZERO_SETTLE_S + 2 + AUTO_ZERO_COOLDOWN_S + 1
        reader._check_auto_zero(avg_grams=2.0, stable=True, now=t2)
        reader._check_auto_zero(avg_grams=2.0, stable=True, now=t2 + AUTO_ZERO_SETTLE_S + 1)
        assert reader.auto_zero_pending

    def test_resets_on_instability(self):
        reader = _make_reader()
        reader._check_auto_zero(avg_grams=3.0, stable=True, now=0.0)
        assert reader._auto_zero_enter == 0.0

        reader._check_auto_zero(avg_grams=3.0, stable=False, now=5.0)
        assert reader._auto_zero_enter is None

        reader._check_auto_zero(avg_grams=3.0, stable=True, now=6.0)
        reader._check_auto_zero(avg_grams=3.0, stable=True, now=6.0 + AUTO_ZERO_SETTLE_S + 1)
        assert reader.auto_zero_pending


class TestCalibrateZero:
    """calibrate_zero() runs NAU7802 system offset cal (mode=2) per datasheet 8.6.1."""

    def test_runs_system_offset_cal_mode_2(self):
        reader = _make_reader(tare=1000)
        nau = reader._scale
        nau.wait_data_ready.return_value = True
        nau.read_raw.return_value = 5  # near-zero after cal

        reader._auto_zero_pending = True
        reader._pending_zero_avg = 3.0

        new_tare = reader.calibrate_zero()

        nau.calibrate_afe.assert_called_once_with(timeout_ms=1000, mode=2)
        nau.flush_readings.assert_called_once_with(count=4, timeout_s=2.0)
        assert new_tare == 5  # average of 5 readings of value 5
        assert reader._tare_offset == 5
        assert not reader.auto_zero_pending

    def test_returns_none_when_not_pending(self):
        reader = _make_reader()
        assert reader.calibrate_zero() is None

    def test_falls_back_to_software_tare_on_hw_failure(self):
        reader = _make_reader(tare=1000, cal=1.0)
        reader._scale.calibrate_afe.side_effect = RuntimeError("timeout")

        reader._auto_zero_pending = True
        reader._pending_zero_avg = 3.0

        new_tare = reader.calibrate_zero()

        assert new_tare == 1003  # software fallback: 1000 + round(3.0 / 1.0)
        assert reader._tare_offset == 1003

    def test_falls_back_to_software_tare_without_hardware(self):
        reader = _make_reader(tare=1000, cal=1.0)
        reader._scale = None

        reader._auto_zero_pending = True
        reader._pending_zero_avg = 4.0

        new_tare = reader.calibrate_zero()
        assert new_tare == 1004

    def test_software_fallback_skips_zero_adjustment(self):
        reader = _make_reader(tare=1000, cal=1.0)
        reader._scale = None

        reader._auto_zero_pending = True
        reader._pending_zero_avg = 0.4  # rounds to 0

        new_tare = reader.calibrate_zero()
        assert new_tare == 1000  # no adjustment


class TestAFERecal:
    """Periodic internal offset calibration (mode=0), safe with spool on scale."""

    def test_recal_due_after_interval(self):
        reader = _make_reader()
        reader._last_afe_recal = 0.0

        with patch("daemon.scale_reader.time") as mock_time:
            mock_time.monotonic.return_value = AFE_RECAL_INTERVAL_S + 1
            assert reader.afe_recal_due

    def test_recal_not_due_before_interval(self):
        reader = _make_reader()

        with patch("daemon.scale_reader.time") as mock_time:
            mock_time.monotonic.return_value = reader._last_afe_recal + 100
            assert not reader.afe_recal_due

    def test_recalibrate_uses_internal_mode_0(self):
        reader = _make_reader()
        nau = reader._scale

        result = reader.recalibrate_afe()

        assert result is True
        nau.calibrate_afe.assert_called_once_with(timeout_ms=1000, mode=0)
        nau.flush_readings.assert_called_once_with(count=2, timeout_s=1.0)
        assert len(reader._samples) == 0

    def test_recalibrate_handles_failure(self):
        reader = _make_reader()
        reader._scale.calibrate_afe.side_effect = RuntimeError("timeout")

        result = reader.recalibrate_afe()
        assert result is False

    def test_no_scale_returns_false(self):
        reader = _make_reader()
        reader._scale = None

        assert not reader.afe_recal_due
        assert not reader.recalibrate_afe()
