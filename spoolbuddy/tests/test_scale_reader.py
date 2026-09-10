"""Tests for ScaleReader periodic AFE recalibration."""

from unittest.mock import MagicMock, patch

from daemon.scale_reader import (
    AFE_RECAL_INTERVAL_S,
    AFE_RECAL_RETRY_S,
    ScaleReader,
)


def _make_reader(tare=1000, cal=1.0):
    reader = ScaleReader(tare_offset=tare, calibration_factor=cal)
    reader._scale = MagicMock()
    reader._ok = True
    return reader


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

    def test_recalibrate_failure_backs_off(self):
        reader = _make_reader()
        reader._last_afe_recal = 0.0
        reader._scale.calibrate_afe.side_effect = RuntimeError("timeout")

        with patch("daemon.scale_reader.time") as mock_time:
            mock_time.monotonic.return_value = AFE_RECAL_INTERVAL_S + 1
            reader.recalibrate_afe()
            next_due = reader._last_afe_recal + AFE_RECAL_INTERVAL_S
            assert next_due > AFE_RECAL_INTERVAL_S + 1
            mock_time.monotonic.return_value = AFE_RECAL_INTERVAL_S + 2
            assert not reader.afe_recal_due
            mock_time.monotonic.return_value = AFE_RECAL_INTERVAL_S + AFE_RECAL_RETRY_S + 2
            assert reader.afe_recal_due

    def test_no_scale_returns_false(self):
        reader = _make_reader()
        reader._scale = None

        assert not reader.afe_recal_due
        assert not reader.recalibrate_afe()

    def test_tare_preserves_through_recal(self):
        """AFE recal (mode=0) must not change the stored tare offset."""
        reader = _make_reader(tare=350000)
        reader.recalibrate_afe()
        assert reader._tare_offset == 350000


class TestReadWait:
    """read_wait waits for data_ready before reading."""

    def test_returns_reading_when_data_ready(self):
        reader = _make_reader(tare=0, cal=1.0)
        reader._scale.wait_data_ready.return_value = True
        reader._scale.read_raw.return_value = 100

        result = reader.read_wait(timeout_s=0.5)
        assert result is not None
        reader._scale.wait_data_ready.assert_called_once_with(timeout_s=0.5)

    def test_returns_none_when_timeout(self):
        reader = _make_reader()
        reader._scale.wait_data_ready.return_value = False

        result = reader.read_wait(timeout_s=0.5)
        assert result is None

    def test_returns_none_without_hardware(self):
        reader = _make_reader()
        reader._scale = None

        result = reader.read_wait(timeout_s=0.5)
        assert result is None
