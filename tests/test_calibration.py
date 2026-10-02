import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_posture import pose

from posture_lab.calibration import CalibrationStore
from posture_lab.config import PostureConfig
from posture_lab.posture import Metrics, PostureMonitor

LAPTOP = Metrics(0, 0.5625, 0, 0.4)
LEFT = Metrics(0.375, 0.5625, 0, 0.4)


def test_both_profiles_survive_restart_and_other_camera_context(tmp_path):
    path = tmp_path / "calibrations.json"
    laptop = CalibrationStore(path, {"camera": 0, "backend": "cpu"})
    laptop.save({"laptop": LAPTOP, "left-monitor": LEFT})
    other = CalibrationStore(path, {"camera": 1, "backend": "coral"})
    assert other.load() == {}
    other.save({"laptop": LEFT})
    restored = CalibrationStore(path, laptop.context).load()
    assert restored == {"laptop": LAPTOP, "left-monitor": LEFT}
    monitor = PostureMonitor(PostureConfig(smoothing_seconds=0))
    monitor.set_profiles(restored)
    assert monitor.update(pose(), 0).status == "good"
    assert monitor.update(pose(nose_x=0.65), 1).status == "good"
    assert monitor.active_profile == "left-monitor"


def test_failed_atomic_save_keeps_previous_calibration(tmp_path, monkeypatch):
    path = tmp_path / "calibrations.json"
    store = CalibrationStore(path, {})
    store.save({"laptop": LAPTOP})
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("Simulated write failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        store.save({"laptop": LEFT})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_file_is_not_overwritten(tmp_path):
    path = tmp_path / "calibrations.json"
    path.write_text("broken json")
    store = CalibrationStore(path, {})
    with pytest.raises(ValueError):
        store.load()
    with pytest.raises(ValueError):
        store.save({"laptop": LAPTOP})
    assert path.read_text() == "broken json"


def test_nonfinite_saved_metrics_rejected(tmp_path):
    path = tmp_path / "calibrations.json"
    store = CalibrationStore(path, {})
    store.save({"laptop": LAPTOP})
    data = json.loads(path.read_text())
    data["contexts"][store.key]["profiles"]["laptop"]["metrics"]["head_height"] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="finite"):
        store.load()


def test_manual_missing_profile_does_not_use_other_saved_position():
    monitor = PostureMonitor(PostureConfig())
    monitor.set_profiles({"laptop": LAPTOP}, "left-monitor")
    assert monitor.update(pose(), 0).status == "uncalibrated"


def test_automatic_mode_does_not_mix_axes_from_different_baselines():
    monitor = PostureMonitor(PostureConfig(smoothing_seconds=0))
    monitor.set_profiles(
        {
            "laptop": Metrics(0, 0.8, 0, 0.4),
            "left-monitor": Metrics(0.5, 0.4, 0, 0.4),
        }
    )
    assert monitor.update(pose(nose_y=0.6 - 0.4 * 256 / 480), 0).status == "pending"


def test_recalibration_preserves_other_profile_and_resets_bad_evidence():
    monitor = PostureMonitor(PostureConfig(calibration_seconds=0.4, smoothing_seconds=0))
    monitor.set_profiles({"laptop": LAPTOP, "left-monitor": LEFT})
    monitor.update(pose(nose_y=0.5), 0)
    monitor.select_profile("laptop")
    assert monitor.reminder_wait_seconds(0) is None
    monitor.start_calibration()
    for now in (1, 1.1, 1.2, 1.3, 1.5):
        monitor.update(pose(nose_x=0.52), now)
    assert monitor.calibration_revision == 1
    assert monitor.profiles["left-monitor"] == LEFT
    assert asdict(monitor.profiles["laptop"])["head_offset"] == pytest.approx(0.05)


def test_countdown_matches_five_second_dwell_and_thirty_second_repeat():
    monitor = PostureMonitor(PostureConfig(smoothing_seconds=0))
    monitor.set_profiles({"laptop": LAPTOP})
    bad = pose(nose_y=0.5)
    for now in range(6):
        result = monitor.update(bad, now)
        assert result.nudge == (now == 5)
        assert monitor.reminder_wait_seconds(now) == (30 if now == 5 else 5 - now)
    for now in range(6, 36):
        result = monitor.update(bad, now)
        assert result.nudge == (now == 35)
    assert monitor.update(pose(), 36).status == "good"
    assert monitor.reminder_wait_seconds(36) is None
