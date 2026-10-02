import math

import pytest

from posture_lab.config import PostureConfig
from posture_lab.posture import PostureMonitor, extract_metrics
from posture_lab.types import Keypoint, Pose


def pose(nose_x=0.5, nose_y=0.3, confidence=0.99, width=640, height=480):
    return Pose(
        {
            "nose": Keypoint(nose_x, nose_y, confidence),
            "left_shoulder": Keypoint(0.3, 0.6, confidence),
            "right_shoulder": Keypoint(0.7, 0.6, confidence),
        },
        0,
        width,
        height,
    )


def calibrated(**overrides):
    options = {
        "calibration_seconds": 0.4,
        "bad_seconds": 2,
        "cooldown_seconds": 5,
        "smoothing_seconds": 0,
    }
    options.update(overrides)
    monitor = PostureMonitor(PostureConfig(**options))
    monitor.start_calibration()
    for now in (0, 0.1, 0.2, 0.3, 0.4):
        monitor.update(pose(), now)
    assert monitor.baseline is not None
    return monitor


def test_nudge_requires_continuous_dwell_and_respects_cooldown():
    monitor = calibrated()
    bad = pose(nose_x=0.65)
    assert not monitor.update(bad, 0.5).nudge
    for now in (1, 1.5, 2):
        assert not monitor.update(bad, now).nudge
    assert monitor.update(bad, 2.5).nudge
    for now in (3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7):
        assert not monitor.update(bad, now).nudge
    assert monitor.update(bad, 7.5).nudge


def test_five_second_threshold_delays_adjust_status_and_reminder():
    monitor = calibrated(bad_seconds=5)
    bad = pose(nose_x=0.65)
    for now in (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5):
        result = monitor.update(bad, now)
        assert result.status == "pending"
        assert not result.nudge
    result = monitor.update(bad, 5.5)
    assert result.status == "adjust"
    assert result.nudge


def test_brief_change_cannot_accumulate_toward_later_reminder():
    monitor = calibrated(bad_seconds=5)
    bad = pose(nose_y=0.5)
    for now in (0.5, 1, 1.5, 2, 2.5, 3, 3.5):
        assert not monitor.update(bad, now).nudge
    assert monitor.update(pose(), 4).status == "good"
    for now in (4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9):
        result = monitor.update(bad, now)
        assert result.status == "pending"
        assert not result.nudge
    assert monitor.update(bad, 9.5).nudge


def test_unknown_and_low_confidence_reset_bad_timer():
    monitor = calibrated()
    bad = pose(nose_y=0.5)
    monitor.update(bad, 0.5)
    monitor.update(bad, 1)
    assert monitor.update(pose(confidence=0.2), 1.5).status == "unknown"
    assert monitor.update(bad, 2).bad_seconds == 0
    monitor.update(None, 2.5)
    assert monitor.update(bad, 3).bad_seconds == 0


def test_long_gap_cannot_trigger_nudge_from_old_evidence():
    monitor = calibrated()
    monitor.update(pose(nose_x=0.65), 0.5)
    result = monitor.update(pose(nose_x=0.65), 30)
    assert result.bad_seconds == 0
    assert not result.nudge


def test_returning_to_baseline_resets_dwell():
    monitor = calibrated()
    monitor.update(pose(nose_y=0.5), 0.5)
    assert monitor.update(pose(), 1).status == "good"
    assert monitor.update(pose(nose_y=0.5), 1.5).bad_seconds == 0


def test_calibration_requires_visible_stable_samples():
    monitor = PostureMonitor(PostureConfig(calibration_seconds=0.4))
    monitor.start_calibration()
    for now, x in ((0, 0.3), (0.1, 0.7), (0.2, 0.3), (0.3, 0.7), (0.4, 0.3)):
        monitor.update(pose(nose_x=x), now)
    assert monitor.baseline is None
    assert monitor.update(None, 0.5).status == "unknown"
    for now in (0.6, 0.7, 0.8, 0.9, 1.01):
        monitor.update(pose(), now)
    assert monitor.baseline is not None


def test_pixel_geometry_accounts_for_aspect_ratio():
    metrics = extract_metrics(pose(), 0.6)
    assert metrics.head_height == pytest.approx(0.5625)
    assert metrics.head_offset == pytest.approx(0)
    assert metrics.shoulder_tilt == pytest.approx(0)


def test_out_of_frame_landmarks_are_unknown():
    assert extract_metrics(pose(nose_x=1.2), 0.6) is None


def test_nonfinite_and_invalid_confidence_rejected():
    for value in (math.nan, math.inf):
        with pytest.raises(ValueError):
            Keypoint(value, 0, 1)
    with pytest.raises(ValueError):
        Keypoint(0, 0, 1.5)
