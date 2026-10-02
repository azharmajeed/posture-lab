import json

import numpy as np
import pytest
from test_posture import calibrated, pose

from posture_lab.app import draw_preview
from posture_lab.config import PostureConfig, load_config
from posture_lab.feedback import FeedbackRecorder, can_label, feedback_hit
from posture_lab.posture import Assessment, PostureMonitor
from posture_lab.types import Pose


def at_time(now, **kwargs):
    sample = pose(**kwargs)
    return Pose(sample.keypoints, int(now * 1000), sample.width, sample.height)


def test_clicked_button_records_current_fn_and_preserves_baseline(tmp_path):
    monitor = calibrated()
    sample = at_time(1, nose_y=0.36)
    assessment = monitor.update(sample, 1)
    assert assessment.status == "good"
    original_profiles = dict(monitor.profiles)
    recorder = FeedbackRecorder(tmp_path / "feedback.jsonl", {"backend": "cpu"})
    recorder.observe(sample, assessment, monitor)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_preview(
        frame,
        sample,
        assessment,
        "CPU",
        62,
        False,
        0.6,
        feedback_enabled=True,
        diagnostic_line=monitor.diagnostic_line,
    )
    assert rendered.shape == (570, 640, 3)
    label = feedback_hit(100, 500, 640, 480)
    assert label == "FN"
    recorder.record(label, sample, assessment, monitor, 1.07, 62)
    saved = json.loads(recorder.path.read_text())
    assert saved["label"] == "FN"
    assert saved["expected_posture"] == "bad"
    assert saved["assessment"]["status"] == "good"
    assert saved["diagnostics"]["raw_metrics"]["head_height"] < monitor.baseline.head_height
    assert saved["diagnostics"]["profile_checks"]["laptop"]["reasons"] == []
    assert saved["posture_config"]["head_drop_threshold"] == monitor.config.head_drop_threshold
    assert saved["pose"]["timestamp_ms"] == 1000
    assert saved["recent_history"][-1]["timestamp_ms"] == 1000
    assert recorder.counts == {"FN": 1, "FP": 0}
    assert monitor.profiles == original_profiles
    assert set(tmp_path.iterdir()) == {recorder.path}  # No images or video written.


def test_fp_appends_and_snapshots_do_not_change_when_pose_changes(tmp_path):
    monitor = calibrated(bad_seconds=0)
    recorder = FeedbackRecorder(tmp_path / "feedback.jsonl", {})
    sample = at_time(1, nose_y=0.5)
    assessment = monitor.update(sample, 1)
    assert assessment.status == "adjust"
    label = feedback_hit(500, 500, 640, 480)
    assert label == "FP"
    recorder.observe(sample, assessment, monitor)
    first = recorder.record(label, sample, assessment, monitor, 1.01, 60)
    sample2 = at_time(1.5)
    assessment2 = monitor.update(sample2, 1.5)
    recorder.observe(sample2, assessment2, monitor)
    recorder.record("FN", sample2, assessment2, monitor, 1.51, 65)
    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [item["label"] for item in records] == ["FP", "FN"]
    assert records[0] == first
    assert records[0]["expected_posture"] == "good"
    assert records[0]["assessment"]["status"] == "adjust"
    assert records[0]["diagnostics"]["profile_checks"]["laptop"]["reasons"] == ["head lowered"]
    assert records[0]["id"] != records[1]["id"]


@pytest.mark.parametrize("status", ["unknown", "paused", "uncalibrated", "calibrating"])
def test_missing_or_ineligible_pose_does_not_save_feedback(tmp_path, status):
    monitor = calibrated()
    sample = at_time(1)
    monitor.update(sample, 1)
    recorder = FeedbackRecorder(tmp_path / "feedback.jsonl", {})
    assert recorder.record("FN", sample, Assessment(status, ""), monitor, 1, 50) is None
    assert not recorder.path.exists()


def test_stale_or_missing_pose_and_invalid_labels_rejected(tmp_path):
    monitor = calibrated()
    sample = at_time(1)
    assessment = monitor.update(sample, 1)
    recorder = FeedbackRecorder(tmp_path / "feedback.jsonl", {})
    assert not can_label(assessment, sample, monitor, 3)
    assert recorder.record("FN", sample, assessment, monitor, 3, 50) is None
    assert recorder.record("FP", None, assessment, monitor, 1, 50) is None
    with pytest.raises(ValueError, match="FN or FP"):
        recorder.record("wrong", sample, assessment, monitor, 1, 50)
    assert not recorder.path.exists()


def test_clicking_camera_controls_or_space_between_buttons_does_not_label():
    for x, y in ((100, 100), (100, 460), (320, 500), (100, 535)):
        assert feedback_hit(x, y, 640, 480) is None


def test_recent_numeric_history_is_bounded_to_six_seconds(tmp_path):
    monitor = calibrated()
    recorder = FeedbackRecorder(tmp_path / "feedback.jsonl", {})
    for step in range(100):
        now = 1 + step / 10
        sample = at_time(now)
        recorder.observe(sample, monitor.update(sample, now), monitor)
    assert recorder.history[-1]["timestamp_ms"] - recorder.history[0]["timestamp_ms"] <= 6000
    assert len(recorder.history) <= 61


def test_manual_saved_profile_uses_same_single_baseline_decision_rules():
    # Profile storage must not change the existing side/drop/tilt/closer rule.
    original = calibrated(smoothing_seconds=0)
    restored = PostureMonitor(PostureConfig(**original.config.__dict__))
    restored.set_profiles({"left-monitor": original.baseline}, "left-monitor")
    for now, sample in enumerate((pose(), pose(nose_x=0.65), pose(nose_y=0.5), pose()), 1):
        original_result = original.update(sample, now)
        restored_result = restored.update(sample, now)
        assert original_result.status == restored_result.status
        assert original_result.nudge == restored_result.nudge
        assert original_result.bad_seconds == restored_result.bad_seconds


def test_feedback_path_can_be_configured_and_cannot_be_empty(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"feedback": {"path": str(tmp_path / "labels.jsonl")}}))
    assert load_config(config).feedback.path == str(tmp_path / "labels.jsonl")
    config.write_text(json.dumps({"feedback": {"path": " "}}))
    with pytest.raises(ValueError, match="feedback path"):
        load_config(config)
