"""User-labelled detection examples with numeric evidence; no camera images."""

import json
import os
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from posture_lab.posture import Assessment, PostureMonitor
from posture_lab.types import Pose


def feedback_buttons(width: int, frame_height: int):
    middle = width // 2
    return {
        "FN": (14, frame_height + 8, middle - 7, frame_height + 43),
        "FP": (middle + 7, frame_height + 8, width - 14, frame_height + 43),
    }


def feedback_hit(x: int, y: int, width: int, frame_height: int) -> str | None:
    for label, (left, top, right, bottom) in feedback_buttons(width, frame_height).items():
        if left <= x < right and top <= y < bottom:
            return label
    return None


def can_label(assessment: Assessment, pose: Pose | None, monitor: PostureMonitor, now: float):
    return bool(
        assessment.status in ("good", "pending", "adjust")
        and pose is not None
        and 0 <= now - pose.timestamp_ms / 1000 <= monitor.config.max_gap_seconds
        and monitor.baseline is not None
        and monitor.diagnostic_details()["smoothed_metrics"] is not None
    )


class FeedbackRecorder:
    def __init__(self, path: Path, context: dict):
        self.path = path
        self.context = context
        # Bounded, short history distinguishes a missed detection from dwell/smoothing.
        self.history = deque(maxlen=240)
        self.counts = {"FN": 0, "FP": 0}
        self.message = "N: missed bad posture | F: false alarm (or click below)"

    def observe(self, pose: Pose, assessment: Assessment, monitor: PostureMonitor):
        details = monitor.diagnostic_details()
        self.history.append(
            {
                "timestamp_ms": pose.timestamp_ms,
                "assessment": asdict(assessment),
                "raw_metrics": details["raw_metrics"],
                "smoothed_metrics": details["smoothed_metrics"],
                "active_profile": details["active_profile"],
            }
        )
        while self.history and pose.timestamp_ms - self.history[0]["timestamp_ms"] > 6000:
            self.history.popleft()

    def record(
        self,
        label: str,
        pose: Pose | None,
        assessment: Assessment,
        monitor: PostureMonitor,
        now: float,
        latency_ms: float,
    ) -> dict | None:
        if label not in self.counts:
            raise ValueError("Feedback label must be FN or FP")
        if not can_label(assessment, pose, monitor, now):
            self.message = "Not saved: need a fresh tracked pose and completed calibration"
            return None
        record = {
            "schema_version": 1,
            "id": str(uuid.uuid4()),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "expected_posture": "bad" if label == "FN" else "good",
            "context": self.context,
            "assessment": asdict(assessment),
            "posture_config": asdict(monitor.config),
            "diagnostics": monitor.diagnostic_details(),
            "pose": asdict(pose),
            "inference_latency_ms": latency_ms,
            "pose_age_seconds": now - pose.timestamp_ms / 1000,
            "reminder_wait_seconds": monitor.reminder_wait_seconds(now),
            "recent_history": list(self.history),
        }
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        self.counts[label] += 1
        self.message = f"Saved {label} for review | this session FN {self.counts['FN']} / FP {self.counts['FP']}"
        return record
