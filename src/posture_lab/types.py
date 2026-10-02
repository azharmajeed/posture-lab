import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.confidence)):
            raise ValueError("Keypoint values must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Keypoint confidence must be between 0 and 1")


@dataclass(frozen=True)
class Pose:
    """COCO-17 keypoints; x/y relative to original frame, confidence in [0, 1]."""

    keypoints: dict[str, Keypoint]
    timestamp_ms: int
    width: int
    height: int


class PoseBackend(Protocol):
    name: str

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Pose: ...

    def close(self) -> None: ...
