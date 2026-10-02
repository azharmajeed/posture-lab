import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class InferenceConfig:
    backend: str = "cpu"
    model_path: str = "models/pose_landmarker_lite.task"
    endpoint: str = "http://127.0.0.1:8765"
    token_env: str = "POSTURE_CORAL_TOKEN"
    timeout_seconds: float = 2.0
    jpeg_quality: int = 80


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    inference_fps: float = 10


@dataclass
class PostureConfig:
    confidence: float = 0.6
    calibration_seconds: float = 3
    bad_seconds: float = 5
    cooldown_seconds: float = 30
    max_gap_seconds: float = 1
    smoothing_seconds: float = 0.5
    head_offset_threshold: float = 0.20
    head_drop_threshold: float = 0.18
    shoulder_tilt_degrees: float = 10
    closer_ratio_threshold: float = 0.25


@dataclass
class ReminderConfig:
    mode: str = "notification"
    tts_engine: str = "piper"
    english_model: str = "models/voices/en_US-lessac-medium.onnx"
    hindi_model: str = "models/voices/hi_IN-pratham-medium.onnx"
    cache_directory: str = ".state/voice-cache"
    volume: float = 0.7
    speed: float = 1.0
    roast_intensity: str = "spicy"
    muted: bool = False
    also_notify: bool = False


@dataclass
class CalibrationConfig:
    path: str = ".state/calibrations.json"
    profile: str = "auto"


@dataclass
class FeedbackConfig:
    path: str = ".state/feedback.jsonl"


@dataclass
class Config:
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    reminders: ReminderConfig = field(default_factory=ReminderConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    notifications: bool = True

    def validate(self):
        if self.inference.backend not in ("cpu", "coral"):
            raise ValueError("backend must be cpu or coral")
        for section in (
            self.camera,
            self.posture,
            self.inference,
            self.reminders,
            self.calibration,
            self.feedback,
        ):
            for item in fields(section):
                value = getattr(section, item.name)
                if isinstance(item.default, bool):
                    if type(value) is not bool:
                        raise ValueError(f"{item.name} must be true or false")
                elif isinstance(item.default, (int, float)):
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"{item.name} must be a finite, nonnegative number")
                elif isinstance(item.default, str) and not isinstance(value, str):
                    raise ValueError(f"{item.name} must be a string")
        for name in ("index", "width", "height"):
            if type(getattr(self.camera, name)) is not int:
                raise ValueError(f"camera {name} must be an integer")
        if self.camera.width < 64 or self.camera.height < 64:
            raise ValueError("Camera dimensions must be at least 64")
        if not 0 < self.camera.inference_fps <= 60:
            raise ValueError("inference_fps must be in (0, 60]")
        if not 0 < self.posture.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if not 0 < self.inference.timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be in (0, 30]")
        if (
            type(self.inference.jpeg_quality) is not int
            or not 1 <= self.inference.jpeg_quality <= 100
        ):
            raise ValueError("jpeg_quality must be an integer in [1, 100]")
        if self.posture.calibration_seconds <= 0 or self.posture.max_gap_seconds <= 0:
            raise ValueError("calibration_seconds and max_gap_seconds must be positive")
        if type(self.notifications) is not bool:
            raise ValueError("notifications must be true or false")
        if self.reminders.mode not in ("notification", "voice", "roast"):
            raise ValueError("reminder mode must be notification, voice, or roast")
        if self.reminders.tts_engine not in ("piper", "system"):
            raise ValueError("tts_engine must be piper or system")
        if self.reminders.roast_intensity not in ("gentle", "spicy"):
            raise ValueError("roast_intensity must be gentle or spicy")
        if not 0 <= self.reminders.volume <= 1:
            raise ValueError("reminder volume must be in [0, 1]")
        if not 0.5 <= self.reminders.speed <= 2:
            raise ValueError("reminder speed must be in [0.5, 2]")
        if self.calibration.profile not in ("auto", "laptop", "left-monitor"):
            raise ValueError("calibration profile must be auto, laptop, or left-monitor")
        if not self.calibration.path.strip():
            raise ValueError("calibration path must not be empty")
        if not self.reminders.cache_directory.strip():
            raise ValueError("voice cache directory must not be empty")
        if not self.feedback.path.strip():
            raise ValueError("feedback path must not be empty")


def _section(cls, values):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = values.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return cls(**values)


def load_config(path: Path | None = None) -> Config:
    data = json.loads(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be an object")
    if data.keys() - {
        "inference",
        "camera",
        "posture",
        "reminders",
        "calibration",
        "feedback",
        "notifications",
    }:
        raise ValueError("Unknown configuration section")
    config = Config(
        inference=_section(InferenceConfig, data.get("inference", {})),
        camera=_section(CameraConfig, data.get("camera", {})),
        posture=_section(PostureConfig, data.get("posture", {})),
        reminders=_section(ReminderConfig, data.get("reminders", {})),
        calibration=_section(CalibrationConfig, data.get("calibration", {})),
        feedback=_section(FeedbackConfig, data.get("feedback", {})),
        notifications=data.get("notifications", True),
    )
    config.validate()
    return config
