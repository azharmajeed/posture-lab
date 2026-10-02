from posture_lab.config import InferenceConfig
from posture_lab.types import PoseBackend


def create_backend(config: InferenceConfig) -> PoseBackend:
    """Only the selected runtime is imported; remote users do not need MediaPipe."""
    if config.backend == "cpu":
        from .mediapipe_cpu import MediaPipeCPU

        return MediaPipeCPU(config)
    if config.backend == "coral":
        from .coral_remote import CoralRemote

        return CoralRemote(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
