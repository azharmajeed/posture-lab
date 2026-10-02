from pathlib import Path

import cv2
import mediapipe as mp

from posture_lab.config import InferenceConfig
from posture_lab.types import KEYPOINT_NAMES, Keypoint, Pose

MEDIAPIPE_TO_COCO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


class MediaPipeCPU:
    name = "MediaPipe / CPU"

    def __init__(self, config: InferenceConfig):
        if not Path(config.model_path).is_file():
            raise FileNotFoundError(
                f"Missing model: {config.model_path}. Run posture-lab download-model first."
            )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=config.model_path,
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._model = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp = -1

    def infer(self, frame_bgr, timestamp_ms: int) -> Pose:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(timestamp_ms, self._last_timestamp + 1)
        self._last_timestamp = timestamp_ms
        result = self._model.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        points = {}
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            for name, index in zip(KEYPOINT_NAMES, MEDIAPIPE_TO_COCO):
                point = landmarks[index]
                points[name] = Keypoint(point.x, point.y, min(point.visibility, point.presence))
        height, width = frame_bgr.shape[:2]
        return Pose(points, timestamp_ms, width, height)

    def close(self):
        self._model.close()
