import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse

import cv2

from posture_lab.config import InferenceConfig
from posture_lab.types import KEYPOINT_NAMES, Keypoint, Pose


class CoralRemote:
    name = "Coral Dev Board Mini / remote Edge TPU"

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.endpoint = config.endpoint.rstrip("/")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Coral endpoint must be an HTTP(S) URL")
        self.token = os.environ.get(config.token_env, "")
        if not self.token:
            raise ValueError(f"Set {config.token_env} to the board service token first")
        # Do not send camera frames through environment-configured HTTP proxies.
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def infer(self, frame_bgr, timestamp_ms: int) -> Pose:
        ok, encoded = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("Could not encode frame")
        request = urllib.request.Request(
            self.endpoint + "/v1/pose",
            data=encoded.tobytes(),
            headers={
                "Content-Type": "image/jpeg",
                "Authorization": "Bearer " + self.token,
                "X-Timestamp-Ms": str(timestamp_ms),
            },
            method="POST",
        )
        try:
            with self._http.open(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError("Coral response exceeded size limit")
            data = json.loads(raw)
            height, width = frame_bgr.shape[:2]
            if (
                data["schema_version"] != 1
                or data["timestamp_ms"] != timestamp_ms
                or data["width"] != width
                or data["height"] != height
            ):
                raise ValueError("Coral returned an incompatible frame or schema")
            points = data["keypoints"]
            if not isinstance(points, dict) or points.keys() - set(KEYPOINT_NAMES):
                raise ValueError("Coral returned unknown keypoints")
            return Pose(
                {name: Keypoint(**value) for name, value in points.items()},
                timestamp_ms,
                width,
                height,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Coral service unavailable; check the board and SSH tunnel") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid pose response from Coral service") from exc

    def close(self):
        pass
