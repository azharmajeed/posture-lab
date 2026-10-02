import urllib.request
from pathlib import Path

MODELS = {
    "cpu": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    "coral": (
        "movenet_single_pose_lightning_ptq_edgetpu.tflite",
        "https://raw.githubusercontent.com/google-coral/test_data/master/"
        "movenet_single_pose_lightning_ptq_edgetpu.tflite",
    ),
}


def download_model(backend: str, directory: Path) -> Path:
    filename, url = MODELS[backend]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.is_file():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size < 1024:
            raise RuntimeError("Downloaded model is unexpectedly small")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
