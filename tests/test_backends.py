import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from posture_lab.backends import create_backend
from posture_lab.backends.coral_remote import CoralRemote
from posture_lab.config import InferenceConfig, load_config

SPEC = importlib.util.spec_from_file_location(
    "coral_server", Path(__file__).parents[1] / "board" / "coral_server.py"
)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class FixtureModel:
    def infer(self, image, timestamp_ms):
        assert image.size == (640, 480)
        return {
            "schema_version": 1,
            "timestamp_ms": timestamp_ms,
            "width": image.width,
            "height": image.height,
            "keypoints": {name: {"x": 0.5, "y": 0.5, "confidence": 0.9} for name in worker.NAMES},
        }


@pytest.fixture
def server():
    instance = HTTPServer(("127.0.0.1", 0), worker.handler_for(FixtureModel(), "test-token"))
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_port}"
    instance.shutdown()
    thread.join()
    instance.server_close()


def test_laptop_to_board_protocol_round_trip(server, monkeypatch):
    monkeypatch.setenv("POSTURE_CORAL_TOKEN", "test-token")
    backend = CoralRemote(InferenceConfig(backend="coral", endpoint=server))
    result = backend.infer(np.zeros((480, 640, 3), np.uint8), 1234)
    assert result.timestamp_ms == 1234
    assert result.width == 640 and result.height == 480
    assert len(result.keypoints) == 17
    assert result.keypoints["nose"].confidence == 0.9


def test_board_requires_authentication(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(server + "/health")
    assert error.value.code == 401


def test_remote_requires_token(monkeypatch):
    monkeypatch.delenv("POSTURE_CORAL_TOKEN", raising=False)
    with pytest.raises(ValueError, match="POSTURE_CORAL_TOKEN"):
        create_backend(InferenceConfig(backend="coral"))


def test_backend_selection_does_not_silently_fallback():
    with pytest.raises(ValueError, match="Unsupported"):
        create_backend(InferenceConfig(backend="unknown"))


def test_letterbox_coordinates_return_to_original_frame():
    image = Image.new("RGB", (640, 480))
    canvas, box = worker.letterbox(image, (192, 192))
    assert canvas.size == (192, 192)
    assert box == (0, 24, 192, 144)
    raw = np.tile([0.3125, 0.5, 0.9], (1, 1, 17, 1)).astype(np.float32)
    points = worker.decode_pose(raw, {}, box, (192, 192))
    assert points["nose"]["x"] == pytest.approx(0.5)
    assert points["nose"]["y"] == pytest.approx(0.25)


def test_quantized_output_dequantized_and_padding_rejected():
    raw = np.tile([0, 128, 230], (1, 1, 17, 1)).astype(np.uint8)
    points = worker.decode_pose(raw, {"quantization": (1 / 255, 0)}, (0, 24, 192, 144), (192, 192))
    assert points["nose"]["confidence"] == 0
    assert points["nose"]["y"] < 0


def test_bad_config_is_rejected(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"camera": {"inference_fps": 0}}))
    with pytest.raises(ValueError, match="inference_fps"):
        load_config(path)
    path.write_text(json.dumps({"inference": {"backnd": "cpu"}}))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
