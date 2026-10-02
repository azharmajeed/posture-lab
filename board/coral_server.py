"""Standalone Python 3.7+ worker for Mendel Linux on the Coral Dev Board Mini.

The laptop package and MediaPipe are deliberately not installed on the board.
"""

import argparse
import hmac
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

NAMES = (
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
MAX_BYTES = 2 * 1024 * 1024


def letterbox(image, size):
    target_width, target_height = size
    ratio = min(target_width / image.width, target_height / image.height)
    resized_width = max(1, min(target_width, int(round(image.width * ratio))))
    resized_height = max(1, min(target_height, int(round(image.height * ratio))))
    resized = image.resize((resized_width, resized_height), Image.BILINEAR)
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    canvas = Image.new("RGB", size)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, (pad_x, pad_y, resized_width, resized_height)


def decode_pose(raw, output_details, box, size):
    if np.issubdtype(raw.dtype, np.integer):
        scale, zero_point = output_details["quantization"]
        if scale <= 0:
            raise ValueError("Output tensor has no valid quantization scale")
        raw = (raw.astype(np.float32) - zero_point) * scale
    if raw.shape != (1, 1, 17, 3):
        raise ValueError("Expected MoveNet single-pose output [1, 1, 17, 3]")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite model output")
    pad_x, pad_y, resized_width, resized_height = box
    points = {}
    for name, (y, x, score) in zip(NAMES, raw.reshape(17, 3)):
        x = (float(x) * size[0] - pad_x) / resized_width
        y = (float(y) * size[1] - pad_y) / resized_height
        confidence = min(1.0, max(0.0, float(score)))
        if not (0 <= x <= 1 and 0 <= y <= 1):
            confidence = 0.0
        points[name] = {"x": x, "y": y, "confidence": confidence}
    return points


class MoveNetTPU:
    def __init__(self, model, device=None):
        from tflite_runtime.interpreter import Interpreter, load_delegate

        options = {"device": device} if device else {}
        self.interpreter = Interpreter(
            model_path=model,
            experimental_delegates=[load_delegate("libedgetpu.so.1", options)],
        )
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected a single-input, single-output MoveNet model")
        self.input = inputs[0]
        self.output = outputs[0]
        shape = self.input["shape"]
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ValueError("Expected RGB input [1, height, width, 3]")
        if self.input["dtype"] != np.uint8:
            raise ValueError("Use the Coral MoveNet Lightning PTQ model with uint8 image input")
        self.size = (int(shape[2]), int(shape[1]))

    def infer(self, image, timestamp_ms):
        image = image.convert("RGB")
        canvas, box = letterbox(image, self.size)
        self.interpreter.set_tensor(self.input["index"], np.asarray(canvas)[None, ...])
        self.interpreter.invoke()
        points = decode_pose(
            self.interpreter.get_tensor(self.output["index"]), self.output, box, self.size
        )
        return {
            "schema_version": 1,
            "timestamp_ms": timestamp_ms,
            "width": image.width,
            "height": image.height,
            "keypoints": points,
        }


def handler_for(model, token):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def respond(self, status, data):
            payload = json.dumps(data, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self):
            if not self.authorized():
                return self.respond(401, {"error": "unauthorized"})
            if self.path != "/health":
                return self.respond(404, {"error": "unknown route"})
            self.respond(200, {"schema_version": 1, "backend": "coral", "status": "ready"})

        def do_POST(self):
            if not self.authorized():
                return self.respond(401, {"error": "unauthorized"})
            if self.path != "/v1/pose":
                return self.respond(404, {"error": "unknown route"})
            if self.headers.get("Content-Type") != "image/jpeg":
                return self.respond(415, {"error": "expected image/jpeg"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                timestamp_ms = int(self.headers["X-Timestamp-Ms"])
                if not 0 < length <= MAX_BYTES or timestamp_ms < 0:
                    raise ValueError("invalid frame size or timestamp")
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("incomplete frame")
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format != "JPEG" or image.width * image.height > 1920 * 1080:
                        raise ValueError("expected JPEG no larger than 1080p")
                    image.load()
                    result = model.infer(image, timestamp_ms)
            except (ValueError, KeyError, OSError) as exc:
                return self.respond(400, {"error": str(exc)})
            except Exception:
                return self.respond(503, {"error": "inference failed"})
            self.respond(200, result)

        def log_message(self, format, *args):
            # No frames, tokens, or request headers are logged.
            print("%s %s" % (self.address_string(), format % args), flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", help="Optional Edge TPU device selector")
    args = parser.parse_args()
    token = os.environ.get("POSTURE_CORAL_TOKEN", "")
    if not token or not token.isascii():
        parser.error("Set a nonempty ASCII POSTURE_CORAL_TOKEN environment variable")
    model = MoveNetTPU(args.model, args.device)
    server = HTTPServer((args.host, args.port), handler_for(model, token))
    print("Coral inference ready on %s:%d" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
