"""Atomic storage of named calibration metrics; no frames or photos are saved."""

import hashlib
import json
import math
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from posture_lab.config import Config
from posture_lab.posture import Metrics

PROFILE_NAMES = ("laptop", "left-monitor")


def calibration_context(config: Config) -> dict:
    if config.inference.backend == "cpu":
        model = Path(config.inference.model_path)
        model_id = hashlib.sha256(model.read_bytes()).hexdigest()
    else:
        # The remote model is managed on the board; changing it needs recalibration.
        model_id = config.inference.endpoint.rstrip("/")
    return {
        "backend": config.inference.backend,
        "model_id": model_id,
        "camera_index": config.camera.index,
        "width": config.camera.width,
        "height": config.camera.height,
    }


def decode_metrics(value) -> Metrics:
    if not isinstance(value, dict) or value.keys() != Metrics.__dataclass_fields__.keys():
        raise ValueError("Calibration has missing or unknown metric fields")
    if any(
        type(number) not in (int, float) or not math.isfinite(number) for number in value.values()
    ):
        raise ValueError("Calibration metrics must be finite numbers")
    metrics = Metrics(**value)
    if metrics.shoulder_scale <= 0 or not -90 <= metrics.shoulder_tilt <= 90:
        raise ValueError("Calibration contains invalid shoulder geometry")
    return metrics


class CalibrationStore:
    def __init__(self, path: Path, context: dict):
        self.path = path
        self.context = context
        encoded = json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.key = hashlib.sha256(encoded).hexdigest()

    def _read(self):
        if not self.path.is_file():
            return {"schema_version": 1, "contexts": {}}
        try:
            if self.path.stat().st_size > 1024 * 1024:
                raise ValueError("Calibration file exceeds the size limit")
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                not isinstance(data, dict)
                or data.get("schema_version") != 1
                or not isinstance(data.get("contexts"), dict)
            ):
                raise ValueError("Unsupported calibration file format")
            return data
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot read calibration file {self.path}: {exc}") from exc

    def load(self) -> dict[str, Metrics]:
        slot = self._read()["contexts"].get(self.key)
        if slot is None:
            return {}
        if (
            not isinstance(slot, dict)
            or slot.get("context") != self.context
            or not isinstance(slot.get("profiles"), dict)
        ):
            raise ValueError("Calibration context record is invalid")
        profiles = {}
        for name, record in slot["profiles"].items():
            if name not in PROFILE_NAMES or not isinstance(record, dict):
                raise ValueError("Unknown calibration profile")
            profiles[name] = decode_metrics(record.get("metrics"))
        return profiles

    def save(self, profiles: dict[str, Metrics]):
        # Re-read to preserve calibrations for other inference/camera setups.
        data = self._read()
        existing = data["contexts"].get(self.key, {}).get("profiles", {})
        records = {}
        for name, metrics in profiles.items():
            if name not in PROFILE_NAMES:
                raise ValueError("Unknown calibration profile")
            values = asdict(metrics)
            decode_metrics(values)
            previous = existing.get(name, {})
            timestamp = previous.get("saved_at_utc") if previous.get("metrics") == values else None
            records[name] = {
                "metrics": values,
                "saved_at_utc": timestamp or datetime.now(timezone.utc).isoformat(),
            }
        data["contexts"][self.key] = {"context": self.context, "profiles": records}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, suffix=".tmp", delete=False
            ) as output:
                temporary = Path(output.name)
                json.dump(data, output, indent=2, allow_nan=False)
                output.write("\n")
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
