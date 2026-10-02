import math
import statistics
from dataclasses import asdict, dataclass

from posture_lab.config import PostureConfig
from posture_lab.types import Pose


@dataclass(frozen=True)
class Metrics:
    head_offset: float
    head_height: float
    shoulder_tilt: float
    shoulder_scale: float


@dataclass(frozen=True)
class Assessment:
    status: str
    message: str
    nudge: bool = False
    bad_seconds: float = 0


def extract_metrics(pose: Pose, confidence: float) -> Metrics | None:
    names = ("nose", "left_shoulder", "right_shoulder")
    if pose.width <= 0 or pose.height <= 0:
        return None
    if any(name not in pose.keypoints for name in names):
        return None
    points = [pose.keypoints[name] for name in names]
    if any(p.confidence < confidence or not 0 <= p.x <= 1 or not 0 <= p.y <= 1 for p in points):
        return None
    nose, left, right = points
    nx, ny = nose.x * pose.width, nose.y * pose.height
    lx, ly = left.x * pose.width, left.y * pose.height
    rx, ry = right.x * pose.width, right.y * pose.height
    width = math.hypot(lx - rx, ly - ry)
    if width < pose.width * 0.08:
        return None
    # x sorting makes the angle invariant to camera mirroring and anatomical labels.
    if lx > rx:
        lx, rx, ly, ry = rx, lx, ry, ly
    return Metrics(
        (nx - (lx + rx) / 2) / width,
        ((ly + ry) / 2 - ny) / width,
        math.degrees(math.atan2(ry - ly, rx - lx)),
        width / pose.width,
    )


class PostureMonitor:
    """Personal baseline, time smoothing, continuous dwell, and reminder cooldown."""

    def __init__(self, config: PostureConfig):
        self.config = config
        self.baseline: Metrics | None = None
        self.profiles: dict[str, Metrics] = {}
        self.profile_selection = "laptop"
        self.active_profile = "laptop"
        self.calibration_target = "laptop"
        self.calibration_revision = 0
        self._calibrating = False
        self._samples: list[tuple[float, Metrics]] = []
        self._smoothed: Metrics | None = None
        self._current_metrics: Metrics | None = None
        self._last_time: float | None = None
        self._bad_since: float | None = None
        self._last_nudge = -math.inf

    def set_profiles(self, profiles: dict[str, Metrics], selection: str = "auto"):
        self.profiles = dict(profiles)
        self.select_profile(selection)

    def select_profile(self, selection: str):
        if selection not in ("auto", "laptop", "left-monitor"):
            raise ValueError("Unknown calibration profile")
        self.profile_selection = selection
        self._calibrating = False
        if selection != "auto":
            self.active_profile = selection
        elif self.active_profile not in self.profiles:
            self.active_profile = next(iter(self.profiles), "laptop")
        self.baseline = self.profiles.get(self.active_profile)
        self.invalidate()

    @property
    def profile_label(self):
        if self._calibrating:
            return f"Calibrating {self.calibration_target}"
        if self.profile_selection == "auto":
            saved = "/".join(self.profiles) if self.profiles else "none saved"
            return f"Auto: {self.active_profile} | saved: {saved}"
        state = "saved" if self.active_profile in self.profiles else "needs C"
        return f"{self.active_profile} ({state})"

    def start_calibration(self, profile: str | None = None):
        target = profile or self.active_profile
        if target not in ("laptop", "left-monitor"):
            raise ValueError("Unknown calibration profile")
        self.calibration_target = target
        self.baseline = None
        self._calibrating = True
        self._samples.clear()
        self.invalidate()

    def invalidate(self):
        """Missing, stale, paused, or failed inference breaks continuous evidence."""
        self._last_time = None
        self._smoothed = None
        self._current_metrics = None
        self._bad_since = None
        self._samples.clear()

    def reminder_wait_seconds(self, now: float) -> int | None:
        if self._bad_since is None or self._calibrating:
            return None
        hold = self.config.bad_seconds - (now - self._bad_since)
        cooldown = self.config.cooldown_seconds - (now - self._last_nudge)
        return math.ceil(max(0, hold, cooldown))

    def update(self, pose: Pose | None, now: float) -> Assessment:
        metrics = extract_metrics(pose, self.config.confidence) if pose else None
        if metrics is None:
            self.invalidate()
            return Assessment("unknown", "Keep your head and both shoulders visible")
        gap = now - self._last_time if self._last_time is not None else 0
        if gap < 0 or gap > self.config.max_gap_seconds:
            self.invalidate()
        self._current_metrics = metrics
        self._last_time = now
        if self._calibrating:
            self._samples.append((now, metrics))
            elapsed = now - self._samples[0][0]
            if elapsed < self.config.calibration_seconds or len(self._samples) < 5:
                return Assessment("calibrating", f"Hold comfortable posture: {elapsed:.1f}s")
            # A moving calibration is misleading; retry the window until stable.
            offsets = [m.head_offset for _, m in self._samples]
            heights = [m.head_height for _, m in self._samples]
            scales = [m.shoulder_scale for _, m in self._samples]
            tilts = [m.shoulder_tilt for _, m in self._samples]
            if (
                statistics.pstdev(offsets) > 0.06
                or statistics.pstdev(heights) > 0.06
                or statistics.pstdev(scales) > 0.02
                or statistics.pstdev(tilts) > 3
            ):
                self._samples.clear()
                return Assessment("calibrating", "Hold still; restarting calibration")
            self.baseline = Metrics(
                *(
                    statistics.median(getattr(m, key) for _, m in self._samples)
                    for key in Metrics.__dataclass_fields__
                )
            )
            self._calibrating = False
            self._samples.clear()
            self._smoothed = metrics
            self.active_profile = self.calibration_target
            self.profiles[self.calibration_target] = self.baseline
            self.calibration_revision += 1
            return Assessment("good", f"Calibrated {self.calibration_target}. Comfortable position")
        candidates = (
            self.profiles
            if self.profile_selection == "auto"
            else {
                name: baseline
                for name, baseline in self.profiles.items()
                if name == self.profile_selection
            }
        )
        if not candidates:
            return Assessment("uncalibrated", "Choose 1 laptop / 2 left monitor, then press C")
        if self._smoothed is None:
            self._smoothed = metrics
        else:
            alpha = 1 - math.exp(-gap / max(self.config.smoothing_seconds, 0.001))
            self._smoothed = Metrics(
                *(
                    old + alpha * (new - old)
                    for old, new in zip(self._smoothed.__dict__.values(), metrics.__dict__.values())
                )
            )
        value = self._smoothed
        # Match one complete baseline; never mix forgiving axes from separate profiles.
        self.active_profile = min(candidates, key=lambda name: self._fit(value, candidates[name]))
        self.baseline = candidates[self.active_profile]
        reasons = self._reasons(value, self.baseline)
        if not reasons:
            self._bad_since = None
            return Assessment("good", f"Comfortable position ({self.active_profile})")
        if self._bad_since is None:
            self._bad_since = now
        duration = now - self._bad_since
        if duration < self.config.bad_seconds:
            return Assessment(
                "pending", "Checking posture: " + ", ".join(reasons), bad_seconds=duration
            )
        nudge = (
            duration >= self.config.bad_seconds
            and now - self._last_nudge >= self.config.cooldown_seconds
        )
        if nudge:
            self._last_nudge = now
        return Assessment("adjust", ", ".join(reasons), nudge, duration)

    @staticmethod
    def deviations(value: Metrics, baseline: Metrics):
        return {
            "head_offset": abs(value.head_offset - baseline.head_offset),
            "head_drop": baseline.head_height - value.head_height,
            "shoulder_tilt_degrees": abs(value.shoulder_tilt - baseline.shoulder_tilt),
            "closer_ratio": value.shoulder_scale / baseline.shoulder_scale - 1,
        }

    def diagnostic_details(self) -> dict:
        return {
            "profile_selection": self.profile_selection,
            "active_profile": self.active_profile,
            "raw_metrics": asdict(self._current_metrics) if self._current_metrics else None,
            "smoothed_metrics": asdict(self._smoothed) if self._smoothed else None,
            "profiles": {name: asdict(baseline) for name, baseline in self.profiles.items()},
            "profile_checks": {
                name: {
                    "raw_deviations": self.deviations(self._current_metrics, baseline),
                    "smoothed_deviations": self.deviations(self._smoothed, baseline),
                    "reasons": self._reasons(self._smoothed, baseline),
                }
                for name, baseline in self.profiles.items()
                if self._current_metrics and self._smoothed and not self._calibrating
            },
        }

    @property
    def diagnostic_line(self) -> str:
        if self.baseline is None or self._smoothed is None or self._calibrating:
            return "Calibrate, then use FN for missed bad posture or FP for a false alarm"
        delta = self.deviations(self._smoothed, self.baseline)
        return (
            f"Side {delta['head_offset']:.2f}/{self.config.head_offset_threshold:.2f} "
            f"Drop {delta['head_drop']:.2f}/{self.config.head_drop_threshold:.2f} "
            f"Tilt {delta['shoulder_tilt_degrees']:.1f}/{self.config.shoulder_tilt_degrees:g} "
            f"Closer {delta['closer_ratio']:.0%}/{self.config.closer_ratio_threshold:.0%}"
        )

    def _fit(self, value: Metrics, baseline: Metrics):
        changes = (
            abs(value.head_offset - baseline.head_offset)
            / max(self.config.head_offset_threshold, 1e-6),
            max(0, baseline.head_height - value.head_height)
            / max(self.config.head_drop_threshold, 1e-6),
            abs(value.shoulder_tilt - baseline.shoulder_tilt)
            / max(self.config.shoulder_tilt_degrees, 1e-6),
            max(0, value.shoulder_scale / baseline.shoulder_scale - 1)
            / max(self.config.closer_ratio_threshold, 1e-6),
        )
        return max(changes), sum(changes)

    def _reasons(self, value: Metrics, baseline: Metrics):
        reasons = []
        if abs(value.head_offset - baseline.head_offset) > self.config.head_offset_threshold:
            reasons.append("head shifted sideways")
        if baseline.head_height - value.head_height > self.config.head_drop_threshold:
            reasons.append("head lowered")
        if abs(value.shoulder_tilt - baseline.shoulder_tilt) > self.config.shoulder_tilt_degrees:
            reasons.append("shoulders tilted")
        if value.shoulder_scale / baseline.shoulder_scale - 1 > self.config.closer_ratio_threshold:
            reasons.append("moved closer to camera")
        return reasons
