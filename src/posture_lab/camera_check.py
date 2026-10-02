"""Camera diagnostics without a pose model or saved camera images."""

import math
import sys
import time

import cv2


def camera_api(name: str) -> int:
    if name == "auto":
        return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "avfoundation": cv2.CAP_AVFOUNDATION,
    }[name]


def check_camera(index: int = 0, seconds: float = 3, api: str = "auto", preview: bool = False):
    if index < 0 or not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("Use a nonnegative camera index and a duration in (0, 60] seconds")
    capture = cv2.VideoCapture(index, camera_api(api))
    frames = 0
    failures = 0
    brightness = 0.0
    detail = 0.0
    width = height = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Camera {index} could not open ({api}). Close other camera apps, check "
                "Windows camera privacy settings, or try --api msmf / --camera 1."
            )
        print(f"Camera {index} opened using {capture.getBackendName()}.", flush=True)
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                time.sleep(0.02)
                continue
            frames += 1
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness += float(gray.mean())
            detail += float(gray.std())
            if preview:
                cv2.imshow("Posture Lab - camera diagnostic (Q to close)", cv2.flip(frame, 1))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if (
                    cv2.getWindowProperty(
                        "Posture Lab - camera diagnostic (Q to close)", cv2.WND_PROP_VISIBLE
                    )
                    < 1
                ):
                    break
        elapsed = time.monotonic() - started
        if not frames:
            raise RuntimeError("Camera opened but delivered no readable frames")
        average_brightness = brightness / frames
        average_detail = detail / frames
        warning = None
        if average_brightness < 10 or average_detail < 2:
            warning = "Frames are very dark or flat. Check the privacy shutter and lighting."
        return {
            "camera": index,
            "api": capture.getBackendName(),
            "frames_read": frames,
            "failed_reads": failures,
            "width": width,
            "height": height,
            "elapsed_seconds": round(elapsed, 2),
            "capture_fps": round(frames / max(elapsed, 0.001), 1),
            "mean_brightness_0_to_255": round(average_brightness, 1),
            "image_detail_stddev": round(average_detail, 1),
            "content_warning": warning,
            "images_saved": False,
        }
    finally:
        capture.release()
        if preview:
            cv2.destroyAllWindows()
