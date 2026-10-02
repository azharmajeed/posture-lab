import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from posture_lab.backends import create_backend
from posture_lab.calibration import CalibrationStore, calibration_context
from posture_lab.config import Config
from posture_lab.feedback import FeedbackRecorder, can_label, feedback_buttons, feedback_hit
from posture_lab.posture import Assessment, PostureMonitor
from posture_lab.reminders import ReminderController

CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
)


def draw_preview(
    frame,
    pose,
    assessment,
    backend_name,
    latency_ms,
    banner,
    confidence,
    reminder_status="Notify",
    profile_status="",
    reminder_wait=None,
    feedback_enabled=False,
    feedback_message="",
    diagnostic_line="",
):
    height, width = frame.shape[:2]
    # Mirror only for display; inference always sees the original frame.
    canvas = cv2.flip(frame, 1)
    if pose:
        pixels = {
            name: (int((1 - point.x) * width), int(point.y * height))
            for name, point in pose.keypoints.items()
            if point.confidence >= confidence and 0 <= point.x <= 1 and 0 <= point.y <= 1
        }
        for left, right in CONNECTIONS:
            if left in pixels and right in pixels:
                cv2.line(canvas, pixels[left], pixels[right], (200, 220, 80), 2)
        for point in pixels.values():
            cv2.circle(canvas, point, 4, (180, 240, 110), -1)
    colors = {"good": (130, 220, 110), "adjust": (90, 170, 255), "unknown": (120, 170, 230)}
    cv2.rectangle(canvas, (0, 0), (width, 130), (25, 28, 31), -1)
    cv2.putText(
        canvas,
        f"Posture Lab | {backend_name} | {reminder_status}",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
    )
    cv2.putText(
        canvas,
        assessment.status.upper(),
        (14, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colors.get(assessment.status, (230, 210, 120)),
        2,
    )
    message = assessment.message
    # Wrap once to fit typical 640px preview without hiding long reasons.
    lines = [message[:70], message[70:140]]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line, (14, 74 + 18 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1
        )
    cv2.putText(
        canvas, profile_status[:88], (14, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 215, 210), 1
    )
    cv2.rectangle(canvas, (0, height - 95), (width, height), (25, 28, 31), -1)
    cv2.putText(
        canvas,
        "C calibrate | P pause | Q quit",
        (14, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (225, 225, 225),
        1,
    )
    cv2.putText(
        canvas,
        "1 laptop | 2 left monitor | A auto profiles",
        (14, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (225, 225, 225),
        1,
    )
    cv2.putText(
        canvas,
        "M reminder mode | T test reminder | V mute speech",
        (14, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (225, 225, 225),
        1,
    )
    cv2.putText(
        canvas,
        f"Inference {latency_ms:.0f}ms | changed {assessment.bad_seconds:.0f}s"
        + (f" | next nudge {reminder_wait}s" if reminder_wait is not None else ""),
        (14, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (180, 180, 180),
        1,
    )
    if banner:
        cv2.rectangle(canvas, (0, 137), (width, 180), (40, 75, 160), -1)
        cv2.putText(
            canvas,
            "Time to adjust your position",
            (14, 167),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    canvas = cv2.copyMakeBorder(canvas, 0, 90, 0, 0, cv2.BORDER_CONSTANT, value=(25, 28, 31))
    labels = {"FN": "FN: missed bad posture [N]", "FP": "FP: false alarm [F]"}
    for label, (left, top, right, bottom) in feedback_buttons(width, height).items():
        color = (65, 85, 140) if label == "FN" else (120, 85, 55)
        cv2.rectangle(
            canvas, (left, top), (right, bottom), color if feedback_enabled else (55, 55, 55), -1
        )
        cv2.putText(
            canvas,
            labels[label],
            (left + 8, top + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245) if feedback_enabled else (150, 150, 150),
            1,
        )
    cv2.putText(
        canvas,
        feedback_message[:88],
        (14, height + 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (210, 215, 215),
        1,
    )
    cv2.putText(
        canvas,
        diagnostic_line[:88],
        (14, height + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (170, 215, 210),
        1,
    )
    return canvas


def run(config: Config, *, duration: float | None = None, headless: bool = False):
    store = CalibrationStore(Path(config.calibration.path), calibration_context(config))
    profiles = store.load()
    monitor = PostureMonitor(config.posture)
    monitor.set_profiles(profiles, config.calibration.profile)
    feedback = FeedbackRecorder(Path(config.feedback.path), store.context)
    feedback_clicks = queue.SimpleQueue()

    def on_mouse(event, x, y, flags, frame_size):
        if event == cv2.EVENT_LBUTTONDOWN:
            label = feedback_hit(x, y, *frame_size)
            if label:
                feedback_clicks.put(label)

    saved_revision = 0
    backend = create_backend(config.inference)
    camera = (
        cv2.VideoCapture(config.camera.index, cv2.CAP_DSHOW)
        if sys.platform == "win32"
        else cv2.VideoCapture(config.camera.index)
    )
    if not camera.isOpened():
        camera.release()
        backend.close()
        raise RuntimeError("Cannot open webcam. Check camera permissions and camera index.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    reminders = ReminderController(config.reminders, config.notifications)
    if headless and monitor.baseline is None:
        monitor.start_calibration()
    assessment = Assessment("uncalibrated", "Choose 1 laptop / 2 left monitor, then press C")
    if profiles:
        print(f"Restored calibration profiles: {', '.join(profiles)}", flush=True)
        assessment = Assessment("ready", "Saved calibrations restored; waiting for camera")
    pending = None
    pending_epoch = 0
    epoch = 0
    pose = None
    latency_ms = 0
    latest_result = -float("inf")
    next_inference = 0
    banner_until = 0
    paused = False
    started = time.monotonic()
    last_error = None
    captured_frames = 0
    completed_inferences = 0
    detected_poses = 0
    total_latency_ms = 0.0
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    print(f"Backend: {backend.name}. Camera open. C: calibrate, P: pause, Q: quit.", flush=True)
    try:
        while True:
            now = time.monotonic()
            if duration is not None and now - started >= duration:
                break
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam stopped delivering frames")
            captured_frames += 1
            if pending is not None and pending.done():
                try:
                    result = pending.result()
                    latency_ms = (now - result.timestamp_ms / 1000) * 1000
                    completed_inferences += 1
                    total_latency_ms += latency_ms
                    if result.keypoints:
                        detected_poses += 1
                        if detected_poses == 1:
                            print(
                                "Person detected. C calibrates; 1/2 select profile; A accepts both.",
                                flush=True,
                            )
                    if (
                        not paused
                        and pending_epoch == epoch
                        and latency_ms / 1000 <= config.posture.max_gap_seconds
                    ):
                        pose = result
                        latest_result = now
                        previous_status = assessment.status
                        assessment = monitor.update(pose, result.timestamp_ms / 1000)
                        feedback.observe(pose, assessment, monitor)
                        if monitor.calibration_revision != saved_revision:
                            saved_revision = monitor.calibration_revision
                            try:
                                store.save(monitor.profiles)
                                print(
                                    f"Saved {monitor.active_profile} calibration to {store.path}",
                                    flush=True,
                                )
                                assessment = Assessment(
                                    "good", f"Saved {monitor.active_profile} calibration"
                                )
                            except (ValueError, OSError) as exc:
                                print(f"Calibration save failed: {exc}", flush=True)
                                assessment = Assessment(
                                    "good", "Calibration works; SAVE FAILED, see terminal"
                                )
                        if previous_status == "adjust" and assessment.status != "adjust":
                            reminders.cancel()
                        last_error = None
                        if assessment.nudge:
                            banner_until = now + 6
                            reminders.remind(assessment.message)
                    else:
                        pose = None
                        monitor.invalidate()
                        reminders.cancel()
                        assessment = Assessment("unknown", "Inference too slow or result discarded")
                except Exception as exc:
                    pose = None
                    monitor.invalidate()
                    reminders.cancel()
                    assessment = Assessment("unknown", "Inference unavailable; see terminal")
                    error = str(exc)
                    if error != last_error:
                        print(f"Inference error: {error}", flush=True)
                        last_error = error
                pending = None
            if paused:
                pose = None
                assessment = Assessment("paused", "Press P to resume")
            elif now - latest_result > config.posture.max_gap_seconds:
                pose = None
                monitor.invalidate()
                reminders.cancel()
                if assessment.status not in ("uncalibrated", "unknown"):
                    assessment = Assessment("unknown", "Waiting for fresh pose")
            if not paused and pending is None and now >= next_inference:
                timestamp_ms = int(now * 1000)
                pending_epoch = epoch
                pending = executor.submit(backend.infer, frame.copy(), timestamp_ms)
                next_inference = now + 1 / config.camera.inference_fps
            if not headless:
                if captured_frames == 1:
                    cv2.namedWindow("Posture Lab", cv2.WINDOW_AUTOSIZE)
                    cv2.setMouseCallback("Posture Lab", on_mouse, (frame.shape[1], frame.shape[0]))
                cv2.imshow(
                    "Posture Lab",
                    draw_preview(
                        frame,
                        pose,
                        assessment,
                        backend.name,
                        latency_ms,
                        now < banner_until,
                        config.posture.confidence,
                        reminders.display_status,
                        monitor.profile_label,
                        monitor.reminder_wait_seconds(now),
                        can_label(assessment, pose, monitor, now),
                        feedback.message,
                        monitor.diagnostic_line,
                    ),
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("n"), ord("f")):
                    feedback_clicks.put("FN" if key == ord("n") else "FP")
                while not feedback_clicks.empty():
                    label = feedback_clicks.get()
                    try:
                        recorded = feedback.record(
                            label, pose, assessment, monitor, time.monotonic(), latency_ms
                        )
                        if recorded:
                            print(
                                f"Saved {label} feedback to {feedback.path} ({recorded['id']})",
                                flush=True,
                            )
                    except (OSError, ValueError) as exc:
                        feedback.message = (
                            f"Feedback save failed: {type(exc).__name__}; see terminal"
                        )
                        print(f"Feedback save failed: {exc}", flush=True)
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("Posture Lab", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("c") and not paused:
                    epoch += 1
                    pose = None
                    banner_until = 0
                    monitor.start_calibration()
                    feedback.history.clear()
                    reminders.cancel()
                    assessment = Assessment(
                        "calibrating",
                        f"Hold {monitor.calibration_target} position "
                        f"for {config.posture.calibration_seconds:g}s",
                    )
                if key in (ord("1"), ord("2"), ord("a")) and not paused:
                    selection = {ord("1"): "laptop", ord("2"): "left-monitor", ord("a"): "auto"}[
                        key
                    ]
                    epoch += 1
                    pose = None
                    banner_until = 0
                    monitor.select_profile(selection)
                    feedback.history.clear()
                    reminders.cancel()
                    assessment = Assessment(
                        "ready", f"Profile: {selection}; C to calibrate selected position"
                    )
                if key == ord("p"):
                    paused = not paused
                    epoch += 1
                    monitor.invalidate()
                    feedback.history.clear()
                    reminders.cancel()
                    banner_until = 0
                if key == ord("m"):
                    reminders.cycle_mode()
                if key == ord("v"):
                    reminders.toggle_mute()
                if key == ord("t") and not paused:
                    if reminders.remind("Test posture reminder"):
                        banner_until = now + 6
            else:
                time.sleep(0.01)
    finally:
        reminders.close()
        camera.release()
        if not headless:
            cv2.destroyAllWindows()
        executor.shutdown(wait=True, cancel_futures=True)
        backend.close()
    print("Camera released; session ended.")
    if completed_inferences:
        print(
            f"Session: {captured_frames} camera frames, {completed_inferences} inferences, "
            f"{detected_poses} detected poses, "
            f"{total_latency_ms / completed_inferences:.0f}ms mean result latency."
        )
