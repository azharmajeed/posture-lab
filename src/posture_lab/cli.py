import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

from posture_lab.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Local webcam posture reminders")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Open the webcam posture monitor")
    run.add_argument("--config", type=Path)
    run.add_argument("--backend", choices=("cpu", "coral"))
    run.add_argument("--camera", type=int)
    run.add_argument("--model", type=Path)
    run.add_argument("--endpoint")
    run.add_argument("--no-notifications", action="store_true")
    run.add_argument("--reminder", choices=("notification", "voice", "roast"))
    run.add_argument("--tts-engine", choices=("piper", "system"))
    run.add_argument("--profile", choices=("auto", "laptop", "left-monitor"))
    run.add_argument("--duration", type=float, help="Stop after N seconds (optional smoke check)")
    run.add_argument("--headless", action="store_true", help="No preview; calibrate automatically")
    download = commands.add_parser("download-model", help="Fetch a model from its official source")
    download.add_argument("--backend", choices=("cpu", "coral"), default="cpu")
    download.add_argument("--directory", type=Path, default=Path("models"))
    check = commands.add_parser("check", help="Check inference without opening the webcam")
    check.add_argument("--config", type=Path)
    check.add_argument("--backend", choices=("cpu", "coral"))
    check.add_argument("--image", type=Path, help="Check a still image instead of a blank frame")
    camera = commands.add_parser("camera-check", help="Test webcam access without a pose model")
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--seconds", type=float, default=3)
    camera.add_argument("--api", choices=("auto", "dshow", "msmf", "avfoundation"), default="auto")
    camera.add_argument("--preview", action="store_true", help="Show a temporary camera preview")
    voices = commands.add_parser("download-voices", help="Download local English/Hindi TTS models")
    voices.add_argument("--directory", type=Path, default=Path("models/voices"))
    voices.add_argument("--language", choices=("both", "en", "hi"), default="both")
    prepare = commands.add_parser(
        "prepare-voices", help="Cache reminder audio before opening the camera"
    )
    prepare.add_argument("--config", type=Path)
    speech = commands.add_parser("voice-test", help="Test speech without using the webcam")
    speech.add_argument("--config", type=Path)
    speech.add_argument("--mode", choices=("voice", "roast"), default="voice")
    speech.add_argument("--tts-engine", choices=("piper", "system"))
    speech.add_argument("--text", help="Optional custom phrase (up to 300 characters)")
    speech.add_argument("--output", type=Path, help="Write a WAV instead of playing (Piper only)")
    args = parser.parse_args()
    try:
        if args.command == "prepare-voices":
            from posture_lab.reminders import GENTLE_ROASTS, SPICY_ROASTS, VOICE_LINE
            from posture_lab.speech import PiperSpeaker

            config = load_config(args.config)
            speaker = PiperSpeaker(config.reminders)
            started = time.monotonic()
            phrases = [(VOICE_LINE, "en")] + [
                (text, "hi") for text in (*SPICY_ROASTS, *GENTLE_ROASTS)
            ]
            for index, (text, language) in enumerate(phrases, 1):
                speaker.render(text, language)
                print(f"Prepared reminder {index}/{len(phrases)} ({language})", flush=True)
            print(
                f"Voice cache ready in {time.monotonic() - started:.2f}s: "
                f"{Path(config.reminders.cache_directory).resolve()}"
            )
            return 0
        if args.command == "download-voices":
            from posture_lab.speech import download_voices

            for model in download_voices(args.directory, args.language):
                print(model.resolve())
            return 0
        if args.command == "voice-test":
            from posture_lab.reminders import GENTLE_ROASTS, SPICY_ROASTS, VOICE_LINE
            from posture_lab.speech import create_speaker

            config = load_config(args.config)
            if args.tts_engine:
                config.reminders.tts_engine = args.tts_engine
            if args.output and config.reminders.tts_engine != "piper":
                raise ValueError("WAV output requires the Piper engine")
            language = "hi" if args.mode == "roast" else "en"
            bank = SPICY_ROASTS if config.reminders.roast_intensity == "spicy" else GENTLE_ROASTS
            text = (
                args.text
                if args.text is not None
                else (bank[0] if language == "hi" else VOICE_LINE)
            )
            if not 0 < len(text.strip()) <= 300:
                raise ValueError("Speech text must contain 1 to 300 characters")
            speaker = create_speaker(config.reminders)
            try:
                audio = speaker.render(text, language)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_bytes(audio)
                    print(f"Speech WAV written: {args.output.resolve()}")
                else:
                    speaker.start(audio, language)
                    speaker.wait()
                    print(f"Speech playback completed ({language}, {config.reminders.tts_engine}).")
            finally:
                speaker.stop()
            return 0
        if args.command == "camera-check":
            from posture_lab.camera_check import check_camera

            report = check_camera(args.camera, args.seconds, args.api, args.preview)
            print(json.dumps(report, indent=2))
            print("PASS: camera frames received; camera released.")
            return 0
        if args.command == "download-model":
            from posture_lab.models import download_model

            print(download_model(args.backend, args.directory).resolve())
            return 0
        config = load_config(args.config)
        if args.backend:
            config.inference.backend = args.backend
        if args.command == "run":
            if args.camera is not None:
                config.camera.index = args.camera
            if args.model:
                config.inference.model_path = str(args.model)
            if args.endpoint:
                config.inference.endpoint = args.endpoint
            if args.no_notifications:
                config.notifications = False
            if args.reminder:
                config.reminders.mode = args.reminder
            if args.tts_engine:
                config.reminders.tts_engine = args.tts_engine
            if args.profile:
                config.calibration.profile = args.profile
            if args.duration is not None and (
                not math.isfinite(args.duration) or args.duration <= 0
            ):
                raise ValueError("duration must be positive")
            config.validate()
            from posture_lab.app import run

            run(config, duration=args.duration, headless=args.headless)
        else:
            import numpy as np

            from posture_lab.backends import create_backend

            backend = create_backend(config.inference)
            try:
                frame = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)
                if args.image:
                    import cv2

                    frame = cv2.imread(str(args.image))
                    if frame is None:
                        raise ValueError(f"Cannot read image: {args.image}")
                pose = backend.infer(frame, 1)
                print(f"OK: {backend.name} on {platform.system()} {platform.machine()}")
                source = "Image" if args.image else "Blank-frame"
                print(f"{source} inference returned {len(pose.keypoints)} keypoints")
            finally:
                backend.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
