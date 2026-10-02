import json
import threading
import time

import pytest

from posture_lab.config import ReminderConfig, load_config
from posture_lab.reminders import ReminderController


class FakeSpeaker:
    def __init__(self, block=False, fail=False):
        self.rendering = threading.Event()
        self.release_render = threading.Event()
        if not block:
            self.release_render.set()
        self.started = []
        self.stopped = 0
        self.fail = fail

    def render(self, text, language):
        self.rendering.set()
        if not self.release_render.wait(2):
            raise RuntimeError("Test render timeout")
        if self.fail:
            raise RuntimeError("Missing audio device")
        return (text, language)

    def start(self, audio, language):
        self.started.append((audio, language))

    def wait(self):
        pass

    def stop(self):
        self.stopped += 1


def wait_until(condition):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    pytest.fail("Speech worker did not reach expected state")


def test_notification_mode_does_not_start_speech():
    calls = []
    speaker = FakeSpeaker()
    controller = ReminderController(ReminderConfig(), speaker=speaker, notifier=calls.append)
    try:
        assert controller.remind("head lowered")
        assert calls == ["head lowered"]
        assert not speaker.rendering.is_set()
    finally:
        controller.close()


def test_voice_runs_off_caller_thread_and_drops_repeat_while_busy():
    speaker = FakeSpeaker(block=True)
    notifications = []
    controller = ReminderController(
        ReminderConfig(mode="voice"), speaker=speaker, notifier=notifications.append
    )
    try:
        assert controller.remind()
        assert speaker.rendering.wait(1)
        # Rendering is blocked but the caller and camera scheduler remain free.
        assert not controller.remind()
        speaker.release_render.set()
        wait_until(lambda: controller.status == "ready")
        assert len(speaker.started) == 1
        assert speaker.started[0][1] == "en"
        assert notifications == []
    finally:
        speaker.release_render.set()
        controller.close()


def test_changing_mode_cancels_audio_still_being_generated():
    speaker = FakeSpeaker(block=True)
    controller = ReminderController(ReminderConfig(mode="roast"), False, speaker=speaker)
    try:
        controller.remind()
        assert speaker.rendering.wait(1)
        controller.cycle_mode()
        assert controller.mode == "notification"
        speaker.release_render.set()
        wait_until(lambda: controller._jobs.unfinished_tasks == 0)
        assert speaker.started == []
    finally:
        speaker.release_render.set()
        controller.close()


def test_hindi_roasts_rotate_and_use_hindi_voice():
    controller = ReminderController(ReminderConfig(mode="roast"), False, speaker=FakeSpeaker())
    try:
        previous = None
        for _ in range(20):
            text, language = controller.choose_text()
            assert language == "hi"
            assert text != previous
            assert any("\u0900" <= letter <= "\u097f" for letter in text)
            previous = text
    finally:
        controller.close()


def test_muting_stops_audio_and_keeps_notification_option():
    calls = []
    speaker = FakeSpeaker()
    controller = ReminderController(
        ReminderConfig(mode="voice"), speaker=speaker, notifier=calls.append
    )
    try:
        controller.toggle_mute()
        assert controller.remind("head lowered")
        assert calls == ["head lowered"]
        assert not speaker.rendering.is_set()
        assert speaker.stopped > 0
    finally:
        controller.close()


def test_speech_failure_does_not_crash_reminder_controller():
    speaker = FakeSpeaker(fail=True)
    controller = ReminderController(ReminderConfig(mode="voice"), False, speaker=speaker)
    try:
        controller.remind()
        wait_until(lambda: controller.status == "failed")
        assert "TTS failed" in controller.display_status
        speaker.fail = False
        assert controller.remind()
        wait_until(lambda: controller.status == "ready")
        assert len(speaker.started) == 1
    finally:
        controller.close()


@pytest.mark.parametrize(
    "setting,value", [("volume", 1.5), ("speed", 0), ("mode", "invalid"), ("muted", "false")]
)
def test_invalid_reminder_config_rejected(tmp_path, setting, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reminders": {setting: value}}))
    with pytest.raises(ValueError):
        load_config(path)
