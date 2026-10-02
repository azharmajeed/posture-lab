"""Reminder selection and a cancellable speech worker, independent of inference."""

import queue
import random
import threading
import time

from posture_lab.config import ReminderConfig
from posture_lab.notifications import notify
from posture_lab.speech import create_speaker

MODES = ("notification", "voice", "roast")
VOICE_LINE = "Fix your posture. Sit upright and relax your shoulders."
GENTLE_ROASTS = (
    "अरे भाई, स्क्रीन में घुसना है क्या? थोड़ा सीधे बैठो।",
    "ये लैपटॉप है, तकिया नहीं। गर्दन उठाओ और पोस्चर ठीक करो।",
    "इतना मत झुको, काम करना है, स्क्रीन को प्रणाम नहीं। सीधे बैठो।",
)
SPICY_ROASTS = (
    "अबे चूतिये, स्क्रीन में घुसने का इरादा है क्या? सीधे बैठ।",
    "भोसड़ीके, लैपटॉप है, तकिया नहीं। गर्दन उठा और पोस्चर ठीक कर।",
    "अबे गांडू, रीढ़ की हड्डी को स्प्रिंग मत बना। सीधे बैठ।",
    "साले चूतिये, कोड लिख रहा है या स्क्रीन को प्रणाम कर रहा है? गर्दन उठा।",
    "अबे भोसड़ीके, इतना मत झुक। लैपटॉप तुझे सूँघ नहीं रहा। पोस्चर ठीक कर।",
    "चूतिये, कुर्सी पर बैठा है, आलू की बोरी मत बन। सीधे बैठ।",
)


class ReminderController:
    def __init__(
        self, config: ReminderConfig, notifications: bool = True, *, speaker=None, notifier=notify
    ):
        self.config = config
        self.notifications = notifications
        self.speaker = speaker if speaker is not None else create_speaker(config)
        self.notifier = notifier
        self.mode = config.mode
        self.muted = config.muted
        self.status = "ready"
        self.last_text = ""
        self._last_roast = None
        self._epoch = 0
        self._closed = False
        self._lock = threading.Lock()
        self._jobs = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._worker, name="speech", daemon=True)
        self._thread.start()

    @property
    def display_status(self):
        label = {"notification": "Notify", "voice": "Voice", "roast": "Hindi roast"}[self.mode]
        if self.muted:
            return label + " / muted"
        if self.status == "failed":
            return label + " / TTS failed"
        suffix = {"rendering": " / generating", "speaking": " / speaking"}.get(self.status, "")
        return label + suffix

    def choose_text(self):
        if self.mode == "voice":
            return VOICE_LINE, "en"
        if self.mode == "roast":
            bank = SPICY_ROASTS if self.config.roast_intensity == "spicy" else GENTLE_ROASTS
            text = random.choice([line for line in bank if line != self._last_roast])
            self._last_roast = text
            return text, "hi"
        return "Time to adjust your posture.", "en"

    def remind(self, reason: str = "Posture changed", *, text=None, language=None):
        with self._lock:
            if self._closed:
                return False
            # Drop repeated tests/nudges while busy; do not queue a stale spoken alert.
            if (
                self.mode != "notification"
                and not self.muted
                and self.status in ("queued", "rendering", "speaking")
            ):
                return False
            if self.mode == "notification" or self.muted:
                if self.notifications:
                    self.notifier(reason)
                return True
            phrase, phrase_language = self.choose_text()
            phrase = text if text is not None else phrase
            phrase_language = language if language is not None else phrase_language
            self.last_text = phrase
            self.status = "queued"
            self._jobs.put_nowait((self._epoch, phrase, phrase_language, time.monotonic()))
            if self.notifications and self.config.also_notify:
                self.notifier(reason)
            return True

    def _worker(self):
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                epoch, text, language, queued_at = job
                with self._lock:
                    if epoch != self._epoch or self._closed:
                        continue
                    self.status = "rendering"
                audio = self.speaker.render(text, language)
                with self._lock:
                    if epoch != self._epoch or self._closed:
                        continue
                    self.status = "speaking"
                    self.speaker.start(audio, language)
                print(
                    f"Speech playback started in {(time.monotonic() - queued_at) * 1000:.0f}ms "
                    f"({language}).",
                    flush=True,
                )
                self.speaker.wait()
                with self._lock:
                    if epoch == self._epoch:
                        self.status = "ready"
            except Exception as exc:
                with self._lock:
                    if epoch == self._epoch:
                        self.status = "failed"
                print(f"Speech unavailable: {exc}", flush=True)
            finally:
                self._jobs.task_done()

    def cancel(self):
        with self._lock:
            self._epoch += 1
            self.status = "ready"
            self.speaker.stop()
            while True:
                try:
                    self._jobs.get_nowait()
                    self._jobs.task_done()
                except queue.Empty:
                    break

    def cycle_mode(self):
        self.cancel()
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        print(f"Reminder mode: {self.mode}", flush=True)

    def toggle_mute(self):
        self.cancel()
        self.muted = not self.muted
        print(f"Speech muted: {self.muted}", flush=True)

    def close(self):
        self.cancel()
        with self._lock:
            self._closed = True
        self._jobs.put_nowait(None)
        self._thread.join(timeout=2)
