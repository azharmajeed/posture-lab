"""Local speech engines; camera images never enter the TTS pipeline."""

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

from posture_lab.config import ReminderConfig

VOICE_SOURCES = {
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "hi_IN-pratham-medium": "hi/hi_IN/pratham/medium",
}


def download_voices(directory: Path, language: str = "both") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    selected = list(VOICE_SOURCES)
    if language != "both":
        selected = [selected[0] if language == "en" else selected[1]]
    models = []
    for voice in selected:
        source = "https://huggingface.co/rhasspy/piper-voices/resolve/main/" + VOICE_SOURCES[voice]
        for remote, local in (
            (voice + ".onnx", voice + ".onnx"),
            (voice + ".onnx.json", voice + ".onnx.json"),
            ("MODEL_CARD", voice + ".MODEL_CARD"),
        ):
            target = directory / local
            if target.is_file():
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                with urllib.request.urlopen(source + "/" + remote, timeout=60) as response:
                    with temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                if temporary.stat().st_size == 0:
                    raise RuntimeError(f"Empty voice download: {local}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        models.append(directory / (voice + ".onnx"))
    return models


class PiperSpeaker:
    def __init__(self, config: ReminderConfig):
        self.config = config
        self._voices = {}
        self._cache = {}
        self._audio = None

    def _disk_path(self, text: str, language: str) -> Path:
        path = Path(self.config.hindi_model if language == "hi" else self.config.english_model)
        model_config = Path(str(path) + ".json")
        if not path.is_file() or not model_config.is_file():
            raise FileNotFoundError("Missing voice model. Run posture-lab download-voices first.")
        identity = {
            "version": 1,
            "text": text,
            "language": language,
            "volume": self.config.volume,
            "speed": self.config.speed,
            "files": [
                (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                for p in (path, model_config)
            ],
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return Path(self.config.cache_directory) / (digest + ".wav")

    @staticmethod
    def _read_cached(path: Path) -> bytes | None:
        try:
            if not 44 < path.stat().st_size <= 5 * 1024 * 1024:
                return None
            audio = path.read_bytes()
            with wave.open(io.BytesIO(audio), "rb") as wav:
                if (
                    wav.getsampwidth() != 2
                    or wav.getnchannels() != 1
                    or not 8000 <= wav.getframerate() <= 48000
                    or wav.getnframes() <= 0
                ):
                    return None
                if len(wav.readframes(wav.getnframes())) != wav.getnframes() * 2:
                    return None
            return audio
        except (OSError, EOFError, wave.Error):
            return None

    def render(self, text: str, language: str) -> bytes:
        disk_path = self._disk_path(text, language)
        key = str(disk_path)
        if key in self._cache:
            return self._cache[key]
        # A prepared phrase can play without importing or loading a neural TTS model.
        cached = self._read_cached(disk_path)
        if cached is not None:
            self._cache[key] = cached
            return cached
        import onnxruntime as ort
        from piper import PiperConfig, PiperVoice, SynthesisConfig

        path = Path(self.config.hindi_model if language == "hi" else self.config.english_model)
        if not path.is_file() or not Path(str(path) + ".json").is_file():
            raise FileNotFoundError("Missing voice model. Run posture-lab download-voices first.")
        if language not in self._voices:
            # Limit neural speech CPU usage so it can share a laptop with pose inference.
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            voice_config = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
            self._voices[language] = PiperVoice(
                session=ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                ),
                config=PiperConfig.from_dict(voice_config),
                download_dir=path.parent,
            )
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            self._voices[language].synthesize_wav(
                text,
                output,
                syn_config=SynthesisConfig(
                    volume=self.config.volume, length_scale=1 / self.config.speed
                ),
            )
        result = buffer.getvalue()
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=disk_path.parent, suffix=".wav", delete=False
            ) as out:
                temporary = Path(out.name)
                out.write(result)
            temporary.replace(disk_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if len(self._cache) >= 24:
            self._cache.clear()
        self._cache[key] = result
        return result

    def start(self, audio: bytes, language: str):
        import numpy as np
        import sounddevice

        self._audio = sounddevice
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise RuntimeError("Expected 16-bit PCM speech audio")
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            samples = samples.reshape(-1, wav.getnchannels())
            sounddevice.play(samples, samplerate=wav.getframerate(), blocking=False)

    def wait(self):
        if self._audio:
            self._audio.wait()

    def stop(self):
        if self._audio:
            self._audio.stop()


class SystemSpeaker:
    """Optional installed OS voice; Hindi requires a Hindi voice on that machine."""

    def __init__(self, config: ReminderConfig):
        self.config = config
        self._process = None

    def render(self, text: str, language: str) -> str:
        return text

    def start(self, text: str, language: str):
        if sys.platform == "win32":
            culture = "hi-IN" if language == "hi" else "en-US"
            script = (
                "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$voices = @($s.GetInstalledVoices() | Where-Object {{$_.VoiceInfo.Culture.Name -eq '{culture}'}}); "
                "if ($voices.Count -eq 0) { throw 'Required OS voice missing; use Piper instead.' }; "
                "$s.SelectVoice($voices[0].VoiceInfo.Name); "
                f"$s.Volume = {round(self.config.volume * 100)}; "
                f"$s.Rate = {max(-10, min(10, round((self.config.speed - 1) * 5)))}; "
                "try { $s.Speak([Console]::In.ReadToEnd()) } finally { $s.Dispose() }"
            )
            command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]
        elif sys.platform == "darwin":
            command = ["say", "-r", str(round(180 * self.config.speed)), "-f", "-"]
            if language == "hi":
                command.extend(["-v", "Lekha"])
        else:
            binary = shutil.which("espeak-ng") or shutil.which("espeak")
            if not binary:
                raise RuntimeError("No OS speech engine available; use Piper")
            command = [
                binary,
                "--stdin",
                "-v",
                language,
                "-s",
                str(round(180 * self.config.speed)),
                "-a",
                str(round(self.config.volume * 100)),
            ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        # Small fixed phrases fit the pipe; the camera thread never waits for speech.
        self._process.stdin.write(text.encode("utf-8"))
        self._process.stdin.close()

    def wait(self):
        if self._process:
            try:
                return_code = self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.stop()
                raise RuntimeError("OS speech timed out") from None
            if return_code:
                raise RuntimeError("OS speech failed; check the installed language voice")

    def stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()


def create_speaker(config: ReminderConfig):
    if config.tts_engine == "piper":
        return PiperSpeaker(config)
    if config.tts_engine == "system":
        return SystemSpeaker(config)
    raise ValueError("Unsupported speech engine")
