import builtins
import io
import wave

from posture_lab.config import ReminderConfig
from posture_lab.speech import PiperSpeaker


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes(b"\0\0" * 100)
    return buffer.getvalue()


def make_speaker(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture model")
    (tmp_path / "model.onnx.json").write_text("{}")
    return PiperSpeaker(
        ReminderConfig(english_model=str(model), cache_directory=str(tmp_path / "cache"))
    )


def test_disk_cache_reused_without_importing_neural_runtime(tmp_path, monkeypatch):
    speaker = make_speaker(tmp_path)
    path = speaker._disk_path("Fix posture", "en")
    path.parent.mkdir()
    audio = wav_bytes()
    path.write_bytes(audio)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in ("piper", "onnxruntime"), "Cached speech loaded the neural runtime"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert PiperSpeaker(speaker.config).render("Fix posture", "en") == audio


def test_speed_volume_and_model_change_invalidate_cached_audio(tmp_path):
    speaker = make_speaker(tmp_path)
    original = speaker._disk_path("Fix posture", "en")
    speaker.config.speed = 1.2
    assert speaker._disk_path("Fix posture", "en") != original
    speaker.config.speed = 1
    speaker.config.volume = 0.4
    assert speaker._disk_path("Fix posture", "en") != original
    speaker.config.volume = 0.7
    (tmp_path / "model.onnx").write_bytes(b"changed model contents")
    assert speaker._disk_path("Fix posture", "en") != original


def test_truncated_or_invalid_wav_not_used(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav" * 20)
    assert PiperSpeaker._read_cached(path) is None
    path.write_bytes(wav_bytes()[:-10])
    assert PiperSpeaker._read_cached(path) is None
