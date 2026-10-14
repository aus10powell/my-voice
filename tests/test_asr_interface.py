import numpy as np

from my_voice.asr.base import AsrBackend, create_backend, list_backends


class FakeBackend:
    name = "fake"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return f"n={audio.size},sr={sample_rate}"


def test_protocol_accepts_fake():
    backend: AsrBackend = FakeBackend()
    out = backend.transcribe(np.zeros(4, dtype=np.float32), 16_000)
    assert out.startswith("n=4")


def test_create_backend_unknown():
    try:
        create_backend("not-a-real-backend")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unknown ASR backend" in str(exc)


def test_catalog_lists_engines():
    names = list_backends()
    assert "faster-whisper" in names
    assert "whisper.cpp" in names
    assert "parakeet" in names


def test_create_backend_name_aliases():
    from my_voice.asr import faster_whisper_backend as mod

    assert mod.FasterWhisperBackend.name == "faster-whisper"
