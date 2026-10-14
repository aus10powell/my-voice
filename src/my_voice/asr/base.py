"""Replaceable ASR backend interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# name -> short description (for CLI / docs)
BACKEND_CATALOG = {
    "faster-whisper": "Faster-Whisper (CTranslate2)",
    "whisper.cpp": "whisper.cpp via pywhispercpp",
    "parakeet": "NVIDIA Parakeet via NeMo",
}


@runtime_checkable
class AsrBackend(Protocol):
    """Minimal contract for a local speech-to-text backend."""

    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return transcribed text for mono float32 audio in [-1, 1]."""
        ...


def list_backends() -> dict[str, str]:
    return dict(BACKEND_CATALOG)


def create_backend(name: str = "faster-whisper", **kwargs) -> AsrBackend:
    """Factory for known backends. Swap implementations without touching the app loop."""
    key = name.strip().lower().replace("_", "-")
    # normalize aliases
    aliases = {
        "fasterwhisper": "faster-whisper",
        "whisper": "faster-whisper",
        "fw": "faster-whisper",
        "whispercpp": "whisper.cpp",
        "whisper-cpp": "whisper.cpp",
        "pywhispercpp": "whisper.cpp",
        "nemo": "parakeet",
        "nemo-parakeet": "parakeet",
    }
    key = aliases.get(key, key)

    if key == "faster-whisper":
        from my_voice.asr.faster_whisper_backend import FasterWhisperBackend

        return FasterWhisperBackend(**kwargs)
    if key == "whisper.cpp":
        from my_voice.asr.whisper_cpp_backend import WhisperCppBackend

        return WhisperCppBackend(**kwargs)
    if key == "parakeet":
        from my_voice.asr.parakeet_backend import ParakeetBackend

        return ParakeetBackend(**kwargs)

    known = ", ".join(sorted(BACKEND_CATALOG))
    raise ValueError(f"Unknown ASR backend {name!r}. Built-in: {known}.")
