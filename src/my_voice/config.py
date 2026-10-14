"""Runtime defaults and CLI-shared settings."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HOTKEY = "f8"
DEFAULT_MODEL = "base"
DEFAULT_BACKEND = "faster-whisper"
DEFAULT_SAMPLE_RATE = 16_000

# Eval audio lives outside the repo so clones stay small.
DEFAULT_CACHE_DIR_NAME = ".cache/my-voice"
DEFAULT_LIBRISPEECH_SUBSET = "test-clean"
DEFAULT_LIBRISPEECH_MAX_UTTERANCES = 20


@dataclass(frozen=True)
class AppConfig:
    hotkey: str = DEFAULT_HOTKEY
    backend: str = DEFAULT_BACKEND
    model_size: str = DEFAULT_MODEL
    language: str | None = None
    device: str = "auto"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    print_only: bool = False
