"""whisper.cpp ASR backend via pywhispercpp."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Map common Faster-Whisper size names to ggml model ids used by pywhispercpp.
_MODEL_ALIASES = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large": "large-v3",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
}


class WhisperCppBackend:
    """Offline Whisper inference via whisper.cpp (pywhispercpp bindings)."""

    name = "whisper.cpp"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str | None = None,
        models_dir: str | Path | None = None,
        **_ignored,
    ) -> None:
        try:
            from pywhispercpp.model import Model
        except ImportError as exc:
            raise ImportError(
                "whisper.cpp backend requires pywhispercpp. "
                "Install with: pip install -e '.[whispercpp]' "
                "or: pip install pywhispercpp"
            ) from exc

        model_id = _MODEL_ALIASES.get(model_size.strip().lower(), model_size)
        if models_dir is None:
            models_dir = Path.home() / ".cache" / "my-voice" / "whispercpp-models"
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Force CPU on this binding unless caller opts into GPU.
        use_gpu = device not in {"cpu", "auto"} and device != "cpu"
        if device == "auto":
            use_gpu = False  # keep clone-and-run CPU-safe by default

        logger.info(
            "Loading whisper.cpp model=%s models_dir=%s use_gpu=%s",
            model_id,
            models_dir,
            use_gpu,
        )
        self._model = Model(
            model_id,
            models_dir=str(models_dir),
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
        )
        self._language = language
        self.model_size = model_id

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16_000:
            samples = _resample_linear(samples, sample_rate, 16_000)

        params: dict = {"print_progress": False, "print_realtime": False}
        if self._language:
            params["language"] = self._language

        segments = self._model.transcribe(samples, **params)
        parts: list[str] = []
        for seg in segments:
            text = getattr(seg, "text", str(seg)).strip()
            if text and text not in {"[BLANK_AUDIO]", "[Silence]"}:
                parts.append(text)
        return " ".join(parts).strip()


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    duration = audio.shape[0] / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
