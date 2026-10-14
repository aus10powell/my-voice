"""Faster-Whisper ASR backend (default)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FasterWhisperBackend:
    """Offline Whisper inference via faster-whisper (CTranslate2)."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        if device == "auto":
            device = self._pick_device()
        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading Faster-Whisper model=%s device=%s compute_type=%s",
            model_size,
            device,
            compute_type,
        )
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language
        self.model_size = model_size

    @staticmethod
    def _pick_device() -> str:
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16_000:
            samples = _resample_linear(samples, sample_rate, 16_000)

        segments, _info = self._model.transcribe(
            samples,
            language=self._language,
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    duration = audio.shape[0] / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
