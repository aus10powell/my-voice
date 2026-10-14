"""NVIDIA Parakeet ASR backend via NeMo."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Default to a smaller English CTC/TDT hybrid that is more realistic on CPU VMs.
# Override with --model nvidia/parakeet-tdt-0.6b-v2 (much heavier).
DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt_ctc-110m"

_SIZE_ALIASES = {
    "tiny": DEFAULT_PARAKEET_MODEL,
    "base": DEFAULT_PARAKEET_MODEL,
    "small": DEFAULT_PARAKEET_MODEL,
    "110m": DEFAULT_PARAKEET_MODEL,
    "parakeet-tdt_ctc-110m": DEFAULT_PARAKEET_MODEL,
    "0.6b": "nvidia/parakeet-tdt-0.6b-v2",
    "parakeet-tdt-0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",
}


class ParakeetBackend:
    """Offline Parakeet inference via NVIDIA NeMo."""

    name = "parakeet"

    def __init__(
        self,
        model_size: str = DEFAULT_PARAKEET_MODEL,
        device: str = "cpu",
        language: str | None = None,
        **_ignored,
    ) -> None:
        try:
            import nemo.collections.asr as nemo_asr
            import torch
        except ImportError as exc:
            raise ImportError(
                "parakeet backend requires NeMo + PyTorch. Install with:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                '  pip install "nemo_toolkit[asr]"'
            ) from exc

        model_id = _SIZE_ALIASES.get(model_size.strip().lower(), model_size)
        if not model_id.startswith("nvidia/") and "/" not in model_id:
            # bare HF-style short name
            model_id = f"nvidia/{model_id}" if not model_id.startswith("stt_") else model_id

        logger.info("Loading Parakeet/NeMo model=%s device=%s", model_id, device)
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_id)

        if device in {"cpu", "auto"}:
            self._model = self._model.to(torch.device("cpu"))
            self._model.eval()
        elif device == "cuda":
            self._model = self._model.cuda()
            self._model.eval()

        self.model_size = model_id
        self._language = language  # Parakeet EN models ignore; kept for interface parity

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16_000:
            samples = _resample_linear(samples, sample_rate, 16_000)

        # NeMo accepts paths or arrays depending on version; path is the most portable.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            sf.write(str(path), samples, 16_000)
            outputs = self._model.transcribe([str(path)], batch_size=1)
            return _extract_text(outputs).strip()
        finally:
            path.unlink(missing_ok=True)


def _extract_text(outputs) -> str:
    if outputs is None:
        return ""
    first = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    if first is None:
        return ""
    if isinstance(first, str):
        return first
    text = getattr(first, "text", None)
    if text is not None:
        return str(text)
    # older NeMo sometimes returned Hypothesis-like objects
    return str(first)


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    duration = audio.shape[0] / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
