"""Microphone capture helpers."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1


class MicRecorder:
    """Record mono float32 audio while a session is active."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        if self._recording:
            return
        import sounddevice as sd

        self._chunks = []
        self._recording = True

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            if status:
                logger.warning("Audio status: %s", status)
            if not self._recording:
                return
            mono = indata.copy()
            if mono.ndim > 1:
                mono = mono.mean(axis=1)
            with self._lock:
                self._chunks.append(mono.astype(np.float32, copy=False))

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=_callback,
        )
        self._stream.start()
        logger.debug("Mic recording started (rate=%s)", self.sample_rate)

    def stop(self) -> np.ndarray:
        """Stop capture and return concatenated mono float32 samples."""
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks)
            self._chunks = []
        logger.debug("Mic recording stopped (%s samples)", audio.shape[0])
        return audio


def list_input_devices() -> Iterator[str]:
    """Yield human-readable input device lines (requires sounddevice)."""
    import sounddevice as sd

    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) > 0:
            yield f"{idx}: {dev['name']} ({dev['hostapi']})"
