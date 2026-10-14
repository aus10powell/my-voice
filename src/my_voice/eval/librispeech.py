"""LibriSpeech test-clean sample download + WER/latency bench.

Audio is cached under ~/.cache/my-voice (or --cache-dir), never vendored into git.
Default uses a small capped subset so clone-and-run benches stay lightweight.
"""

from __future__ import annotations

import logging
import tarfile
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from my_voice.config import (
    DEFAULT_CACHE_DIR_NAME,
    DEFAULT_LIBRISPEECH_MAX_UTTERANCES,
    DEFAULT_LIBRISPEECH_SUBSET,
)
from my_voice.eval.metrics import edit_distance, normalize_transcript, wer

logger = logging.getLogger(__name__)

LIBRISPEECH_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
}


@dataclass
class Utterance:
    utt_id: str
    audio_path: Path
    reference: str


@dataclass
class EvalResult:
    subset: str
    backend: str
    model_size: str
    utterances: int
    wer: float
    total_audio_sec: float
    total_infer_sec: float
    rtf: float
    cache_dir: Path
    per_utt: list[dict] = field(default_factory=list)
    download_only: bool = False

    def format_report(self) -> str:
        if self.download_only:
            return (
                f"Downloaded/prepared LibriSpeech {self.subset} under {self.cache_dir}\n"
                f"Ready utterances: {self.utterances}\n"
                "Re-run without --download-only to measure WER + latency."
            )
        lines = [
            "=== My Voice ASR eval (LibriSpeech) ===",
            f"subset:          {self.subset}",
            f"backend:         {self.backend}",
            f"model:           {self.model_size}",
            f"utterances:      {self.utterances}",
            f"WER:             {self.wer:.3f}",
            f"audio seconds:   {self.total_audio_sec:.2f}",
            f"infer seconds:   {self.total_infer_sec:.2f}",
            f"RTF (infer/audio): {self.rtf:.3f}",
            f"cache:           {self.cache_dir}",
            "",
            "Note: Common Voice is a good follow-on corpus; not wired yet.",
        ]
        return "\n".join(lines)


def default_cache_dir(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (Path.home() / DEFAULT_CACHE_DIR_NAME).resolve()


def run_librispeech_eval(
    *,
    backend_name: str = "faster-whisper",
    model_size: str = "base",
    device: str = "auto",
    language: str | None = "en",
    subset: str = DEFAULT_LIBRISPEECH_SUBSET,
    max_utterances: int = DEFAULT_LIBRISPEECH_MAX_UTTERANCES,
    cache_dir: str | Path | None = None,
    download_only: bool = False,
) -> EvalResult:
    cache = default_cache_dir(cache_dir)
    utterances = prepare_librispeech_sample(
        subset=subset,
        max_utterances=max_utterances,
        cache_dir=cache,
    )
    if download_only:
        return EvalResult(
            subset=subset,
            backend=backend_name,
            model_size=model_size,
            utterances=len(utterances),
            wer=0.0,
            total_audio_sec=0.0,
            total_infer_sec=0.0,
            rtf=0.0,
            cache_dir=cache,
            download_only=True,
        )

    from my_voice.asr import create_backend

    backend = create_backend(
        backend_name,
        model_size=model_size,
        device=device,
        language=language,
    )
    resolved_model = getattr(backend, "model_size", model_size)

    total_edits_weight = 0.0
    total_ref_tokens = 0
    total_audio = 0.0
    total_infer = 0.0
    rows: list[dict] = []

    for utt in utterances:
        audio, sr = load_wav_mono(utt.audio_path)
        audio_sec = float(audio.shape[0]) / float(sr)
        t0 = time.perf_counter()
        hyp = backend.transcribe(audio, sr)
        infer_sec = time.perf_counter() - t0
        u_wer = wer(utt.reference, hyp)
        ref_n = normalize_transcript(utt.reference)
        hyp_n = normalize_transcript(hyp)
        edits = edit_distance(ref_n, hyp_n)
        total_edits_weight += edits
        total_ref_tokens += max(len(ref_n), 1)
        total_audio += audio_sec
        total_infer += infer_sec
        rows.append(
            {
                "id": utt.utt_id,
                "wer": u_wer,
                "audio_sec": audio_sec,
                "infer_sec": infer_sec,
                "ref": utt.reference,
                "hyp": hyp,
            }
        )
        logger.info(
            "%s wer=%.3f audio=%.2fs infer=%.2fs",
            utt.utt_id,
            u_wer,
            audio_sec,
            infer_sec,
        )

    overall = total_edits_weight / max(total_ref_tokens, 1)
    rtf = total_infer / total_audio if total_audio > 0 else 0.0
    return EvalResult(
        subset=subset,
        backend=backend.name,
        model_size=str(resolved_model),
        utterances=len(utterances),
        wer=overall,
        total_audio_sec=total_audio,
        total_infer_sec=total_infer,
        rtf=rtf,
        cache_dir=cache,
        per_utt=rows,
    )


def prepare_librispeech_sample(
    *,
    subset: str = DEFAULT_LIBRISPEECH_SUBSET,
    max_utterances: int = DEFAULT_LIBRISPEECH_MAX_UTTERANCES,
    cache_dir: Path,
) -> list[Utterance]:
    if subset not in LIBRISPEECH_URLS:
        raise ValueError(
            f"Unsupported subset {subset!r}. Choose from: {sorted(LIBRISPEECH_URLS)}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_root = cache_dir / "librispeech"
    subset_dir = extract_root / "LibriSpeech" / subset
    archive_path = cache_dir / f"librispeech-{subset}.tar.gz"

    if not subset_dir.is_dir():
        if not archive_path.is_file():
            url = LIBRISPEECH_URLS[subset]
            logger.info("Downloading %s → %s", url, archive_path)
            _download(url, archive_path)
        logger.info("Extracting %s", archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_root, filter="data")

    utterances = _index_subset(subset_dir, max_utterances=max_utterances)
    if not utterances:
        raise RuntimeError(f"No utterances found under {subset_dir}")
    logger.info(
        "Prepared %d / requested %d utterances from %s",
        len(utterances),
        max_utterances,
        subset,
    )
    return utterances


def _index_subset(subset_dir: Path, *, max_utterances: int) -> list[Utterance]:
    found: list[Utterance] = []
    for transcript in sorted(subset_dir.rglob("*.trans.txt")):
        chapter_dir = transcript.parent
        for line in transcript.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            wav = chapter_dir / f"{utt_id}.flac"
            if not wav.is_file():
                alt = chapter_dir / f"{utt_id}.wav"
                wav = alt if alt.is_file() else wav
            if not wav.is_file():
                continue
            found.append(Utterance(utt_id=utt_id, audio_path=wav, reference=text.strip()))
            if len(found) >= max_utterances:
                return found
    return found


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load flac/wav as mono float32. Prefers soundfile; falls back for wav."""
    suffix = path.suffix.lower()
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception:
        if suffix != ".wav":
            raise
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(n)
    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width {width} for {path}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, int(sr)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
