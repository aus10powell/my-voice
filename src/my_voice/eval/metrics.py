"""Simple WER without heavyweight metric packages."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_transcript(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over token sequences."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if r == h else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = edits / reference tokens. Empty ref → 0 if hyp empty else 1."""
    ref = normalize_transcript(reference)
    hyp = normalize_transcript(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
