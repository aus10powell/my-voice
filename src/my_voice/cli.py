"""Command-line entry points for run, devices, and eval."""

from __future__ import annotations

import argparse
import logging
import sys

from my_voice import __version__
from my_voice.config import (
    DEFAULT_BACKEND,
    DEFAULT_HOTKEY,
    DEFAULT_LIBRISPEECH_MAX_UTTERANCES,
    DEFAULT_LIBRISPEECH_SUBSET,
    DEFAULT_MODEL,
    AppConfig,
)

_COMMANDS = frozenset({"run", "devices", "eval", "backends"})
_GLOBAL_FLAGS_NO_VALUE = frozenset({"-v", "--verbose"})


def _normalize_argv(argv: list[str]) -> list[str]:
    """Default to `run` while allowing global flags before the subcommand."""
    if not argv:
        return ["run"]
    if argv[0] in {"-h", "--help", "--version"}:
        return argv

    i = 0
    while i < len(argv) and argv[i] in _GLOBAL_FLAGS_NO_VALUE:
        i += 1
    rest = argv[i:]
    prefix = argv[:i]
    if not rest:
        return [*prefix, "run"]
    if rest[0] in _COMMANDS:
        return argv
    return [*prefix, "run", *rest]


def main(argv: list[str] | None = None) -> int:
    argv = _normalize_argv(list(sys.argv[1:] if argv is None else argv))

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    parser = argparse.ArgumentParser(
        prog="my-voice",
        description="Local push-to-talk voice-to-text (offline ASR).",
        parents=[shared],
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Start hold-to-record", parents=[shared])
    _add_runtime_args(run_p)
    run_p.add_argument(
        "--print-only",
        action="store_true",
        help="Print transcripts only; do not paste into the focused app",
    )

    sub.add_parser("devices", help="List microphone input devices", parents=[shared])
    sub.add_parser("backends", help="List ASR backends", parents=[shared])

    eval_p = sub.add_parser(
        "eval",
        help="Download a small LibriSpeech sample and report WER + latency",
        parents=[shared],
    )
    _add_runtime_args(eval_p)
    eval_p.add_argument(
        "--subset",
        default=DEFAULT_LIBRISPEECH_SUBSET,
        help=f"LibriSpeech subset (default: {DEFAULT_LIBRISPEECH_SUBSET})",
    )
    eval_p.add_argument(
        "--max-utterances",
        type=int,
        default=DEFAULT_LIBRISPEECH_MAX_UTTERANCES,
        help=f"Cap utterances for a lightweight bench (default: {DEFAULT_LIBRISPEECH_MAX_UTTERANCES})",
    )
    eval_p.add_argument(
        "--cache-dir",
        default=None,
        help="Where to store downloaded audio (default: ~/.cache/my-voice)",
    )
    eval_p.add_argument(
        "--download-only",
        action="store_true",
        help="Fetch/prepare eval audio and exit without running ASR",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "devices":
        return _cmd_devices()
    if args.command == "backends":
        return _cmd_backends()
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "run":
        return _cmd_run(args)
    parser.error("a subcommand is required (run | devices | eval | backends)")
    return 2


def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--hotkey",
        default=DEFAULT_HOTKEY,
        help=f"Hold-to-record key (default: {DEFAULT_HOTKEY})",
    )
    p.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=(
            f"ASR backend: faster-whisper | whisper.cpp | parakeet "
            f"(default: {DEFAULT_BACKEND})"
        ),
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Model size / id for the backend (default: {DEFAULT_MODEL}). "
            "Parakeet accepts HF ids like nvidia/parakeet-tdt_ctc-110m."
        ),
    )
    p.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. en); default is auto-detect where supported",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    from my_voice.app import VoiceApp
    from my_voice.asr import create_backend

    config = AppConfig(
        hotkey=args.hotkey,
        backend=args.backend,
        model_size=args.model,
        language=args.language,
        device=args.device,
        print_only=bool(getattr(args, "print_only", False)),
    )
    backend = create_backend(
        config.backend,
        model_size=config.model_size,
        device=config.device,
        language=config.language,
    )
    try:
        VoiceApp(config, backend).run()
    except KeyboardInterrupt:
        print("\nBye.", flush=True)
    return 0


def _cmd_devices() -> int:
    from my_voice.audio import list_input_devices

    found = False
    for line in list_input_devices():
        print(line)
        found = True
    if not found:
        print("No input devices found.", file=sys.stderr)
        return 1
    return 0


def _cmd_backends() -> int:
    from my_voice.asr import list_backends

    for name, desc in list_backends().items():
        print(f"{name:16} {desc}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from my_voice.eval.librispeech import run_librispeech_eval

    result = run_librispeech_eval(
        backend_name=args.backend,
        model_size=args.model,
        device=args.device,
        language=args.language or "en",
        subset=args.subset,
        max_utterances=args.max_utterances,
        cache_dir=args.cache_dir,
        download_only=args.download_only,
    )
    print(result.format_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
