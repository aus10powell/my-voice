from my_voice.cli import _normalize_argv


def test_default_run():
    assert _normalize_argv([]) == ["run"]


def test_global_verbose_before_eval():
    assert _normalize_argv(["-v", "eval", "--download-only"]) == [
        "-v",
        "eval",
        "--download-only",
    ]


def test_flags_default_to_run():
    assert _normalize_argv(["--model", "tiny"]) == ["run", "--model", "tiny"]
    assert _normalize_argv(["-v"]) == ["-v", "run"]


def test_help_untouched():
    assert _normalize_argv(["--help"]) == ["--help"]
