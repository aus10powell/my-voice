from my_voice.cli import main


def test_help_exits_zero():
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_eval_help():
    try:
        main(["eval", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
