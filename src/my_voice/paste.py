"""Insert transcribed text into the focused application via clipboard paste."""

from __future__ import annotations

import logging
import platform
import time

logger = logging.getLogger(__name__)


def paste_text(text: str, *, restore_clipboard: bool = True) -> None:
    """Copy *text* to the clipboard and send the OS paste shortcut."""
    if not text:
        return

    import pyperclip
    from pynput.keyboard import Controller, Key

    previous: str | None = None
    if restore_clipboard:
        try:
            previous = pyperclip.paste()
        except Exception:
            previous = None

    pyperclip.copy(text)
    time.sleep(0.05)

    keyboard = Controller()
    modifier = Key.cmd if platform.system() == "Darwin" else Key.ctrl
    with keyboard.pressed(modifier):
        keyboard.tap("v")

    logger.info("Pasted %d characters", len(text))

    if restore_clipboard and previous is not None:
        time.sleep(0.08)
        try:
            pyperclip.copy(previous)
        except Exception:
            logger.debug("Could not restore previous clipboard contents", exc_info=True)
