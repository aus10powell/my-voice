"""Hold-to-record loop: hotkey down → mic → hotkey up → ASR → paste."""

from __future__ import annotations

import logging
import threading

from my_voice.asr.base import AsrBackend
from my_voice.audio import MicRecorder
from my_voice.config import AppConfig
from my_voice.paste import paste_text

logger = logging.getLogger(__name__)


class VoiceApp:
    def __init__(self, config: AppConfig, backend: AsrBackend) -> None:
        self.config = config
        self.backend = backend
        self.recorder = MicRecorder(sample_rate=config.sample_rate)
        self._busy = threading.Lock()

    def run(self) -> None:
        from pynput import keyboard

        hotkey = self.config.hotkey.lower().strip()
        logger.info(
            "My Voice ready — hold %s to talk, release to transcribe (%s / %s)",
            hotkey.upper(),
            self.backend.name,
            self.config.model_size,
        )
        print(
            f"Hold [{hotkey.upper()}] to record, release to insert text. "
            "Ctrl+C to quit.",
            flush=True,
        )

        def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
            if not _key_matches(key, hotkey):
                return
            if self.recorder.is_recording:
                return
            if not self._busy.acquire(blocking=False):
                return
            try:
                self.recorder.start()
                print("● recording…", flush=True)
            except Exception:
                self._busy.release()
                logger.exception("Failed to start recording")

        def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
            if not _key_matches(key, hotkey):
                return
            if not self.recorder.is_recording:
                return
            audio = self.recorder.stop()
            threading.Thread(
                target=self._transcribe_and_insert,
                args=(audio,),
                daemon=True,
            ).start()

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def _transcribe_and_insert(self, audio) -> None:
        try:
            if audio.size == 0:
                print("(no audio captured)", flush=True)
                return
            print("… transcribing", flush=True)
            text = self.backend.transcribe(audio, self.config.sample_rate)
            if not text:
                print("(empty transcript)", flush=True)
                return
            print(f"→ {text}", flush=True)
            if self.config.print_only:
                return
            paste_text(text)
        except Exception:
            logger.exception("Transcription failed")
            print("! transcription error (see logs)", flush=True)
        finally:
            if self._busy.locked():
                self._busy.release()


def _key_matches(key, hotkey: str) -> bool:
    from pynput import keyboard

    name = hotkey.lower()
    special = {
        "f8": keyboard.Key.f8,
        "f9": keyboard.Key.f9,
        "f10": keyboard.Key.f10,
        "f11": keyboard.Key.f11,
        "f12": keyboard.Key.f12,
        "rctrl": keyboard.Key.ctrl_r,
        "rightctrl": keyboard.Key.ctrl_r,
        "ralt": keyboard.Key.alt_r,
        "rightalt": keyboard.Key.alt_r,
        "rshift": keyboard.Key.shift_r,
        "caps": keyboard.Key.caps_lock,
        "capslock": keyboard.Key.caps_lock,
    }
    if name in special:
        return key == special[name]
    if isinstance(key, keyboard.KeyCode) and key.char:
        return key.char.lower() == name
    return False
