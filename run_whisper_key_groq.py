import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import whisper_key.main as wk_main

from groq_engine import GroqWhisperEngine


def _enable_vt():
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass


_enable_vt()

wk_main.WhisperEngine = GroqWhisperEngine

_original_pipeline = wk_main.StateManager._transcription_pipeline


def _async_transcription_pipeline(self, audio_data, use_auto_enter: bool = False):
    with self._state_lock:
        if self.is_processing:
            return
        self.is_processing = True

    def worker():
        try:
            _original_pipeline(self, audio_data, use_auto_enter)
        except Exception as e:
            self.logger.error(f"Async pipeline error: {e}")

    threading.Thread(target=worker, daemon=True).start()


wk_main.StateManager._transcription_pipeline = _async_transcription_pipeline

wk_main.main()