import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import whisper_key.main as wk_main
from whisper_key.whisper_engine import WhisperEngine

from fixups import post_fix

# Keep the real local faster-whisper engine; only apply the shared post-fix
# layer (filename/extension repairs + hotword normalization) to its output.
_original_transcribe = WhisperEngine.transcribe_audio


def _transcribe_with_fix(self, audio_data):
    text = _original_transcribe(self, audio_data)
    if not text:
        return None
    words = tuple(w.strip() for w in (self.hotwords or "").split(",") if w.strip())
    return post_fix(text, words) or None


WhisperEngine.transcribe_audio = _transcribe_with_fix

# Run the transcription pipeline on a background thread so the hotkey poller
# is never blocked (fixes the "Ctrl+Win does nothing after a recording" freeze).
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