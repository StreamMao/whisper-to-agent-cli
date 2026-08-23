import io
import logging
import os
import time
import wave

import numpy as np
import httpx

from fixups import post_fix as _post_fix

SAMPLE_RATE = 16000
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"
HTTP_TIMEOUT = httpx.Timeout(15.0, connect=8.0)
MAX_ATTEMPTS = 3
RETRYABLE_STATUS = (429, 500, 502, 503, 504)

from fixups import post_fix as _post_fix


def _load_dotenv():
    """Lightweight .env loader - no external dep. Single source: repo root .env"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(p):
        return
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


# load .env on import so GROQ_API_KEY is available immediately
try:
    _load_dotenv()
    import dotenv  # type: ignore  # optional: handles quoted/multiline if installed

    dotenv.load_dotenv(override=False)  # type: ignore
except Exception:
    pass


def _load_api_key():
    return os.environ.get("GROQ_API_KEY", "").strip()


def _encode_wav(audio_data):
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
    pcm = np.clip(audio_data, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class GroqWhisperEngine:
    def __init__(self,
                 model_key: str = "groq",
                 device: str = "cpu",
                 compute_type: str = "int8",
                 language: str = None,
                 beam_size: int = 5,
                 initial_prompt: str = "",
                 hotwords: list = None,
                 vad_manager=None,
                 model_registry=None):
        self.model_key = model_key
        self.language = None if language in (None, "auto") else language
        self.initial_prompt = initial_prompt or None
        self.words = tuple(w for w in (hotwords or []) if isinstance(w, str) and w.strip())
        self.vad_manager = vad_manager
        self.api_key = _load_api_key()
        self.model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.logger = logging.getLogger(__name__)

    def is_loading(self) -> bool:
        return False

    def change_model(self, new_model_key: str, progress_callback=None):
        self.model_key = new_model_key
        if progress_callback:
            progress_callback("ready")

    def transcribe_audio(self, audio_data: np.ndarray):
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("No audio data to transcribe")
            return None

        if not self.api_key:
            print("   [X] GROQ_API_KEY not set. Create .env from .env.example and set GROQ_API_KEY=...", flush=True)
            return None

        if self.vad_manager and self.vad_manager.is_available():
            if not self.vad_manager.check_audio_for_speech(audio_data):
                print("   [x] No speech detected, skipping transcription", flush=True)
                return None

        start_time = time.time()
        wav = _encode_wav(audio_data)

        data = {"model": self.model, "temperature": 0.0}
        if self.language:
            data["language"] = self.language
        prompt = self.initial_prompt or ""
        if self.words:
            extra = "重点词语（请准确转录，保留原样）：" + "、".join(self.words)
            prompt = (prompt + "\n" + extra).strip() if prompt else extra
        if prompt:
            data["prompt"] = prompt

        try:
            resp = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    resp = httpx.post(
                        GROQ_URL,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data=data,
                        files={"file": ("audio.wav", wav, "audio/wav")},
                        timeout=HTTP_TIMEOUT,
                    )
                except Exception as e:
                    if attempt < MAX_ATTEMPTS:
                        wait = 1.0 * (2 ** (attempt - 1))
                        print(f"   [!] Groq request failed ({e}), retrying in {wait:.0f}s...", flush=True)
                        time.sleep(wait)
                        continue
                    self.logger.error(f"Groq request failed after {MAX_ATTEMPTS} attempts: {e}")
                    print(f"   [X] Groq request failed: {e}", flush=True)
                    return None

                if resp.status_code == 200:
                    break

                if resp.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                    wait = 1.0 * (2 ** (attempt - 1))
                    print(f"   [!] Groq HTTP {resp.status_code}, retrying in {wait:.0f}s...", flush=True)
                    time.sleep(wait)
                    continue

                self.logger.error(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
                print(f"   [X] Groq error {resp.status_code}: {resp.text[:200]}", flush=True)
                return None

        except Exception as e:
            self.logger.error(f"Groq request failed: {e}")
            print(f"   [X] Groq request failed: {e}", flush=True)
            return None

        try:
            text = resp.json().get("text", "").strip()
        except Exception:
            text = resp.text.strip()

        elapsed = time.time() - start_time
        self.logger.info(f"Groq transcription complete in {elapsed:.2f}s")
        return _post_fix(text, self.words) or None
