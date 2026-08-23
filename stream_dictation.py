import ctypes
import os
import sys
import threading
import time
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
FEED_SECONDS = 0.4
MODEL_DIR = r"D:\Ethan_WorkingSpace\tools\whisperkey\sherpa\sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
QUANT = os.environ.get("WK_QUANT", "fp32")
HOTWORDS_FILE = os.environ.get("WK_HOTWORDS", os.path.join(MODEL_DIR, "hotwords.txt"))
HOTWORDS_SCORE = float(os.environ.get("WK_HOTWORDS_SCORE", "3.0"))
MAX_ACTIVE_PATHS = int(os.environ.get("WK_BEAM", "8"))
CPU_THREADS = max(2, (os.cpu_count() or 4) // 2)

user32 = ctypes.WinDLL("user32", use_last_error=True)
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class InputI(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", InputI)]


def type_unicode(text):
    for ch in text:
        code = ord(ch)
        for up in (0, 0x0002):
            extra = ctypes.c_ulong(0)
            ii = InputI()
            ii.ki = KeyBdInput(0, code, up | 0x0004, 0, ctypes.pointer(extra))
            x = Input(1, ii)
            user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


class Recorder:
    def __init__(self, sr):
        self.sr = sr
        self.chunks = deque()
        self.total = 0
        self.lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        with self.lock:
            self.chunks.append(indata.copy())
            self.total += frames

    def take(self):
        with self.lock:
            parts = list(self.chunks)
            self.chunks.clear()
        if not parts:
            return None
        return np.concatenate(parts).astype(np.float32).ravel()

    def clear(self):
        with self.lock:
            self.chunks.clear()
            self.total = 0


class StreamingEngine:
    def __init__(self, model_dir, threads):
        import sherpa_onnx
        q = "" if QUANT == "fp32" else ".int8"
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(model_dir, "tokens.txt"),
            encoder=os.path.join(model_dir, "encoder-epoch-99-avg-1%s.onnx" % q),
            decoder=os.path.join(model_dir, "decoder-epoch-99-avg-1%s.onnx" % q),
            joiner=os.path.join(model_dir, "joiner-epoch-99-avg-1%s.onnx" % q),
            num_threads=threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
            decoding_method="modified_beam_search",
            max_active_paths=MAX_ACTIVE_PATHS,
            modeling_unit="cjkchar+bpe",
            bpe_vocab=os.path.join(model_dir, "bpe.vocab"),
            hotwords_file=HOTWORDS_FILE if os.path.exists(HOTWORDS_FILE) else "",
            hotwords_score=HOTWORDS_SCORE,
        )
        self.stream = self.recognizer.create_stream()
        self.prev_text = ""
        self.feed_count = 0
        self.recreations = 0

    def _fresh_stream(self):
        old = self.stream
        self.stream = self.recognizer.create_stream()
        try:
            self.recognizer.reset(old)
        except Exception:
            pass
        self.prev_text = ""
        self.feed_count = 0

    def recreate(self):
        import sherpa_onnx
        q = "" if QUANT == "fp32" else ".int8"
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(MODEL_DIR, "tokens.txt"),
            encoder=os.path.join(MODEL_DIR, "encoder-epoch-99-avg-1%s.onnx" % q),
            decoder=os.path.join(MODEL_DIR, "decoder-epoch-99-avg-1%s.onnx" % q),
            joiner=os.path.join(MODEL_DIR, "joiner-epoch-99-avg-1%s.onnx" % q),
            num_threads=CPU_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
            decoding_method="modified_beam_search",
            max_active_paths=MAX_ACTIVE_PATHS,
            modeling_unit="cjkchar+bpe",
            bpe_vocab=os.path.join(MODEL_DIR, "bpe.vocab"),
            hotwords_file=HOTWORDS_FILE if os.path.exists(HOTWORDS_FILE) else "",
            hotwords_score=HOTWORDS_SCORE,
        )
        self.stream = self.recognizer.create_stream()
        self.prev_text = ""
        self.feed_count = 0
        self.recreations += 1

    def reset(self):
        self._fresh_stream()

    def feed(self, samples):
        if len(samples) == 0:
            return ""
        self.stream.accept_waveform(SAMPLE_RATE, samples)
        self.recognizer.decode_stream(self.stream)
        self.feed_count += 1
        text = self.recognizer.get_result(self.stream)
        tail = ""
        if text.startswith(self.prev_text):
            tail = text[len(self.prev_text):]
            self.prev_text = text
        elif len(text) >= len(self.prev_text):
            self.prev_text = text
        else:
            self.prev_text = text
        if self.recognizer.is_endpoint(self.stream):
            self._fresh_stream()
        return tail


def main():
    try:
        ctypes.windll.kernel32.SetConsoleTitleW("LIVE Dictation (stream) - Ctrl+Win")
    except Exception:
        pass
    print("==============================================", flush=True)
    print("  LIVE streaming dictation (sherpa-onnx)", flush=True)
    print("  Ctrl+Win (or Ctrl+Alt+Space) to start/stop.", flush=True)
    print("  In this console: 'r' = toggle, 'q' = quit.", flush=True)
    print("==============================================", flush=True)
    print("Loading streaming model (bilingual zh/en, %s)... (set WK_QUANT=int8 for a faster, less accurate build)" % QUANT, flush=True)
    engine = StreamingEngine(MODEL_DIR, CPU_THREADS)
    print("Model ready.", flush=True)

    import sounddevice as sd

    recorder = Recorder(SAMPLE_RATE)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.1), callback=recorder.callback,
    )
    stream.start()

    VK_LCONTROL, VK_RCONTROL = 0xA2, 0xA3
    VK_LWIN, VK_RWIN = 0x5B, 0x5C
    VK_MENU, VK_SPACE = 0x12, 0x20

    def key_down(vk):
        return bool(user32.GetAsyncKeyState(vk) & 0x8000)

    def hotkey_pressed():
        ctrl = key_down(VK_LCONTROL) or key_down(VK_RCONTROL)
        win = key_down(VK_LWIN) or key_down(VK_RWIN)
        alt = key_down(VK_MENU)
        return (ctrl and win) or (ctrl and alt and key_down(VK_SPACE))

    MIN_FEED = int(SAMPLE_RATE * FEED_SECONDS)
    MAX_FEED = int(SAMPLE_RATE * 1.5)
    state = {"recording": False, "reset": False, "flush": False}
    pending = np.zeros(0, dtype=np.float32)
    prev_hot = False

    def safe_toggle():
        try:
            state["recording"] = not state["recording"]
            if state["recording"]:
                state["reset"] = True
                print("[REC] speak now (Ctrl+Win to stop)", flush=True)
            else:
                state["flush"] = True
                print("[STOP]", flush=True)
        except Exception as e:
            print("[toggle error]", e, flush=True)

    def console_control():
        while True:
            try:
                line = input()
            except EOFError:
                break
            except Exception:
                break
            line = line.strip().lower()
            if line == "r":
                safe_toggle()
            elif line == "q":
                try:
                    stream.stop()
                except Exception:
                    pass
                os._exit(0)

    threading.Thread(target=console_control, daemon=True).start()
    print("Ready. Press Ctrl+Win (or Ctrl+Alt+Space) to start/stop.", flush=True)
    t_start = time.time()

    try:
        last_hb = time.time()
        while True:
            hot = hotkey_pressed()
            if hot and not prev_hot:
                prev_hot = True
                safe_toggle()
            elif not hot:
                prev_hot = False

            if time.time() - last_hb >= 3:
                last_hb = time.time()
                print("[hb %ds rec=%s pend=%d feeds=%d recreated=%d]" % (
                    time.time() - t_start, state["recording"], len(pending),
                    engine.feed_count, engine.recreations), flush=True)

            if state["reset"]:
                state["reset"] = False
                try:
                    recorder.clear()
                    engine.reset()
                    pending = np.zeros(0, dtype=np.float32)
                except Exception as e:
                    print("[reset error]", e, flush=True)
            if state["flush"]:
                state["flush"] = False
                try:
                    while len(pending) > 0:
                        chunk = pending[:MIN_FEED]
                        pending = pending[MIN_FEED:]
                        if len(chunk) < MIN_FEED:
                            chunk = np.concatenate(
                                [chunk, np.zeros(MIN_FEED - len(chunk), dtype=np.float32)]
                            )
                        tail = engine.feed(chunk)
                        if tail:
                            type_unicode(tail)
                            print("[typed]", tail, flush=True)
                    pending = np.zeros(0, dtype=np.float32)
                except Exception as e:
                    pending = np.zeros(0, dtype=np.float32)
                    try:
                        engine.reset()
                    except Exception:
                        pass
                    print("[flush error]", e, flush=True)
            if state["recording"]:
                try:
                    audio = recorder.take()
                    if audio is not None:
                        pending = np.concatenate([pending, audio])
                    if len(pending) > MAX_FEED:
                        pending = pending[-MAX_FEED:]
                    if len(pending) >= MIN_FEED:
                        chunk = pending[:MIN_FEED]
                        pending = pending[MIN_FEED:]
                        t_feed = time.time()
                        tail = engine.feed(chunk)
                        dt_feed = time.time() - t_feed
                        if dt_feed > 2.0:
                            print("[slow feed %.1fs -> recreating recognizer]" % dt_feed, flush=True)
                            try:
                                engine.recreate()
                            except Exception as e:
                                print("[recreate error]", e, flush=True)
                        if tail:
                            type_unicode(tail)
                            print("[typed]", tail, flush=True)
                except Exception as e:
                    pending = np.zeros(0, dtype=np.float32)
                    try:
                        engine.reset()
                    except Exception:
                        pass
                    print("[feed error]", e, flush=True)
            else:
                try:
                    recorder.take()
                except Exception:
                    pass
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()


if __name__ == "__main__":
    if "--check" in sys.argv:
        StreamingEngine(MODEL_DIR, CPU_THREADS)
        print("OK: model loads", flush=True)
        sys.exit(0)
    main()
