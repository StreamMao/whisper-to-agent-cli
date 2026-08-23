import ctypes
import os
import sys
import threading
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from groq_engine import GroqWhisperEngine, _load_api_key

SAMPLE_RATE = 16000

# Tuning (can be overridden via env vars):
#  WK_REFRESH      seconds between live preview refreshes (console [LIVE] line only)
#  WK_MIN_AUDIO    minimum seconds of audio before the first preview is sent
#  WK_MAX_AUDIO    hard cap on the audio buffer (older audio is dropped)
#  WK_LANG         override language ('zh', 'en', ...); defaults to user_settings.yaml
#  GROQ_MODEL      model name, defaults to whisper-large-v3-turbo
REFRESH_SECONDS = float(os.environ.get("WK_REFRESH", "2.5"))
MIN_AUDIO_SECONDS = float(os.environ.get("WK_MIN_AUDIO", "1.0"))
MAX_AUDIO_SECONDS = float(os.environ.get("WK_MAX_AUDIO", "60.0"))
MODEL = os.environ.get("GROQ_MODEL", "whisper-large-v3-turbo")
LANGUAGE = os.environ.get("WK_LANG", "") or None

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)
PUL = ctypes.POINTER(ctypes.c_ulong)

# The window text is typed into; captured when recording starts.
_active_target = None


def _enable_vt():
    try:
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def _set_title(title):
    try:
        kernel32.SetConsoleTitleW(title)
    except Exception:
        pass


def _get_console_window():
    try:
        return kernel32.GetConsoleWindow()
    except Exception:
        return None


def _restore_target(hwnd):
    """Bring the recorded target window to the foreground so injected text
    lands there even if the tool's own console stole focus."""
    global _active_target
    _active_target = hwnd
    if not hwnd:
        return
    try:
        if user32.GetForegroundWindow() == hwnd:
            return
        # Release the Windows foreground lock (simulate an ALT press), then switch.
        extra = ctypes.c_ulong(0)
        for flags in (0, KEYEVENTF_KEYUP):
            ii = InputI()
            ii.ki = KeyBdInput(0xA4, 0, flags, 0, ctypes.pointer(extra))
            user32.SendInput(1, ctypes.byref(Input(1, ii)), ctypes.sizeof(Input))
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


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


def _send(inputs):
    n = len(inputs)
    array = (Input * n)(*inputs)
    return user32.SendInput(n, ctypes.byref(array), ctypes.sizeof(Input))


def _unicode_inp(cp, flags=0):
    extra = ctypes.c_ulong(0)
    ii = InputI()
    ii.ki = KeyBdInput(0, cp, KEYEVENTF_UNICODE | flags, 0, ctypes.pointer(extra))
    return Input(1, ii)


def _vk_inp(vk, flags=0):
    extra = ctypes.c_ulong(0)
    ii = InputI()
    ii.ki = KeyBdInput(vk, 0, flags, 0, ctypes.pointer(extra))
    return Input(1, ii)


def type_text(text):
    _restore_target(_active_target)
    inputs = []
    for ch in text:
        cp = ord(ch)
        if cp > 0xFFFF:
            hi = 0xD800 + ((cp - 0x10000) >> 10)
            lo = 0xDC00 + ((cp - 0x10000) & 0x3FF)
            inputs.append(_unicode_inp(hi))
            inputs.append(_unicode_inp(hi, KEYEVENTF_KEYUP))
            inputs.append(_unicode_inp(lo))
            inputs.append(_unicode_inp(lo, KEYEVENTF_KEYUP))
        else:
            inputs.append(_unicode_inp(cp))
            inputs.append(_unicode_inp(cp, KEYEVENTF_KEYUP))
    if inputs:
        _send(inputs)


def backspace(count):
    _restore_target(_active_target)
    if count <= 0:
        return
    inputs = []
    for _ in range(count):
        inputs.append(_vk_inp(0x08))  # VK_BACK
        inputs.append(_vk_inp(0x08, KEYEVENTF_KEYUP))
    _send(inputs)


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PreviewTyper:
    """Types the live preview into the focused chat APPEND-ONLY: each refresh
    only the new tail (past the longest common prefix with what's already
    typed) is written, so old text is never deleted or re-typed. On stop the
    whole preview is replaced once by the final transcript."""

    def __init__(self):
        self.lock = threading.Lock()
        self.committed = ""

    def append_preview(self, new_text):
        with self.lock:
            new_text = new_text or ""
            if not new_text:
                return
            if not self.committed:
                type_text(new_text)
                self.committed = new_text
                return
            cp = _common_prefix_len(self.committed, new_text)
            # Skip refreshes where the transcript rewrote too much of what we
            # already typed (avoids visible duplication); a later refresh (with
            # more audio) typically stabilizes, and the final pass fixes it.
            if cp < 0.4 * min(len(self.committed), len(new_text)):
                return
            tail = new_text[cp:]
            if tail:
                type_text(tail)
                self.committed = new_text

    def finalize(self, final_text):
        with self.lock:
            final_text = final_text or ""
            if self.committed:
                backspace(len(self.committed))
                self.committed = ""
            if final_text:
                type_text(final_text)


class TrayIndicator:
    """Shows an icon in the notification area / task bar while recording, and
    flips the console window title so the task bar button shows the state."""

    NIM_ADD = 0x0
    NIM_DELETE = 0x2
    NIF_ICON = 0x2
    NIF_TIP = 0x4

    def __init__(self):
        self._hwnd = None
        self._icon = None
        self._nid = None
        self._active = False

    def _build_nid(self, tip):
        class NOTIFYICONDATAW(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_ulong),
                ("hWnd", ctypes.c_void_p),
                ("uID", ctypes.c_uint),
                ("uFlags", ctypes.c_uint),
                ("uCallbackMessage", ctypes.c_uint),
                ("hIcon", ctypes.c_void_p),
                ("szTip", ctypes.c_wchar * 128),
                ("dwState", ctypes.c_ulong),
                ("dwStateMask", ctypes.c_ulong),
                ("szInfo", ctypes.c_wchar * 256),
                ("uVersion", ctypes.c_uint),
                ("szInfoTitle", ctypes.c_wchar * 64),
                ("dwInfoFlags", ctypes.c_ulong),
            ]

        if self._hwnd is None:
            self._hwnd = _get_console_window()
            if not self._hwnd:
                return None
        if self._icon is None:
            self._icon = user32.LoadIconW(None, 32512)  # IDI_APPLICATION
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(nid)
        nid.hWnd = self._hwnd
        nid.uID = 1
        nid.uFlags = self.NIF_ICON | self.NIF_TIP
        nid.uCallbackMessage = 0x400 + 1
        nid.hIcon = self._icon
        nid.szTip = tip[:127]
        return nid

    def show(self, tip):
        try:
            self._nid = self._build_nid(tip)
            if self._nid is None:
                return
            shell32.Shell_NotifyIconW(self.NIM_ADD, ctypes.byref(self._nid))
            self._active = True
        except Exception:
            pass

    def hide(self):
        try:
            if self._nid is not None:
                shell32.Shell_NotifyIconW(self.NIM_DELETE, ctypes.byref(self._nid))
                self._nid = None
            self._active = False
        except Exception:
            pass


def _load_config():
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "appdata", "whisperkey", "user_settings.yaml",
    )
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    w = cfg.get("whisper", {})
    lang = LANGUAGE
    if not lang:
        val = w.get("language")
        lang = None if val in (None, "auto") else str(val)
    return {
        "language": lang,
        "initial_prompt": w.get("initial_prompt", "") or "",
        "hotwords": w.get("hotwords", []) or [],
    }


def _make_engine(cfg):
    key = _load_api_key()
    if not key:
        print("[X] No Groq API key found. Create .env from .env.example and set GROQ_API_KEY=...", flush=True)
        sys.exit(1)
    return GroqWhisperEngine(
        model_key=MODEL, device="cpu", compute_type="int8",
        language=cfg["language"], beam_size=5,
        initial_prompt=cfg["initial_prompt"], hotwords=cfg["hotwords"],
        vad_manager=None, model_registry=None,
    )


class Recorder:
    def __init__(self, sr):
        self.sr = sr
        self.chunks = deque()
        self.lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        with self.lock:
            self.chunks.append(indata.copy())

    def take(self):
        with self.lock:
            parts = list(self.chunks)
            self.chunks.clear()
        if not parts:
            return None
        return np.concatenate(parts).astype(np.float32).ravel()


class AudioBuffer:
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.parts = []
        self.samples = 0
        self.lock = threading.Lock()

    def add(self, audio):
        if audio is None or len(audio) == 0:
            return
        with self.lock:
            self.parts.append(audio)
            self.samples += len(audio)
            while self.samples > self.max_samples:
                head = self.parts[0]
                excess = self.samples - self.max_samples
                if len(head) <= excess:
                    self.parts.pop(0)
                    self.samples -= len(head)
                else:
                    self.parts[0] = head[excess:]
                    self.samples -= excess

    def clear(self):
        with self.lock:
            self.parts = []
            self.samples = 0

    def snapshot(self):
        with self.lock:
            if self.samples == 0:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.parts).astype(np.float32).ravel()


def clear_line():
    sys.stdout.write("\r\033[2K")
    sys.stdout.flush()


def show_preview(text):
    sys.stdout.write("\r\033[2K[LIVE] " + text + "\r")
    sys.stdout.flush()


def main():
    _enable_vt()
    base_title = "LIVE Dictation (Groq) - Ctrl+Win"
    _set_title(base_title)
    print("==============================================", flush=True)
    print("  LIVE streaming dictation (Groq whisper)", flush=True)
    print("  Ctrl+Win (or Ctrl+Alt+Space) to start/stop.", flush=True)
    print("  In this console: 'r' = toggle, 'q' = quit.", flush=True)
    print("==============================================", flush=True)

    cfg = _load_config()
    print("Loading engine (model=%s, lang=%s)..." % (MODEL, cfg["language"] or "auto"), flush=True)
    engine = _make_engine(cfg)
    print("Engine ready.", flush=True)

    import sounddevice as sd

    recorder = Recorder(SAMPLE_RATE)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.1), callback=recorder.callback,
    )
    stream.start()

    tray = TrayIndicator()
    typer = PreviewTyper()

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

    MIN_SAMPLES = int(SAMPLE_RATE * MIN_AUDIO_SECONDS)
    buffer = AudioBuffer(int(SAMPLE_RATE * MAX_AUDIO_SECONDS))
    work_lock = threading.Lock()
    state = {
        "recording": False,
        "reset": False,
        "stop": False,
        "finalizing": False,
        "busy": False,
    }

    def start_recording_ui():
        tray.show("● REC — LIVE dictation (Groq)")
        _set_title("● REC — " + base_title)

    def stop_recording_ui():
        tray.hide()
        _set_title(base_title)

    def preview_job():
        with work_lock:
            if not state["recording"]:
                state["busy"] = False
                return
            snap = buffer.snapshot()
            if len(snap) < MIN_SAMPLES:
                state["busy"] = False
                return
            try:
                text = engine.transcribe_audio(snap)
                if text:
                    typer.append_preview(text)
                    show_preview(text)
            except Exception as e:
                print("\r[preview error] %s" % e, flush=True)
            finally:
                state["busy"] = False

    def do_final():
        try:
            with work_lock:
                snap = buffer.snapshot()
                text = engine.transcribe_audio(snap) if len(snap) >= MIN_SAMPLES else None
            clear_line()
            if text:
                print("✓ FINAL: %s" % text, flush=True)
                typer.finalize(text)
                print("✓ Final typed into the focused window", flush=True)
            else:
                print("[no speech detected]", flush=True)
        except Exception as e:
            clear_line()
            print("[finalize error] %s" % e, flush=True)
        finally:
            buffer.clear()
            stop_recording_ui()
            state["finalizing"] = False

    def safe_toggle():
        if state["recording"]:
            state["recording"] = False
            state["stop"] = True
            print("\r[STOP] finalizing...", flush=True)
        elif not state["finalizing"]:
            state["recording"] = True
            state["reset"] = True
            target = user32.GetForegroundWindow()
            if target and target == _get_console_window():
                print("\r\u26a0  Focus the target terminal (e.g. opencode) before speaking!", flush=True)
            _restore_target(target)
            start_recording_ui()
            print("\r[REC] speak now (Ctrl+Win to stop)", flush=True)
        else:
            print("\r[busy finalizing, wait a moment]", flush=True)

    def console_control():
        while True:
            try:
                line = input()
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
    print("Keep the target terminal (e.g. opencode) focused — the final message", flush=True)
    print("is typed there once you stop. A task bar icon marks recording.", flush=True)

    prev_hot = False
    last_refresh = time.time()
    try:
        while True:
            hot = hotkey_pressed()
            if hot and not prev_hot:
                prev_hot = True
                safe_toggle()
            elif not hot:
                prev_hot = False

            if state["reset"]:
                state["reset"] = False
                buffer.clear()
                last_refresh = time.time()

            if state["stop"]:
                state["stop"] = False
                state["finalizing"] = True
                audio = recorder.take()
                if audio is not None:
                    buffer.add(audio)
                threading.Thread(target=do_final, daemon=True).start()

            if state["recording"]:
                audio = recorder.take()
                if audio is not None:
                    buffer.add(audio)
                if (not state["busy"] and not state["finalizing"]
                        and buffer.samples >= MIN_SAMPLES
                        and time.time() - last_refresh >= REFRESH_SECONDS):
                    last_refresh = time.time()
                    state["busy"] = True
                    threading.Thread(target=preview_job, daemon=True).start()
            else:
                try:
                    recorder.take()
                except Exception:
                    pass
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        tray.hide()
        _set_title(base_title)
        stream.stop()


if __name__ == "__main__":
    if "--check" in sys.argv:
        cfg = _load_config()
        engine = _make_engine(cfg)
        print("OK: engine ready (model=%s, lang=%s, hotwords=%s)" % (
            MODEL, cfg["language"] or "auto", cfg["hotwords"]), flush=True)
        sys.exit(0)
    main()