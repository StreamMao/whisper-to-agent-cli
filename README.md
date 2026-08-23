# whisper-to-agent-cli — Voice Input for Any Agent CLI (WSL / Windows)

Local + cloud speech-to-text so you can dictate prompts and commands straight into
any agent CLI running in Windows Terminal — **opencode**, **Claude Code**, **Codex**, **Gemini CLI**, and any other terminal. Works via global hotkey + `SendInput` typing into the focused window, not tied to opencode.

> Formerly `whisperkey` / `Voice Input for opencode`. Renamed to reflect universal terminal support.

Location (example): `D:\Ethan_WorkingSpace\tools\whisper-to-agent-cli\` (or wherever you clone it — `.bat` launchers now use relative `%~dp0` paths so the repo is portable)

---

## Three tools

| Tool | Behaviour | When to use |
|---|---|---|
| `whisper-key` (batch, local) | Speak, stop, then the full transcript is pasted at your cursor | Short, precise prompts where you review before sending |
| `whisper-key-groq` (batch, cloud) | Same flow but transcribed by Groq's `whisper-large-v3-turbo` (~0.3s, much more accurate) | When you want speed + accuracy and have internet |
| `stream_dictation.py` (live, offline) | sherpa-onnx streams text in ~2–4s rolling chunks *while* you speak | Long dictation, fully offline |
| `stream_dictation_groq.py` (live, cloud) | Groq re-transcribes the audio-so-far every ~2.5s and the transcript **grows incrementally** in the chat (never re-typed); a task-bar icon marks recording; on stop the final message replaces it | Live preview with the accurate online model |

> Run only one tool at a time — they all share the `Ctrl+Win` hotkey.

---

## 1. whisper-key (batch push-to-talk)

### Launch
Double-click `run-whisper-key.bat`.

### Hotkeys
| Key | Action |
|---|---|
| `Ctrl+Win` | Start / stop recording |
| `Ctrl` | Stop & paste transcript at cursor |
| `Alt` | Stop, paste, and press Enter (auto-send) |
| `Esc` | Cancel recording |
| `Alt+Win` | Voice-command mode (triggers configured in `commands.yaml`) |

### Config
`appdata\whisperkey\user_settings.yaml` — e.g. model size (`whisper.model`),
language (`whisper.language: zh` is set because your dictation is Chinese-dominant:
forcing `zh` roughly halves transcription time and improves accuracy; set it back
to `auto` if you dictate mostly English), hotwords. Changes take effect after
restart.

> **Accuracy/speed expectations (local, on a dual-core CPU):** with `base` model
> + forced `zh`, a ~6s clip takes ~2–4s to transcribe and loads in ~1.5s (vs
> `small`: ~6–8s + ~4.5s load). The `base` model still has real mixed zh/en
> limits (e.g. `识别得出来` may come out as `实际的出来`, `file吗` as `发文吗`).
> For the fastest accurate mixed-language results prefer `run-whisper-key-groq`;
> the local tool is the offline/simple option.

### How it runs
`run-whisper-key.bat` launches the same whisper-key package via
`run_whisper_key_local.py` (Python) rather than the old compiled `whisper-key.exe`.
That wrapper:
- runs the transcribe→paste pipeline on a background thread, so hotkeys stay
  responsive and `Ctrl+Win` works again right after a recording (fixes the
  previous freeze-until-Ctrl+C bug);
- applies the same post-fix layer as the Groq tool (`fixups.py`): `X的ext` →
  `X.ext` for any known extension, ASCII extensions lowercased, and tokens
  listed under `whisper.hotwords` normalized to your spelling;
- enables faster-whisper's `vad_filter` (bundled Silero VAD trims silence before
  transcription — small accuracy + speed win).
The `whisper_engine.py` VAD patch lives in `venv\Lib\site-packages\whisper_key\`;
it reverts if that package is ever reinstalled.

---

## 1b. whisper-key-groq (batch push-to-talk, cloud transcription)

Identical flow to whisper-key but sends the recorded audio to Groq's
`whisper-large-v3-turbo` (a hosted Whisper) instead of running a small model
locally. On this dual-core laptop it's both **much faster** (~0.3s) and **much
more accurate**, especially for mixed Chinese/English.

### One-time setup
1. Get a free key at <https://console.groq.com/keys>.
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY=gsk_...` inside (or `setx GROQ_API_KEY ...`).
3. Run `run-whisper-key-groq.bat`.

Requires internet; no local model to download.

### Notes
- Hotkeys and clipboard behaviour are identical to whisper-key (same config file).
- Model defaults to `whisper-large-v3-turbo`; override with `GROQ_MODEL` env var.
- The `whisper.model`, `beam_size`, and `post_processing.corrections` settings in
  `user_settings.yaml` are ignored by the Groq backend (Groq picks its own
  parameters); `whisper.language`, `whisper.initial_prompt`, and
  `whisper.hotwords` are respected.
- Network calls are bounded: a Groq request times out after ~15s and is retried
  (up to 2 more times, 1s/2s backoff) on connection resets (`[WinError 10054]`),
  rate limits (429), or server errors (5xx).
- Groq runs with `temperature=0.0` (deterministic) and the response passes
  through a post-fix layer in `groq_engine.py` that:
  - collapses spaces inside ASCII filenames (`README . md` → `README.md`),
  - turns `X的YYY` into `X.YYY` when `YYY` is a known file extension
    (`readme的md` → `readme.md`; works for *any* extension in `KNOWN_EXTENSIONS`),
  - lowercases ASCII extensions (`file.PY` → `file.py`),
  - normalizes tokens listed under `whisper.hotwords` in `user_settings.yaml`
    to your exact spelling, and feeds them into the transcription prompt so
    they're more likely to be transcribed correctly in the first place. Add any
    word/filename you dictate often there.
- Transcription runs on a background thread so hotkeys stay responsive even while
  a Groq request is in flight. If you press a key mid-transcription you may see
  `⏳ Still processing previous recording...` — that's expected, press again.
 - **Focus matters for the paste, not for the hotkeys:** hotkeys are global, so
  they work no matter which window is focused. But the simulated `Ctrl+V` pastes
  into whatever window has *focus*. Keep the target agent terminal (opencode / Claude Code / Codex / etc.) focused before you
  press `Ctrl` to stop — if you're watching the `.bat` console instead, the text
  will land in that console, not in the agent.

---

## 2. stream_dictation.py (live typing — sherpa-onnx)

Uses a **real-time streaming** bilingual (Mandarin + English) model that runs on CPU
with ~0.4s latency — no GPU needed, fully offline.

### Launch
Double-click `run-stream-dictation.bat`. A console opens, loads the model, then:

1. Click into the **target agent terminal** (opencode / Claude Code / Codex / Gemini CLI — it must have focus).
2. Press **`Ctrl+Win`** (or **`Ctrl+Alt+Space`**) to start recording.
3. Speak in **English or Mandarin** (auto-detected) — text types in as you talk.
4. Press **`Ctrl+Win`** again to stop.

What was typed also prints in the console as `[typed] ...`. Fallback controls in the
console window itself: type `r` + Enter to toggle recording, `q` + Enter to quit.

### Tuning
Edit `stream_dictation.py`:

| Setting | Default | Notes |
|---|---|---|
| `FEED_SECONDS` | `0.4` | audio fed to the engine per pass (do not go below ~0.4 — the encoder needs ≥39 feature frames) |
| `WK_QUANT` | `fp32` | set `WK_QUANT=int8` for a ~2x faster but noticeably less accurate build (helps a lot with mixed zh/en accuracy to keep fp32) |
| `WK_HOTWORDS` | `sherpa\...\hotwords.txt` | file of English words/phrases to bias the decoder toward (e.g. `README`, `MD`). Empty if file missing |
| `WK_HOTWORDS_SCORE` | `3.0` | boost strength for hotwords; raise if a word is still mis-heard, lower if it over-corrects |
| `WK_BEAM` | `8` | `max_active_paths` for modified beam search — higher = better accuracy, slower |
| `CPU_THREADS` | half your cores | `num_threads` for the model |
| `MODEL_DIR` | `sherpa\...bilingual-zh-en...` | point at another sherpa-onnx streaming model dir (must contain `tokens.txt`, `*-encoder*.onnx`, etc.) |
| `rule1/2_min_trailing_silence` | `2.4` / `1.2` | seconds of silence before the utterance is committed/reset |

The recognizer emits words continuously as you speak; it auto-resets after a pause
(rule1/2), so short phrases flow without pressing any extra key.

---

## 2b. stream_dictation_groq.py (live preview — Groq, cloud)

Groq has no true streaming endpoint, so this tool does **chunked pseudo-streaming**:
while you speak it re-sends the accumulated audio to Groq every ~2.5s. Each refresh
**appends only the new tail** of the transcript to the focused terminal — old text is
never deleted or re-typed, so the chat box only grows (no "refresh" flicker). When
you stop, the full audio is transcribed once more and the preview is replaced by the
**final message** (backspaced once, then typed in full).

It reuses the exact engine as `whisper-key-groq` — same model, `temperature=0.0`,
the bilingual prompt, `whisper.hotwords`, and the post-fix layer — so the preview
and final text get the same accuracy benefits.

### Launch
Double-click `run-stream-dictation-groq.bat` (requires internet + your Groq key).
It opens in Windows Terminal when available, otherwise a normal console.

1. Click into the **target agent terminal** (opencode / Claude Code / Codex / Gemini CLI) — the live preview and the final message
    are typed into the focused window.
2. Press **`Ctrl+Win`** (or **`Ctrl+Alt+Space`**) to start recording — a task bar /
    notification-area icon appears and the console window title shows `● REC`.
3. Speak — the transcript appears in the chat, **growing incrementally** with each
    ~2.5s refresh (a `[LIVE]` mirror also updates in the tool's own console).
4. Press **`Ctrl+Win`** again to stop — the preview is replaced by the full,
    corrected **final text** (`✓ FINAL:` is printed in the console).

Fallback controls in the console window: `r` + Enter to toggle, `q` + Enter to quit.
Run `python stream_dictation_groq.py --check` to verify the API key and config.

### Tuning
Edit `stream_dictation_groq.py` or set env vars:

| Setting | Default | Notes |
|---|---|---|
| `WK_REFRESH` | `2.5` | seconds between live preview refreshes (each re-transcribes audio-so-far) |
| `WK_MIN_AUDIO` | `1.0` | minimum seconds of audio before the first preview is sent |
| `WK_MAX_AUDIO` | `60.0` | hard cap on the audio buffer (older audio is dropped) |
| `WK_LANG` | from `user_settings.yaml` | override language (`zh`, `en`, …) |
| `GROQ_MODEL` | `whisper-large-v3-turbo` | model override |

---

## Structure

```
whisper-to-agent-cli\
├── venv\                    # Python 3.13 venv (faster-whisper, sherpa-onnx, etc.)
├── models\                  # faster-whisper model cache (HF_HOME)
├── sherpa\                  # sherpa-onnx streaming model (bilingual zh/en)
├── pip-cache\               # pip download cache
├── appdata\whisperkey\      # whisper-key config + logs (kept for compat with upstream)
├── run-whisper-key.bat      # launch batch tool (→ run_whisper_key_local.py)
├── run-whisper-key-groq.bat # launch cloud batch tool (→ run_whisper_key_groq.py)
├── run-stream-dictation.bat  # launch live tool (offline sherpa)
├── run-stream-dictation-groq.bat # launch live tool (cloud Groq)
├── run_whisper_key_local.py # local wrapper: async pipeline + post-fix
├── run_whisper_key_groq.py  # groq wrapper: async pipeline + post-fix
├── groq_engine.py           # Groq API engine
├── fixups.py                # shared post-fix rules (both engines)
├── stream_dictation.py      # live streaming script (sherpa-onnx)
├── stream_dictation_groq.py # live preview script (Groq, cloud)
└── README.md
```

---

## Troubleshooting

- **Hotkeys stop working / nothing happens after a recording (esp. with `-groq`):**
  the transcription pipeline now runs on a background thread (see Notes above), and
  Groq requests are bounded + retried, so a slow/flaky network call can no longer
  freeze hotkey detection. If you still feel it is stuck, it's usually because the
  text was pasted somewhere unexpected — see the focus note above.
- **Nothing typed / text lands elsewhere:** the app must run from **Windows**, not WSL
  (env vars don't propagate across the WSL boundary — the `.bat` sets them natively),
  and the target terminal must have **focus**.
- **Model downloaded to C:** — that only happens if env vars are missing; the `.bat`
  files set `HF_HOME` to `D:\...\models`. Old C: copies can be deleted from
  `C:\Users\maosi\.cache\huggingface\hub\`.
- **Poor Mandarin accuracy:** for the streaming tool use the bilingual model (it handles
  zh/en automatically); for whisper-key set `whisper.model` to `small` in
  `user_settings.yaml`.
- **Duplicated words:** if the streaming engine re-emits a word, check `FEED_SECONDS`
  is ≥0.4 (too-small feeds can cause odd behaviour).
- **English words mis-heard (e.g. `readme` → `REDMIT`):** the streaming model outputs
  English in all-caps and never emits punctuation like `.`/`_`. Add the word to
  `sherpa\sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20\hotwords.txt`
  (uppercase, one per line) and restart — e.g. `README`, `MD`. You'll get
  `README`/`MD` instead of `REDMIT`, but the `.` must be typed manually. For
  accurate filenames/code, use **whisper-key** (batch) instead.
- **Symlink warning:** harmless (NTFS without Developer Mode). Enable Windows
  Developer Mode to silence it.
