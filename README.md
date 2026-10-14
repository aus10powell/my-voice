# My Voice

Local push-to-talk voice-to-text. Hold a global hotkey, speak, release — offline ASR transcribes and pastes at the cursor.

Hobby / clone-and-run utility. No cloud API keys. Default backend: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). Alternate engines (whisper.cpp, NVIDIA Parakeet) plug in behind the same `AsrBackend` interface.

## Quickstart

```bash
# System deps (Debian/Ubuntu example)
sudo apt install -y python3-venv python3-dev portaudio19-dev libsndfile1

git clone https://github.com/aus10powell/my-voice.git
cd my-voice
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"         # app + pytest; runtime-only: pip install -e .

my-voice devices
my-voice backends
my-voice --print-only           # hold F8, speak, release (no paste)
pytest
```

**Default hotkey:** hold `F8` to record, release to transcribe and paste.

### Optional ASR engines

```bash
# whisper.cpp (pywhispercpp wheels)
pip install -e ".[whispercpp]"

# NVIDIA Parakeet via NeMo (heavy: CPU PyTorch + nemo-toolkit)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "nemo_toolkit[asr]"
```

macOS: `brew install portaudio libsndfile`, then grant **Accessibility**, **Input Monitoring**, and **Microphone** to your terminal (see [OS caveats](#os-permissions--caveats)).

## Cache directories (outside the repo)

| Cache | Default path | Contents |
|-------|--------------|----------|
| Eval datasets | `~/.cache/my-voice/` | LibriSpeech archives + extract (`--cache-dir` to override) |
| whisper.cpp weights | `~/.cache/my-voice/whispercpp-models/` | ggml `*.bin` |
| Faster-Whisper / HF | `~/.cache/huggingface/` | CTranslate2 + NeMo/HF weights |

## Backends

| `--backend` | Engine | Typical `--model` |
|-------------|--------|-------------------|
| `faster-whisper` (default) | Faster-Whisper / CTranslate2 | `tiny`, `base`, `small`, … |
| `whisper.cpp` | whisper.cpp via pywhispercpp | `tiny`, `base`, `small`, … |
| `parakeet` | NVIDIA Parakeet via NeMo | `nvidia/parakeet-tdt_ctc-110m` (default), or `nvidia/parakeet-tdt-0.6b-v2` |

```bash
my-voice run --backend faster-whisper --model base
my-voice run --backend whisper.cpp --model base
my-voice run --backend parakeet --model nvidia/parakeet-tdt_ctc-110m
```

## Run

```bash
my-voice                         # same as: my-voice run
my-voice run --model base --hotkey f8
my-voice run --print-only
my-voice devices
my-voice -v eval --download-only
```

| Flag | Meaning |
|------|---------|
| `--hotkey f8` | Hold-to-record key |
| `--backend …` | ASR engine (see above) |
| `--model …` | Size or HF model id |
| `--print-only` | Skip clipboard paste |
| `--device cpu` | Force CPU (default `auto`) |
| `-v` | Debug logging |

## OS permissions / caveats

### macOS

1. **Accessibility** — `pynput` key observe + Cmd+V
2. **Input Monitoring** — global hotkeys outside the terminal
3. **Microphone** — grant to Terminal / iTerm

### Linux

- Wayland: global hotkeys via `pynput` can be unreliable; X11 is better tested.
- `python3-dev` is required so `pynput` can build `evdev`.

### Windows

- Mic permission prompt on first run; AV may block keyboard hooks.

### CI / headless

Unit tests need no mic. Full push-to-talk needs a desktop session.

## Manual smoke test

1. `pip install -e ".[dev]"`
2. `my-voice devices`
3. `my-voice --print-only`, hold `F8`, speak, release
4. Re-run without `--print-only` to confirm paste

## ASR eval (LibriSpeech)

Same protocol across backends. Audio is **not** vendored — cached under `~/.cache/my-voice/`.

**Default:** LibriSpeech `test-clean`, first **20** utterances, `--language en`, CPU.

```bash
my-voice eval --download-only

my-voice eval --backend faster-whisper --model tiny --max-utterances 20 --device cpu --language en
my-voice eval --backend faster-whisper --model base --max-utterances 20 --device cpu --language en
my-voice eval --backend whisper.cpp --model base --max-utterances 20 --device cpu --language en
my-voice eval --backend parakeet --model nvidia/parakeet-tdt_ctc-110m --max-utterances 20 --device cpu --language en
```

**Follow-on corpus:** Mozilla Common Voice (not wired yet).

## Project layout

```
src/my_voice/
  app.py              # hold-to-record loop
  audio.py / paste.py
  asr/                # replaceable backends
  eval/               # LibriSpeech download + WER/latency
```

## License

MIT
