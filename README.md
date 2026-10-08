# 🌐 Offline AI Voice Translator

A **completely offline** speech + text translator desktop application, built for
final-year engineering project demonstration, GitHub portfolio and placement
presentations.

No cloud APIs. No internet requirement at runtime. Everything runs locally on
your machine once the required AI models have been downloaded.

---

## 📖 Project Overview

The **Offline AI Voice Translator** lets you record your voice, automatically
transcribe it using OpenAI's Whisper, translate the resulting text between 16
languages (including Tamil, Telugu, Malayalam and Kannada) using Meta's
NLLB-200 model, and speak the translation back to you — all without a single
network call during actual translation.

| Capability            | Technology Used                         | Runs Offline? |
|------------------------|------------------------------------------|:---:|
| Speech-to-Text         | OpenAI Whisper (`tiny`/`base`/`small`/`medium`) | ✅ |
| Machine Translation    | Meta NLLB-200, served via CTranslate2   | ✅ |
| Text-to-Speech         | pyttsx3 (native OS voices)               | ✅ |
| Desktop GUI            | CustomTkinter                            | ✅ |

> ⚠️ Argos Translate is intentionally **not** used, since it does not
> reliably support Tamil, Telugu, Malayalam or Kannada. NLLB-200 was chosen
> specifically because Meta trained and validated it on all four languages.

---

## 🏗️ Architecture Diagram

```
                       ┌───────────────────────────┐
                       │   whisper_voice_translator │
                       │        (CustomTkinter GUI) │
                       └──────────────┬─────────────┘
                                      │ orchestrates
        ┌────────────────┬───────────┼───────────┬────────────────┐
        ▼                ▼           ▼           ▼                ▼
 ┌─────────────┐  ┌─────────────┐ ┌─────────┐ ┌───────────┐ ┌───────────┐
 │  audio.py   │  │   stt.py    │ │translate│ │  tts.py   │ │ utils.py  │
 │ sounddevice │  │  Whisper    │ │  .py    │ │ pyttsx3   │ │ history/  │
 │ record/play │  │ (local pt)  │ │ NLLB+CT2│ │  speech   │ │ export/   │
 └─────────────┘  └─────────────┘ └─────────┘ └───────────┘ └───────────┘
        │                │              │            │
        ▼                ▼              ▼            ▼
   audio/*.wav     models/whisper/  models/nllb/   OS voice engine
```

---

## 📁 Folder Structure

```
offline-translator/
│
├── app/
│   ├── __init__.py
│   ├── audio.py          # Recording / playback (sounddevice)
│   ├── language.py       # Supported languages + code mappings
│   ├── stt.py             # Offline Whisper speech-to-text
│   ├── translate.py      # Offline NLLB-200 translation (CTranslate2)
│   ├── tts.py             # Offline pyttsx3 text-to-speech
│   └── utils.py           # History, export, clipboard, helpers
│
├── models/
│   ├── whisper/            # Place tiny.pt / base.pt / small.pt / medium.pt here
│   └── nllb/                # Place the CTranslate2-converted NLLB-200 model here
│
├── model-en/                # Reserved for optional English-specific assets
├── audio/                   # Auto-saved recordings (timestamped .wav files)
├── Output/                   # Default location for exported TXT/JSON files
├── history.json              # Persistent translation history
├── whisper_voice_translator.py  # Application entry point
├── requirements.txt
└── README.md
```

---

## ✨ Features

- 🎙️ Unlimited-length recording with record / pause / resume / stop
- 📈 Live waveform visualisation and recording timer
- 🧠 Automatic spoken-language detection with a low-confidence fallback prompt
- 🌍 16 supported languages, easily extended in `app/language.py`
- 🔁 Offline translation cached in memory (model loaded once, never reloaded)
- 🔊 Text-to-speech with voice, speed and volume controls, plus stop/replay
- 🌗 Dark / Light theme switch with a modern, rounded, responsive UI
- 📜 Full history log with TXT/JSON export
- 📋 Copy / paste / clear shortcuts
- 🛡️ Defensive error handling — the app never crashes; every failure shows a
  friendly dialog instead (missing mic, missing models, missing ffmpeg, etc.)
- ⚡ GPU (CUDA) auto-detection for both Whisper and NLLB, with automatic CPU fallback

---

## 🖼️ Screenshots

> _Add your own screenshots here before submitting your project report._

```
[ Screenshot: Main window - Dark theme ]
[ Screenshot: Recording in progress with waveform ]
[ Screenshot: Translation result with target language TTS ]
[ Screenshot: History window ]
```

---

## 🔧 Installation Guide

### 1. Python 3.11 Setup

Download and install **Python 3.11** from [python.org](https://www.python.org/downloads/).

```bash
python3.11 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. FFmpeg Setup

Whisper requires `ffmpeg` to decode audio.

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin/` folder to your PATH.
- **macOS:** `brew install ffmpeg`
- **Linux (Debian/Ubuntu):** `sudo apt install ffmpeg`

### 3. Whisper Model Setup

Download one or more official Whisper model files (`tiny.pt`, `base.pt`,
`small.pt`, `medium.pt`) and place them inside:

```
models/whisper/
```

Official model download links are listed at:
https://github.com/openai/whisper#available-models-and-languages

> The application will **never** download a model automatically. If a
> selected model is missing, a dialog box will tell you exactly where to
> place the file.

### 4. NLLB Model Setup

Download the `facebook/nllb-200-distilled-600M` (or a larger NLLB-200
checkpoint) from Hugging Face, then convert it to CTranslate2 format (see
below) and place the resulting folder contents inside:

```
models/nllb/
```

The folder must contain `model.bin` plus the original tokenizer files
(`sentencepiece.bpe.model`, `tokenizer.json`, `special_tokens_map.json`,
`tokenizer_config.json`).

### 5. How to Convert NLLB to CTranslate2

```bash
pip install ctranslate2 transformers sentencepiece huggingface_hub

ct2-transformers-converter \
    --model facebook/nllb-200-distilled-600M \
    --output_dir models/nllb \
    --quantization int8

# Copy the tokenizer files alongside the converted model:
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
tok.save_pretrained('models/nllb')
"
```

> `--quantization int8` significantly reduces model size and speeds up CPU
> inference with minimal accuracy loss — recommended for laptops without a GPU.

---

## ▶️ Running the Project

```bash
python whisper_voice_translator.py
```

The application will:
1. Launch the CustomTkinter GUI.
2. Wait for you to select source/target languages and a Whisper model size.
3. Record audio, transcribe it, translate it and optionally speak it back —
   loading each AI model only once and reusing it for the rest of the session.

---

## 🩺 Troubleshooting

| Problem | Solution |
|---|---|
| "No microphone was detected" | Connect a microphone and check OS privacy/audio permissions. |
| "Whisper model was not found" | Download the `.pt` file for the selected size into `models/whisper/`. |
| "NLLB-200 translation model was not found" | Follow the conversion steps above and place files in `models/nllb/`. |
| Transcription fails / audio decoding error | Confirm `ffmpeg` is installed and available on your PATH. |
| No sound during playback/TTS | Check system audio output device and OS volume mixer. |
| Slow translation on CPU | Use the `int8` quantized NLLB export, or run on a CUDA-enabled GPU. |

---

## 🚀 Future Improvements

- Batch translation of subtitle/document files
- Real-time streaming translation (speech-to-speech pipeline)
- Additional Indian regional languages (Bengali, Marathi, Punjabi, Gujarati)
- Packaging as a standalone `.exe` / `.app` installer with PyInstaller
- Speaker diarization for multi-speaker conversations

---

## 📄 License

This project is released under the MIT License. You are free to use, modify
and distribute it for educational and personal purposes.
