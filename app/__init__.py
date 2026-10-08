"""
Offline AI Voice Translator
============================
Application package containing all business-logic modules:

    - audio.py      : Microphone recording / playback (sounddevice)
    - stt.py        : Offline speech-to-text (OpenAI Whisper)
    - translate.py  : Offline neural machine translation (Meta NLLB-200 + CTranslate2)
    - tts.py        : Offline text-to-speech (pyttsx3)
    - language.py   : Supported language catalogue and code mappings
    - utils.py      : History persistence, export, clipboard and file helpers

The GUI (whisper_voice_translator.py) only orchestrates these modules and
never contains business logic itself, keeping a clean separation of concerns.
"""

__version__ = "1.0.0"
__author__ = "Offline AI Voice Translator Team"
