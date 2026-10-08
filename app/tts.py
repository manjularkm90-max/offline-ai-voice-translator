"""
app/tts.py
==========
Offline text-to-speech powered exclusively by ``pyttsx3`` (which itself
wraps native OS engines: SAPI5 on Windows, NSSpeechSynthesizer on macOS,
and espeak on Linux) - no network access is ever required.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except Exception:  # pragma: no cover
    _PYTTSX3_AVAILABLE = False


class TextToSpeechError(Exception):
    """Raised for any recoverable TTS failure."""


@dataclass
class VoiceInfo:
    """Simplified description of a voice exposed by the OS engine."""

    id: str
    name: str
    gender: str  # "male", "female" or "unknown"


class TTSEngine:
    """Thin, thread-safe wrapper around pyttsx3 with replay support."""

    def __init__(self) -> None:
        self._engine = None
        self._lock = threading.Lock()
        self._speak_thread: Optional[threading.Thread] = None
        self._last_text: str = ""
        self._rate = 175
        self._volume = 1.0
        self._voice_id: Optional[str] = None
        self._is_speaking = False

        if _PYTTSX3_AVAILABLE:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self._rate)
                self._engine.setProperty("volume", self._volume)
            except Exception:
                self._engine = None

    @property
    def is_available(self) -> bool:
        return self._engine is not None

    def get_voices(self) -> List[VoiceInfo]:
        """Return every voice the OS speech engine exposes."""
        if self._engine is None:
            return []
        voices = []
        try:
            for v in self._engine.getProperty("voices"):
                name_lower = (v.name or "").lower()
                gender = "unknown"
                if "female" in name_lower or "zira" in name_lower or "susan" in name_lower:
                    gender = "female"
                elif "male" in name_lower or "david" in name_lower or "mark" in name_lower:
                    gender = "male"
                voices.append(VoiceInfo(id=v.id, name=v.name, gender=gender))
        except Exception as exc:
            raise TextToSpeechError(f"Could not read available voices: {exc}") from exc
        return voices

    def set_voice(self, voice_id: str) -> None:
        """Select a specific voice by its engine-provided id."""
        if self._engine is None:
            raise TextToSpeechError("Text-to-speech engine is not available.")
        try:
            self._engine.setProperty("voice", voice_id)
            self._voice_id = voice_id
        except Exception as exc:
            raise TextToSpeechError(f"Could not set voice: {exc}") from exc

    def set_rate(self, words_per_minute: int) -> None:
        """Set speaking speed (typical range: 100-250)."""
        if self._engine is None:
            return
        self._rate = words_per_minute
        self._engine.setProperty("rate", words_per_minute)

    def set_volume(self, volume: float) -> None:
        """Set speaking volume, from 0.0 (silent) to 1.0 (full)."""
        if self._engine is None:
            return
        self._volume = max(0.0, min(1.0, volume))
        self._engine.setProperty("volume", self._volume)

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def speak(self, text: str, on_finished=None) -> None:
        """Speak the given text asynchronously on a background thread."""
        if self._engine is None:
            raise TextToSpeechError("Text-to-speech engine is not available.")
        if not text or not text.strip():
            return

        self._last_text = text
        self.stop()

        def _worker():
            with self._lock:
                self._is_speaking = True
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass
                finally:
                    self._is_speaking = False
                    if on_finished:
                        on_finished()

        self._speak_thread = threading.Thread(target=_worker, daemon=True)
        self._speak_thread.start()

    def replay(self, on_finished=None) -> None:
        """Speak the most recently spoken text again."""
        if not self._last_text:
            raise TextToSpeechError("There is no previous speech to replay.")
        self.speak(self._last_text, on_finished=on_finished)

    def stop(self) -> None:
        """Stop any speech currently in progress."""
        if self._engine is None:
            return
        try:
            self._engine.stop()
        except Exception:
            pass
        self._is_speaking = False
