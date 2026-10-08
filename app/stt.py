"""
app/stt.py
==========
Offline speech-to-text powered by OpenAI Whisper.

The model is loaded exactly once (see :class:`WhisperSTT`) and reused for
every subsequent transcription, keeping memory usage and latency low.

Models are expected to live inside ``models/whisper``. If the requested
model file is missing, :class:`WhisperModelNotFoundError` is raised so the
GUI layer can show a friendly dialog instead of crashing - the application
never attempts to auto-download a model on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    import whisper
    _WHISPER_AVAILABLE = True
except Exception:  # pragma: no cover
    _WHISPER_AVAILABLE = False

from app.language import get_by_whisper_code

SUPPORTED_MODEL_SIZES = ("tiny", "base", "small", "medium")

# Confidence below this threshold means the GUI should ask the user to
# manually confirm/select the spoken language instead of trusting Whisper.
LOW_CONFIDENCE_THRESHOLD = 0.55


class WhisperModelNotFoundError(Exception):
    """Raised when the requested Whisper model is not present locally."""


class SpeechToTextError(Exception):
    """Raised for any other recoverable STT failure."""


@dataclass
class TranscriptionResult:
    """Result returned after transcribing an audio file."""

    text: str
    detected_language_code: str
    detected_language_name: str
    confidence: float
    is_low_confidence: bool


class WhisperSTT:
    """Thin, GUI-agnostic wrapper around the offline Whisper model."""

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self._model = None
        self._loaded_size: Optional[str] = None
        self._device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    @property
    def device(self) -> str:
        """Return 'cuda' or 'cpu', whichever this engine is running on."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_size(self) -> Optional[str]:
        return self._loaded_size

    def _model_file_path(self, model_size: str) -> str:
        return os.path.join(self.models_dir, f"{model_size}.pt")

    def is_model_available(self, model_size: str) -> bool:
        """Check whether the given model's weight file exists locally."""
        return os.path.exists(self._model_file_path(model_size))

    def load_model(self, model_size: str) -> None:
        """Load (once) the requested Whisper model from ``models/whisper``.

        Raises:
            WhisperModelNotFoundError: if the weight file is missing.
            SpeechToTextError: if the whisper/torch libraries are missing
                or loading fails for any other reason.
        """
        if model_size not in SUPPORTED_MODEL_SIZES:
            raise SpeechToTextError(f"Unsupported Whisper model size: {model_size}")

        if self._model is not None and self._loaded_size == model_size:
            return  # already loaded - never reload

        if not _WHISPER_AVAILABLE:
            raise SpeechToTextError(
                "The 'openai-whisper' package is not installed. "
                "Please install it with: pip install openai-whisper"
            )

        if not self.is_model_available(model_size):
            raise WhisperModelNotFoundError(
                f"Whisper '{model_size}' model was not found.\n\n"
                f"Please download '{model_size}.pt' and place it inside:\n"
                f"{os.path.abspath(self.models_dir)}\n\n"
                "Official model files are published by OpenAI at:\n"
                "https://github.com/openai/whisper#available-models-and-languages"
            )

        try:
            self._model = whisper.load_model(
                model_size,
                device=self._device,
                download_root=self.models_dir,
            )
            self._loaded_size = model_size
        except Exception as exc:
            self._model = None
            self._loaded_size = None
            raise SpeechToTextError(f"Failed to load Whisper model: {exc}") from exc

    def transcribe(self, audio_path: str, language_hint: Optional[str] = None) -> TranscriptionResult:
        """Transcribe an audio file and detect its spoken language.

        Args:
            audio_path: Path to a WAV file to transcribe.
            language_hint: Optional Whisper language code to force a
                particular language instead of auto-detecting it.
        """
        if self._model is None:
            raise SpeechToTextError("Whisper model has not been loaded yet.")
        if not os.path.exists(audio_path):
            raise SpeechToTextError(f"Audio file not found: {audio_path}")

        try:
            result = self._model.transcribe(
                audio_path,
                language=language_hint,
                fp16=(self._device == "cuda"),
            )
        except Exception as exc:
            raise SpeechToTextError(f"Transcription failed: {exc}") from exc

        text = (result.get("text") or "").strip()
        detected_code = result.get("language", "en")

        # Whisper does not return a single scalar confidence value, so we
        # derive an approximate score from the average of the per-segment
        # "no speech" probabilities (inverted) and average log-probability.
        segments = result.get("segments") or []
        if segments:
            avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
            no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
            # avg_logprob is typically in [-1, 0]; map to [0, 1]
            confidence = max(0.0, min(1.0, 1.0 + avg_logprob)) * (1.0 - no_speech)
        else:
            confidence = 0.5

        language_entry = get_by_whisper_code(detected_code)
        language_name = language_entry.name if language_entry else detected_code

        return TranscriptionResult(
            text=text,
            detected_language_code=detected_code,
            detected_language_name=language_name,
            confidence=round(confidence, 3),
            is_low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
        )
