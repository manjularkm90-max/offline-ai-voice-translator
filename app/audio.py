"""
app/audio.py
============
Microphone recording and WAV playback, built entirely on top of
``sounddevice`` + ``soundfile`` so the whole pipeline stays 100% offline.

Design notes
------------
* Recording happens on a background thread so the GUI never freezes.
* The recorder supports record / pause / resume / stop with unlimited
  duration - audio chunks are appended to an in-memory list and only
  written to disk once recording stops.
* A small ring-buffer of the most recent samples is exposed through
  :meth:`AudioRecorder.get_waveform` so the GUI can draw a live waveform.
* Every recording is saved with a timestamped filename inside ``audio/``.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    _AUDIO_BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover - triggered only if PortAudio is missing
    _AUDIO_BACKEND_AVAILABLE = False


class AudioError(Exception):
    """Raised for any recoverable audio related failure."""


@dataclass
class RecordingResult:
    """Container returned once a recording has been stopped and saved."""

    file_path: str
    duration_seconds: float
    sample_rate: int


class AudioRecorder:
    """Handles microphone capture, pause/resume/stop and WAV playback."""

    def __init__(self, output_dir: str, sample_rate: int = 16000, channels: int = 1) -> None:
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.channels = channels

        os.makedirs(self.output_dir, exist_ok=True)

        self._stream: Optional["sd.InputStream"] = None
        self._frames: List[np.ndarray] = []
        self._recording = False
        self._paused = False
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._elapsed_before_pause = 0.0
        self._waveform_buffer = np.zeros(400, dtype=np.float32)

        # Playback state
        self._play_thread: Optional[threading.Thread] = None
        self._stop_playback = threading.Event()

    # ------------------------------------------------------------------
    # Device checks
    # ------------------------------------------------------------------
    @staticmethod
    def is_backend_available() -> bool:
        """Return True if sounddevice/PortAudio could be imported."""
        return _AUDIO_BACKEND_AVAILABLE

    @staticmethod
    def has_input_device() -> bool:
        """Return True if at least one microphone is detected."""
        if not _AUDIO_BACKEND_AVAILABLE:
            return False
        try:
            devices = sd.query_devices()
            return any(d.get("max_input_channels", 0) > 0 for d in devices)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Recording controls
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Begin recording from the default microphone."""
        if not _AUDIO_BACKEND_AVAILABLE:
            raise AudioError("Audio backend (PortAudio) is not available on this system.")
        if not self.has_input_device():
            raise AudioError("No microphone was detected. Please connect one and try again.")
        if self._recording:
            return

        self._frames = []
        self._recording = True
        self._paused = False
        self._elapsed_before_pause = 0.0
        self._start_time = time.time()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if self._recording and not self._paused:
                with self._lock:
                    self._frames.append(indata.copy())
                    mono = indata[:, 0] if indata.ndim > 1 else indata
                    step = max(1, len(mono) // 50)
                    sample = mono[::step][:50]
                    self._waveform_buffer = np.roll(self._waveform_buffer, -len(sample))
                    self._waveform_buffer[-len(sample):] = sample

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=_callback,
            )
            self._stream.start()
        except Exception as exc:
            self._recording = False
            raise AudioError(f"Could not start recording: {exc}") from exc

    def pause(self) -> None:
        """Pause an in-progress recording (frames stop accumulating)."""
        if self._recording and not self._paused:
            self._paused = True
            self._elapsed_before_pause += time.time() - self._start_time

    def resume(self) -> None:
        """Resume a paused recording."""
        if self._recording and self._paused:
            self._paused = False
            self._start_time = time.time()

    def stop(self) -> RecordingResult:
        """Stop recording, persist the WAV file and return its metadata."""
        if not self._recording:
            raise AudioError("No active recording to stop.")

        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if not self._frames:
            raise AudioError("Recording produced no audio data.")

        audio_data = np.concatenate(self._frames, axis=0)
        filename = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        file_path = os.path.join(self.output_dir, filename)

        try:
            sf.write(file_path, audio_data, self.sample_rate)
        except Exception as exc:
            raise AudioError(f"Failed to save recording: {exc}") from exc

        duration = len(audio_data) / float(self.sample_rate)
        return RecordingResult(file_path=file_path, duration_seconds=duration, sample_rate=self.sample_rate)

    def get_elapsed_seconds(self) -> float:
        """Return total elapsed recording time, accounting for pauses."""
        if not self._recording:
            return self._elapsed_before_pause
        if self._paused:
            return self._elapsed_before_pause
        return self._elapsed_before_pause + (time.time() - self._start_time)

    def get_waveform(self) -> np.ndarray:
        """Return a small array of recent samples for waveform visualisation."""
        with self._lock:
            return self._waveform_buffer.copy()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def play_file(self, file_path: str, on_finished=None) -> None:
        """Play back a WAV file asynchronously on a background thread."""
        if not _AUDIO_BACKEND_AVAILABLE:
            raise AudioError("Audio backend (PortAudio) is not available on this system.")
        if not os.path.exists(file_path):
            raise AudioError(f"Audio file not found: {file_path}")

        self.stop_playback()
        self._stop_playback.clear()

        def _worker():
            try:
                data, samplerate = sf.read(file_path, dtype="float32")
                sd.play(data, samplerate)
                sd.wait()
            except Exception:
                pass
            finally:
                if on_finished:
                    on_finished()

        self._play_thread = threading.Thread(target=_worker, daemon=True)
        self._play_thread.start()

    def stop_playback(self) -> None:
        """Stop any audio currently being played back."""
        self._stop_playback.set()
        try:
            sd.stop()
        except Exception:
            pass
