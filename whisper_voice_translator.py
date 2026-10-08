"""
whisper_voice_translator.py
============================
Offline AI Voice Translator - Main Application Entry Point.

A fully offline speech + text translator built with:
    * OpenAI Whisper       -> Speech-to-text (local models/whisper)
    * Meta NLLB-200 (CT2)  -> Neural machine translation (local models/nllb)
    * pyttsx3              -> Text-to-speech
    * CustomTkinter        -> Modern desktop GUI

Run with:
    python whisper_voice_translator.py

No internet connection is required once the Whisper and NLLB models have
been placed in their respective ``models/`` sub-folders (see README.md).
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from typing import Optional

import customtkinter as ctk

# Make sure `app` package is importable regardless of the current working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.audio import AudioError, AudioRecorder
from app.language import LANGUAGES, get_by_name, get_language_names
from app.stt import SpeechToTextError, WhisperModelNotFoundError, WhisperSTT, SUPPORTED_MODEL_SIZES
from app.translate import NLLBModelNotFoundError, NLLBTranslator, TranslationError
from app.tts import TextToSpeechError, TTSEngine
from app.utils import HistoryManager, HistoryRecord, copy_to_clipboard, current_date_string, \
    current_time_string, ensure_directory, paste_from_clipboard

# --------------------------------------------------------------------------
# Paths (project structure must stay exactly as documented in the README)
# --------------------------------------------------------------------------
WHISPER_MODELS_DIR = os.path.join(BASE_DIR, "models", "whisper")
NLLB_MODEL_DIR = os.path.join(BASE_DIR, "models", "nllb")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
HISTORY_PATH = os.path.join(BASE_DIR, "history.json")

APP_NAME = "Offline AI Voice Translator"
APP_VERSION = "1.0.0"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class OfflineTranslatorApp(ctk.CTk):
    """Root application window - orchestrates GUI, audio, STT, MT and TTS."""

    def __init__(self) -> None:
        super().__init__()

        ensure_directory(WHISPER_MODELS_DIR)
        ensure_directory(NLLB_MODEL_DIR)
        ensure_directory(AUDIO_DIR)
        ensure_directory(OUTPUT_DIR)

        self.title(APP_NAME)
        self.geometry("1150x760")
        self.minsize(980, 660)

        # Business-logic objects (business logic never lives in the GUI layer)
        self.recorder = AudioRecorder(output_dir=AUDIO_DIR)
        self.stt_engine = WhisperSTT(models_dir=WHISPER_MODELS_DIR)
        self.translator = NLLBTranslator(model_dir=NLLB_MODEL_DIR)
        self.tts_engine = TTSEngine()
        self.history = HistoryManager(history_path=HISTORY_PATH)

        # State
        self.current_audio_path: Optional[str] = None
        self.detected_language_name: Optional[str] = None
        self.selected_whisper_size = tk.StringVar(value="base")
        self.recording_timer_job = None
        self.appearance_mode = tk.StringVar(value="Dark")

        self._build_layout()
        self._update_status("Ready. Select languages and press the microphone to begin.")

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_top_bar()
        self._build_center_section()
        self._build_bottom_bar()

    def _build_top_bar(self) -> None:
        top = ctk.CTkFrame(self, corner_radius=0, height=90)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)

        logo = ctk.CTkLabel(top, text="🌐", font=ctk.CTkFont(size=36))
        logo.grid(row=0, column=0, rowspan=2, padx=(20, 10), pady=15)

        title = ctk.CTkLabel(top, text=APP_NAME, font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=1, sticky="w", pady=(15, 0))

        subtitle = ctk.CTkLabel(
            top,
            text="100% Offline Speech & Text Translation - Whisper + NLLB-200",
            font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray70"),
        )
        subtitle.grid(row=1, column=1, sticky="w", pady=(0, 15))

        # Whisper model selector
        model_frame = ctk.CTkFrame(top, fg_color="transparent")
        model_frame.grid(row=0, column=2, rowspan=2, padx=20)
        ctk.CTkLabel(model_frame, text="Whisper Model:").pack(side="left", padx=(0, 6))
        model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=list(SUPPORTED_MODEL_SIZES),
            variable=self.selected_whisper_size,
            width=110,
        )
        model_menu.pack(side="left")

        theme_switch = ctk.CTkSwitch(
            top, text="Light Mode", command=self._toggle_theme, onvalue="Light", offvalue="Dark",
            variable=self.appearance_mode,
        )
        theme_switch.grid(row=0, column=3, rowspan=2, padx=20)

    def _build_center_section(self) -> None:
        center = ctk.CTkFrame(self, corner_radius=0)
        center.grid(row=1, column=0, sticky="nsew", padx=15, pady=10)
        center.grid_columnconfigure((0, 1), weight=1)
        center.grid_rowconfigure(3, weight=1)

        # --- Language selectors -----------------------------------------
        lang_frame = ctk.CTkFrame(center, fg_color="transparent")
        lang_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(10, 5), padx=10)
        lang_frame.grid_columnconfigure((0, 1, 2), weight=1)

        names = get_language_names()
        ctk.CTkLabel(lang_frame, text="Source Language").grid(row=0, column=0, sticky="w")
        self.source_lang_var = tk.StringVar(value="English")
        ctk.CTkOptionMenu(lang_frame, values=names, variable=self.source_lang_var).grid(
            row=1, column=0, sticky="ew", padx=(0, 10)
        )

        swap_btn = ctk.CTkButton(lang_frame, text="⇄ Swap", width=70, command=self._swap_languages)
        swap_btn.grid(row=1, column=1)

        ctk.CTkLabel(lang_frame, text="Target Language").grid(row=0, column=2, sticky="w")
        self.target_lang_var = tk.StringVar(value="Tamil")
        ctk.CTkOptionMenu(lang_frame, values=names, variable=self.target_lang_var).grid(
            row=1, column=2, sticky="ew", padx=(10, 0)
        )

        # --- Recording controls ------------------------------------------
        controls = ctk.CTkFrame(center, fg_color="transparent")
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10, padx=10)

        self.mic_button = ctk.CTkButton(
            controls, text="🎙 Record", width=120, height=40, command=self._on_record_clicked
        )
        self.mic_button.pack(side="left", padx=5)

        self.pause_button = ctk.CTkButton(
            controls, text="⏸ Pause", width=100, height=40, command=self._on_pause_clicked, state="disabled"
        )
        self.pause_button.pack(side="left", padx=5)

        self.resume_button = ctk.CTkButton(
            controls, text="▶ Resume", width=100, height=40, command=self._on_resume_clicked, state="disabled"
        )
        self.resume_button.pack(side="left", padx=5)

        self.stop_button = ctk.CTkButton(
            controls, text="⏹ Stop", width=100, height=40, command=self._on_stop_clicked, state="disabled"
        )
        self.stop_button.pack(side="left", padx=5)

        self.timer_label = ctk.CTkLabel(controls, text="00:00", font=ctk.CTkFont(size=16, weight="bold"))
        self.timer_label.pack(side="left", padx=20)

        self.play_button = ctk.CTkButton(
            controls, text="🔊 Play Audio", width=120, height=40, command=self._on_play_audio_clicked
        )
        self.play_button.pack(side="right", padx=5)

        # --- Waveform canvas -----------------------------------------------
        self.waveform_canvas = tk.Canvas(center, height=60, bg="#1a1a1a", highlightthickness=0)
        self.waveform_canvas.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        # --- Text areas ------------------------------------------------------
        text_frame = ctk.CTkFrame(center, fg_color="transparent")
        text_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        text_frame.grid_columnconfigure((0, 1), weight=1)
        text_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(text_frame, text="Source Text").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(text_frame, text="Translated Text").grid(row=0, column=1, sticky="w")

        self.source_text = ctk.CTkTextbox(text_frame, wrap="word", font=ctk.CTkFont(size=14))
        self.source_text.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        self.translated_text = ctk.CTkTextbox(text_frame, wrap="word", font=ctk.CTkFont(size=14))
        self.translated_text.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        # --- Action buttons ---------------------------------------------------
        action_frame = ctk.CTkFrame(center, fg_color="transparent")
        action_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        ctk.CTkButton(action_frame, text="Translate ➜", width=140, height=40,
                      command=self._on_translate_clicked).pack(side="left", padx=5)
        ctk.CTkButton(action_frame, text="🔈 Speak", width=110, height=40,
                      command=self._on_speak_clicked).pack(side="left", padx=5)
        ctk.CTkButton(action_frame, text="⟲ Replay", width=110, height=40,
                      command=self._on_replay_clicked).pack(side="left", padx=5)
        ctk.CTkButton(action_frame, text="■ Stop Speech", width=130, height=40,
                      fg_color="#8a2c2c", hover_color="#6e2323",
                      command=self._on_stop_speech_clicked).pack(side="left", padx=5)

        self.progress_bar = ctk.CTkProgressBar(center, mode="indeterminate")
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5))
        self.progress_bar.set(0)

    def _build_bottom_bar(self) -> None:
        bottom = ctk.CTkFrame(self, corner_radius=0, height=110)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(8, 4))

        buttons = [
            ("📜 History", self._show_history_window),
            ("⬇ Export", self._show_export_window),
            ("📋 Copy", self._on_copy_clicked),
            ("📥 Paste", self._on_paste_clicked),
            ("🗑 Clear", self._on_clear_clicked),
            ("⚙ Settings", self._show_settings_window),
            ("ℹ About", self._show_about_window),
        ]
        for text, cmd in buttons:
            ctk.CTkButton(btn_row, text=text, width=110, command=cmd).pack(side="left", padx=4)

        self.status_bar = ctk.CTkLabel(bottom, text="", anchor="w", font=ctk.CTkFont(size=12))
        self.status_bar.pack(fill="x", padx=15, pady=(2, 8))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_status(self, message: str) -> None:
        self.status_bar.configure(text=message)

    def _toggle_theme(self) -> None:
        ctk.set_appearance_mode(self.appearance_mode.get())

    def _swap_languages(self) -> None:
        src, tgt = self.source_lang_var.get(), self.target_lang_var.get()
        self.source_lang_var.set(tgt)
        self.target_lang_var.set(src)

    def _show_error(self, title: str, message: str) -> None:
        messagebox.showerror(title, message)

    def _show_info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message)

    def _run_in_background(self, target, *args) -> None:
        threading.Thread(target=target, args=args, daemon=True).start()

    def _start_progress(self) -> None:
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()

    def _stop_progress(self) -> None:
        self.progress_bar.stop()
        self.progress_bar.set(0)

    # ------------------------------------------------------------------
    # Recording controls
    # ------------------------------------------------------------------
    def _on_record_clicked(self) -> None:
        if self.recorder.is_recording:
            return
        try:
            self.recorder.start()
        except AudioError as exc:
            self._show_error("Recording Error", str(exc))
            return

        self.mic_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.stop_button.configure(state="normal")
        self.resume_button.configure(state="disabled")
        self._update_status("Recording... speak now.")
        self._tick_timer()
        self._tick_waveform()

    def _on_pause_clicked(self) -> None:
        self.recorder.pause()
        self.pause_button.configure(state="disabled")
        self.resume_button.configure(state="normal")
        self._update_status("Recording paused.")

    def _on_resume_clicked(self) -> None:
        self.recorder.resume()
        self.pause_button.configure(state="normal")
        self.resume_button.configure(state="disabled")
        self._update_status("Recording resumed.")

    def _on_stop_clicked(self) -> None:
        try:
            result = self.recorder.stop()
        except AudioError as exc:
            self._show_error("Recording Error", str(exc))
            self._reset_recording_buttons()
            return

        self.current_audio_path = result.file_path
        self._reset_recording_buttons()
        self._update_status(f"Recording saved ({result.duration_seconds:.1f}s). Transcribing...")
        self._run_in_background(self._transcribe_worker, result.file_path)

    def _reset_recording_buttons(self) -> None:
        self.mic_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.resume_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.timer_label.configure(text="00:00")
        if self.recording_timer_job:
            self.after_cancel(self.recording_timer_job)
            self.recording_timer_job = None

    def _tick_timer(self) -> None:
        if not self.recorder.is_recording:
            return
        elapsed = int(self.recorder.get_elapsed_seconds())
        minutes, seconds = divmod(elapsed, 60)
        self.timer_label.configure(text=f"{minutes:02d}:{seconds:02d}")
        self.recording_timer_job = self.after(500, self._tick_timer)

    def _tick_waveform(self) -> None:
        if not self.recorder.is_recording:
            self.waveform_canvas.delete("all")
            return
        data = self.recorder.get_waveform()
        self.waveform_canvas.delete("all")
        width = max(self.waveform_canvas.winfo_width(), 400)
        height = max(self.waveform_canvas.winfo_height(), 60)
        mid = height / 2
        step = width / max(1, len(data))
        for i, sample in enumerate(data):
            x = i * step
            amp = float(sample) * mid * 4
            self.waveform_canvas.create_line(x, mid - amp, x, mid + amp, fill="#3b8ed0")
        self.after(80, self._tick_waveform)

    def _on_play_audio_clicked(self) -> None:
        if not self.current_audio_path:
            self._show_info("No Recording", "There is no recorded audio to play yet.")
            return
        try:
            self.recorder.play_file(self.current_audio_path)
            self._update_status("Playing recorded audio...")
        except AudioError as exc:
            self._show_error("Playback Error", str(exc))

    # ------------------------------------------------------------------
    # Speech-to-text
    # ------------------------------------------------------------------
    def _transcribe_worker(self, audio_path: str) -> None:
        self.after(0, self._start_progress)
        model_size = self.selected_whisper_size.get()
        try:
            self.stt_engine.load_model(model_size)
            result = self.stt_engine.transcribe(audio_path)
        except WhisperModelNotFoundError as exc:
            self.after(0, lambda: self._show_error("Whisper Model Missing", str(exc)))
            self.after(0, self._stop_progress)
            return
        except SpeechToTextError as exc:
            self.after(0, lambda: self._show_error("Speech Recognition Error", str(exc)))
            self.after(0, self._stop_progress)
            return

        def _apply_result():
            self.source_text.delete("1.0", "end")
            self.source_text.insert("1.0", result.text)
            self.detected_language_name = result.detected_language_name
            self._stop_progress()

            if result.is_low_confidence:
                self._update_status(
                    f"Low confidence ({result.confidence:.0%}) detecting language - "
                    "please confirm the source language manually."
                )
            else:
                if result.detected_language_name in get_language_names():
                    self.source_lang_var.set(result.detected_language_name)
                self._update_status(
                    f"Transcribed successfully. Detected language: "
                    f"{result.detected_language_name} ({result.confidence:.0%} confidence)."
                )

        self.after(0, _apply_result)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def _on_translate_clicked(self) -> None:
        text = self.source_text.get("1.0", "end").strip()
        if not text:
            self._show_info("Nothing to Translate", "Please record or type some text first.")
            return

        source_lang = get_by_name(self.source_lang_var.get())
        target_lang = get_by_name(self.target_lang_var.get())
        if source_lang is None or target_lang is None:
            self._show_error("Language Error", "Please select valid source and target languages.")
            return

        self._update_status("Translating...")
        self._run_in_background(self._translate_worker, text, source_lang, target_lang)

    def _translate_worker(self, text: str, source_lang, target_lang) -> None:
        self.after(0, self._start_progress)
        try:
            self.translator.load()
            translated = self.translator.translate(text, source_lang.nllb_code, target_lang.nllb_code)
        except NLLBModelNotFoundError as exc:
            self.after(0, lambda: self._show_error("NLLB Model Missing", str(exc)))
            self.after(0, self._stop_progress)
            return
        except TranslationError as exc:
            self.after(0, lambda: self._show_error("Translation Error", str(exc)))
            self.after(0, self._stop_progress)
            return

        def _apply_result():
            self.translated_text.delete("1.0", "end")
            self.translated_text.insert("1.0", translated)
            self._stop_progress()
            self._update_status(f"Translated {source_lang.name} -> {target_lang.name} successfully.")

            record = HistoryRecord(
                date=current_date_string(),
                time=current_time_string(),
                detected_language=self.detected_language_name or source_lang.name,
                source_language=source_lang.name,
                target_language=target_lang.name,
                original_text=text,
                translated_text=translated,
                audio_file=self.current_audio_path or "",
            )
            self.history.add(record)

        self.after(0, _apply_result)

    # ------------------------------------------------------------------
    # Text-to-speech
    # ------------------------------------------------------------------
    def _on_speak_clicked(self) -> None:
        text = self.translated_text.get("1.0", "end").strip()
        if not text:
            self._show_info("Nothing to Speak", "Please translate some text first.")
            return
        try:
            self.tts_engine.speak(text)
            self._update_status("Speaking translated text...")
        except TextToSpeechError as exc:
            self._show_error("Text-to-Speech Error", str(exc))

    def _on_replay_clicked(self) -> None:
        try:
            self.tts_engine.replay()
            self._update_status("Replaying last speech...")
        except TextToSpeechError as exc:
            self._show_error("Text-to-Speech Error", str(exc))

    def _on_stop_speech_clicked(self) -> None:
        self.tts_engine.stop()
        self._update_status("Speech stopped.")

    # ------------------------------------------------------------------
    # Clipboard / clear
    # ------------------------------------------------------------------
    def _on_copy_clicked(self) -> None:
        text = self.translated_text.get("1.0", "end").strip()
        if text:
            copy_to_clipboard(self, text)
            self._update_status("Translated text copied to clipboard.")

    def _on_paste_clicked(self) -> None:
        text = paste_from_clipboard(self)
        if text:
            self.source_text.delete("1.0", "end")
            self.source_text.insert("1.0", text)
            self._update_status("Pasted text from clipboard.")

    def _on_clear_clicked(self) -> None:
        self.source_text.delete("1.0", "end")
        self.translated_text.delete("1.0", "end")
        self.current_audio_path = None
        self.detected_language_name = None
        self._update_status("Cleared.")

    # ------------------------------------------------------------------
    # History window
    # ------------------------------------------------------------------
    def _show_history_window(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Translation History")
        win.geometry("800x500")

        textbox = ctk.CTkTextbox(win, wrap="word", font=ctk.CTkFont(size=13))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)

        records = self.history.all()
        if not records:
            textbox.insert("1.0", "No history yet. Translate something to see it here.")
        else:
            for r in records:
                textbox.insert(
                    "end",
                    f"[{r.date} {r.time}]  {r.source_language} -> {r.target_language} "
                    f"(detected: {r.detected_language})\n"
                    f"  Original   : {r.original_text}\n"
                    f"  Translated : {r.translated_text}\n"
                    f"  Audio      : {r.audio_file}\n"
                    f"{'-' * 70}\n",
                )
        textbox.configure(state="disabled")

        clear_btn = ctk.CTkButton(
            win, text="Clear All History", fg_color="#8a2c2c", hover_color="#6e2323",
            command=lambda: self._confirm_clear_history(win),
        )
        clear_btn.pack(pady=(0, 10))

    def _confirm_clear_history(self, window) -> None:
        if messagebox.askyesno("Confirm", "Are you sure you want to delete ALL history?"):
            self.history.clear()
            window.destroy()
            self._update_status("History cleared.")

    # ------------------------------------------------------------------
    # Export window
    # ------------------------------------------------------------------
    def _show_export_window(self) -> None:
        choice = messagebox.askquestion(
            "Export History", "Export as TXT? (choose 'No' for JSON instead)"
        )
        if choice == "yes":
            path = filedialog.asksaveasfilename(
                initialdir=OUTPUT_DIR, defaultextension=".txt",
                filetypes=[("Text file", "*.txt")], title="Export History as TXT",
            )
            if path:
                self.history.export_txt(path)
                self._show_info("Export Complete", f"History exported to:\n{path}")
        else:
            path = filedialog.asksaveasfilename(
                initialdir=OUTPUT_DIR, defaultextension=".json",
                filetypes=[("JSON file", "*.json")], title="Export History as JSON",
            )
            if path:
                self.history.export_json(path)
                self._show_info("Export Complete", f"History exported to:\n{path}")

    # ------------------------------------------------------------------
    # Settings window
    # ------------------------------------------------------------------
    def _show_settings_window(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Settings")
        win.geometry("420x420")

        ctk.CTkLabel(win, text="Voice Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))

        voices = self.tts_engine.get_voices()
        voice_names = [v.name for v in voices] or ["Default"]
        voice_var = tk.StringVar(value=voice_names[0])
        ctk.CTkLabel(win, text="Voice").pack(pady=(10, 0))
        voice_menu = ctk.CTkOptionMenu(win, values=voice_names, variable=voice_var)
        voice_menu.pack(pady=5)

        def apply_voice(choice):
            for v in voices:
                if v.name == choice:
                    try:
                        self.tts_engine.set_voice(v.id)
                    except TextToSpeechError as exc:
                        self._show_error("Voice Error", str(exc))
                    break

        voice_menu.configure(command=apply_voice)

        ctk.CTkLabel(win, text="Speech Speed").pack(pady=(20, 0))
        rate_slider = ctk.CTkSlider(win, from_=80, to=300, command=lambda v: self.tts_engine.set_rate(int(v)))
        rate_slider.set(175)
        rate_slider.pack(pady=5, padx=30, fill="x")

        ctk.CTkLabel(win, text="Volume").pack(pady=(20, 0))
        volume_slider = ctk.CTkSlider(win, from_=0, to=1, command=lambda v: self.tts_engine.set_volume(float(v)))
        volume_slider.set(1.0)
        volume_slider.pack(pady=5, padx=30, fill="x")

        ctk.CTkLabel(
            win,
            text=f"Whisper device: {self.stt_engine.device.upper()}\n"
                 f"NLLB device: {(self.translator.device or 'not loaded yet').upper()}",
            font=ctk.CTkFont(size=12),
        ).pack(pady=20)

    # ------------------------------------------------------------------
    # About window
    # ------------------------------------------------------------------
    def _show_about_window(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("About")
        win.geometry("460x320")

        ctk.CTkLabel(win, text=APP_NAME, font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 5))
        ctk.CTkLabel(win, text=f"Version {APP_VERSION}").pack()
        ctk.CTkLabel(
            win,
            text=(
                "A completely offline speech and text translator.\n\n"
                "Speech Recognition : OpenAI Whisper\n"
                "Translation Engine  : Meta NLLB-200 (CTranslate2)\n"
                "Text-to-Speech      : pyttsx3\n"
                "Interface           : CustomTkinter\n\n"
                "Built as a final year engineering project."
            ),
            justify="left",
        ).pack(pady=15, padx=20)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    def on_close(self) -> None:
        try:
            if self.recorder.is_recording:
                self.recorder.stop()
        except Exception:
            pass
        self.tts_engine.stop()
        self.destroy()


def main() -> None:
    """Application entry point."""
    app = OfflineTranslatorApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
