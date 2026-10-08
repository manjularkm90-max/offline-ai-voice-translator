"""
app/utils.py
============
Cross-cutting helper utilities used throughout the application:

    * :class:`HistoryManager` - persists every translation to ``history.json``
      and supports exporting the history to TXT or JSON.
    * Clipboard helpers built on top of Tkinter (no extra dependency).
    * Timestamp and generic file-system helpers.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional


# --------------------------------------------------------------------------
# Timestamp helpers
# --------------------------------------------------------------------------
def current_date_string() -> str:
    """Return today's date formatted as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def current_time_string() -> str:
    """Return the current time formatted as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def timestamped_filename(prefix: str, extension: str) -> str:
    """Build a unique, sortable filename such as ``prefix_20260717_142233.ext``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{extension.lstrip('.')}"


# --------------------------------------------------------------------------
# File-system helpers
# --------------------------------------------------------------------------
def ensure_directory(path: str) -> None:
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def safe_read_json(path: str, default):
    """Read JSON from ``path``, returning ``default`` on any failure."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def safe_write_json(path: str, data) -> None:
    """Write ``data`` to ``path`` as pretty-printed UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# History record + manager
# --------------------------------------------------------------------------
@dataclass
class HistoryRecord:
    """A single translation event, as persisted inside ``history.json``."""

    date: str
    time: str
    detected_language: str
    source_language: str
    target_language: str
    original_text: str
    translated_text: str
    audio_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class HistoryManager:
    """Loads, appends to, persists and exports the translation history."""

    def __init__(self, history_path: str) -> None:
        self.history_path = history_path
        self._records: List[HistoryRecord] = []
        self.load()

    def load(self) -> None:
        """Load history records from disk, tolerating a missing/corrupt file."""
        raw = safe_read_json(self.history_path, default=[])
        records = []
        for item in raw if isinstance(raw, list) else []:
            try:
                records.append(HistoryRecord(**item))
            except TypeError:
                # Skip malformed / legacy entries rather than crashing.
                continue
        self._records = records

    def save(self) -> None:
        """Persist all in-memory records back to ``history.json``."""
        safe_write_json(self.history_path, [r.to_dict() for r in self._records])

    def add(self, record: HistoryRecord) -> None:
        """Append a new record and persist immediately."""
        self._records.append(record)
        self.save()

    def all(self) -> List[HistoryRecord]:
        """Return all records, most recent first."""
        return list(reversed(self._records))

    def clear(self) -> None:
        """Remove every history record."""
        self._records = []
        self.save()

    def export_txt(self, destination_path: str) -> None:
        """Export the full history as a readable plain-text file."""
        lines = []
        for r in self.all():
            lines.append(f"Date/Time     : {r.date} {r.time}")
            lines.append(f"Detected Lang : {r.detected_language}")
            lines.append(f"Source -> Target : {r.source_language} -> {r.target_language}")
            lines.append(f"Original      : {r.original_text}")
            lines.append(f"Translated    : {r.translated_text}")
            lines.append(f"Audio File    : {r.audio_file}")
            lines.append("-" * 60)
        with open(destination_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) if lines else "No history records found.")

    def export_json(self, destination_path: str) -> None:
        """Export the full history as a JSON array."""
        safe_write_json(destination_path, [r.to_dict() for r in self.all()])


# --------------------------------------------------------------------------
# Clipboard helpers (Tkinter based, no extra dependency required)
# --------------------------------------------------------------------------
def copy_to_clipboard(widget, text: str) -> None:
    """Copy ``text`` to the system clipboard using any Tk widget instance."""
    widget.clipboard_clear()
    widget.clipboard_append(text)
    widget.update()


def paste_from_clipboard(widget) -> str:
    """Return the current clipboard contents, or an empty string on failure."""
    try:
        return widget.clipboard_get()
    except Exception:
        return ""
