"""
app/language.py
================
Central catalogue of every language supported by the Offline AI Voice
Translator.

Each entry maps a human readable language name to the three technical
codes required by the different offline engines used in this project:

    * ``code``         -> Simple internal ISO-639-1 style code (used for
                           dropdown values / history records).
    * ``nllb_code``     -> FLORES-200 code required by the Meta NLLB-200
                           translation model.
    * ``whisper_code``  -> ISO-639-1 code understood by OpenAI Whisper.

Adding a new language only requires appending a new ``Language`` instance
to the ``LANGUAGES`` list below - nothing else in the application needs
to change.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Language:
    """Immutable definition of a single supported language."""

    name: str
    code: str
    nllb_code: str
    whisper_code: str


# --------------------------------------------------------------------------
# Supported languages
# --------------------------------------------------------------------------
LANGUAGES: List[Language] = [
    Language("English", "en", "eng_Latn", "en"),
    Language("Tamil", "ta", "tam_Taml", "ta"),
    Language("Telugu", "te", "tel_Telu", "te"),
    Language("Malayalam", "ml", "mal_Mlym", "ml"),
    Language("Kannada", "kn", "kan_Knda", "kn"),
    Language("Hindi", "hi", "hin_Deva", "hi"),
    Language("French", "fr", "fra_Latn", "fr"),
    Language("German", "de", "deu_Latn", "de"),
    Language("Spanish", "es", "spa_Latn", "es"),
    Language("Italian", "it", "ita_Latn", "it"),
    Language("Portuguese", "pt", "por_Latn", "pt"),
    Language("Japanese", "ja", "jpn_Jpan", "ja"),
    Language("Chinese", "zh", "zho_Hans", "zh"),
    Language("Arabic", "ar", "arb_Arab", "ar"),
    Language("Russian", "ru", "rus_Cyrl", "ru"),
    Language("Korean", "ko", "kor_Hang", "ko"),
]


def get_language_names() -> List[str]:
    """Return the list of display names for dropdown widgets."""
    return [lang.name for lang in LANGUAGES]


def get_by_name(name: str) -> Optional[Language]:
    """Look up a :class:`Language` by its display name."""
    for lang in LANGUAGES:
        if lang.name == name:
            return lang
    return None


def get_by_whisper_code(code: str) -> Optional[Language]:
    """Look up a :class:`Language` by the code Whisper returns for detection."""
    for lang in LANGUAGES:
        if lang.whisper_code == code:
            return lang
    return None


def get_by_nllb_code(code: str) -> Optional[Language]:
    """Look up a :class:`Language` by its NLLB / FLORES-200 code."""
    for lang in LANGUAGES:
        if lang.nllb_code == code:
            return lang
    return None
