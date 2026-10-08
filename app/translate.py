"""
app/translate.py
================
Offline neural machine translation using Meta's NLLB-200 model, converted
to the CTranslate2 format for fast CPU/GPU inference.

The model is expected to live inside ``models/nllb`` as a CTranslate2
export (produced with ``ct2-transformers-converter``, see the README for
the exact conversion command) together with the original HuggingFace
tokenizer files (``sentencepiece.bpe.model``, ``tokenizer.json``,
``special_tokens_map.json`` etc.) so the SentencePiece tokenizer can be
loaded alongside it.

The translator is loaded lazily and cached - it is only initialised once,
on first use, and reused for every subsequent call.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import ctranslate2
    _CTRANSLATE2_AVAILABLE = True
except Exception:  # pragma: no cover
    _CTRANSLATE2_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


class NLLBModelNotFoundError(Exception):
    """Raised when the local CTranslate2 NLLB model cannot be found."""


class TranslationError(Exception):
    """Raised for any other recoverable translation failure."""


class NLLBTranslator:
    """Loads and caches a local CTranslate2 export of NLLB-200 for translation."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._translator: Optional["ctranslate2.Translator"] = None
        self._tokenizer = None
        self._device: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._translator is not None

    @property
    def device(self) -> Optional[str]:
        return self._device

    def is_model_available(self) -> bool:
        """Check that the CTranslate2 model directory looks populated."""
        if not os.path.isdir(self.model_dir):
            return False
        required_any = ["model.bin"]
        has_model_bin = any(
            os.path.exists(os.path.join(self.model_dir, f)) for f in required_any
        )
        has_tokenizer = any(
            os.path.exists(os.path.join(self.model_dir, f))
            for f in ("sentencepiece.bpe.model", "tokenizer.json")
        )
        return has_model_bin and has_tokenizer

    def load(self) -> None:
        """Load the CTranslate2 model + tokenizer once, and cache them.

        Raises:
            NLLBModelNotFoundError: if the local model files are missing.
            TranslationError: if required libraries are missing or
                loading otherwise fails.
        """
        if self._translator is not None:
            return  # already loaded - never reload

        if not _CTRANSLATE2_AVAILABLE:
            raise TranslationError(
                "The 'ctranslate2' package is not installed. "
                "Please install it with: pip install ctranslate2"
            )
        if not _TRANSFORMERS_AVAILABLE:
            raise TranslationError(
                "The 'transformers' package is not installed. "
                "Please install it with: pip install transformers"
            )

        if not self.is_model_available():
            raise NLLBModelNotFoundError(
                "NLLB-200 translation model was not found.\n\n"
                f"Please place a CTranslate2-converted NLLB-200 model "
                f"(model.bin + tokenizer files) inside:\n"
                f"{os.path.abspath(self.model_dir)}\n\n"
                "See the README section 'How to Convert NLLB to CTranslate2' "
                "for the exact conversion command."
            )

        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            self._translator = ctranslate2.Translator(self.model_dir, device=device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._device = device
        except Exception as exc:
            self._translator = None
            self._tokenizer = None
            raise TranslationError(f"Failed to load NLLB model: {exc}") from exc

    def translate(self, text: str, source_nllb_code: str, target_nllb_code: str) -> str:
        """Translate ``text`` from one NLLB/FLORES-200 language code to another."""
        if self._translator is None or self._tokenizer is None:
            raise TranslationError("Translation model has not been loaded yet.")
        if not text or not text.strip():
            return ""

        try:
            self._tokenizer.src_lang = source_nllb_code
            source_tokens = self._tokenizer.convert_ids_to_tokens(
                self._tokenizer(text).input_ids
            )
            target_prefix = [target_nllb_code]

            results = self._translator.translate_batch(
                [source_tokens],
                target_prefix=[target_prefix],
                beam_size=4,
            )

            output_tokens = results[0].hypotheses[0][1:]  # drop the target-language prefix token
            output_ids = self._tokenizer.convert_tokens_to_ids(output_tokens)
            translated = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            return translated.strip()
        except Exception as exc:
            raise TranslationError(f"Translation failed: {exc}") from exc
