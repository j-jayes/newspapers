"""Modular OCR backends for multi-model transcription comparison."""

from newspapers.ocr.backends import (
    GeminiOCR,
    HuggingFaceOCR,
    OCRBackend,
    TRANSCRIPTION_PROMPT,
    get_all_backends,
)

__all__ = [
    "GeminiOCR",
    "HuggingFaceOCR",
    "OCRBackend",
    "TRANSCRIPTION_PROMPT",
    "get_all_backends",
]
