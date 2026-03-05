"""Modular OCR backends for multi-model transcription comparison."""

from newspapers.ocr.backends import (
    EndpointManager,
    GeminiOCR,
    HuggingFaceOCR,
    OCRBackend,
    TRANSCRIPTION_PROMPT,
    get_all_backends,
)

__all__ = [
    "EndpointManager",
    "GeminiOCR",
    "HuggingFaceOCR",
    "OCRBackend",
    "TRANSCRIPTION_PROMPT",
    "get_all_backends",
]
