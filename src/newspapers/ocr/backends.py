"""Modular OCR backends for comparing transcription quality across models.

Each backend wraps a different vision-language model and exposes a unified
``transcribe(image) -> str`` interface so the comparison notebook can
swap models trivially.

Supported backends
------------------
- **GeminiOCR** — Google Gemini 2.5 Flash via ``google-genai``
- **HuggingFaceOCR** — Any HF Inference Endpoint (DeepSeek-OCR-2, LightOnOCR-2, GLM-OCR, …)
"""

from __future__ import annotations

import io
import base64
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared transcription prompt
# ---------------------------------------------------------------------------

TRANSCRIPTION_PROMPT = """\
Perform a strict, literal diplomatic transcription of the text in this image.
The text is from a 19th-century Swedish newspaper and may contain Fraktur (blackletter) \
or Antiqua (Roman) typefaces, or a mixture.

CRITICAL INSTRUCTIONS:
1. Do NOT normalize spelling or grammar.
2. Transcribe EXACTLY what you see.
3. Do NOT hallucinate medieval or archaic characters that are not physically present.
4. Preserve the exact line breaks and spacing.
5. Provide ONLY the transcribed text. No commentary.
"""


def _load_dotenv() -> None:
    """Load .env from the repo root so API keys are available."""
    for candidate in [Path(".env"), *[p / ".env" for p in Path(__file__).parents]]:
        if candidate.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(candidate, override=False)
                return
            except ImportError:
                pass
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
            return


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class OCRBackend(ABC):
    """Base class for OCR transcription backends."""

    name: str

    @abstractmethod
    def transcribe(self, image: Image.Image) -> str:
        """Return the transcribed text for *image*."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------


class GeminiOCR(OCRBackend):
    """OCR via Google Gemini (uses ``google-genai``)."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        self.name = f"Gemini ({model_name})"
        _load_dotenv()

        from google import genai  # noqa: PLC0415

        api_key = (
            os.environ.get("GEMINI_FLASH_API_KEY")
            or os.environ.get("GEMINI_PRO_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        self._client = genai.Client(api_key=api_key)

    def transcribe(self, image: Image.Image) -> str:
        from google.genai import types  # noqa: PLC0415

        _is_pro = "pro" in self.model_name.lower()
        cfg = types.GenerateContentConfig(
            **({} if _is_pro else {"thinking_config": types.ThinkingConfig(thinking_budget=0)}),
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[image, TRANSCRIPTION_PROMPT],
            config=cfg,
        )
        text = response.text or ""
        logger.info("%s: transcribed %d chars", self.name, len(text))
        return text


# ---------------------------------------------------------------------------
# HuggingFace Inference Endpoint backend
# ---------------------------------------------------------------------------


class HuggingFaceOCR(OCRBackend):
    """OCR via a HuggingFace Inference Endpoint (dedicated or serverless).

    Parameters
    ----------
    name:
        Human-readable name for display (e.g. "DeepSeek-OCR-2").
    model_id:
        HuggingFace model identifier (e.g. "deepseek-ai/DeepSeek-OCR-2").
    endpoint_url:
        URL of the dedicated Inference Endpoint.  If *None*, falls back to
        serverless inference (which may not be available for all models).
    token:
        HuggingFace API token.  Defaults to ``HF_TOKEN`` env var.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        endpoint_url: str | None = None,
        token: str | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.endpoint_url = endpoint_url
        _load_dotenv()

        from huggingface_hub import InferenceClient  # noqa: PLC0415

        self._token = token or os.environ.get("HF_TOKEN")
        self._client = InferenceClient(
            model=endpoint_url or model_id,
            token=self._token,
        )

    def transcribe(self, image: Image.Image) -> str:
        # Encode image as base64 data URI for the chat completion API
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_url = f"data:image/png;base64,{b64}"

        response = self._client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    ],
                }
            ],
            max_tokens=4096,
        )
        text = response.choices[0].message.content or ""
        logger.info("%s: transcribed %d chars", self.name, len(text))
        return text


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

# Default model registry: (display_name, hf_model_id, env_var_for_endpoint)
_HF_MODELS = [
    ("DeepSeek-OCR-2 (3B)", "deepseek-ai/DeepSeek-OCR-2", "DEEPSEEK_OCR_ENDPOINT"),
    ("LightOnOCR-2 (1B)", "lightonai/LightOnOCR-2-1B", "LIGHTON_OCR_ENDPOINT"),
    ("GLM-OCR (0.9B)", "zai-org/GLM-OCR", "GLM_OCR_ENDPOINT"),
]


def get_all_backends(
    *,
    gemini_model: str = "gemini-2.5-flash",
    skip_missing: bool = True,
) -> list[OCRBackend]:
    """Instantiate all configured OCR backends.

    Parameters
    ----------
    gemini_model:
        Gemini model to use.
    skip_missing:
        If *True*, silently skip HF backends whose endpoint URL env var
        is not set.  If *False*, raise on missing config.
    """
    _load_dotenv()
    backends: list[OCRBackend] = []

    # Gemini (always available if GEMINI_API_KEY is set)
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_FLASH_API_KEY")
    if gemini_key:
        backends.append(GeminiOCR(model_name=gemini_model))
    elif not skip_missing:
        raise EnvironmentError("GEMINI_API_KEY not set")

    # HuggingFace endpoints
    hf_token = os.environ.get("HF_TOKEN")
    for display_name, model_id, env_var in _HF_MODELS:
        endpoint = os.environ.get(env_var)
        if endpoint:
            backends.append(
                HuggingFaceOCR(
                    name=display_name,
                    model_id=model_id,
                    endpoint_url=endpoint,
                    token=hf_token,
                )
            )
        elif not skip_missing:
            raise EnvironmentError(
                f"{env_var} not set — deploy {model_id} as an Inference Endpoint "
                f"and set {env_var} in .env"
            )

    return backends
