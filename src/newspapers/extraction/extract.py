"""Vision LLM structured extraction of job advertisements.

Sends cropped advertisement images to a multimodal LLM and enforces
a strict Pydantic schema on the response to produce structured records.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from newspapers.models import JobAdvertisement

logger = logging.getLogger(__name__)


def encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64 encoding."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


SYSTEM_PROMPT = """\
You are a specialist in reading 19th-century Swedish newspaper job advertisements.
Given an image of a single job advertisement cropped from a historical Swedish
newspaper (1880–1926), extract the structured fields described in the JSON schema.
The text may use Fraktur (blackletter) or Antiqua (Roman) typefaces, or a mixture.
Return ONLY the requested JSON — no commentary.
"""


def extract_job_ad(
    image_path: Path,
    *,
    model_name: str = "gemini-2.5-flash",
) -> JobAdvertisement:
    """Extract structured data from a cropped job advertisement image.

    Parameters
    ----------
    image_path:
        Path to the cropped advertisement image (PNG).
    model_name:
        Gemini model identifier to use for extraction.

    Returns
    -------
    JobAdvertisement
        Validated Pydantic model with the extracted fields.
    """
    try:
        import google.generativeai as genai
        import instructor
    except ImportError as exc:
        raise ImportError(
            "instructor and google-generativeai are required for extraction. "
            "Install them with: pip install instructor google-generativeai"
        ) from exc

    client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name=model_name),
    )

    image_data = encode_image_base64(image_path)

    result = client.chat.completions.create(
        response_model=JobAdvertisement,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract job advertisement data from this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ],
            },
        ],
    )

    logger.info(
        "Extracted ad from %s — title=%r, confidence=%.2f",
        image_path.name,
        result.job_title,
        result.confidence_score,
    )
    return result
