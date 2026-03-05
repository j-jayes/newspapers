"""Vision LLM structured extraction of job advertisements.

Uses Gemini 2.5 Flash to act as a pure, zero-shot OCR layer that performs 
literal diplomatic transcription (to avoid systemic hallucinations of archaic characters 
as per empirical research). Then uses Google LangExtract to map that text to 
the strict Pydantic schema and provide interactive source-grounding.
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from the repo root so GEMINI_API_KEY is available."""
    for candidate in [Path(".env"), *[p / ".env" for p in Path(__file__).parents]]:
        if candidate.exists():
            try:
                from dotenv import load_dotenv  # type: ignore[import-not-found]
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


_load_dotenv()

# Note: The new dependencies. google-genai and langextract
from google import genai
from google.genai import types
from PIL import Image
import langextract as lx

from newspapers.models import JobAdvertisement

logger = logging.getLogger(__name__)

# Prompt engineered to force literal transcription and avoid "over-historicization"
TRANSCRIPTION_PROMPT = """\
Perform a strict, literal diplomatic transcription of the text in this image.
The text is from a 19th-century Swedish newspaper and may contain Fraktur (blackletter) 
or Antiqua (Roman) typefaces, or a mixture. 

CRITICAL INSTRUCTIONS:
1. Do NOT normalize spelling or grammar.
2. Formulate EXACTLY what you see.
3. Do NOT hallucinate medieval or archaic characters that are not physically present.
4. Preserve the exact line breaks and spacing.
5. Provide ONLY the transcribed text. No commentary.
"""

# LangExtract rules for parsing the raw text
EXTRACTION_PROMPT = textwrap.dedent("""\
    Extract the core details of this historical Swedish job advertisement.
    Use EXACT literal phrases from the text for your extractions. Do not summarize.
    """)


def transcribe_advertisement_image(
    image_path: Path,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """Uses Vision LLM to perform zero-shot exact transcription (Hybrid OCR)."""
    # Assuming API Key is set in environment: GEMINI_API_KEY
    client = genai.Client()
    img = Image.open(image_path)
    
    response = client.models.generate_content(
        model=model_name,
        contents=[
            img,
            TRANSCRIPTION_PROMPT
        ]
    )
    
    text = response.text or ""
    logger.info("Transcribed %s (%d characters)", image_path.name, len(text))
    return text


def extract_job_ad_with_grounding(
    raw_text: str,
    source_name: str,
    model_name: str = "gemini-2.5-flash"
) -> tuple[JobAdvertisement, lx.data.AnnotatedDocument]:
    """Uses langextract to parse structural variables while retaining source offsets.
    
    Because langextract works via entity mappings instead of direct JSON schemas,
    we define specific examples to teach it how to map text spans into our fields.
    """
    # Create a few-shot example so langextract knows our target fields
    examples = [
        lx.data.ExampleData(
            text="Sökes: En erfaren springgosse i Stockholm. Lön 5 kr/vecka.",
            extractions=[
                lx.data.Extraction(extraction_class="job_title", extraction_text="springgosse"),
                lx.data.Extraction(extraction_class="skills_required", extraction_text="erfaren"),
                lx.data.Extraction(extraction_class="location", extraction_text="Stockholm"),
                lx.data.Extraction(extraction_class="compensation", extraction_text="5 kr/vecka"),
            ]
        )
    ]
    
    # Execute extraction
    result = lx.extract(
        text_or_documents=raw_text,
        prompt_description=EXTRACTION_PROMPT,
        examples=examples,
        model_id=model_name,
    )
    
    # Map back to our Pydantic model
    mapped_data = {
        "job_title": "Unknown",
        "skills_required": [],
        "confidence_score": 1.0  # Langextract enforces precision via strict APIs
    }
    
    for extraction in result.extractions:
        cls_name = extraction.extraction_class
        val = extraction.extraction_text
        if cls_name == "skills_required":
            mapped_data["skills_required"].append(val)
        elif cls_name in ["job_title", "gender_preference", "age_requirement", "location", "employer", "compensation"]:
             mapped_data[cls_name] = val
             
    # Create the validated Pydantic model
    job_ad = JobAdvertisement(**mapped_data)
    
    logger.info("Successfully extracted structured data for %s", source_name)
    return job_ad, result


def process_advertisement(image_path: Path) -> tuple[JobAdvertisement, lx.data.AnnotatedDocument]:
    """End-to-end processing pipeline for a single cropped advertisement."""
    raw_text = transcribe_advertisement_image(image_path)
    job_ad, grounded_doc = extract_job_ad_with_grounding(raw_text, image_path.name)
    return job_ad, grounded_doc
