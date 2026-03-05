"""Gemini-powered auto-annotation for historical newspaper page segmentation.

Workflow
--------
1. Send a full-page newspaper JPG to Gemini 2.5 Flash.
2. Receive a structured list of bounding boxes with class labels.
3. Convert Gemini's [y_min, x_min, y_max, x_max] (0–1000 normalised space)
   → YOLO format: ``<class_id> <cx> <cy> <w> <h>`` (0–1 normalised).
4. Write a ``.txt`` label file alongside the image.
5. Save a PIL overlay visualisation PNG for human review.

Usage
-----
Auto-annotate a single image::

    uv run python -m newspapers.segmentation.annotate \\
        --input  data/processed/page.jpg \\
        --labels data/annotations/labels/train \\
        --images data/annotations/images/train \\
        --vis    data/annotations/visualizations

Batch-annotate a directory::

    uv run python -m newspapers.segmentation.annotate \\
        --input  data/processed \\
        --labels data/annotations/labels/train \\
        --images data/annotations/images/train \\
        --vis    data/annotations/visualizations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Annotated, Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, RootModel


def _load_dotenv(dotenv_path: Path | None = None) -> None:
    """Load ``GEMINI_API_KEY`` (and any other vars) from a ``.env`` file.

    Tries ``python-dotenv`` first; falls back to a simple line parser so the
    project has no hard dependency on that package.
    """
    path = dotenv_path or Path(".env")
    if not path.exists():
        # Walk up to repo root in case cwd is a subdirectory
        for parent in Path(__file__).parents:
            candidate = parent / ".env"
            if candidate.exists():
                path = candidate
                break
        else:
            return

    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass

    # Minimal fallback parser (handles KEY=value and KEY="value")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class taxonomy – must stay consistent with data/annotations/classes.txt
#                  and data/annotations/dataset.yaml
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "job_advertisement",       # 0 – primary extraction target
    "article_text",            # 1 – general news columns
    "headline",                # 2 – section/article headings
    "commercial_advertisement",# 3 – non-job ads
    "financial_table",         # 4 – price lists, stock tables
    "masthead",                # 5 – newspaper title/date banner
]
CLASS_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Distinct colours for the overlay visualisation
_VIS_COLOURS: dict[str, str] = {
    "job_advertisement":        "#FF4136",
    "article_text":             "#2ECC40",
    "headline":                 "#0074D9",
    "commercial_advertisement": "#FF851B",
    "financial_table":          "#B10DC9",
    "masthead":                 "#FFDC00",
}

# ---------------------------------------------------------------------------
# Gemini prompt
# ---------------------------------------------------------------------------
_ANNOTATION_PROMPT = """\
You are a document layout analyser specialising in historical Swedish newspapers \
from the 1880–1926 era. Examine this full-page newspaper scan and detect every \
distinct content region.

For each region output a JSON object with these exact keys:
  "label"  – one of: job_advertisement, article_text, headline,
             commercial_advertisement, financial_table, masthead
  "box"    – [y_min, x_min, y_max, x_max] as integers in 0–1000 space
             (i.e. 0 = top/left edge, 1000 = bottom/right edge of the image)

Respond with ONLY a JSON array of these objects.  No markdown fences, no prose.

Rules:
- Detect EVERY distinct region; do not skip small ads or short headlines.
- job_advertisement: any block seeking employees, servants, apprentices, etc.
- commercial_advertisement: shops, products, announcements (not job-seeking).
- financial_table: price lists, stock quotes, shipping manifests, timetables.
- masthead: the newspaper name/date header at the very top of the page.
- headline: a heading that introduces an article or section (not the masthead).
- article_text: running prose body text columns.
- Boxes must be tight around the content; avoid large empty margins.
"""

_CRITIQUE_PROMPT_TEMPLATE = """\
You are an expert annotation quality reviewer for historical Swedish newspaper \
layout detection (1880–1926 era).

A first-pass annotation has been produced for this full-page newspaper scan.
Your task: critique it rigorously and return a REFINED, improved annotation.

FIRST-PASS ANNOTATION ({n_regions} regions):
{annotations_json}

Critique checklist — apply ALL of the following:
1. TIGHT BOXES  – Re-fit any box that has excessive whitespace margins.
2. MERGE        – Combine boxes of the same class that form one contiguous block.
3. SPLIT        – Divide any single box that spans two clearly different content types.
4. RELABEL      – Fix any misclassified region.
5. ADD          – Insert boxes for important regions that were completely missed,
                  especially: mastheads, job advertisements, financial tables.
6. REMOVE       – Delete near-duplicate boxes (same label, overlapping >80 %%).

Respond with ONLY a JSON array of the refined boxes in exactly this format:
[{{"label": "...", "box": [y_min, x_min, y_max, x_max]}}, ...]
No markdown fences, no prose, no explanation.
"""

_STRIP_ANNOTATION_PROMPT_TEMPLATE = """\
You are a document layout analyser specialising in historical Swedish newspapers \
from the 1880–1926 era.

This image is a VERTICAL STRIP representing column {column_index} of {n_columns} \
from a full newspaper page (total page size: {page_width}×{page_height} px).
The strip covers x-positions {x_start}–{x_end} of the full page.

Detect every distinct content region WITHIN THIS STRIP.

For each region output a JSON object with these exact keys:
  "label"  – one of: job_advertisement, article_text, headline,
             commercial_advertisement, financial_table, masthead
  "box"    – [y_min, x_min, y_max, x_max] as integers in 0–1000 space
             (relative to THIS strip image, not the full page)

Respond with ONLY a JSON array of these objects.  No markdown fences, no prose.

Rules:
- Detect EVERY distinct region; do not skip small ads or short headlines.
- job_advertisement: any block seeking employees, servants, apprentices, etc.
- commercial_advertisement: shops, products, announcements (not job-seeking).
- financial_table: price lists, stock quotes, shipping manifests, timetables.
- masthead: the newspaper name/date header at the very top of the page.
- headline: a heading that introduces an article or section (not the masthead).
- article_text: running prose body text columns.
- Boxes must be tight around the content; avoid large empty margins.
- This is column {column_index} of {n_columns}: expect mostly vertical text runs.
"""

_MASTHEAD_ANNOTATION_PROMPT = """\
You are a document layout analyser for historical Swedish newspapers (1880–1926 era).
This image shows the TOP STRIP of a newspaper page (masthead area, full page width).

Detect every region visible in this strip.
For each region output:
  "label"  – one of: masthead, headline, commercial_advertisement, job_advertisement,
             article_text, financial_table
  "box"    – [y_min, x_min, y_max, x_max] in 0–1000 space (relative to this strip).
Respond with ONLY a JSON array.  No markdown fences, no prose.
"""

_FULL_PAGE_CROSS_COL_PROMPT = """\
You are a document layout analyser for historical Swedish newspapers (1880–1926 era).
This is a reduced-resolution full-page view.

Focus ONLY on content regions that SPAN MULTIPLE COLUMNS (wider than a single column):
- Multi-column display advertisements
- Wide decorative headings / banners
- Full-width horizontal rules or dividers with labels

Ignore single-column text regions (those are handled separately).

For each region output:
  "label"  – one of: commercial_advertisement, headline, masthead, job_advertisement
  "box"    – [y_min, x_min, y_max, x_max] in 0–1000 space.
Respond with ONLY a JSON array.  No markdown fences, no prose.
"""

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BBoxRegion(BaseModel):
    """A single detected region returned by Gemini."""

    label: str = Field(..., description="One of the six class names.")
    box: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[y_min, x_min, y_max, x_max] in 0–1000 space.",
    )

    def to_yolo(self) -> tuple[int, float, float, float, float]:
        """Convert to YOLO format: (class_id, cx, cy, w, h) all 0–1."""
        y_min, x_min, y_max, x_max = (v / 1000.0 for v in self.box)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        class_id = CLASS_ID.get(self.label, -1)
        return class_id, cx, cy, w, h


# ---------------------------------------------------------------------------
# Structured output schema (for Gemini)
# ---------------------------------------------------------------------------

_AllowedLabel = Literal[
    "job_advertisement",
    "article_text",
    "headline",
    "commercial_advertisement",
    "financial_table",
    "masthead",
]


class _GeminiRegion(BaseModel):
    label: _AllowedLabel
    box: list[Annotated[int, Field(ge=0, le=1000)]] = Field(min_length=4, max_length=4)


class _GeminiRegionList(RootModel[list[_GeminiRegion]]):
    pass


_LABEL_SYNONYMS: dict[str, str] = {
    "article": "article_text",
    "body": "article_text",
    "body_text": "article_text",
    "text": "article_text",
    "advertisement": "commercial_advertisement",
    "advert": "commercial_advertisement",
    "ad": "commercial_advertisement",
    "subheadline": "headline",
    "dateline": "headline",
    "caption": "headline",
    "table": "financial_table",
    "nameplate": "masthead",
}


def _normalize_label(label: str) -> str | None:
    cleaned = label.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned in CLASS_ID:
        return cleaned
    cleaned = _LABEL_SYNONYMS.get(cleaned, cleaned)
    if cleaned in CLASS_ID:
        return cleaned
    if "job" in cleaned or "wanted" in cleaned or "classified" in cleaned:
        return "job_advertisement"
    if "mast" in cleaned:
        return "masthead"
    if "head" in cleaned:
        return "headline"
    if "table" in cleaned:
        return "financial_table"
    if "ad" in cleaned:
        return "commercial_advertisement"
    if "article" in cleaned or "text" in cleaned:
        return "article_text"
    return None


def _coerce_boxes(box_value: Any) -> list[list[int]]:
    def _coerce_one(box: Any) -> list[int] | None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        if not all(isinstance(v, (int, float)) for v in box):
            return None
        out = [int(round(float(v))) for v in box]
        return [max(0, min(1000, v)) for v in out]

    if isinstance(box_value, dict):
        # Common alternative formats.
        keys = [
            ("y_min", "x_min", "y_max", "x_max"),
            ("y1", "x1", "y2", "x2"),
        ]
        for k in keys:
            if all(key in box_value for key in k):
                coerced = _coerce_one([box_value[k[0]], box_value[k[1]], box_value[k[2]], box_value[k[3]]])
                return [coerced] if coerced is not None else []
        return []

    one = _coerce_one(box_value)
    if one is not None:
        return [one]

    if isinstance(box_value, list) and box_value and all(isinstance(b, (list, tuple)) for b in box_value):
        boxes: list[list[int]] = []
        for b in box_value:
            coerced = _coerce_one(b)
            if coerced is not None:
                boxes.append(coerced)
        return boxes

    return []


def _parse_regions_json(raw: str, *, source: str) -> list[BBoxRegion]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.splitlines() if not line.startswith("``` ") and not line.startswith("```") ).strip()

    # Preferred path: schema-validated JSON.
    try:
        parsed = _GeminiRegionList.model_validate_json(raw).root
        return [BBoxRegion(label=r.label, box=list(map(int, r.box))) for r in parsed]
    except Exception:
        pass

    # Fallback: best-effort parse + normalization.
    try:
        parsed_any: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("%s returned non-JSON response: %s", source, raw[:500])
        raise ValueError(f"Could not parse {source} response as JSON: {exc}") from exc

    if isinstance(parsed_any, dict) and "regions" in parsed_any:
        parsed_any = parsed_any["regions"]

    if not isinstance(parsed_any, list):
        logger.error("%s returned JSON that is not a list: %s", source, str(parsed_any)[:200])
        return []

    regions: list[BBoxRegion] = []
    for item in parsed_any:
        if not isinstance(item, dict):
            continue
        raw_label = str(item.get("label", ""))
        label = _normalize_label(raw_label)
        if label is None:
            logger.warning("%s: unknown label '%s' – skipping.", source, raw_label)
            continue

        boxes = _coerce_boxes(item.get("box", []))
        if not boxes:
            logger.warning("%s: invalid box %s for '%s' – skipping.", source, item.get("box", []), label)
            continue

        for box in boxes:
            if len(box) != 4:
                logger.warning("%s: invalid box %s for '%s' – skipping.", source, box, label)
                continue
            regions.append(BBoxRegion(label=label, box=box))

    return regions


# ---------------------------------------------------------------------------
# Core annotation logic
# ---------------------------------------------------------------------------


def _call_gemini(image_path: Path, model_name: str) -> list[BBoxRegion]:
    """Send the image to Gemini and parse the JSON response into BBoxRegion list."""
    _load_dotenv()

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is required. Install with: uv pip install google-genai"
        ) from exc

    from google.genai import types  # noqa: PLC0415

    api_key = (
        os.environ.get("GEMINI_FLASH_API_KEY")
        or os.environ.get("GEMINI_PRO_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    client = genai.Client(api_key=api_key)
    img = Image.open(image_path)

    _is_pro = "pro" in model_name.lower()
    _cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=_GeminiRegionList.model_json_schema(),
        **({} if _is_pro else {"thinking_config": types.ThinkingConfig(thinking_budget=0)}),
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[img, _ANNOTATION_PROMPT],
        config=_cfg,
    )

    raw = response.text or ""
    regions = _parse_regions_json(raw, source="Gemini")

    logger.info("Gemini returned %d valid regions for %s", len(regions), image_path.name)
    return regions


def _call_gemini_with_prompt(
    image_path: Path,
    model_name: str,
    prompt: str,
) -> list[BBoxRegion]:
    """Like :func:`_call_gemini` but accepts an explicit *prompt* string."""
    _load_dotenv()

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is required. Install with: uv pip install google-genai"
        ) from exc

    from google.genai import types  # noqa: PLC0415

    api_key = (
        os.environ.get("GEMINI_FLASH_API_KEY")
        or os.environ.get("GEMINI_PRO_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    client = genai.Client(api_key=api_key)
    img = Image.open(image_path)

    _is_pro = "pro" in model_name.lower()
    _cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=_GeminiRegionList.model_json_schema(),
        **({} if _is_pro else {"thinking_config": types.ThinkingConfig(thinking_budget=0)}),
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[img, prompt],
        config=_cfg,
    )

    raw = response.text or ""
    regions = _parse_regions_json(raw, source="Gemini")

    logger.info(
        "_call_gemini_with_prompt: %d valid regions for %s", len(regions), image_path.name
    )
    return regions


def _critique_annotations(
    image_path: Path,
    regions: list["BBoxRegion"],
    model_name: str,
) -> list["BBoxRegion"]:
    """Send image + first-pass annotations to the critic model and return refined regions."""
    _load_dotenv()

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is required. Install with: uv pip install google-genai"
        ) from exc

    from google.genai import types  # noqa: PLC0415

    api_key = os.environ.get("GEMINI_PRO_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    img = Image.open(image_path)

    annotations_json = json.dumps(
        [{"label": r.label, "box": r.box} for r in regions], indent=2
    )
    prompt = _CRITIQUE_PROMPT_TEMPLATE.format(
        n_regions=len(regions),
        annotations_json=annotations_json,
    )

    _is_pro = "pro" in model_name.lower()
    _cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=_GeminiRegionList.model_json_schema(),
        **({} if _is_pro else {"thinking_config": types.ThinkingConfig(thinking_budget=0)}),
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[img, prompt],
        config=_cfg,
    )

    raw = response.text or ""
    refined = _parse_regions_json(raw, source="Critic")

    logger.info(
        "Critic refined %d → %d regions for %s",
        len(regions), len(refined), image_path.name,
    )
    return refined


def _write_yolo_label(regions: list[BBoxRegion], label_path: Path) -> None:
    """Write a YOLO-format .txt label file."""
    lines: list[str] = []
    for region in regions:
        class_id, cx, cy, w, h = region.to_yolo()
        if class_id < 0:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.debug("Wrote %d YOLO labels to %s", len(lines), label_path)


def _write_visualisation(
    image_path: Path,
    regions: list[BBoxRegion],
    output_path: Path,
    *,
    round_label: str = "",
    column_bounds: list[int] | None = None,
) -> None:
    """Draw bounding boxes + optional column boundary lines on the image and save as PNG."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        # Pillow < 10 doesn't support size kwarg on load_default
        font = ImageFont.load_default()

    for region in regions:
        y_min_n, x_min_n, y_max_n, x_max_n = (v / 1000.0 for v in region.box)
        x0 = int(x_min_n * width)
        y0 = int(y_min_n * height)
        x1 = int(x_max_n * width)
        y1 = int(y_max_n * height)
        colour = _VIS_COLOURS.get(region.label, "#FFFFFF")

        # Semi-transparent fill
        draw.rectangle([x0, y0, x1, y1], fill=colour + "33", outline=colour, width=2)

        # Label tag background
        tag_text = region.label.replace("_", " ")
        bbox_text = draw.textbbox((x0 + 2, y0 + 2), tag_text, font=font)
        draw.rectangle(bbox_text, fill=colour + "CC")
        draw.text((x0 + 2, y0 + 2), tag_text, fill="#FFFFFF", font=font)

    if column_bounds:
        img_w, _ = img.size
        for bx in column_bounds:
            draw.line([(bx, 0), (bx, height)], fill="#E63946", width=2)

    if round_label:
        try:
            font_banner = ImageFont.load_default(size=16)
        except TypeError:
            font_banner = ImageFont.load_default()
        banner_h = 28
        draw.rectangle([0, 0, width, banner_h], fill="#000000DD")
        draw.text((6, 6), round_label, fill="#FFFFFF", font=font_banner)

    img.save(output_path, format="PNG")
    logger.info("Saved visualisation → %s", output_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_page(
    image_path: Path,
    labels_dir: Path,
    images_dir: Path,
    vis_dir: Path,
    *,
    generator_model: str = "gemini-2.5-flash",
    critic_model: str = "gemini-2.5-pro",
    critique_rounds: int = 1,
    model_name: str | None = None,
    overwrite: bool = False,
) -> list[BBoxRegion]:
    """Auto-annotate a single newspaper page with a Generator/Critic pattern.

    Parameters
    ----------
    image_path:
        Path to the low-resolution ``.jpg`` preprocessed page.
    labels_dir:
        Directory where the YOLO ``.txt`` label file will be written.
    images_dir:
        Directory where a copy of the image will be placed for YOLO training.
    vis_dir:
        Directory for human-review overlay PNGs (one per round + final).
    generator_model:
        Gemini model used for the first-pass annotation.
    critic_model:
        Gemini model used to review and refine each round.
    critique_rounds:
        Number of critic refinement passes (0 = generator only).
    model_name:
        Deprecated shorthand; overrides *generator_model* when set.
    overwrite:
        If ``False`` (default), skip images that already have a label file.

    Returns
    -------
    list[BBoxRegion]
        The final (critic-refined) regions.
    """
    if model_name is not None:
        generator_model = model_name

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    label_path = labels_dir / (image_path.stem + ".txt")
    if label_path.exists() and not overwrite:
        logger.info("Skipping %s – label already exists.", image_path.name)
        return []

    dest_image = images_dir / image_path.name
    if not dest_image.exists() or overwrite:
        shutil.copy2(image_path, dest_image)

    stem = image_path.stem
    annotated_at = datetime.now(timezone.utc).isoformat()

    # ── Round 0: generator ────────────────────────────────────────────────
    regions = _call_gemini(image_path, generator_model)
    _write_visualisation(
        image_path,
        regions,
        vis_dir / f"{stem}_r0_vis.png",
        round_label=f"Round 0 | Generator: {generator_model} | {len(regions)} regions",
    )
    round_stats = [
        {
            "round": 0,
            "model": generator_model,
            "role": "generator",
            "n_regions": len(regions),
            "class_counts": dict(Counter(r.label for r in regions)),
        }
    ]

    current_regions = regions

    # ── Critique passes ───────────────────────────────────────────────────
    for i in range(critique_rounds):
        refined = _critique_annotations(image_path, current_regions, critic_model)
        _write_visualisation(
            image_path,
            refined,
            vis_dir / f"{stem}_r{i + 1}_vis.png",
            round_label=(
                f"Round {i + 1} | Critic: {critic_model}"
                f" | {len(refined)} regions (was {len(current_regions)})"
            ),
        )
        round_stats.append(
            {
                "round": i + 1,
                "model": critic_model,
                "role": "critic",
                "n_regions": len(refined),
                "class_counts": dict(Counter(r.label for r in refined)),
            }
        )
        current_regions = refined

    # ── Write final YOLO label + canonical visualisation ─────────────────
    _write_yolo_label(current_regions, label_path)
    _write_visualisation(
        image_path,
        current_regions,
        vis_dir / f"{stem}_vis.png",
        round_label=f"FINAL | {len(current_regions)} regions",
    )

    # ── Per-page stats sidecar ────────────────────────────────────────────
    stats = {
        "image": image_path.name,
        "annotated_at": annotated_at,
        "generator_model": generator_model,
        "critic_model": critic_model,
        "critique_rounds": critique_rounds,
        "rounds": round_stats,
    }
    (vis_dir / f"{stem}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(
        "annotate_page done: %s — %d final regions after %d round(s)",
        image_path.name, len(current_regions), critique_rounds,
    )
    return current_regions


def annotate_directory(
    input_dir: Path,
    labels_dir: Path,
    images_dir: Path,
    vis_dir: Path,
    *,
    generator_model: str = "gemini-2.5-flash",
    critic_model: str = "gemini-2.5-pro",
    critique_rounds: int = 1,
    model_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Batch-annotate all ``.jpg`` files in *input_dir*.

    Returns a summary dict ``{image_stem: region_count}``.
    """
    jpg_files = sorted(input_dir.glob("*.jpg"))
    if not jpg_files:
        logger.warning("No .jpg files found in %s", input_dir)
        return {}

    summary: dict[str, int] = {}
    for jpg in jpg_files:
        try:
            regions = annotate_page(
                jpg,
                labels_dir=labels_dir,
                images_dir=images_dir,
                vis_dir=vis_dir,
                generator_model=generator_model,
                critic_model=critic_model,
                critique_rounds=critique_rounds,
                model_name=model_name,
                overwrite=overwrite,
            )
            summary[jpg.stem] = len(regions)
        except Exception:
            logger.exception("Failed to annotate %s – continuing.", jpg.name)
            summary[jpg.stem] = -1

    logger.info(
        "Batch annotation complete: %d images, %d total regions.",
        len(summary),
        sum(v for v in summary.values() if v > 0),
    )
    return summary


# ---------------------------------------------------------------------------
# Structured (column-aware) annotation
# ---------------------------------------------------------------------------


def annotate_page_structured(
    image_path: Path,
    labels_dir: Path,
    images_dir: Path,
    vis_dir: Path,
    *,
    generator_model: str = "gemini-2.5-flash",
    critic_model: str = "gemini-2.5-pro",
    critique_rounds: int = 1,
    n_columns_hint: int = 8,
    masthead_frac: float = 0.12,
    overlap_frac: float = 0.05,
    show_vis: bool = False,
    overwrite: bool = False,
) -> list[BBoxRegion]:
    """Column-aware annotation: detect page structure then annotate per strip.

    Workflow
    --------
    1. Detect skew, correct if needed.
    2. Compute vertical projection profile → detect column boundaries.
    3. Supplement boundary detection with printed vertical-rule detection.
    4. Decompose page into masthead strip, N column strips, and a
       full-page thumbnail.
    5. Annotate each strip with Gemini (generator → optional critic).
    6. Convert all strip-local boxes to full-page coordinates.
    7. IoU-deduplicate across strips.
    8. Write YOLO labels, per-round visualisations (with column boundary lines),
       and a stats JSON sidecar.

    Parameters
    ----------
    image_path:
        Path to the ``.jpg`` processed image.  The corresponding high-res
        ``.png`` (same stem, same folder) is used for structure analysis and
        strip extraction if available; falls back to the JPG.
    labels_dir, images_dir, vis_dir:
        Output directories (same semantics as :func:`annotate_page`).
    generator_model:
        Gemini model for first-pass per-strip annotation.
    critic_model:
        Gemini model for per-strip critique/refinement.
    critique_rounds:
        Number of critic passes per strip (0 = generator only).
    n_columns_hint:
        Expected number of columns (passed to structure detection).
    masthead_frac:
        Fraction of page height treated as masthead zone.
    overlap_frac:
        Overlap fraction added to each side of a column strip.
    overwrite:
        Skip if final label file already exists.

    Returns
    -------
    list[BBoxRegion]
        Final merged, deduplicated regions in full-page 0-1000 space.
    """
    from newspapers.segmentation.structure import (
        analyse_page_structure,
        merge_strip_annotations,
        draw_column_bounds,
    )

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    label_path = labels_dir / (image_path.stem + ".txt")
    if label_path.exists() and not overwrite:
        logger.info("Skipping %s – label already exists.", image_path.name)
        return []

    # Prefer the high-res PNG for structure analysis
    png_path = image_path.with_suffix(".png")
    struct_image = png_path if png_path.exists() else image_path

    # ── Step 1: page structure analysis ────────────────────────────────
    column_bounds, strips, profile, skew_angle = analyse_page_structure(
        struct_image,
        n_columns_hint=n_columns_hint,
        masthead_frac=masthead_frac,
        overlap_frac=overlap_frac,
    )
    logger.info(
        "Page structure: skew=%.2f°, %d column bounds, %d strips",
        skew_angle, len(column_bounds), len(strips),
    )

    # ── Checkpoint setup ────────────────────────────────────────────
    checkpoint_path = vis_dir / (image_path.stem + "_checkpoint.json")
    strip_cache_dir = vis_dir / (image_path.stem + "_strips_cache")
    completed_strip_ids: set[str] = set()
    if not overwrite and checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            completed_strip_ids = set(ckpt.get("completed_strips", []))
            if completed_strip_ids:
                logger.info(
                    "Resuming from checkpoint: %d strips already done.",
                    len(completed_strip_ids),
                )
        except (json.JSONDecodeError, OSError):
            completed_strip_ids = set()

    # ── Step 2: annotate each strip ──────────────────────────────────
    strip_results: list[tuple] = []
    annotated_at = datetime.now(timezone.utc).isoformat()

    strip_cache_dir.mkdir(parents=True, exist_ok=True)

    for strip in strips:
        cache_file = strip_cache_dir / f"{strip.strip_id}.json"

        # Resume: load cached result for already-completed strips
        if strip.strip_id in completed_strip_ids and cache_file.exists():
            logger.info("Loading cached annotation for strip '%s'.", strip.strip_id)
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                regions: list[BBoxRegion] = [
                    BBoxRegion(label=item["label"], box=item["box"])
                    for item in cached
                    if item.get("label") in CLASS_ID and len(item.get("box", [])) == 4
                ]
            except (json.JSONDecodeError, OSError, KeyError):
                logger.warning("Could not load cache for strip '%s'; re-annotating.", strip.strip_id)
                completed_strip_ids.discard(strip.strip_id)
                regions = []  # will fall through to annotation below
            else:
                strip_results.append((strip, regions))
                continue

        logger.info("Annotating strip '%s': %s", strip.strip_id, strip.image_path.name)

        # Choose the appropriate prompt based on strip type
        if strip.strip_id == "masthead":
            prompt = _MASTHEAD_ANNOTATION_PROMPT
        elif strip.strip_id == "full":
            prompt = _FULL_PAGE_CROSS_COL_PROMPT
        else:
            # Column strip
            prompt = _STRIP_ANNOTATION_PROMPT_TEMPLATE.format(
                column_index=strip.column_index,
                n_columns=strip.column_count,
                page_width=strip.page_width,
                page_height=strip.page_height,
                x_start=strip.x_offset,
                x_end=strip.x_offset + strip.strip_width,
            )

        regions = _call_gemini_with_prompt(strip.image_path, generator_model, prompt)

        if critique_rounds > 0:
            for _ in range(critique_rounds):
                regions = _critique_annotations(strip.image_path, regions, critic_model)

        # Persist strip result to cache and update checkpoint
        cache_file.write_text(
            json.dumps([{"label": r.label, "box": r.box} for r in regions], indent=2),
            encoding="utf-8",
        )
        completed_strip_ids.add(strip.strip_id)
        checkpoint_path.write_text(
            json.dumps({"completed_strips": list(completed_strip_ids)}, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Checkpoint saved: %d/%d strips done.", len(completed_strip_ids), len(strips)
        )

        strip_results.append((strip, regions))

    # ── Step 3: merge into full-page coordinates ──────────────────────
    page_w = strips[0].page_width
    page_h = strips[0].page_height
    final_regions = merge_strip_annotations(
        strip_results, page_w, page_h, column_bounds
    )

    # ── Step 4: write outputs ──────────────────────────────────────
    dest_image = images_dir / image_path.name
    if not dest_image.exists() or overwrite:
        shutil.copy2(image_path, dest_image)

    _write_yolo_label(final_regions, label_path)
    vis_png = vis_dir / (image_path.stem + "_structured_vis.png")
    _write_visualisation(
        image_path,
        final_regions,
        vis_png,
        round_label=f"STRUCTURED | {len(strips)} strips | {len(final_regions)} regions",
        column_bounds=column_bounds,
    )
    if show_vis and vis_png.exists():
        import os as _os  # noqa: PLC0415
        _os.startfile(str(vis_png))

    # Stats sidecar
    strip_stats = [
        {
            "strip_id": s.strip_id,
            "column_index": s.column_index,
            "n_regions": len(r),
            "class_counts": dict(Counter(reg.label for reg in r)),
        }
        for s, r in strip_results
    ]
    stats = {
        "image": image_path.name,
        "annotated_at": annotated_at,
        "mode": "structured",
        "generator_model": generator_model,
        "critic_model": critic_model,
        "critique_rounds": critique_rounds,
        "skew_angle_deg": skew_angle,
        "n_columns": len(column_bounds) + 1,
        "column_bounds": column_bounds,
        "overlap_frac": overlap_frac,
        "final_n_regions": len(final_regions),
        "final_class_counts": dict(Counter(r.label for r in final_regions)),
        "strips": strip_stats,
    }
    (vis_dir / (image_path.stem + "_structured_stats.json")).write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    # Clean up checkpoint and strip cache now that the run completed successfully
    checkpoint_path.unlink(missing_ok=True)
    shutil.rmtree(strip_cache_dir, ignore_errors=True)

    logger.info(
        "annotate_page_structured done: %s — %d final regions from %d strips",
        image_path.name, len(final_regions), len(strips),
    )
    return final_regions


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Auto-annotate newspaper page images using a Gemini Generator/Critic pattern."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed"),
        help="Single .jpg image or directory of .jpg images to annotate.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/labels/train"),
        help="Output directory for YOLO .txt label files.",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("data/annotations/images/train"),
        help="Output directory for image copies (sibling of labels/ for YOLO).",
    )
    p.add_argument(
        "--vis",
        type=Path,
        default=Path("data/annotations/visualizations"),
        help="Output directory for human-review overlay PNGs.",
    )
    p.add_argument(
        "--generator-model",
        default="gemini-2.5-flash",
        help="Gemini model ID for first-pass annotation (generator).",
    )
    p.add_argument(
        "--critic-model",
        default="gemini-2.5-pro",
        help="Gemini model ID for annotation review/refinement (critic).",
    )
    p.add_argument(
        "--critique-rounds",
        type=int,
        default=1,
        help="Number of critic refinement passes (0 = generator only).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-annotate images that already have a label file.",
    )
    p.add_argument(
        "--structured",
        action="store_true",
        help="Use structured (column-aware) annotation via structure.py pipeline.",
    )
    p.add_argument(
        "--n-columns",
        type=int,
        default=8,
        help="Hint for number of columns (used only with --structured).",
    )
    p.add_argument(
        "--overlap-frac",
        type=float,
        default=0.05,
        help="Overlap fraction added to each side of a column strip (--structured only).",
    )
    p.add_argument(
        "--show-vis",
        action="store_true",
        help="Open the visualisation PNG in the default viewer after each image (--structured only).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )

    inp: Path = args.input
    if inp.is_dir():
        if args.structured:
            # Batch structured run over all JPGs
            jpg_files = sorted(inp.glob("*.jpg"))
            for jpg in jpg_files:
                try:
                    regions = annotate_page_structured(
                        jpg,
                        labels_dir=args.labels,
                        images_dir=args.images,
                        vis_dir=args.vis,
                        generator_model=args.generator_model,
                        critic_model=args.critic_model,
                        critique_rounds=args.critique_rounds,
                        n_columns_hint=args.n_columns,
                        overlap_frac=args.overlap_frac,
                        show_vis=args.show_vis,
                        overwrite=args.overwrite,
                    )
                    print(f"  {jpg.stem}: {len(regions)} regions (structured)")
                except Exception:
                    logger.exception("Failed to annotate %s", jpg.name)
                    print(f"  {jpg.stem}: FAILED")
        else:
            summary = annotate_directory(
                inp,
                labels_dir=args.labels,
                images_dir=args.images,
                vis_dir=args.vis,
                generator_model=args.generator_model,
                critic_model=args.critic_model,
                critique_rounds=args.critique_rounds,
                overwrite=args.overwrite,
            )
            for stem, count in summary.items():
                status = f"{count} regions" if count >= 0 else "FAILED"
                print(f"  {stem}: {status}")
    else:
        if args.structured:
            regions = annotate_page_structured(
                inp,
                labels_dir=args.labels,
                images_dir=args.images,
                vis_dir=args.vis,
                generator_model=args.generator_model,
                critic_model=args.critic_model,
                critique_rounds=args.critique_rounds,
                n_columns_hint=args.n_columns,
                overlap_frac=args.overlap_frac,
                show_vis=args.show_vis,
                overwrite=args.overwrite,
            )
        else:
            regions = annotate_page(
                inp,
                labels_dir=args.labels,
                images_dir=args.images,
                vis_dir=args.vis,
                generator_model=args.generator_model,
                critic_model=args.critic_model,
                critique_rounds=args.critique_rounds,
                overwrite=args.overwrite,
            )
        print(f"Annotated {inp.name}: {len(regions)} regions detected.")
        for r in regions:
            print(f"  [{r.label}] box={r.box}")
