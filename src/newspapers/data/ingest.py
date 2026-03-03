"""KB archive ingestion — download and convert JPEG2000 scans.

Provides utilities for:
- Querying the KB (tidningar.kb.se) API for newspaper metadata.
- Downloading JPEG2000 (.jp2) page scans.
- Converting .jp2 files to optimised .jpg / .png for downstream processing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JPEG2000 → standard format conversion
# ---------------------------------------------------------------------------


def convert_jp2(
    input_path: Path,
    output_dir: Path,
    *,
    low_res_size: tuple[int, int] = (1280, 1280),
) -> tuple[Path, Path]:
    """Convert a JPEG2000 file to both a low-res JPEG and a high-res PNG.

    Parameters
    ----------
    input_path:
        Path to the source ``.jp2`` file.
    output_dir:
        Directory where the converted files will be written.
    low_res_size:
        Maximum (width, height) for the low-resolution JPEG thumbnail
        used by the segmentation model.

    Returns
    -------
    tuple[Path, Path]
        ``(jpg_path, png_path)`` — paths to the generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    img = Image.open(input_path)

    # High-resolution lossless PNG (for Vision LLM extraction)
    png_path = output_dir / f"{stem}.png"
    img.save(png_path, format="PNG")

    # Low-resolution JPEG (for YOLOv11 segmentation)
    jpg_path = output_dir / f"{stem}.jpg"
    thumbnail = img.copy()
    thumbnail.thumbnail(low_res_size, Image.LANCZOS)
    thumbnail.save(jpg_path, format="JPEG", quality=85)

    logger.info("Converted %s → %s, %s", input_path.name, jpg_path.name, png_path.name)
    return jpg_path, png_path
