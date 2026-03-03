"""YOLOv11 newspaper page segmentation.

Detects individual advertisement regions on a scanned newspaper page
and returns bounding-box metadata for downstream cropping.
"""

from __future__ import annotations

import logging
from pathlib import Path

from newspapers.models import PageSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_segments(
    image_path: Path,
    model_path: Path,
    *,
    confidence_threshold: float = 0.25,
) -> list[PageSegment]:
    """Run YOLOv11 inference on a newspaper page image.

    Parameters
    ----------
    image_path:
        Path to the preprocessed page image (JPEG).
    model_path:
        Path to the YOLOv11 ``.pt`` weights file.
    confidence_threshold:
        Minimum confidence to keep a detection.

    Returns
    -------
    list[PageSegment]
        Detected page segments with bounding boxes and labels.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for segmentation. Install it with: pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    results = model(str(image_path))

    segments: list[PageSegment] = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = (float(c) for c in box.xyxy[0])
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, f"class_{cls_id}")
            segments.append(
                PageSegment(
                    label=label,
                    x_min=x1,
                    y_min=y1,
                    x_max=x2,
                    y_max=y2,
                    confidence=conf,
                )
            )

    logger.info("Detected %d segments in %s", len(segments), image_path.name)
    return segments


def crop_segments(
    image_path: Path,
    segments: list[PageSegment],
    output_dir: Path,
) -> list[Path]:
    """Crop detected segments from the high-resolution page image.

    Parameters
    ----------
    image_path:
        Path to the high-resolution page image (PNG).
    segments:
        Segments produced by :func:`detect_segments`.
    output_dir:
        Directory to write the cropped images.

    Returns
    -------
    list[Path]
        Paths to the cropped image files.
    """
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    stem = image_path.stem

    paths: list[Path] = []
    for idx, seg in enumerate(segments):
        crop = img.crop((seg.x_min, seg.y_min, seg.x_max, seg.y_max))
        out = output_dir / f"{stem}_seg{idx:04d}_{seg.label}.png"
        crop.save(out, format="PNG")
        paths.append(out)

    logger.info("Cropped %d segments from %s", len(paths), image_path.name)
    return paths
