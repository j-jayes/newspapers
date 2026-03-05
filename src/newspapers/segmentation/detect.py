"""YOLOv11 newspaper page segmentation.

Detects individual advertisement regions on a scanned newspaper page
and returns bounding-box metadata for downstream cropping.

Coordinate spaces
-----------------
YOLO **inference** runs on the low-resolution JPEG (max 1280 × 1280 px).
The bounding boxes it returns are in **JPEG pixel space**.

**Cropping** must be done on the full-resolution PNG (which can be
4 000 × 6 000 px or larger) to preserve legibility for the downstream
Vision LLM.  Pass the PNG path as *inference_image_path* to
:func:`crop_segments` and it will automatically scale the coordinates.

Usage
-----
Run detection and crop from CLI::

    uv run python -m newspapers.segmentation.detect \\
        --input   data/processed \\
        --model   models/newspapers_detector.pt \\
        --output  data/interim/crops
"""

from __future__ import annotations

import argparse
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
    *,
    inference_image_path: Path | None = None,
) -> list[Path]:
    """Crop detected segments from a page image.

    Parameters
    ----------
    image_path:
        Path to the image to crop from.  Typically the **full-resolution PNG**
        produced by :func:`~newspapers.data.ingest.convert_jp2` for maximum
        legibility when passed to the downstream Vision LLM.
    segments:
        Segments produced by :func:`detect_segments`.  Their bounding-box
        coordinates are in the pixel space of the image that was fed to YOLO
        (usually a low-res JPEG).
    output_dir:
        Directory to write the cropped images.
    inference_image_path:
        Path to the image that was actually fed to YOLO for inference (the
        low-res JPEG).  When supplied, the function derives scale factors
        ``(crop_w / infer_w, crop_h / infer_h)`` and maps every bounding-box
        coordinate from JPEG pixel space to full-res PNG pixel space before
        cropping.  If ``None``, coordinates are used as-is (suitable when
        both images share identical dimensions).

    Returns
    -------
    list[Path]
        Paths to the cropped image files (PNG).
    """
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    crop_w, crop_h = img.size
    stem = image_path.stem

    # Derive scale factors when the inference image differs from the crop image
    scale_x = scale_y = 1.0
    if inference_image_path is not None and inference_image_path != image_path:
        with Image.open(inference_image_path) as infer_img:
            infer_w, infer_h = infer_img.size
        if infer_w > 0 and infer_h > 0:
            scale_x = crop_w / infer_w
            scale_y = crop_h / infer_h
            logger.debug(
                "Coordinate scaling: infer(%dx%d) -> crop(%dx%d)  sx=%.4f sy=%.4f",
                infer_w, infer_h, crop_w, crop_h, scale_x, scale_y,
            )

    paths: list[Path] = []
    for idx, seg in enumerate(segments):
        x0 = int(seg.x_min * scale_x)
        y0 = int(seg.y_min * scale_y)
        x1 = int(seg.x_max * scale_x)
        y1 = int(seg.y_max * scale_y)

        # Clamp to image bounds
        x0 = max(0, min(x0, crop_w))
        y0 = max(0, min(y0, crop_h))
        x1 = max(0, min(x1, crop_w))
        y1 = max(0, min(y1, crop_h))

        if x1 <= x0 or y1 <= y0:
            logger.warning("Degenerate crop box at idx %d – skipping.", idx)
            continue

        crop = img.crop((x0, y0, x1, y1))
        out = output_dir / f"{stem}_seg{idx:04d}_{seg.label}.png"
        crop.save(out, format="PNG")
        paths.append(out)

    logger.info("Cropped %d segments from %s", len(paths), image_path.name)
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run YOLOv11 detection on preprocessed newspaper pages and crop segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed"),
        help="Single .jpg file or directory of .jpg files to run detection on.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path("models/newspapers_detector.pt"),
        help="Path to the trained YOLOv11 .pt weights.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/crops"),
        help="Directory to save cropped segment PNGs.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence threshold.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


if __name__ == "__main__":
    import logging as _logging

    args = _build_parser().parse_args()
    _logging.basicConfig(
        level=_logging.DEBUG if args.verbose else _logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )

    inp: Path = args.input
    jpg_files = [inp] if inp.is_file() else sorted(inp.glob("*.jpg"))

    if not jpg_files:
        print(f"No .jpg files found in {inp}")
        raise SystemExit(1)

    total_crops = 0
    for jpg in jpg_files:
        # Use the sibling high-res PNG for cropping if it exists
        png = jpg.with_suffix(".png")
        crop_source = png if png.exists() else jpg
        if crop_source != jpg:
            logger.info("Using high-res PNG for cropping: %s", png.name)

        segs = detect_segments(jpg, args.model, confidence_threshold=args.conf)
        if not segs:
            print(f"  {jpg.name}: no segments detected.")
            continue

        crops = crop_segments(
            crop_source,
            segs,
            args.output,
            inference_image_path=jpg if crop_source != jpg else None,
        )
        total_crops += len(crops)
        print(f"  {jpg.name}: {len(segs)} segments → {len(crops)} crops saved.")

    print(f"\nDone. {total_crops} total crops saved to {args.output}.")
