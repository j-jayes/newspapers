"""Newspaper page structure detection.

Classical-CV pipeline that runs *before* any LLM call:
  1. Skew detection & correction (Hough-lines on Canny edges)
  2. Vertical projection profile → column valley detection (SciPy)
  3. Morphological vertical-rule detection (supplementary)
  4. Strip decomposition — masthead, per-column, and full-page thumbnail
  5. Annotation coordinate merging + IoU deduplication

Public API
----------
detect_skew, correct_skew,
compute_projection_profile, detect_column_boundaries,
detect_vertical_rules, finalise_column_bounds,
PageStrip, decompose_into_strips,
strip_to_page_coords, merge_strip_annotations,
draw_column_bounds
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

if TYPE_CHECKING:
    pass  # avoid circular imports

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable defaults (module-level constants so they're easy to find)
# ---------------------------------------------------------------------------

#: Gaussian smoothing sigma for the projection profile, expressed as a
#: fraction of the image width. E.g. 0.005 × 2480 ≈ 12-pixel kernel.
PROFILE_SIGMA_FRAC: float = 0.005

#: How much to extend each column strip beyond its boundaries on each side.
#: 0.05 = 5 % of the column's own width.  Tune via `decompose_into_strips`.
DEFAULT_OVERLAP_FRAC: float = 0.05

#: Page fraction treated as masthead (top portion extracted separately).
DEFAULT_MASTHEAD_FRAC: float = 0.12

#: IoU threshold above which two overlapping boxes are considered duplicates.
IOU_DEDUP_THRESHOLD: float = 0.5

#: Minimum gap between two column boundaries as a fraction of page width.
#: Prevents spurious valleys very close together.
MIN_COLUMN_GAP_FRAC: float = 0.04

#: Tolerance (px) for merging a printed-rule position with a valley position.
RULE_MERGE_TOLERANCE_PX: int = 15


# ---------------------------------------------------------------------------
# 1. Skew detection & correction
# ---------------------------------------------------------------------------


def _to_gray_uint8(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an 8-bit grayscale numpy array."""
    return np.array(pil_image.convert("L"), dtype=np.uint8)


def detect_skew(gray_arr: np.ndarray) -> float:
    """Estimate the page skew angle in degrees using Hough-line analysis.

    Works by finding long near-vertical lines in the Canny edge map and
    returning the median angular deviation from true vertical.

    Parameters
    ----------
    gray_arr:
        8-bit grayscale image as a numpy array.

    Returns
    -------
    float
        Skew angle in degrees.  Positive = clockwise tilt.
        Returns 0.0 if no reliable lines are found.
    """
    blurred = cv2.GaussianBlur(gray_arr, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=150)
    if lines is None:
        logger.debug("detect_skew: no Hough lines found; returning 0.0")
        return 0.0

    angles: list[float] = []
    for line in lines:
        rho, theta = line[0]
        # theta is the angle of the line's *normal*.
        # A true vertical line has theta ≈ 0 or π.
        # We want deviations from vertical — convert to degrees from 90°.
        angle_deg = math.degrees(theta) - 90.0
        # Only keep near-vertical lines (within ±10°)
        if abs(angle_deg) <= 10.0:
            angles.append(angle_deg)

    if not angles:
        logger.debug("detect_skew: no near-vertical lines; returning 0.0")
        return 0.0

    skew = float(np.median(angles))
    logger.info("detect_skew: %.3f° (from %d near-vertical lines)", skew, len(angles))
    return skew


def correct_skew(pil_image: Image.Image, angle: float) -> Image.Image:
    """Rotate the image by *-angle* degrees to correct skew.

    The rotation is about the image centre; the canvas is expanded so no
    content is cropped.  Background is filled with white (255, 255, 255).

    Parameters
    ----------
    pil_image:
        Input PIL image (any mode; RGB preferred).
    angle:
        Skew angle as returned by :func:`detect_skew`.

    Returns
    -------
    PIL.Image.Image
        De-skewed image.
    """
    if abs(angle) < 0.1:
        return pil_image  # not worth resampling

    arr = np.array(pil_image.convert("RGB"))
    h, w = arr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), -angle, scale=1.0)

    # Compute new bounding box size after rotation
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust translation so image is centred in the new canvas
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    corrected = cv2.warpAffine(
        arr, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    result = Image.fromarray(corrected, "RGB")
    logger.info("correct_skew: rotated %.3f° (%dx%d → %dx%d)", angle, w, h, new_w, new_h)
    return result


# ---------------------------------------------------------------------------
# 2. Vertical projection profile & column boundary detection
# ---------------------------------------------------------------------------


def compute_projection_profile(gray_arr: np.ndarray) -> np.ndarray:
    """Compute the vertical ink-density projection profile.

    Binarises using Otsu's threshold, inverts (ink = 1), then sums along
    ``axis=0`` to obtain a 1-D array of ink density at each x-column.
    The result is Gaussian-smoothed to remove high-frequency noise.

    Parameters
    ----------
    gray_arr:
        8-bit grayscale numpy array.

    Returns
    -------
    np.ndarray
        1-D float array of length ``gray_arr.shape[1]``.  Higher values
        indicate denser ink (text columns); lower values indicate gutters.
    """
    _, binary = cv2.threshold(gray_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert: ink pixels become 1
    ink = (binary == 0).astype(np.float32)
    profile = ink.sum(axis=0)

    sigma = max(2.0, gray_arr.shape[1] * PROFILE_SIGMA_FRAC)
    smoothed = gaussian_filter1d(profile, sigma=sigma)
    return smoothed


#: Fraction of page width from each edge within which a valley is treated
#: as a page margin rather than a column divider.
MARGIN_EXCLUSION_FRAC: float = 0.03


def detect_column_boundaries(
    profile: np.ndarray,
    n_hint: int | None = None,
    *,
    page_height: int = 1,
    margin_exclusion_frac: float = MARGIN_EXCLUSION_FRAC,
) -> list[int]:
    """Find *inter-column gutter* x-positions from a projection profile.

    Uses SciPy :func:`~scipy.signal.find_peaks` on the *negated* profile
    to locate valleys.  When *n_hint* is provided the algorithm targets
    ``n_hint - 1`` valleys.

    Valleys within *margin_exclusion_frac* × page_width of the left or right
    edge are treated as outer-margin gutters and discarded, since they divide
    the blank margin from column 1 (or last column from the right margin)
    rather than separating two text columns.

    Parameters
    ----------
    profile:
        1-D array from :func:`compute_projection_profile`.
    n_hint:
        Expected number of columns (helps constrain detection).
    page_height:
        Original image height (px).  Unused currently; reserved for future
        prominence scaling.
    margin_exclusion_frac:
        Fraction of page width within which edge valleys are excluded.

    Returns
    -------
    list[int]
        Sorted list of x-positions of column boundary centres.
        Does NOT include the left (0) or right (page_width) edge.
    """
    page_width = len(profile)
    margin_px = int(page_width * margin_exclusion_frac)
    target_valleys = (n_hint - 1) if (n_hint and n_hint > 1) else None
    min_distance = int(page_width * MIN_COLUMN_GAP_FRAC)

    # Find all peaks in the negated profile (valleys in the original)
    valleys, props = find_peaks(
        -profile,
        distance=min_distance,
        prominence=0,
    )

    # Exclude near-edge valleys (these are outer-margin gaps, not column dividers)
    mask = (valleys >= margin_px) & (valleys <= page_width - margin_px)
    valleys = valleys[mask]
    props = {k: v[mask] for k, v in props.items()}

    if target_valleys is not None and len(valleys) > target_valleys:
        # Keep the `target_valleys` deepest valleys (highest prominence)
        prominences = props["prominences"]
        order = np.argsort(-prominences)  # descending prominence
        keep = sorted(order[:target_valleys].tolist())
        valleys = valleys[keep]

    boundaries = sorted(int(v) for v in valleys)
    logger.info(
        "detect_column_boundaries: %d boundaries detected (n_hint=%s, margin_px=%d)",
        len(boundaries), n_hint, margin_px,
    )
    return boundaries


# ---------------------------------------------------------------------------
# 3. Morphological vertical-rule detection
# ---------------------------------------------------------------------------


def detect_vertical_rules(gray_arr: np.ndarray) -> list[int]:
    """Detect printed vertical separator lines using morphological operations.

    Erodes with a tall, narrow structuring element to isolate continuous
    vertical ink, then finds contour centres.

    Parameters
    ----------
    gray_arr:
        8-bit grayscale numpy array.

    Returns
    -------
    list[int]
        Sorted x-positions of detected rule centre-lines.
    """
    h, w = gray_arr.shape
    _, binary = cv2.threshold(gray_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Structuring element: 1 pixel wide, at least page_height/8 tall
    element_h = max(30, h // 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, element_h))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rule_xs: list[int] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Only keep very thin, tall regions
        if ch >= h // 6 and cw <= 8:
            rule_xs.append(x + cw // 2)

    rule_xs = sorted(set(rule_xs))
    logger.info("detect_vertical_rules: %d rules detected", len(rule_xs))
    return rule_xs


# ---------------------------------------------------------------------------
# 4. Merge projection valleys + printed rules
# ---------------------------------------------------------------------------


def finalise_column_bounds(
    projection_valleys: list[int],
    rule_positions: list[int],
    page_width: int,
    n_hint: int | None = None,
) -> list[int]:
    """Merge valley and rule signals into a final column boundary list.

    Algorithm:
      - For each rule, if there's a valley within RULE_MERGE_TOLERANCE_PX,
        replace the valley with the rule position (rules are more precise).
      - Add any rules that had no matching valley.
      - Re-sort and re-apply minimum-gap filter.
      - Clip to (0, page_width).

    Returns
    -------
    list[int]
        Sorted list of column boundary x-positions (interior only, no edges).
    """
    merged = list(projection_valleys)

    for rx in rule_positions:
        close = [v for v in merged if abs(v - rx) <= RULE_MERGE_TOLERANCE_PX]
        if close:
            # Replace the nearest valley with the rule position
            nearest = min(close, key=lambda v: abs(v - rx))
            merged = [rx if v == nearest else v for v in merged]
        else:
            merged.append(rx)

    merged = sorted(set(merged))

    # Remove boundaries too close together
    min_gap = int(page_width * MIN_COLUMN_GAP_FRAC)
    filtered: list[int] = []
    for x in merged:
        if not filtered or (x - filtered[-1]) >= min_gap:
            filtered.append(x)

    # If we still have too many and n_hint is given, select the
    # most evenly-spaced subset rather than blindly taking the first N.
    target = n_hint - 1 if (n_hint and n_hint > 1) else None
    if target and len(filtered) > target:
        filtered = _select_evenly_spaced(filtered, target, page_width)

    logger.info(
        "finalise_column_bounds: %d boundaries from %d valleys + %d rules",
        len(filtered), len(projection_valleys), len(rule_positions),
    )
    return filtered


def _select_evenly_spaced(candidates: list[int], n: int, page_width: int) -> list[int]:
    """Select *n* boundaries from *candidates* that best approximate  equal spacing.

    Uses a greedy nearest-neighbour matching against ideal evenly-spaced
    positions (``page_width / (n+1) * i`` for i in 1..n).
    """
    ideal_spacing = page_width / (n + 1)
    ideal_positions = [int(ideal_spacing * (i + 1)) for i in range(n)]
    selected: list[int] = []
    used: set[int] = set()
    for ideal in ideal_positions:
        # Nearest unused candidate to the ideal position
        best = min(
            (c for c in candidates if c not in used),
            key=lambda c: abs(c - ideal),
            default=None,
        )
        if best is not None:
            selected.append(best)
            used.add(best)
    return sorted(selected)


# ---------------------------------------------------------------------------
# 5. Strip decomposition
# ---------------------------------------------------------------------------


@dataclass
class PageStrip:
    """A rectangular crop of a newspaper page for targeted LLM annotation."""

    strip_id: str
    """Unique identifier, e.g. ``'col_3'``, ``'masthead'``, ``'full'``."""

    image_path: Path
    """Path to the saved strip PNG."""

    x_offset: int
    """Left edge of this strip in the original page's pixel space."""

    y_offset: int
    """Top edge of this strip in the original page's pixel space."""

    strip_width: int
    """Width of the crop in original page pixels."""

    strip_height: int
    """Height of the crop in original page pixels."""

    page_width: int
    """Full original page width (for coordinate conversion)."""

    page_height: int
    """Full original page height (for coordinate conversion)."""

    column_index: int | None = None
    """1-based column number for column strips; None for others."""

    column_count: int | None = None
    """Total number of columns detected on the page."""

    meta: dict = field(default_factory=dict)
    """Extra metadata (skew angle, boundary positions, etc.)."""


def decompose_into_strips(
    image_path: Path,
    column_bounds: list[int],
    *,
    output_dir: Path | None = None,
    masthead_frac: float = DEFAULT_MASTHEAD_FRAC,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
    full_thumb_max: int = 800,
) -> list[PageStrip]:
    """Slice a full-page image into annotatable strips.

    Produces:
    - 1 masthead strip (full width, top ``masthead_frac`` of page height)
    - N column strips (full height, one per detected column, with overlap)
    - 1 full-page thumbnail (≤ ``full_thumb_max`` px on longest side)

    Parameters
    ----------
    image_path:
        Path to the full-page PNG (should already be skew-corrected).
    column_bounds:
        Sorted interior boundary x-positions from :func:`finalise_column_bounds`.
    output_dir:
        Where to save strip PNGs.  Defaults to
        ``data/interim/strips/{image_path.stem}/``.
    masthead_frac:
        Fraction of page height reserved for the masthead strip.
    overlap_frac:
        Fraction of *column width* to extend each strip on each side.
    full_thumb_max:
        Maximum pixel dimension for the full-page thumbnail.

    Returns
    -------
    list[PageStrip]
        Ordered: [masthead, col_1, col_2, ..., col_N, full]
    """
    img = Image.open(image_path).convert("RGB")
    page_w, page_h = img.size

    if output_dir is None:
        output_dir = (
            Path("data") / "interim" / "strips" / image_path.stem
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    strips: list[PageStrip] = []
    stem = image_path.stem

    # ── Extended column boundary list (including edges) ───────────────────
    bounds = [0] + list(column_bounds) + [page_w]
    n_cols = len(bounds) - 1

    # ── Masthead strip ────────────────────────────────────────────────────
    masthead_h = int(page_h * masthead_frac)
    masthead_crop = img.crop((0, 0, page_w, masthead_h))
    masthead_path = output_dir / f"{stem}_masthead.png"
    masthead_crop.save(masthead_path, format="PNG")
    strips.append(PageStrip(
        strip_id="masthead",
        image_path=masthead_path,
        x_offset=0, y_offset=0,
        strip_width=page_w, strip_height=masthead_h,
        page_width=page_w, page_height=page_h,
        column_index=None, column_count=n_cols,
    ))
    logger.debug("Masthead strip: %s", masthead_path.name)

    # ── Column strips ─────────────────────────────────────────────────────
    for i in range(n_cols):
        col_x0 = bounds[i]
        col_x1 = bounds[i + 1]
        col_w = col_x1 - col_x0
        overlap_px = max(0, int(col_w * overlap_frac))

        x0 = max(0, col_x0 - overlap_px)
        x1 = min(page_w, col_x1 + overlap_px)

        col_crop = img.crop((x0, 0, x1, page_h))
        col_path = output_dir / f"{stem}_col{i + 1:02d}.png"
        col_crop.save(col_path, format="PNG")

        strips.append(PageStrip(
            strip_id=f"col_{i + 1}",
            image_path=col_path,
            x_offset=x0, y_offset=0,
            strip_width=(x1 - x0), strip_height=page_h,
            page_width=page_w, page_height=page_h,
            column_index=i + 1, column_count=n_cols,
        ))
        logger.debug("Column %d strip: x=%d–%d  %s", i + 1, x0, x1, col_path.name)

    # ── Full-page thumbnail ───────────────────────────────────────────────
    thumb = img.copy()
    thumb.thumbnail((full_thumb_max, full_thumb_max), Image.LANCZOS)
    thumb_path = output_dir / f"{stem}_full_thumb.png"
    thumb.save(thumb_path, format="PNG")
    strips.append(PageStrip(
        strip_id="full",
        image_path=thumb_path,
        x_offset=0, y_offset=0,
        strip_width=page_w, strip_height=page_h,
        page_width=page_w, page_height=page_h,
        column_index=None, column_count=n_cols,
    ))
    logger.debug("Full-page thumbnail: %s", thumb_path.name)

    logger.info(
        "decompose_into_strips: %d strips (1 masthead + %d cols + 1 thumb) for %s",
        len(strips), n_cols, image_path.name,
    )
    return strips


# ---------------------------------------------------------------------------
# 6. Annotation coordinate conversion & merging
# ---------------------------------------------------------------------------


def strip_to_page_coords(
    box_0_1000: list[int],
    strip: PageStrip,
) -> list[int]:
    """Convert a box from strip-local 0-1000 space to full-page 0-1000 space.

    The Gemini annotation prompt uses 0–1000 normalised coordinates relative
    to the *strip* image.  This function converts to 0–1000 relative to the
    *full page*.

    Parameters
    ----------
    box_0_1000:
        ``[y_min, x_min, y_max, x_max]`` in strip 0-1000 space.
    strip:
        The :class:`PageStrip` the region came from.

    Returns
    -------
    list[int]
        ``[y_min, x_min, y_max, x_max]`` in full-page 0-1000 space.
    """
    y_min_s, x_min_s, y_max_s, x_max_s = box_0_1000

    # Convert strip 0-1000 → strip absolute pixels
    x_abs_min = x_min_s / 1000.0 * strip.strip_width + strip.x_offset
    x_abs_max = x_max_s / 1000.0 * strip.strip_width + strip.x_offset
    y_abs_min = y_min_s / 1000.0 * strip.strip_height + strip.y_offset
    y_abs_max = y_max_s / 1000.0 * strip.strip_height + strip.y_offset

    # Clamp to page boundaries
    x_abs_min = max(0.0, min(x_abs_min, strip.page_width))
    x_abs_max = max(0.0, min(x_abs_max, strip.page_width))
    y_abs_min = max(0.0, min(y_abs_min, strip.page_height))
    y_abs_max = max(0.0, min(y_abs_max, strip.page_height))

    # Convert absolute pixels → full-page 0-1000 space
    y_min_p = int(y_abs_min / strip.page_height * 1000)
    x_min_p = int(x_abs_min / strip.page_width * 1000)
    y_max_p = int(y_abs_max / strip.page_height * 1000)
    x_max_p = int(x_abs_max / strip.page_width * 1000)

    return [y_min_p, x_min_p, y_max_p, x_max_p]


def _box_iou(a: list[int], b: list[int]) -> float:
    """Compute intersection-over-union for two boxes in [y1, x1, y2, x2] format."""
    iy1 = max(a[0], b[0])
    ix1 = max(a[1], b[1])
    iy2 = min(a[2], b[2])
    ix2 = min(a[3], b[3])
    inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def merge_strip_annotations(
    strip_results: list[tuple["PageStrip", list]],
    page_w: int,
    page_h: int,
    column_bounds: list[int],
    *,
    iou_threshold: float = IOU_DEDUP_THRESHOLD,
    cross_col_min_frac: float = 1.5,
) -> list:
    """Merge per-strip annotations back into a single full-page set.

    Strategy
    --------
    - **Masthead strip** → all detected boxes accepted as-is.
    - **Column strips** → boxes clipped to 130 % of that column's width
      (drop boxes that spill far into the neighbouring column — those are
      artefacts of the overlap region).
    - **Full-page thumbnail** → only keep boxes whose x-span is at least
      ``cross_col_min_frac`` × average column width (i.e. cross-column
      elements like wide ads and banners).
    - All boxes converted to full-page 0-1000 coords.
    - IoU deduplication: if two boxes (same label) overlap > ``iou_threshold``
      keep the larger one.

    Parameters
    ----------
    strip_results:
        List of ``(PageStrip, [BBoxRegion, ...])`` pairs.  The BBoxRegion
        objects must have ``.box`` in strip-local 0-1000 space and a
        ``.label`` attribute.  They are imported lazily to avoid circular
        dep on ``annotate``.
    page_w, page_h:
        Full original page dimensions in pixels.
    column_bounds:
        Interior column boundary x-positions (from
        :func:`finalise_column_bounds`).
    iou_threshold:
        IoU above which two same-label boxes are considered duplicates.
    cross_col_min_frac:
        Minimum x-width (in column-widths) for a full-page-pass box to be
        kept.

    Returns
    -------
    list[BBoxRegion]
        Merged, deduplicated region list in full-page 0-1000 space.
    """
    from newspapers.segmentation.annotate import BBoxRegion  # lazy import

    bounds = [0] + list(column_bounds) + [page_w]
    avg_col_w_norm = (1000 / len(bounds) - 1) if len(bounds) > 1 else 1000

    all_regions: list[BBoxRegion] = []

    for strip, regions in strip_results:
        if not regions:
            continue

        for region in regions:
            page_box = strip_to_page_coords(region.box, strip)

            if strip.strip_id == "masthead":
                all_regions.append(BBoxRegion(label=region.label, box=page_box))
                continue

            if strip.strip_id == "full":
                # Only keep wide cross-column boxes
                box_w_norm = page_box[3] - page_box[1]
                if box_w_norm >= avg_col_w_norm * cross_col_min_frac:
                    all_regions.append(BBoxRegion(label=region.label, box=page_box))
                continue

            # Column strip — check x-extent doesn't stray too far from column
            if strip.column_index is not None:
                col_idx = strip.column_index - 1
                col_x0_norm = int(bounds[col_idx] / page_w * 1000)
                col_x1_norm = int(bounds[col_idx + 1] / page_w * 1000)
                col_w_norm = col_x1_norm - col_x0_norm
                box_w_norm = page_box[3] - page_box[1]
                if box_w_norm <= col_w_norm * 1.3:
                    all_regions.append(BBoxRegion(label=region.label, box=page_box))
            else:
                all_regions.append(BBoxRegion(label=region.label, box=page_box))

    # NMS deduplication via supervision (class-aware, keeps largest box per overlap)
    try:
        import numpy as _np
        import supervision as sv  # noqa: PLC0415

        _CLASS_IDS = {lbl: i for i, lbl in enumerate(
            ["job_advertisement", "article_text", "headline",
             "commercial_advertisement", "financial_table", "masthead"]
        )}
        # Sort by area descending so NMS retains larger boxes (greedy behaviour)
        areas = _np.array(
            [(r.box[2] - r.box[0]) * (r.box[3] - r.box[1]) for r in all_regions],
            dtype=_np.float32,
        )
        sort_idx = _np.argsort(-areas)
        sorted_regions = [all_regions[i] for i in sort_idx]

        area_sorted = areas[sort_idx]
        denom = float(area_sorted.max()) if area_sorted.max() > 0 else 1.0
        conf = (area_sorted / denom).astype(_np.float32)

        # supervision uses [x_min, y_min, x_max, y_max] (xyxy)
        xyxy = _np.array(
            [[r.box[1], r.box[0], r.box[3], r.box[2]] for r in sorted_regions],
            dtype=_np.float32,
        )
        class_ids = _np.array(
            [_CLASS_IDS.get(r.label, 0) for r in sorted_regions], dtype=int
        )
        # Store original positions so we can recover them after NMS filtering
        orig_idx = _np.arange(len(sorted_regions))

        dets = sv.Detections(
            xyxy=xyxy,
            class_id=class_ids,
            confidence=conf,
            data={"orig_idx": orig_idx},
        )
        dets = dets.with_nms(threshold=iou_threshold, class_agnostic=False)
        kept = [sorted_regions[i] for i in dets.data["orig_idx"]]

    except Exception as exc:  # noqa: BLE001  # fallback if supervision unavailable
        logger.debug("supervision NMS unavailable (%s); using manual greedy NMS.", exc)
        all_regions.sort(key=lambda r: -(r.box[2] - r.box[0]) * (r.box[3] - r.box[1]))
        kept = []
        for candidate in all_regions:
            suppress = False
            for accepted in kept:
                if accepted.label == candidate.label:
                    if _box_iou(candidate.box, accepted.box) > iou_threshold:
                        suppress = True
                        break
            if not suppress:
                kept.append(candidate)

    logger.info(
        "merge_strip_annotations: %d candidate regions → %d after deduplication",
        len(all_regions), len(kept),
    )
    return kept


# ---------------------------------------------------------------------------
# 7. Visualisation helper
# ---------------------------------------------------------------------------


def draw_column_bounds(
    pil_image: Image.Image,
    column_bounds: list[int],
    profile: np.ndarray | None = None,
    *,
    line_colour: str = "#E63946",
    line_width: int = 3,
    profile_height_frac: float = 0.08,
) -> Image.Image:
    """Overlay column boundary lines (and optionally a projection profile bar) on *pil_image*.

    Parameters
    ----------
    pil_image:
        The full-page PIL image (will be copied, not modified in place).
    column_bounds:
        Interior x-positions to mark as vertical lines.
    profile:
        If provided, a scaled waveform is drawn at the base of the image.
    line_colour:
        HTML colour for boundary lines.
    line_width:
        Thickness of boundary lines in pixels.
    profile_height_frac:
        Fraction of image height devoted to the profile waveform.

    Returns
    -------
    PIL.Image.Image
        Annotated copy.
    """
    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for x in column_bounds:
        draw.line([(x, 0), (x, h)], fill=line_colour, width=line_width)

    if profile is not None and len(profile) == w:
        prof_h = int(h * profile_height_frac)
        prof_max = profile.max() or 1.0
        for px in range(w):
            bar_h = int(profile[px] / prof_max * prof_h)
            y_top = h - prof_h
            y_bar = h - bar_h
            # Background
            draw.line([(px, y_top), (px, h)], fill="#DDDDDD")
            # Bar
            draw.line([(px, y_bar), (px, h)], fill="#4A90D9")
        # Redraw boundary lines over the profile
        for x in column_bounds:
            draw.line([(x, h - prof_h), (x, h)], fill=line_colour, width=line_width)

    return img


# ---------------------------------------------------------------------------
# 8. Convenience: full structure analysis for one image
# ---------------------------------------------------------------------------


def analyse_page_structure(
    image_path: Path,
    *,
    n_columns_hint: int | None = 8,
    masthead_frac: float = DEFAULT_MASTHEAD_FRAC,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
    correct_skew_flag: bool = True,
    interim_dir: Path | None = None,
) -> tuple[list[int], list["PageStrip"], np.ndarray, float]:
    """Run the full structure-detection pipeline for a single page.

    Parameters
    ----------
    image_path:
        Path to the page PNG (high-res preferred).
    n_columns_hint:
        Expected number of columns (passed to :func:`detect_column_boundaries`).
    masthead_frac:
        Passed to :func:`decompose_into_strips`.
    overlap_frac:
        Passed to :func:`decompose_into_strips`.
    correct_skew_flag:
        Whether to run skew correction before analysis.
    interim_dir:
        Where to store corrected page + strips. Defaults to
        ``data/interim/strips/{stem}/``.

    Returns
    -------
    tuple
        ``(column_bounds, strips, profile, skew_angle)``
        where *column_bounds* is the interior list, *strips* the list of
        :class:`PageStrip` objects, *profile* the smoothed 1-D projection
        array, and *skew_angle* the detected (and corrected) angle.
    """
    img = Image.open(image_path).convert("RGB")
    gray = _to_gray_uint8(img)

    # Skew detection & optional correction
    skew_angle = detect_skew(gray)
    if correct_skew_flag and abs(skew_angle) >= 0.1:
        img = correct_skew(img, skew_angle)
        gray = _to_gray_uint8(img)

        # Save corrected page to interim so strips come from it
        if interim_dir is None:
            interim_dir = Path("data") / "interim" / "strips" / image_path.stem
        interim_dir.mkdir(parents=True, exist_ok=True)
        corrected_path = interim_dir / f"{image_path.stem}_deskewed.png"
        img.save(corrected_path, format="PNG")
        logger.info("Saved deskewed page → %s", corrected_path)
        working_path = corrected_path
    else:
        working_path = image_path

    # Projection profile → column boundaries
    profile = compute_projection_profile(gray)
    valleys = detect_column_boundaries(
        profile, n_hint=n_columns_hint, page_height=img.size[1]
    )
    rules = detect_vertical_rules(gray)
    column_bounds = finalise_column_bounds(
        valleys, rules, page_width=img.size[0], n_hint=n_columns_hint
    )

    # Strip decomposition
    strips = decompose_into_strips(
        working_path,
        column_bounds,
        output_dir=interim_dir,
        masthead_frac=masthead_frac,
        overlap_frac=overlap_frac,
    )

    return column_bounds, strips, profile, skew_angle
