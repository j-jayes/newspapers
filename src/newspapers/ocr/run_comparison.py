"""Run OCR comparison across all pages and cache results.

Usage::

    uv run python -m newspapers.ocr.run_comparison
    uv run python -m newspapers.ocr.run_comparison --strips masthead,col_1
    uv run python -m newspapers.ocr.run_comparison --pages "*0002*,*0003*"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from newspapers.ocr.backends import EndpointManager
from newspapers.segmentation.structure import analyse_page_structure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
PROCESSED = REPO / "data" / "processed"
INTERIM = REPO / "data" / "interim"
STRIPS_DIR = INTERIM / "strips"

DEFAULT_STRIPS = ["masthead", "col_1", "col_2", "col_3"]


def _find_pages(patterns: list[str] | None) -> list[Path]:
    """Find page images in data/processed/, optionally filtered by glob patterns."""
    all_pages = sorted(PROCESSED.glob("*.jpg"))
    if not patterns:
        return all_pages
    matched = []
    for p in all_pages:
        for pat in patterns:
            if p.match(pat) or pat in p.stem:
                matched.append(p)
                break
    return matched


def run(
    pages: list[Path],
    strip_ids: list[str],
    backends,
) -> None:
    """Run OCR on all pages × strips × backends, caching results."""
    total_new = 0

    for page_path in pages:
        stem = page_path.stem
        png_path = page_path.with_suffix(".png")
        img_path = png_path if png_path.exists() else page_path

        # Segment page into strips
        _bounds, strips, _profile, skew = analyse_page_structure(
            img_path, n_columns_hint=8
        )
        logger.info("Page %s: %d strips, skew=%.2f°", stem, len(strips), skew)

        # Load cache
        cache_path = INTERIM / f"ocr_comparison_{stem}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            cached = {}

        results: dict[str, dict[str, str]] = defaultdict(dict)
        results.update({k: dict(v) for k, v in cached.items()})

        # Filter strips
        run_strips = [s for s in strips if s.strip_id in strip_ids]

        page_new = 0
        for strip in run_strips:
            img = Image.open(strip.image_path).convert("RGB")
            for backend in backends:
                key = strip.strip_id
                if key in results and backend.name in results[key]:
                    continue

                print(
                    f"  {stem} / {strip.strip_id} / {backend.name}...",
                    end=" ",
                    flush=True,
                )
                t0 = time.time()
                try:
                    text = backend.transcribe(img)
                    results[key][backend.name] = text
                    elapsed = time.time() - t0
                    print(f"{len(text)} chars in {elapsed:.1f}s")
                except Exception as exc:
                    results[key][backend.name] = f"[ERROR] {exc}"
                    print(f"FAILED: {exc}")
                page_new += 1

        # Save cache
        if page_new:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(dict(results), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        total_new += page_new
        cached_count = sum(len(v) for v in results.values()) - page_new
        logger.info(
            "  %s: %d new, %d cached", stem, page_new, cached_count
        )

    logger.info("Done! %d new transcriptions.", total_new)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR comparison across newspaper pages."
    )
    parser.add_argument(
        "--pages",
        help="Comma-separated page name patterns (default: all)",
    )
    parser.add_argument(
        "--strips",
        default=",".join(DEFAULT_STRIPS),
        help=f"Comma-separated strip IDs (default: {','.join(DEFAULT_STRIPS)})",
    )
    args = parser.parse_args(argv)

    page_patterns = args.pages.split(",") if args.pages else None
    strip_ids = [s.strip() for s in args.strips.split(",")]

    pages = _find_pages(page_patterns)
    if not pages:
        logger.error("No pages found in %s", PROCESSED)
        sys.exit(1)

    logger.info("%d page(s), strips: %s", len(pages), strip_ids)

    with EndpointManager() as mgr:
        backends = mgr.get_backends()
        logger.info(
            "%d backends: %s", len(backends), [b.name for b in backends]
        )
        run(pages, strip_ids, backends)


if __name__ == "__main__":
    main()
