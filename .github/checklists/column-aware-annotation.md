# Column-Aware Annotation Pipeline — Implementation Checklist

## Overview
Adds classical-CV page structure detection (skew correction, column boundary detection via
projection profiles + vertical rule detection) to pre-process pages before sending narrow
per-column strips to Gemini, improving bounding-box precision substantially.

---

## Phase 1 — Dependencies
- [x] Add `analysis` optional-dep group to `pyproject.toml` (opencv-python-headless, scipy, numpy)
- [x] Add `analysis` to `[all]` extra
- [x] `uv sync --extra all` — verify opencv importable

## Phase 2 — `src/newspapers/segmentation/structure.py`
- [x] `detect_skew(gray_arr)` — HoughLines on Canny edges, return median angle
- [x] `correct_skew(pil_image, angle)` — warpAffine rotation
- [x] `compute_projection_profile(gray_arr)` — vertical ink-density projection
- [x] `detect_column_boundaries(profile, n_hint)` — SciPy valley-finding with margin exclusion
- [x] `detect_vertical_rules(binary_inv_arr)` — morphological tall-line detection
- [x] `finalise_column_bounds(valleys, rules, page_width, n_hint)` — merge + evenly-spaced selection
- [x] `PageStrip` dataclass
- [x] `decompose_into_strips(image_path, column_bounds, ...)` — slice + save to `data/interim/`
- [x] `strip_to_page_coords(region, strip, page_w, page_h)`
- [x] `merge_strip_annotations(strip_results, page_w, page_h, column_bounds)` — IoU dedup
- [x] `draw_column_bounds(pil_image, column_bounds, profile)` — visualisation helper

## Phase 3 — Tests for structure.py
- [x] Skew detection returns plausible angle (0.0° on clean scan ✓)
- [x] Column detection returns 7 boundaries (8 columns) on sample ✓
- [x] Strip decomposition writes exactly 10 files (1 masthead + 8 col + 1 thumb) ✓
- [ ] Unit tests for coordinate round-trip (strip→page→strip within tolerance)

## Phase 4 — `src/newspapers/segmentation/annotate.py` updates
- [ ] `_STRIP_ANNOTATION_PROMPT_TEMPLATE` constant
- [ ] `annotate_page_structured()` — full gen/critic loop on per-strip basis
- [ ] `_write_visualisation()` — new `column_bounds` kwarg, draws boundary lines
- [ ] `--structured` CLI flag in `_build_parser()` and `__main__`

## Phase 5 — `pyproject.toml` / dependency cleanup
- [ ] Verify `opencv-python-headless` in `analysis` group resolves without conflicts

## Phase 6 — Notebook updates
- [ ] Cell: column detection visualisation (projection profile + boundary overlay)
- [ ] Cell: strip browser (N+2 strip images in a grid)
- [ ] Cell: `annotate_page_structured()` runner
- [ ] Cell: merge diagnostic (region provenance scatter)

## Phase 7 — Tunable parameters to revisit
- [ ] Strip overlap fraction (default 5%) — tune based on end-to-end results
- [ ] Projection profile Gaussian σ (currently `page_width // 200`)
- [ ] Valley `prominence` threshold for column boundary detection
- [ ] Masthead fraction (default 12% of page height)
- [ ] IoU deduplication threshold (default 0.5)

## End-to-End
- [ ] Run structured pipeline on `bib13991099_19000124_0_10721a_0002.png`
- [ ] Assert column bounds detected at correct x positions
- [ ] Assert final annotation has tighter column-aligned boxes than legacy full-page pass
- [ ] Visually inspect `_vis.png` with column boundary lines
