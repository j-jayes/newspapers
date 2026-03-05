# Development Plan: Architecting the Historical Newspapers AI Pipeline
*Date: March 3, 2026* (Updated: March 4, 2026)

This plan outlines the end-to-end development of a multimodal pipeline for extracting structured economic data from historical Swedish newspapers (1880-1926).

**Crucial Directive:** Always use `uv` for package management throughout this project.

## Phase 1: Data Ingestion and Preprocessing (Completed)
- [x] Set up package management with `uv` and install dependencies. 
- [x] Set up Google Drive API integration to map the dataset and download `.jp2` samples using Application Default Credentials (quota project `newspapers-454313`).
- [x] Create `ingest.py` script to fetch files programmatically using gcloud credentials.
- [x] Develop preprocessing infrastructure to convert `.jp2` to `.jpg` (for YOLO layout analysis) and `.png` (for high-fidelity text extraction).

## Phase 2: Text Transcription & Structured Extraction (Completed)
*Note: This phase was pulled forward to validate feasibility before investing in model training.*
- [x] Finalize Architectural Decision: Rejected Hybrid OCR (Tesseract/ByT5) due to "cascading errors"; adopted Vision LLM zero-shot transcription.
- [x] Migrate dependency stack to `google-genai` and `langextract` (removed `instructor` and legacy SDKs).
- [x] Resolved API access issues by migrating to standard Google AI Studio Key via `.env` file instead of GCP Vertex AI.
- [x] Engineer anti-hallucination VLM prompts to strictly enforce literal diplomatic transcription without "over-historicization".
- [x] Establish and successfully test the `extract_job_ad_with_grounding` pipeline using Gemini 2.5 Flash and Pydantic schema mapping via few-shot `langextract`.
- [x] Validate end-to-end extraction (Image -> Raw Strict Text -> Parsed Grounded JSON) on a sample image.

## Phase 3: Layout Analysis & Semantic Segmentation (Next Steps)
*Current focus: The base YOLOv11 nano model (`yolo11n.pt`) doesn't recognize 19th-century Swedish job layout bounding boxes. We need to train it.*
- [ ] Set up an annotation pipeline/script to create ground-truth bounding box data for isolating classified advertisements in multi-column layouts.
- [ ] Gather and annotate a micro-dataset of historical newspaper crops for fine-tuning.
- [ ] Write the training pipeline to fine-tune a custom `yolo11n` model into a newspaper-layout detection model.
- [ ] Implement OpenCV/Pillow cropping module to slice full-page `.jpg` files using our custom YOLO coordinates and pass them forward as individual ad images.

## Phase 4: Integration, Validation and Storage (To Do)
- [ ] Stitch the master pipeline script: `ingest.py` -> `detect.py` (Fine-Tuned YOLO) -> `extract.py` -> `JSON`.
- [ ] Run the complete automated extraction successfully on the local processed sample dataset.
- [ ] Audit output against the `langextract` source grounding HTML visualizations.
- [ ] Build mapping tools for linking raw extraction strings to unified historical taxonomy architectures.