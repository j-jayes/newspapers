.PHONY: install lint test clean data annotate train segment extract pipeline help

# Configurable paths (override on the command line if needed)
PROCESSED_DIR   ?= data/processed
ANNO_LABELS_DIR ?= data/annotations/labels/train
ANNO_IMAGES_DIR ?= data/annotations/images/train
VIS_DIR         ?= data/annotations/visualizations
VAL_LABELS_DIR  ?= data/annotations/labels/val
VAL_IMAGES_DIR  ?= data/annotations/images/val
CROPS_DIR       ?= data/interim/crops
MODEL_PATH      ?= models/newspapers_detector.pt
BASE_WEIGHTS    ?= yolo11n.pt
GEMINI_MODEL    ?= gemini-2.5-flash

## Install the package in editable mode with all extras
install:
	uv pip install -e ".[all]"

## Run linters
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

## Run tests
test:
	pytest tests/ -v

## Remove compiled Python files and caches
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/

## Download .jp2 samples from Google Drive and convert to JPG + PNG
data:
	uv run python -m newspapers.data.ingest

## Auto-annotate preprocessed JPGs with Gemini → YOLO .txt labels + review PNGs
## Review overlays in data/annotations/visualizations/ before running 'train'.
## Add --overwrite to re-annotate already-labelled images.
annotate:
	uv run python -m newspapers.segmentation.annotate \
		--input   $(PROCESSED_DIR) \
		--labels  $(ANNO_LABELS_DIR) \
		--images  $(ANNO_IMAGES_DIR) \
		--vis     $(VIS_DIR) \
		--model   $(GEMINI_MODEL)

## Fine-tune YOLOv11 on the annotated dataset.
## Requires at least a few labelled images in data/annotations/images/{train,val}.
## Best checkpoint is saved to models/newspapers_detector.pt.
train:
	uv run python -m newspapers.segmentation.train \
		--data    data/annotations/dataset.yaml \
		--weights $(BASE_WEIGHTS) \
		--epochs  50 \
		--imgsz   1280 \
		--batch   4

## Run the trained detector on preprocessed pages and save cropped segment PNGs.
## Crops are taken from the full-res PNG (if present) for Vision LLM quality.
segment:
	uv run python -m newspapers.segmentation.detect \
		--input  $(PROCESSED_DIR) \
		--model  $(MODEL_PATH) \
		--output $(CROPS_DIR)

## Run Vision LLM structured extraction on all cropped advertisement PNGs
extract:
	uv run python -c "\
import logging, sys; logging.basicConfig(level=logging.INFO); \
from pathlib import Path; \
from newspapers.extraction.extract import process_advertisement; \
crops = sorted(Path('$(CROPS_DIR)').glob('*job_advertisement*.png')); \
print(f'Extracting from {len(crops)} job advertisement crops...'); \
[print(process_advertisement(c)[0].model_dump_json(indent=2)) for c in crops]"

## Run full pipeline end-to-end: ingest → annotate → train → detect → extract
pipeline: data annotate train segment extract
	@echo "Pipeline complete."

## Show available commands
help:
	@echo "Available targets:"
	@echo "  install   – Install the package (uses uv)"
	@echo "  lint      – Run ruff linter and formatter checks"
	@echo "  test      – Run pytest test suite"
	@echo "  clean     – Remove caches and build artifacts"
	@echo "  data      – Download .jp2 samples and convert to JPG/PNG"
	@echo "  annotate  – Auto-annotate pages with Gemini 2.5 Flash"
	@echo "  train     – Fine-tune YOLOv11 on annotated dataset"
	@echo "  segment   – Detect regions and crop segments from pages"
	@echo "  extract   – Run Vision LLM extraction on cropped ads"
	@echo "  pipeline  – Run full end-to-end pipeline"
