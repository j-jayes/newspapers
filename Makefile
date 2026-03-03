.PHONY: install lint test clean data segment extract pipeline

## Install the package in editable mode with all extras
install:
	pip install -e ".[all]"

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

## Convert raw JP2 images to JPG/PNG (placeholder)
data:
	@echo "TODO: run data ingestion pipeline"

## Run YOLOv11 segmentation on preprocessed pages (placeholder)
segment:
	@echo "TODO: run segmentation pipeline"

## Run Vision LLM extraction on cropped ads (placeholder)
extract:
	@echo "TODO: run extraction pipeline"

## Run full pipeline end-to-end (placeholder)
pipeline: data segment extract
	@echo "Pipeline complete."

## Show available commands
help:
	@echo "Available targets:"
	@echo "  install   – Install the package in editable mode"
	@echo "  lint      – Run ruff linter and formatter checks"
	@echo "  test      – Run pytest test suite"
	@echo "  clean     – Remove caches and build artifacts"
	@echo "  data      – Run data ingestion (placeholder)"
	@echo "  segment   – Run page segmentation (placeholder)"
	@echo "  extract   – Run structured extraction (placeholder)"
	@echo "  pipeline  – Run full end-to-end pipeline (placeholder)"
