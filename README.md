# newspapers

Multimodal AI pipeline for structured extraction of historical Swedish labor market data (1880–1926) from digitized newspaper archives held by the National Library of Sweden (Kungliga biblioteket).

## Project Organisation

```
├── Makefile           <- Makefile with commands: `make install`, `make test`, `make pipeline`
├── README.md
├── data
│   ├── external       <- Data from third-party sources.
│   ├── interim        <- Intermediate transformed data.
│   ├── processed      <- Final canonical datasets.
│   └── raw            <- Original immutable data (KB archive scans).
│
├── models             <- Trained YOLOv11 weights and model artefacts.
│
├── notebooks          <- Jupyter notebooks for exploration and prototyping.
│
├── references         <- Data dictionaries, manuals, and explanatory materials.
│
├── reports
│   └── figures        <- Generated graphics and figures.
│
├── requirements.txt   <- pip requirements for reproducing the environment.
├── pyproject.toml     <- Python package configuration.
│
├── src/newspapers     <- Source code for the pipeline.
│   ├── data           <- Data ingestion and JP2 conversion.
│   ├── segmentation   <- YOLOv11 page segmentation.
│   ├── extraction     <- Vision LLM structured extraction.
│   ├── visualization  <- Exploratory and results visualizations.
│   └── models.py      <- Pydantic schemas (JobAdvertisement, PageSegment, …).
│
└── tests              <- Unit and integration tests.
```

## Pipeline Overview

1. **Ingest** – Download JPEG2000 scans from the KB archive (via Google Drive scraper) and convert to JPG/PNG.
2. **Segment** – Run a fine-tuned YOLOv11 model to detect and crop individual job advertisements on each page.
3. **Transcribe & Extract (Hybrid)** – Process the cropped ad images via high-fidelity transcription (Hybrid OCR / Vision LLM) to text. Pass the text to Google's `langextract` library to produce typed `JobAdvertisement` records with precise source grounding and HTML visualizations.
4. **Analyse** – Aggregate and visualise the structured dataset for quantitative economic history research.

## Quickstart

```bash
# Create a virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with all dependencies using uv
uv pip install -e ".[all]"

# Run the test suite
make test
```