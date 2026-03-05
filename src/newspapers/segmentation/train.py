"""Fine-tune YOLOv11 on annotated historical newspaper pages.

Trains a custom object-detection model that recognises six content regions:
  0  job_advertisement
  1  article_text
  2  headline
  3  commercial_advertisement
  4  financial_table
  5  masthead

The best checkpoint is automatically copied to ``models/newspapers_detector.pt``
so the rest of the pipeline can reference a stable path.

Usage
-----
Quick fine-tune with defaults::

    uv run python -m newspapers.segmentation.train

Custom run::

    uv run python -m newspapers.segmentation.train \\
        --data     data/annotations/dataset.yaml \\
        --weights  yolo11n.pt \\
        --epochs   100 \\
        --imgsz    1280 \\
        --batch    4 \\
        --project  models/runs \\
        --name     newspapers_v1

The final model is always saved to ``models/newspapers_detector.pt``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Stable output path consumed by the rest of the pipeline
DEFAULT_OUTPUT_PATH = Path("models/newspapers_detector.pt")


def train_model(
    data_yaml: Path,
    base_weights: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    project_dir: Path,
    run_name: str,
    *,
    device: str = "",
) -> Path:
    """Fine-tune YOLOv11 on the annotated newspaper dataset.

    Parameters
    ----------
    data_yaml:
        Path to the YOLO dataset YAML (``data/annotations/dataset.yaml``).
    base_weights:
        Path to the starting checkpoint (``.pt`` file).  Typically
        ``yolo11n.pt`` (base nano weights) or a previous custom checkpoint.
    epochs:
        Number of training epochs.
    imgsz:
        Input image size for training (pixels, square).  Should match the
        resolution used during annotation (1280 recommended for newspaper pages).
    batch:
        Batch size.  Reduce if you hit GPU OOM; 4 works on a 16 GB GPU.
    project_dir:
        Parent directory for Ultralytics experiment outputs.
    run_name:
        Sub-directory name under *project_dir* for this training run.
    device:
        PyTorch device string (e.g. ``"0"`` for first GPU, ``"cpu"`` for CPU,
        ``""`` to let Ultralytics auto-detect).

    Returns
    -------
    Path
        The path to the best model weights produced by this run *and* copied
        to ``models/newspapers_detector.pt``.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required. Install with: uv pip install ultralytics"
        ) from exc

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Run annotation first: uv run python -m newspapers.segmentation.annotate"
        )
    if not base_weights.exists():
        raise FileNotFoundError(
            f"Base weights not found: {base_weights}\n"
            "Download the YOLOv11 nano weights: https://docs.ultralytics.com/models/yolo11/"
        )

    project_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting training: weights=%s  data=%s  epochs=%d  imgsz=%d",
        base_weights,
        data_yaml,
        epochs,
        imgsz,
    )

    model = YOLO(str(base_weights))

    train_kwargs: dict = dict(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
    if device:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)  # noqa: F841

    # Locate the best checkpoint Ultralytics saved
    best_pt = project_dir / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback: last checkpoint
        best_pt = project_dir / run_name / "weights" / "last.pt"

    if not best_pt.exists():
        raise FileNotFoundError(
            f"Training finished but no checkpoint found under {project_dir / run_name}."
        )

    # Copy to the stable pipeline path
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, DEFAULT_OUTPUT_PATH)
    logger.info("Best model saved → %s", DEFAULT_OUTPUT_PATH)

    return DEFAULT_OUTPUT_PATH


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv11 on annotated historical newspaper pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/annotations/dataset.yaml"),
        help="Path to the YOLO dataset YAML.",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("yolo11n.pt"),
        help="Base YOLOv11 weights to fine-tune from.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size (pixels, square).",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training batch size.",
    )
    p.add_argument(
        "--project",
        type=Path,
        default=Path("models/runs"),
        help="Parent directory for Ultralytics experiment outputs.",
    )
    p.add_argument(
        "--name",
        default="newspapers_detector",
        help="Run name (sub-directory under --project).",
    )
    p.add_argument(
        "--device",
        default="",
        help="PyTorch device (e.g. '0', 'cpu'). Auto-detected if empty.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )

    best_model = train_model(
        data_yaml=args.data,
        base_weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project_dir=args.project,
        run_name=args.name,
        device=args.device,
    )
    print(f"\nTraining complete. Best model: {best_model}")
