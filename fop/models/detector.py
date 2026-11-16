"""
phase2_train_original.py - Phase 2: Train the Original 6-Class YOLOv8n Detector (M0).

This script establishes the unlearning baseline model M0:
1. Loads pretrained YOLOv8n backbone (yolov8n.pt).
2. Configures and trains on the 6-class DIOR dataset (airplane, airport, vehicle, ship, harbor, windmill).
3. Saves the best model checkpoint to checkpoints/original_6class/best.pt.
4. Evaluates performance on the held-out test split and writes baseline metrics to results/original_metrics.json.
5. Supports a fast --pilot mode (e.g., 5 epochs) for pipeline validation.
"""

import sys
import argparse
import shutil
from pathlib import Path

# Add project root to path
from fop.config import (
    DATA_YAML_PATH,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    BASE_MODEL,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS_ORIGINAL,
    SEED,
    DEVICE,
    LR_ORIGINAL,
)
from fop.utils import set_seed, setup_logging, ensure_dirs, save_results
from fop.evaluation.metrics import evaluate_yolo_model

logger = setup_logging("phase2_train_original.log")


def train_original_detector(
    data_yaml: Path = DATA_YAML_PATH,
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS_ORIGINAL,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    seed: int = SEED,
    pilot: bool = False,
) -> Path:
    """
    Train the original 6-class detector M0.

    Returns:
        Path: Path to saved best model checkpoint.
    """
    from ultralytics import YOLO

    set_seed(seed)
    ensure_dirs()

    if pilot:
        epochs = min(epochs, 5)
        logger.info("Running in PILOT mode: training for %d epochs.", epochs)

    save_dir = CHECKPOINTS_DIR / "original_6class"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing YOLOv8n from pretrained weights: %s", base_model)
    model = YOLO(base_model)

    # Project run directory for Ultralytics
    run_project = CHECKPOINTS_DIR / "yolo_runs"
    run_name = "train_original_6class"

    logger.info("Starting training: data=%s, epochs=%d, batch=%d, img_size=%d, device=%s",
                data_yaml, epochs, batch_size, img_size, device)

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        seed=seed,
        project=str(run_project),
        name=run_name,
        exist_ok=True,
        lr0=LR_ORIGINAL,
        verbose=True,
        val=True,
    )

    # Locate and copy best checkpoint
    trained_best_pt = run_project / run_name / "weights" / "best.pt"
    if not trained_best_pt.exists():
        # Fallback to last.pt if best.pt is not generated (e.g., in short pilot runs)
        trained_best_pt = run_project / run_name / "weights" / "last.pt"

    target_best_pt = save_dir / "best.pt"
    if trained_best_pt.exists():
        shutil.copy2(trained_best_pt, target_best_pt)
        logger.info("Saved original 6-class model checkpoint to %s", target_best_pt)
    else:
        logger.warning("No checkpoint found at %s. Saving current model instance...", trained_best_pt)
        model.save(str(target_best_pt))

    # Evaluate on held-out test split
    logger.info("Evaluating original detector M0 on test split...")
    metrics_path = RESULTS_DIR / "original_metrics.json"
    eval_metrics = evaluate_yolo_model(
        model_path_or_instance=target_best_pt,
        data_yaml=data_yaml,
        img_size=img_size,
        batch_size=batch_size,
        device=device,
        split="test",
        save_json_path=metrics_path,
    )

    logger.info("Phase 2 complete! Baseline test mAP50: %.4f | Retain mAP50: %.4f | Forget AP50: %.4f",
                eval_metrics["mAP50"], eval_metrics["retain_mAP50"], eval_metrics["forget_AP50"])
    return target_best_pt


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train Original 6-Class YOLOv8n Detector (M0).")
    parser.add_argument("--data", type=str, default=str(DATA_YAML_PATH), help="Path to 6-class dataset YAML")
    parser.add_argument("--epochs", type=int, default=EPOCHS_ORIGINAL, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Image resolution")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--pilot", action="store_true", help="Run quick 5-epoch pilot validation")
    args = parser.parse_args()

    train_original_detector(
        data_yaml=Path(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
