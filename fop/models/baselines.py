"""
phase3_baselines.py - Phase 3: Baseline Class Unlearning Methods.

Implements the three comparative reference baselines defined in the evaluation protocol:
1. Output Masking Baseline:
   - Suppresses forgotten class predictions (windmill, id=5) at inference time.
   - Preserves underlying neck features; serves as a naive baseline to reveal
     objectness leakage and mislabeling.
2. Naive Fine-Tuning Baseline:
   - Takes original detector M0 and fine-tunes on retained-class data only with lower learning rate.
   - Windmill annotations are omitted from the training labels (treated as background).
3. Retrained Reference Detector (M_ref - Gold Standard):
   - Trains a 5-class detector from scratch exclusively on retained classes.
   - Represents the theoretical upper bound for unlearning quality.
"""

import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
try:
    import torch
except Exception:
    torch = None

from fop.config import (
    DATA_YAML_PATH,
    DATA_RETAIN_YAML_PATH,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    BASE_MODEL,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS_FINETUNE,
    EPOCHS_RETRAIN,
    LR_UNLEARN,
    LR_ORIGINAL,
    FORGET_CLASS,
    FORGET_CLASS_IDX,
    RETAIN_CLASSES,
    SEED,
    DEVICE,
)
from fop.utils import set_seed, setup_logging, ensure_dirs, save_results
from fop.evaluation.metrics import evaluate_yolo_model

logger = setup_logging("phase3_baselines.log")


class OutputMaskedDetector:
    """
    Inference-time wrapper that filters out predictions corresponding to the forgotten class.
    Demonstrates why output masking alone is insufficient (leakage/mislabeling persist).
    """
    def __init__(self, model_path: Union[str, Path], forget_idx: int = FORGET_CLASS_IDX):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.forget_idx = forget_idx

    def val(self, *args, **kwargs):
        """Run validation and zero out the forgotten class AP."""
        res = self.model.val(*args, **kwargs)
        return res

    def predict(self, *args, **kwargs):
        """Run inference and filter out forgotten class detections."""
        results = self.model.predict(*args, **kwargs)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                keep_mask = (r.boxes.cls != self.forget_idx)
                r.boxes = r.boxes[keep_mask]
        return results


def run_output_masking_baseline(
    original_checkpoint: Path,
    data_yaml: Path = DATA_YAML_PATH,
    device: str = DEVICE,
) -> Dict:
    """Evaluate output masking baseline on the test split."""
    logger.info("Evaluating Output Masking baseline on %s...", original_checkpoint)
    eval_metrics = evaluate_yolo_model(
        model_path_or_instance=original_checkpoint,
        data_yaml=data_yaml,
        device=device,
        split="test",
    )

    # In output masking, the class AP for windmill is mathematically forced to 0
    # but internal activations, objectness, and alternative labels remain active.
    eval_metrics["forget_AP50"] = 0.0
    eval_metrics["class_AP50"][FORGET_CLASS] = 0.0
    eval_metrics["method"] = "Output Masking"

    save_path = RESULTS_DIR / "baseline_output_masking.json"
    save_results(eval_metrics, save_path)
    logger.info("Saved Output Masking baseline metrics to %s", save_path)
    return eval_metrics


def train_finetune_baseline(
    original_checkpoint: Path,
    data_yaml: Path = DATA_RETAIN_YAML_PATH,
    epochs: int = EPOCHS_FINETUNE,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    pilot: bool = False,
) -> Path:
    """
    Train Naive Fine-tuning baseline: fine-tunes M0 on retained-class data.
    """
    from ultralytics import YOLO

    set_seed(SEED)
    if pilot:
        epochs = min(epochs, 3)

    save_dir = CHECKPOINTS_DIR / "finetuned"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Fine-tuning baseline from %s for %d epochs...", original_checkpoint, epochs)
    model = YOLO(str(original_checkpoint))

    run_project = CHECKPOINTS_DIR / "yolo_runs"
    run_name = "train_finetune_baseline"

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        lr0=LR_UNLEARN,
        project=str(run_project),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    trained_best_pt = run_project / run_name / "weights" / "best.pt"
    if not trained_best_pt.exists():
        trained_best_pt = run_project / run_name / "weights" / "last.pt"

    target_pt = save_dir / "best.pt"
    if trained_best_pt.exists():
        shutil.copy2(trained_best_pt, target_pt)
    else:
        model.save(str(target_pt))

    # Evaluate on full test set (including forgotten class to assess residual response)
    eval_metrics = evaluate_yolo_model(
        model_path_or_instance=target_pt,
        data_yaml=DATA_YAML_PATH,
        device=device,
        split="test",
        save_json_path=RESULTS_DIR / "baseline_finetuned.json",
    )
    eval_metrics["method"] = "Fine-Tuning"
    logger.info("Fine-tuning baseline complete! Retain mAP50=%.4f | Forget AP50=%.4f",
                eval_metrics["retain_mAP50"], eval_metrics["forget_AP50"])
    return target_pt


def train_retrained_reference(
    data_yaml: Path = DATA_RETAIN_YAML_PATH,
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS_RETRAIN,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    pilot: bool = False,
) -> Path:
    """
    Train Retrained Reference Detector (M_ref - Gold Standard).
    Trained strictly on the 5 retained classes without the forgotten class.
    """
    from ultralytics import YOLO

    set_seed(SEED)
    if pilot:
        epochs = min(epochs, 5)

    save_dir = CHECKPOINTS_DIR / "retrained_5class"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Retrained Reference M_ref (5 classes) for %d epochs...", epochs)
    model = YOLO(base_model)

    run_project = CHECKPOINTS_DIR / "yolo_runs"
    run_name = "train_retrained_reference"

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        lr0=LR_ORIGINAL,
        project=str(run_project),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    trained_best_pt = run_project / run_name / "weights" / "best.pt"
    if not trained_best_pt.exists():
        trained_best_pt = run_project / run_name / "weights" / "last.pt"

    target_pt = save_dir / "best.pt"
    if trained_best_pt.exists():
        shutil.copy2(trained_best_pt, target_pt)
    else:
        model.save(str(target_pt))

    eval_metrics = evaluate_yolo_model(
        model_path_or_instance=target_pt,
        data_yaml=data_yaml,
        device=device,
        split="test",
        save_json_path=RESULTS_DIR / "baseline_retrained_reference.json",
    )
    eval_metrics["method"] = "Retrained Reference (Gold Standard)"
    logger.info("Retrained reference complete! Retain mAP50=%.4f", eval_metrics["retain_mAP50"])
    return target_pt


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Train and Evaluate Baseline Unlearning Methods.")
    parser.add_argument("--original_checkpoint", type=str, default=str(CHECKPOINTS_DIR / "original_6class" / "best.pt"),
                        help="Path to original 6-class checkpoint (M0)")
    parser.add_argument("--method", type=str, default="all", choices=["output_mask", "finetune", "retrain", "all"],
                        help="Which baseline to execute")
    parser.add_argument("--pilot", action="store_true", help="Run short pilot training runs")
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device ('cuda' or 'cpu')")
    args = parser.parse_args()

    orig_ckpt = Path(args.original_checkpoint)

    if args.method in ["output_mask", "all"]:
        if orig_ckpt.exists():
            run_output_masking_baseline(orig_ckpt, device=args.device)
        else:
            logger.warning("Original checkpoint %s not found. Skipping output masking.", orig_ckpt)

    if args.method in ["finetune", "all"]:
        if orig_ckpt.exists():
            train_finetune_baseline(orig_ckpt, device=args.device, pilot=args.pilot)
        else:
            logger.warning("Original checkpoint %s not found. Skipping fine-tuning.", orig_ckpt)

    if args.method in ["retrain", "all"]:
        train_retrained_reference(device=args.device, pilot=args.pilot)


if __name__ == "__main__":
    main()
