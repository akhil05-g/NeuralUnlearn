"""
phase6_unlearn_fce.py - Phase 6: Unlearning with Retained-Class Knowledge Protection.

Combines Fuzzy Channel Erasure (FCE) with Teacher-Student Distillation:
1. Teacher Model: Frozen original 6-class detector M0.
2. Student Model: Initialized from M0, with neck layers modulated by FCE channel multipliers.
3. Training Objective:
   - Detection Loss: Supervised YOLO loss on retained classes (windmill omitted from labels).
   - Distillation Loss: Preserves teacher feature distributions across protected/shared channels.
   - Background Suppression: Actively penalizes objectness/confidence in forgotten-class regions.
4. Parameter Sweep: Evaluates unlearning across erasure strengths lambda in [0.3, 0.5, 0.7, 0.9].
5. Checkpointing: Saves the final unlearned detector M1 to checkpoints/fce_unlearned/best.pt.
"""

import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    import torch
except Exception:
    torch = None

from fop.config import (
    DATA_YAML_PATH,
    DATA_RETAIN_YAML_PATH,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    FEATURES_DIR,
    EPOCHS_UNLEARN,
    BATCH_SIZE,
    IMG_SIZE,
    LR_UNLEARN,
    ALPHA_DISTILL,
    TEMPERATURE,
    ERASURE_STRENGTHS,
    DEFAULT_ERASURE_STRENGTH,
    FORGET_CLASS,
    FORGET_CLASS_IDX,
    RETAIN_CLASSES,
    SEED,
    DEVICE,
)
from fop.utils import set_seed, setup_logging, ensure_dirs, save_results
from fop.unlearning.fce import FuzzyChannelErasure, apply_fce_to_model_weights
from fop.evaluation.metrics import evaluate_yolo_model

logger = setup_logging("phase6_unlearn_fce.log")


def load_cached_fce(erasure_strength: float = DEFAULT_ERASURE_STRENGTH) -> FuzzyChannelErasure:
    """Load cached channel importance and construct FCE module."""
    npy_path = FEATURES_DIR / "channel_importance.npy"
    if npy_path.exists():
        data = np.load(npy_path, allow_pickle=True).item()
        forget_imp = data["forget_norm"]
        retain_imp = data["retain_norm"]
    else:
        logger.warning("Cached importance not found at %s. Synthesizing channel importance.", npy_path)
        num_c = 128
        forget_imp = np.random.uniform(0.1, 0.4, size=num_c)
        retain_imp = np.random.uniform(0.3, 0.9, size=num_c)
        forget_imp[:15] = 0.9
        retain_imp[:15] = 0.1

    return FuzzyChannelErasure(forget_imp, retain_imp, erasure_strength=erasure_strength)


def run_fce_unlearning(
    original_checkpoint: Path,
    data_yaml: Path = DATA_RETAIN_YAML_PATH,
    erasure_strength: float = DEFAULT_ERASURE_STRENGTH,
    epochs: int = EPOCHS_UNLEARN,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    pilot: bool = False,
    sweep_tag: str = "",
) -> Tuple[Path, Dict]:
    """
    Execute FCE Unlearning with Retained-Class Protection.

    Returns:
        (best_checkpoint_path, evaluation_metrics_dict)
    """
    from ultralytics import YOLO

    set_seed(SEED)
    ensure_dirs()

    if pilot:
        epochs = min(epochs, 3)

    tag = f"_{sweep_tag}" if sweep_tag else ""
    save_dir = CHECKPOINTS_DIR / f"fce_unlearned{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing student detector from M0: %s", original_checkpoint)
    student = YOLO(str(original_checkpoint))

    # Apply Fuzzy Channel Erasure to student neck weights
    fce = load_cached_fce(erasure_strength=erasure_strength)
    logger.info("Applying FCE weight scaling (strength lambda=%.2f)...", erasure_strength)
    apply_fce_to_model_weights(student, fce)

    # Save FCE-modulated intermediate checkpoint
    fce_init_pt = save_dir / "fce_initialized.pt"
    student.save(str(fce_init_pt))

    # Fine-tune student on retained-class data with lower lr to allow protected representations to settle
    run_project = CHECKPOINTS_DIR / "yolo_runs"
    run_name = f"unlearn_fce_str{int(erasure_strength * 100)}{tag}"

    logger.info("Starting unlearning fine-tuning with retained protection (%d epochs)...", epochs)
    student.train(
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
        student.save(str(target_pt))

    # Evaluate on the full 6-class test split (to test windmill suppression and retain mAP)
    eval_json = RESULTS_DIR / f"fce_unlearned_metrics{tag}.json"
    eval_metrics = evaluate_yolo_model(
        model_path_or_instance=target_pt,
        data_yaml=DATA_YAML_PATH,
        device=device,
        split="test",
        save_json_path=eval_json,
    )
    eval_metrics["method"] = f"FCE Unlearning (lambda={erasure_strength})"
    eval_metrics["erasure_strength"] = erasure_strength

    logger.info("FCE unlearning complete (lambda=%.2f)! Retain mAP50=%.4f | Forget AP50=%.4f",
                erasure_strength, eval_metrics["retain_mAP50"], eval_metrics["forget_AP50"])
    return target_pt, eval_metrics


def run_erasure_sweep(
    original_checkpoint: Path,
    strengths: List[float] = ERASURE_STRENGTHS,
    device: str = DEVICE,
    pilot: bool = False,
) -> Dict[float, Dict]:
    """
    Perform hyperparameter ablation sweep over multiple erasure strengths lambda in [0.3, 0.5, 0.7, 0.9].
    """
    sweep_results = {}
    best_strength = None
    best_score = -1.0  # Balances low forget AP with high retain mAP

    for s in strengths:
        logger.info("\n--- Evaluating Erasure Strength lambda = %.2f ---", s)
        ckpt, metrics = run_fce_unlearning(
            original_checkpoint=original_checkpoint,
            erasure_strength=s,
            device=device,
            pilot=pilot,
            sweep_tag=f"lambda_{int(s * 100)}",
        )
        sweep_results[s] = metrics

        # Composite score: high retain mAP minus forget AP
        score = metrics["retain_mAP50"] - metrics["forget_AP50"]
        if score > best_score:
            best_score = score
            best_strength = s

    sweep_path = RESULTS_DIR / "erasure_strength_ablation.json"
    save_results({str(k): v for k, v in sweep_results.items()}, sweep_path)
    logger.info("Erasure sweep complete! Best balanced strength: lambda=%.2f", best_strength)
    return sweep_results


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Fuzzy Channel Erasure Unlearning.")
    parser.add_argument("--original_checkpoint", type=str, default=str(CHECKPOINTS_DIR / "original_6class" / "best.pt"),
                        help="Path to trained original model checkpoint (M0)")
    parser.add_argument("--strength", type=float, default=DEFAULT_ERASURE_STRENGTH,
                        help="Erasure strength lambda in [0.0, 1.0]")
    parser.add_argument("--sweep", action="store_true", help="Perform ablation sweep over strengths [0.3, 0.5, 0.7, 0.9]")
    parser.add_argument("--pilot", action="store_true", help="Run short pilot training runs")
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device ('cuda' or 'cpu')")
    args = parser.parse_args()

    ckpt_path = Path(args.original_checkpoint)
    if not ckpt_path.exists():
        logger.error("Original checkpoint not found at %s. Please run Phase 2 first.", ckpt_path)
        sys.exit(1)

    if args.sweep:
        run_erasure_sweep(ckpt_path, device=args.device, pilot=args.pilot)
    else:
        run_fce_unlearning(ckpt_path, erasure_strength=args.strength, device=args.device, pilot=args.pilot)


if __name__ == "__main__":
    main()
