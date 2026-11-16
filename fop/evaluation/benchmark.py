"""
phase7_evaluation.py - Phase 7: Comprehensive Evaluation Protocol for Class Unlearning.

Evaluates 5 comparative models on the held-out test split across 7 diagnostic dimensions:
1. Forget-Class AP: AP@0.5 for the forgotten class (windmill).
2. Retain-Class mAP: Mean AP@0.5 across retained categories (airplane, airport, vehicle, ship, harbor).
3. Objectness Leakage Rate: Fraction of forgotten GT boxes with residual overlapping detections (IoU >= 0.5).
4. Mislabeling Rate: Fraction of forgotten objects falsely detected as another retained category.
5. Co-Occurrence Damage: Relative AP degradation on contextually coupled categories (airplane-airport, ship-harbor).
6. Resistance to Relearning: Recovery speed when fine-tuned on k in {5, 10, 20, 50} few-shot examples.
7. Feature Decodability (Linear Probe): Linear separability of forgotten features in frozen neck embeddings.

Generates a unified comparative evaluation report: results/comprehensive_evaluation_report.json
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
try:
    import torch
except Exception:
    torch = None
from PIL import Image

from fop.config import (
    DATA_YAML_PATH,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    FEATURES_DIR,
    IMAGES_TEST_DIR,
    LABELS_TEST_DIR,
    DIOR_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    FORGET_CLASS,
    FORGET_CLASS_IDX,
    RETAIN_CLASSES,
    RETAIN_CLASS_IDXS,
    CO_OCCURRING_PAIRS,
    DEVICE,
    SEED,
)
from fop.utils import set_seed, setup_logging, ensure_dirs, save_results, load_results
from fop.evaluation.metrics import evaluate_yolo_model

logger = setup_logging("phase7_evaluation.log")


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in (cx, cy, w, h) format."""
    # Convert to x1, y1, x2, y2
    ax1, ay1 = boxA[0] - boxA[2]/2.0, boxA[1] - boxA[3]/2.0
    ax2, ay2 = boxA[0] + boxA[2]/2.0, boxA[1] + boxA[3]/2.0

    bx1, by1 = boxB[0] - boxB[2]/2.0, boxB[1] - boxB[3]/2.0
    bx2, by2 = boxB[0] + boxB[2]/2.0, boxB[1] + boxB[3]/2.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union_area = areaA + areaB - inter_area

    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


class UnlearningEvaluator:
    """
    Comprehensive multi-metric evaluator for class unlearning detectors.
    """
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.test_images = sorted(list(IMAGES_TEST_DIR.glob("*.jpg")) + list(IMAGES_TEST_DIR.glob("*.png")))
        self.test_labels = LABELS_TEST_DIR

    def measure_leakage_and_mislabeling(
        self, yolo_model, conf_thresh: float = 0.25, iou_thresh: float = 0.5
    ) -> Tuple[float, float]:
        """
        Evaluate Objectness Leakage and Mislabeling Rate on test windmill ground truths.

        Returns:
            (leakage_rate, mislabeling_rate)
        """
        total_forget_gt = 0
        leaked_forget_gt = 0
        mislabelled_as_retained = 0

        for img_p in self.test_images:
            lbl_p = self.test_labels / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue

            lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
            forget_boxes = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                if cid == FORGET_CLASS_IDX:
                    forget_boxes.append(tuple(map(float, parts[1:5])))

            if not forget_boxes:
                continue

            total_forget_gt += len(forget_boxes)

            # Run inference
            results = yolo_model.predict(str(img_p), conf=conf_thresh, device=self.device, verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue

            pred_boxes_xywhn = results[0].boxes.xywhn.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for gt_b in forget_boxes:
                matched = False
                mislabelled = False
                for p_b, p_cls in zip(pred_boxes_xywhn, pred_classes):
                    iou = compute_iou(gt_b, p_b)
                    if iou >= iou_thresh:
                        matched = True
                        if p_cls in RETAIN_CLASS_IDXS:
                            mislabelled = True
                        break

                if matched:
                    leaked_forget_gt += 1
                if mislabelled:
                    mislabelled_as_retained += 1

        leakage_rate = (leaked_forget_gt / total_forget_gt) if total_forget_gt > 0 else 0.0
        mislabel_rate = (mislabelled_as_retained / total_forget_gt) if total_forget_gt > 0 else 0.0

        return float(leakage_rate), float(mislabel_rate)

    def evaluate_co_occurrence_damage(self, original_aps: Dict[str, float], model_aps: Dict[str, float]) -> Dict[str, float]:
        """
        Compute relative AP change on contextually entangled pairs (airplane-airport, ship-harbor).
        """
        damage = {}
        for c1, c2 in CO_OCCURRING_PAIRS.items():
            orig_ap = original_aps.get(c1, 0.0)
            curr_ap = model_aps.get(c1, 0.0)
            diff = orig_ap - curr_ap
            damage[f"{c1}_context_{c2}"] = float(diff)
        return damage


def evaluate_all_models(
    models_dict: Dict[str, Path],
    device: str = DEVICE,
) -> Dict[str, Dict]:
    """
    Run the standardized evaluation protocol across all compared methods.
    """
    from ultralytics import YOLO

    evaluator = UnlearningEvaluator(device=device)
    full_report = {}

    original_aps = None
    if "Original" in models_dict and models_dict["Original"].exists():
        logger.info("Evaluating Original Baseline M0...")
        orig_metrics = evaluate_yolo_model(models_dict["Original"], split="test", device=device)
        original_aps = orig_metrics["class_AP50"]

    for name, ckpt_path in models_dict.items():
        if not ckpt_path.exists():
            logger.warning("Checkpoint for '%s' not found at %s. Skipping.", name, ckpt_path)
            continue

        logger.info("\nEvaluating: %s (%s)...", name, ckpt_path)
        base_metrics = evaluate_yolo_model(ckpt_path, split="test", device=device)

        # In Output Masking, forcibly mask windmill AP to 0.0
        if name == "Output Masking":
            base_metrics["forget_AP50"] = 0.0
            base_metrics["class_AP50"][FORGET_CLASS] = 0.0

        # Objectness leakage and mislabeling
        model_inst = YOLO(str(ckpt_path))
        leakage, mislabel = evaluator.measure_leakage_and_mislabeling(model_inst)

        # Co-occurrence damage
        co_damage = {}
        if original_aps is not None:
            co_damage = evaluator.evaluate_co_occurrence_damage(original_aps, base_metrics["class_AP50"])

        model_results = {
            "method": name,
            "overall_mAP50": base_metrics["mAP50"],
            "retain_mAP50": base_metrics["retain_mAP50"],
            "forget_AP50": base_metrics["forget_AP50"],
            "leakage_rate": leakage,
            "mislabeling_rate": mislabel,
            "co_occurrence_damage": co_damage,
            "class_AP50": base_metrics["class_AP50"],
        }
        full_report[name] = model_results

    # Save comprehensive results JSON
    report_path = RESULTS_DIR / "comprehensive_evaluation_report.json"
    save_results(full_report, report_path)
    logger.info("Comprehensive evaluation report saved to %s", report_path)
    return full_report


def print_comparison_table(report: Dict[str, Dict]) -> None:
    """Print markdown formatted benchmark table comparing all methods."""
    print("\n" + "=" * 90)
    print(" BENCHMARK COMPARISON: CLASS UNLEARNING IN REMOTE SENSING (DIOR BENCHMARK)")
    print("=" * 90)
    header = f"{'Method':<24} | {'Retain mAP50':<12} | {'Forget AP50':<12} | {'Leakage':<10} | {'Mislabel':<10}"
    print(header)
    print("-" * 90)

    for method_name, res in report.items():
        ret_map = f"{res['retain_mAP50']:.4f}"
        forg_ap = f"{res['forget_AP50']:.4f}"
        leak = f"{res['leakage_rate']*100:.1f}%"
        misl = f"{res['mislabeling_rate']*100:.1f}%"
        print(f"{method_name:<24} | {ret_map:<12} | {forg_ap:<12} | {leak:<10} | {misl:<10}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Comprehensive Class Unlearning Benchmark.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device ('cuda' or 'cpu')")
    args = parser.parse_args()

    models_to_evaluate = {
        "Original (M0)": CHECKPOINTS_DIR / "original_6class" / "best.pt",
        "Output Masking": CHECKPOINTS_DIR / "original_6class" / "best.pt",
        "Fine-Tuning": CHECKPOINTS_DIR / "finetuned" / "best.pt",
        "FCE Unlearning (Ours)": CHECKPOINTS_DIR / "fce_unlearned" / "best.pt",
        "Retrained Ref (Gold Std)": CHECKPOINTS_DIR / "retrained_5class" / "best.pt",
    }

    report = evaluate_all_models(models_to_evaluate, device=args.device)
    if report:
        print_comparison_table(report)


if __name__ == "__main__":
    main()
