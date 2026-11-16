"""
tools/evaluate.py - CLI tool for model evaluation and benchmarking.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fop.config import CHECKPOINTS_DIR, DATA_YAML_PATH, DEVICE
from fop.evaluation.metrics import evaluate_yolo_model
from fop.evaluation.benchmark import evaluate_all_models, print_comparison_table
from fop.evaluation.relearning import run_relearning_experiment
from fop.evaluation.linear_probe import evaluate_linear_probe


def main():
    parser = argparse.ArgumentParser(description="FOP Evaluation CLI.")
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=["eval_single", "benchmark", "relearning", "linear_probe"],
        help="Evaluation mode",
    )
    parser.add_argument("--weights", type=str, default=None, help="Model weights path (for eval_single/relearning/linear_probe)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device ('cuda' or 'cpu')")
    args = parser.parse_args()

    if args.mode == "eval_single":
        if not args.weights:
            print("Error: --weights required for eval_single mode.")
            sys.exit(1)
        metrics = evaluate_yolo_model(args.weights, device=args.device)
        print(f"Retain mAP50: {metrics['retain_mAP50']:.4f} | Forget AP50: {metrics['forget_AP50']:.4f}")

    elif args.mode == "benchmark":
        models = {
            "Original (M0)": CHECKPOINTS_DIR / "original_6class" / "best.pt",
            "Output Masking": CHECKPOINTS_DIR / "original_6class" / "best.pt",
            "Fine-Tuning": CHECKPOINTS_DIR / "finetuned" / "best.pt",
            "FCE Unlearning (Ours)": CHECKPOINTS_DIR / "fce_unlearned" / "best.pt",
            "Retrained Ref (Gold Std)": CHECKPOINTS_DIR / "retrained_5class" / "best.pt",
        }
        report = evaluate_all_models(models, device=args.device)
        if report:
            print_comparison_table(report)

    elif args.mode == "relearning":
        ckpt = Path(args.weights) if args.weights else (CHECKPOINTS_DIR / "fce_unlearned" / "best.pt")
        run_relearning_experiment(ckpt, device=args.device)

    elif args.mode == "linear_probe":
        ckpt = Path(args.weights) if args.weights else (CHECKPOINTS_DIR / "fce_unlearned" / "best.pt")
        evaluate_linear_probe(ckpt, device=args.device)


if __name__ == "__main__":
    main()
