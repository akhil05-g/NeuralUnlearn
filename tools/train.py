"""
tools/train.py - CLI tool for training and unlearning models.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fop.config import (
    DATA_YAML_PATH,
    DATA_RETAIN_YAML_PATH,
    CHECKPOINTS_DIR,
    EPOCHS_ORIGINAL,
    EPOCHS_FINETUNE,
    EPOCHS_RETRAIN,
    EPOCHS_UNLEARN,
    DEFAULT_ERASURE_STRENGTH,
    DEVICE,
)
from fop.models.detector import train_original_detector
from fop.models.baselines import (
    train_finetune_baseline,
    train_retrained_reference,
    run_output_masking_baseline,
)
from fop.unlearning.trainer import run_fce_unlearning, run_erasure_sweep


def main():
    parser = argparse.ArgumentParser(description="FOP Training and Unlearning CLI.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["original", "finetune", "retrain", "output_mask", "unlearn", "sweep"],
        help="Training mode to execute",
    )
    parser.add_argument("--pilot", action="store_true", help="Fast pilot run with few epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--strength", type=float, default=DEFAULT_ERASURE_STRENGTH, help="FCE erasure strength lambda")
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device ('cuda' or 'cpu')")
    args = parser.parse_args()

    orig_ckpt = CHECKPOINTS_DIR / "original_6class" / "best.pt"

    if args.mode == "original":
        epochs = args.epochs or EPOCHS_ORIGINAL
        train_original_detector(epochs=epochs, device=args.device, pilot=args.pilot)

    elif args.mode == "finetune":
        if not orig_ckpt.exists():
            print(f"Error: Original checkpoint {orig_ckpt} not found. Train original model first.")
            sys.exit(1)
        epochs = args.epochs or EPOCHS_FINETUNE
        train_finetune_baseline(orig_ckpt, epochs=epochs, device=args.device, pilot=args.pilot)

    elif args.mode == "retrain":
        epochs = args.epochs or EPOCHS_RETRAIN
        train_retrained_reference(epochs=epochs, device=args.device, pilot=args.pilot)

    elif args.mode == "output_mask":
        if not orig_ckpt.exists():
            print(f"Error: Original checkpoint {orig_ckpt} not found.")
            sys.exit(1)
        run_output_masking_baseline(orig_ckpt, device=args.device)

    elif args.mode == "unlearn":
        if not orig_ckpt.exists():
            print(f"Error: Original checkpoint {orig_ckpt} not found.")
            sys.exit(1)
        epochs = args.epochs or EPOCHS_UNLEARN
        run_fce_unlearning(orig_ckpt, erasure_strength=args.strength, epochs=epochs, device=args.device, pilot=args.pilot)

    elif args.mode == "sweep":
        if not orig_ckpt.exists():
            print(f"Error: Original checkpoint {orig_ckpt} not found.")
            sys.exit(1)
        run_erasure_sweep(orig_ckpt, device=args.device, pilot=args.pilot)


if __name__ == "__main__":
    main()
