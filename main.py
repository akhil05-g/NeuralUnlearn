"""
main.py - Master Pipeline Orchestrator for 'Forgetting on Purpose' Project.

Executes the complete research pipeline through the modular `fop` package:
  Stage 1: Dataset preparation & annotation conversion (DIOR VOC -> YOLO).
  Stage 2: Dataset quality verification & bounding box audit.
  Stage 3: Train original 6-class YOLOv8n detector M0.
  Stage 4: Train comparative baselines (Masking, Fine-Tuning, Retrained Reference).
  Stage 5: Neck feature analysis & channel importance profiling.
  Stage 6: Fuzzy Channel Erasure (FCE) weight calculation.
  Stage 7: FCE Unlearning with distillation & retained protection.
  Stage 8: Comprehensive evaluation benchmark & diagnostics.
  Stage 9: Publication visualizations & LaTeX table generation.
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fop.config import (
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    print_config,
)
from fop.utils import setup_logging, ensure_dirs

logger = setup_logging("main_pipeline.log")

PIPELINE_STAGES = [
    (1, "Dataset Preparation", "fop.data.preparation"),
    (2, "Dataset Quality Audit", "fop.data.verification"),
    (3, "Train Original Detector (M0)", "fop.models.detector"),
    (4, "Train Comparative Baselines", "fop.models.baselines"),
    (5, "Neck Feature Analysis", "fop.unlearning.feature_analysis"),
    (6, "Fuzzy Channel Erasure Engine", "fop.unlearning.fce"),
    (7, "FCE Unlearning Training", "fop.unlearning.trainer"),
    (8, "Comprehensive Evaluation", "fop.evaluation.benchmark"),
    (9, "Publication Visualizations", "fop.visualization.plots"),
]


def run_stage(module_name: str, extra_args: list = None) -> bool:
    """Run a Python module via -m flag."""
    cmd = [sys.executable, "-m", module_name]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Executing: %s", " ".join(cmd))
    start_t = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        elapsed = time.time() - start_t
        logger.info("Completed %s successfully in %.1fs", module_name, elapsed)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Stage failed in %s (exit code %d)", module_name, e.returncode)
        return False


def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Runner for 'Forgetting on Purpose' Project.")
    parser.add_argument("--start_stage", type=int, default=1, help="Start stage number (1 to 9)")
    parser.add_argument("--end_stage", type=int, default=9, help="End stage number (1 to 9)")
    parser.add_argument("--pilot", action="store_true", help="Run short pilot training runs for testing")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic DIOR dataset if raw data unavailable")
    parser.add_argument("--dry_run", action="store_true", help="Print execution plan without running")
    args = parser.parse_args()

    ensure_dirs()
    print_config()

    print("\n" + "=" * 70)
    print(" RESEARCH PIPELINE EXECUTION PLAN:")
    for step_num, stage_title, module in PIPELINE_STAGES:
        status = "SCHEDULED" if args.start_stage <= step_num <= args.end_stage else "SKIPPED"
        print(f"  Stage {step_num}: {stage_title:<32} [python -m {module}] -> {status}")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("Dry run requested. Exiting without execution.")
        return

    for step_num, stage_title, module in PIPELINE_STAGES:
        if step_num < args.start_stage or step_num > args.end_stage:
            continue

        print(f"\n>>> Running Stage {step_num}: {stage_title} ({module})...")
        extra_args = []
        if args.pilot and module in ["fop.models.detector", "fop.models.baselines", "fop.unlearning.trainer"]:
            extra_args.append("--pilot")
        if args.synthetic and module == "fop.data.preparation":
            extra_args.append("--synthetic")

        success = run_stage(module, extra_args)
        if not success:
            print(f"\n[ERROR] Pipeline stopped due to failure at Stage {step_num} ({module}).")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(" ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"Results directory   : {RESULTS_DIR}")
    print(f"Plots directory     : {PLOTS_DIR}")
    print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
