"""
phase1_verify_dataset.py - DIOR Dataset Quality and Statistics Verifier.

This script audits the generated YOLO-format dataset:
1. Validates coordinate normalization (all bounding box parameters strictly in [0, 1]).
2. Analyzes class distribution across Train, Val, and Test splits.
3. Computes class imbalance ratios and highlights forgotten class (windmill) counts.
4. Renders sample images with color-coded bounding boxes to plots/dataset_samples/ for manual visual verification.
"""

import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
from fop.config import (
    DIOR_YOLO_DIR,
    IMAGES_TRAIN_DIR,
    IMAGES_VAL_DIR,
    IMAGES_TEST_DIR,
    LABELS_TRAIN_DIR,
    LABELS_VAL_DIR,
    LABELS_TEST_DIR,
    PLOTS_DIR,
    DIOR_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    FORGET_CLASS,
    FORGET_CLASS_IDX,
    SEED,
)
from fop.utils import setup_logging, save_results

logger = setup_logging("phase1_verify.log")

CLASS_COLORS = {
    "airplane": "#00FFCC",
    "airport": "#FF9900",
    "vehicle": "#FF3366",
    "ship": "#3399FF",
    "harbor": "#9933FF",
    "windmill": "#FFFF00",  # Yellow for target unlearning class
}


def audit_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read and validate a single YOLO label file."""
    boxes = []
    if not label_path.exists():
        return boxes

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line_idx, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 5:
            logger.warning("Invalid token count (%d) in %s at line %d", len(parts), label_path.name, line_idx)
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            logger.warning("Parse error in %s line %d: '%s'", label_path.name, line_idx, line)
            continue

        # Check bounds
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            logger.warning("Out-of-bounds coords in %s line %d: (%f, %f, %f, %f)", label_path.name, line_idx, cx, cy, w, h)
            continue
        if cid not in IDX_TO_CLASS:
            logger.warning("Unrecognized class ID %d in %s line %d", cid, label_path.name, line_idx)
            continue

        boxes.append((cid, cx, cy, w, h))
    return boxes


def audit_dataset_split(img_dir: Path, lbl_dir: Path, split_name: str) -> Dict:
    """Audit an entire split (train/val/test) and return summary metrics."""
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total_images = len(img_files)
    total_annotations = 0
    class_counts = Counter()
    images_with_forget_class = 0
    empty_label_files = 0

    for img_path in img_files:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes = audit_label_file(lbl_path)
        if not boxes:
            empty_label_files += 1
            continue

        total_annotations += len(boxes)
        has_forget = False
        for cid, _, _, _, _ in boxes:
            cname = IDX_TO_CLASS[cid]
            class_counts[cname] += 1
            if cid == FORGET_CLASS_IDX:
                has_forget = True

        if has_forget:
            images_with_forget_class += 1

    return {
        "split": split_name,
        "total_images": total_images,
        "total_annotations": total_annotations,
        "empty_label_files": empty_label_files,
        "images_with_forget_class": images_with_forget_class,
        "class_counts": dict(class_counts),
    }


def visualize_annotated_samples(
    img_dir: Path,
    lbl_dir: Path,
    output_dir: Path,
    num_samples: int = 5,
) -> None:
    """Render sample images with color-coded bounding boxes and labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_files:
        return

    random.seed(SEED)
    sample_imgs = random.sample(img_files, min(num_samples, len(img_files)))

    for img_path in sample_imgs:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes = audit_label_file(lbl_path)
        if not boxes:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            w_img, h_img = img.size

            for cid, cx, cy, bw, bh in boxes:
                cname = IDX_TO_CLASS[cid]
                color = CLASS_COLORS.get(cname, "#FFFFFF")

                # Denormalize box
                x1 = (cx - bw / 2.0) * w_img
                y1 = (cy - bh / 2.0) * h_img
                x2 = (cx + bw / 2.0) * w_img
                y2 = (cy + bh / 2.0) * h_img

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label_txt = f"{cname}"
                draw.text((x1 + 4, max(0, y1 - 12)), label_txt, fill=color)

            save_path = output_dir / f"sample_{img_path.stem}.png"
            img.save(save_path)
            logger.info("Saved visual sample: %s", save_path.name)
        except Exception as e:
            logger.warning("Could not render sample for %s: %s", img_path.name, str(e))


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Dataset Quality & Verification Audit.")
    parser.add_argument("--samples", type=int, default=6, help="Number of sample visualizations to save.")
    args = parser.parse_args()

    splits = [
        ("train", IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR),
        ("val", IMAGES_VAL_DIR, LABELS_VAL_DIR),
        ("test", IMAGES_TEST_DIR, LABELS_TEST_DIR),
    ]

    all_audits = {}
    print("\n" + "=" * 70)
    print(" DIOR DATASET QUALITY AUDIT REPORT")
    print("=" * 70)

    for sname, idir, ldir in splits:
        if not idir.exists():
            logger.error("Split directory missing: %s", idir)
            continue
        audit = audit_dataset_split(idir, ldir, sname)
        all_audits[sname] = audit

        print(f"\nSplit: {sname.upper()}")
        print(f"  - Total Images      : {audit['total_images']}")
        print(f"  - Total Annotations : {audit['total_annotations']}")
        print(f"  - Empty Label Files : {audit['empty_label_files']}")
        print(f"  - Images with '{FORGET_CLASS}' (forget target): {audit['images_with_forget_class']}")
        print("  - Per-class Object Counts:")
        for cname in DIOR_CLASSES:
            cnt = audit["class_counts"].get(cname, 0)
            flag = " [FORGET TARGET]" if cname == FORGET_CLASS else ""
            print(f"      {cname:<12}: {cnt:4d}{flag}")

    # Generate visual verification plots
    vis_dir = PLOTS_DIR / "dataset_samples"
    visualize_annotated_samples(IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR, vis_dir, num_samples=args.samples)
    print(f"\nVisual inspection samples saved to: {vis_dir}")

    # Save summary report to results/
    audit_report_path = PROJECT_ROOT / "results" / "dataset_audit.json"
    save_results(all_audits, audit_report_path)
    print(f"Full audit report saved to: {audit_report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
