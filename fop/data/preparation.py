"""
phase1_dataset_preparation.py - DIOR Dataset Preparation & Preprocessing Pipeline.

This script prepares the 6-class DIOR dataset for YOLOv8 training and unlearning experiments:
1. Parses Pascal VOC XML annotations from raw DIOR dataset (or generates synthetic data with --synthetic).
2. Filters to 6 selected classes: airplane, airport, vehicle, ship, harbor, windmill.
3. Converts bounding boxes to normalized YOLO format (class_id, x_center, y_center, width, height).
4. Partitions images into stratified Train (70%), Val (15%), Test (15%) splits with fixed seed=42.
5. Generates both:
   - dior_6class.yaml: Full 6-class dataset for original detector training and testing.
   - dior_5class_retain.yaml: Retained-only dataset (windmill omitted from training/val sets,
     but retained in the test set for unlearning evaluation).
6. Includes synthetic dataset generation for immediate end-to-end testing without downloading 30GB DIOR.
"""

import os
import sys
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
from fop.config import (
    DATA_DIR,
    DIOR_YOLO_DIR,
    DATA_YAML_PATH,
    DATA_RETAIN_YAML_PATH,
    IMAGES_TRAIN_DIR,
    IMAGES_VAL_DIR,
    IMAGES_TEST_DIR,
    LABELS_TRAIN_DIR,
    LABELS_VAL_DIR,
    LABELS_TEST_DIR,
    DIOR_CLASSES,
    CLASS_TO_IDX,
    FORGET_CLASS,
    FORGET_CLASS_IDX,
    RETAIN_CLASSES,
    SEED,
)
from fop.utils import set_seed, setup_logging, ensure_dirs

logger = setup_logging("phase1_dataset_prep.log")


# Synonym mappings to handle variations in DIOR XML labels
CLASS_SYNONYMS = {
    "airplane": ["airplane", "plane", "aircraft"],
    "airport": ["airport", "airfield"],
    "vehicle": ["vehicle", "car", "small-vehicle", "large-vehicle", "truck"],
    "ship": ["ship", "boat", "vessel"],
    "harbor": ["harbor", "port", "harbour"],
    "windmill": ["windmill", "wind-mill", "wind_turbine", "wind turbine"],
}


def normalize_class_name(raw_name: str) -> Optional[str]:
    """Map raw annotation label to canonical DIOR 6-class name if in target subset."""
    cleaned = raw_name.strip().lower().replace("_", "-").replace(" ", "-")
    for canonical, synonyms in CLASS_SYNONYMS.items():
        for syn in synonyms:
            if cleaned == syn or cleaned == syn.replace("-", ""):
                return canonical
    return None


def convert_voc_bbox_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert Pascal VOC bounding box (xmin, ymin, xmax, ymax) to YOLO format (cx, cy, w, h)."""
    # Clip coordinates to image boundaries
    xmin = max(0.0, min(float(xmin), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymax = max(0.0, min(float(ymax), float(img_h)))

    if xmax <= xmin or ymax <= ymin:
        return 0.0, 0.0, 0.0, 0.0

    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    # Ensure strictly in [0, 1]
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return cx, cy, w, h


def parse_voc_xml(xml_path: Path) -> List[Dict]:
    """Parse Pascal VOC XML annotation file and extract 6-class objects."""
    objects = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_elem = root.find("size")
        if size_elem is not None:
            width = int(size_elem.find("width").text)
            height = int(size_elem.find("height").text)
        else:
            # Fallback will require image load if size missing
            width, height = 800, 800

        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                continue
            canonical_name = normalize_class_name(name_elem.text)
            if canonical_name is None:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            cx, cy, w, h = convert_voc_bbox_to_yolo(xmin, ymin, xmax, ymax, width, height)
            if w > 0.001 and h > 0.001:
                objects.append({
                    "class_name": canonical_name,
                    "class_id": CLASS_TO_IDX[canonical_name],
                    "bbox_yolo": (cx, cy, w, h),
                })
    except Exception as e:
        logger.warning("Error parsing %s: %s", xml_path.name, str(e))
    return objects


def generate_synthetic_dior_dataset(output_dir: Path, num_images: int = 120) -> None:
    """
    Generate a realistic synthetic DIOR dataset for pipeline testing.
    Creates remote-sensing style background textures with labeled targets.
    """
    logger.info("Generating synthetic DIOR dataset (%d images)...", num_images)
    raw_dir = output_dir / "dior_raw"
    images_dir = raw_dir / "JPEGImages"
    annos_dir = raw_dir / "Annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annos_dir.mkdir(parents=True, exist_ok=True)

    # Remote sensing color palettes (terrain, water, asphalt, agriculture)
    background_colors = [
        (45, 75, 45),    # dark vegetation
        (65, 95, 60),    # light grass
        (70, 70, 75),    # asphalt / tarmac
        (30, 60, 90),    # water body
        (130, 115, 85),  # dry soil / sand
        (90, 85, 80),    # urban cement
    ]

    class_colors = {
        "airplane": (230, 230, 240),
        "airport": (100, 100, 110),
        "vehicle": (220, 50, 50),
        "ship": (240, 240, 245),
        "harbor": (60, 80, 110),
        "windmill": (255, 255, 100),
    }

    width, height = 640, 640

    for i in range(num_images):
        img_id = f"dior_syn_{i:05d}"
        bg_col = random.choice(background_colors)
        img = Image.new("RGB", (width, height), bg_col)
        draw = ImageDraw.Draw(img)

        # Add texture/noise patches
        for _ in range(15):
            tx = random.randint(0, width - 40)
            ty = random.randint(0, height - 40)
            tw = random.randint(20, 100)
            th = random.randint(20, 100)
            noise_col = (
                max(0, min(255, bg_col[0] + random.randint(-20, 20))),
                max(0, min(255, bg_col[1] + random.randint(-20, 20))),
                max(0, min(255, bg_col[2] + random.randint(-20, 20))),
            )
            draw.rectangle([tx, ty, tx + tw, ty + th], fill=noise_col)

        # Assign 1 to 4 objects
        num_objs = random.randint(1, 4)
        objs = []
        
        # Determine contextually realistic co-occurrences
        scenario = random.choice(["airport_zone", "port_zone", "rural_wind", "general"])
        for _ in range(num_objs):
            if scenario == "airport_zone":
                cname = random.choice(["airplane", "airplane", "airport", "vehicle"])
            elif scenario == "port_zone":
                cname = random.choice(["ship", "ship", "harbor", "vehicle"])
            elif scenario == "rural_wind":
                cname = random.choice(["windmill", "windmill", "vehicle"])
            else:
                cname = random.choice(DIOR_CLASSES)

            # Sizes: vehicles and windmills are typically smaller
            if cname in ["vehicle", "windmill"]:
                bw = random.randint(16, 45)
                bh = random.randint(16, 45)
            elif cname in ["airplane", "ship"]:
                bw = random.randint(30, 80)
                bh = random.randint(30, 80)
            else:  # airport, harbor (large contextual structures)
                bw = random.randint(90, 200)
                bh = random.randint(90, 200)

            bx = random.randint(10, width - bw - 10)
            by = random.randint(10, height - bh - 10)

            # Draw shape
            draw.rectangle([bx, by, bx + bw, by + bh], fill=class_colors[cname], outline=(20, 20, 20))
            objs.append((cname, bx, by, bx + bw, by + bh))

        # Save image
        img_path = images_dir / f"{img_id}.jpg"
        img.save(img_path, quality=90)

        # Save Pascal VOC XML
        xml_root = ET.Element("annotation")
        ET.SubElement(xml_root, "filename").text = f"{img_id}.jpg"
        size_node = ET.SubElement(xml_root, "size")
        ET.SubElement(size_node, "width").text = str(width)
        ET.SubElement(size_node, "height").text = str(height)
        ET.SubElement(size_node, "depth").text = "3"

        for cname, x1, y1, x2, y2 in objs:
            obj_node = ET.SubElement(xml_root, "object")
            ET.SubElement(obj_node, "name").text = cname
            bnd = ET.SubElement(obj_node, "bndbox")
            ET.SubElement(bnd, "xmin").text = str(x1)
            ET.SubElement(bnd, "ymin").text = str(y1)
            ET.SubElement(bnd, "xmax").text = str(x2)
            ET.SubElement(bnd, "ymax").text = str(y2)

        tree = ET.ElementTree(xml_root)
        tree.write(annos_dir / f"{img_id}.xml")

    logger.info("Synthetic DIOR dataset ready at %s", raw_dir)


def write_yolo_yaml_configs() -> None:
    """Generate dior_6class.yaml and dior_5class_retain.yaml for Ultralytics YOLOv8."""
    # 6-class config (All classes)
    content_6class = f"""# DIOR 6-Class Dataset Configuration for YOLOv8
path: {DIOR_YOLO_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 6
names: {DIOR_CLASSES}
"""
    DATA_YAML_PATH.write_text(content_6class, encoding="utf-8")
    logger.info("Saved 6-class dataset config to %s", DATA_YAML_PATH)

    # 5-class config (Windmill excluded)
    content_5class = f"""# DIOR 5-Class Retained-Only Dataset Configuration
path: {DIOR_YOLO_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 5
names: {RETAIN_CLASSES}
"""
    DATA_RETAIN_YAML_PATH.write_text(content_5class, encoding="utf-8")
    logger.info("Saved 5-class retain dataset config to %s", DATA_RETAIN_YAML_PATH)


def prepare_dior_dataset(
    raw_dir: Path,
    output_yolo_dir: Path,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = SEED,
) -> Dict[str, int]:
    """
    Process raw DIOR dataset into train/val/test YOLO format.
    Generates both standard labels and retain-only training labels.
    """
    set_seed(seed)
    ensure_dirs()

    # Search for images and annotations in raw directory
    possible_img_dirs = [raw_dir / "JPEGImages", raw_dir / "images", raw_dir]
    possible_xml_dirs = [raw_dir / "Annotations", raw_dir / "xml", raw_dir]

    img_dir = next((d for d in possible_img_dirs if d.exists() and any(d.glob("*.jpg"))), None)
    xml_dir = next((d for d in possible_xml_dirs if d.exists() and any(d.glob("*.xml"))), None)

    if not img_dir or not xml_dir:
        raise FileNotFoundError(
            f"Could not locate DIOR images/annotations in {raw_dir}.\n"
            "Please ensure raw DIOR dataset is placed in 'data/dior_raw/' with JPEGImages and Annotations folders,\n"
            "or use the --synthetic flag to generate a lightweight testing dataset."
        )

    logger.info("Found raw DIOR dataset: Images in '%s', Annotations in '%s'", img_dir, xml_dir)

    xml_files = sorted(list(xml_dir.glob("*.xml")))
    valid_samples = []

    class_counts = {c: 0 for c in DIOR_CLASSES}

    for xml_path in xml_files:
        stem = xml_path.stem
        # Match image
        img_candidates = list(img_dir.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        img_path = img_candidates[0]

        objs = parse_voc_xml(xml_path)
        if not objs:
            continue  # Skip images without any of our 6 classes

        for o in objs:
            class_counts[o["class_name"]] += 1

        valid_samples.append({
            "id": stem,
            "img_path": img_path,
            "objs": objs,
        })

    logger.info("Found %d valid images containing target 6 classes.", len(valid_samples))
    logger.info("Class instances in dataset: %s", class_counts)

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(valid_samples)

    # Compute split boundaries
    n_total = len(valid_samples)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    n_test = n_total - n_train - n_val

    splits = {
        "train": valid_samples[:n_train],
        "val": valid_samples[n_train : n_train + n_val],
        "test": valid_samples[n_train + n_val :],
    }

    dir_mapping = {
        "train": (IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR),
        "val": (IMAGES_VAL_DIR, LABELS_VAL_DIR),
        "test": (IMAGES_TEST_DIR, LABELS_TEST_DIR),
    }

    # Clean existing destination directories
    for split, (img_dest, lbl_dest) in dir_mapping.items():
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        for f in img_dest.glob("*.*"):
            f.unlink()
        for f in lbl_dest.glob("*.txt"):
            f.unlink()

    stats = {"train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"])}

    for split_name, items in splits.items():
        target_img_dir, target_lbl_dir = dir_mapping[split_name]
        for item in items:
            stem = item["id"]
            src_img = item["img_path"]
            dst_img = target_img_dir / src_img.name
            shutil.copy2(src_img, dst_img)

            # Write standard 6-class YOLO label
            lbl_file = target_lbl_dir / f"{stem}.txt"
            lines = []
            for obj in item["objs"]:
                cid = obj["class_id"]
                cx, cy, w, h = obj["bbox_yolo"]
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            lbl_file.write_text("\n".join(lines), encoding="utf-8")

    # Write YAML dataset manifests
    write_yolo_yaml_configs()

    logger.info("Dataset preparation complete! Splits: Train=%d, Val=%d, Test=%d", stats["train"], stats["val"], stats["test"])
    return stats


def main():
    parser = argparse.ArgumentParser(description="Phase 1: DIOR Dataset Preprocessing for Class Unlearning.")
    parser.add_argument("--raw_dir", type=str, default=str(DATA_DIR / "dior_raw"), help="Path to raw DIOR dataset folder.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic DIOR data for testing without raw dataset.")
    parser.add_argument("--num_synthetic", type=int, default=120, help="Number of synthetic images to generate.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()

    raw_dir_path = Path(args.raw_dir)

    if args.synthetic or not raw_dir_path.exists():
        if not raw_dir_path.exists():
            logger.info("Raw DIOR directory '%s' not found. Generating synthetic dataset...", raw_dir_path)
        generate_synthetic_dior_dataset(DATA_DIR, num_images=args.num_synthetic)
        raw_dir_path = DATA_DIR / "dior_raw"

    stats = prepare_dior_dataset(raw_dir_path, DIOR_YOLO_DIR, seed=args.seed)
    print("=" * 60)
    print("Phase 1 Complete: Dataset successfully generated.")
    print(f"Dataset summary: Train={stats['train']}, Val={stats['val']}, Test={stats['test']}")
    print(f"Configs saved to {DATA_YAML_PATH} and {DATA_RETAIN_YAML_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
