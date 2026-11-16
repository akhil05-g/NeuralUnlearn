"""
fop.config - Central Configuration for 'Forgetting on Purpose' Project.

Defines all paths, class mappings, hyperparameters, and experiment settings
for class unlearning in remote-sensing small-object detection using DIOR and YOLOv8n.
"""

from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False


# ==============================================================================
# Path Configurations
# ==============================================================================
# PROJECT_ROOT is the top-level forgetting_on_purpose/ directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
FEATURES_DIR: Path = PROJECT_ROOT / "features"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"

# DIOR dataset YOLO directory structure
DIOR_YOLO_DIR: Path = DATA_DIR / "dior_yolo"
IMAGES_DIR: Path = DIOR_YOLO_DIR / "images"
LABELS_DIR: Path = DIOR_YOLO_DIR / "labels"

IMAGES_TRAIN_DIR: Path = IMAGES_DIR / "train"
IMAGES_VAL_DIR: Path = IMAGES_DIR / "val"
IMAGES_TEST_DIR: Path = IMAGES_DIR / "test"

LABELS_TRAIN_DIR: Path = LABELS_DIR / "train"
LABELS_VAL_DIR: Path = LABELS_DIR / "val"
LABELS_TEST_DIR: Path = LABELS_DIR / "test"

DATA_YAML_PATH: Path = DATA_DIR / "dior_6class.yaml"
DATA_RETAIN_YAML_PATH: Path = DATA_DIR / "dior_5class_retain.yaml"


# ==============================================================================
# Class Definitions & Unlearning Settings
# ==============================================================================
DIOR_CLASSES: List[str] = [
    "airplane",   # index 0
    "airport",    # index 1
    "vehicle",    # index 2
    "ship",       # index 3
    "harbor",     # index 4
    "windmill",   # index 5 (target forget class)
]

NUM_CLASSES: int = len(DIOR_CLASSES)

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(DIOR_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(DIOR_CLASSES)}

FORGET_CLASS: str = "windmill"
FORGET_CLASS_IDX: int = CLASS_TO_IDX[FORGET_CLASS]

RETAIN_CLASSES: List[str] = [c for c in DIOR_CLASSES if c != FORGET_CLASS]
RETAIN_CLASS_IDXS: List[int] = [CLASS_TO_IDX[c] for c in RETAIN_CLASSES]
NUM_RETAIN_CLASSES: int = len(RETAIN_CLASSES)

CO_OCCURRING_PAIRS: Dict[str, str] = {
    "airplane": "airport",
    "airport": "airplane",
    "ship": "harbor",
    "harbor": "ship",
}


# ==============================================================================
# Model & Training Configurations
# ==============================================================================
BASE_MODEL: str = "yolov8n.pt"
IMG_SIZE: int = 640
BATCH_SIZE: int = 16
NUM_WORKERS: int = 4

EPOCHS_ORIGINAL: int = 100
EPOCHS_FINETUNE: int = 30
EPOCHS_RETRAIN: int = 100
EPOCHS_UNLEARN: int = 30

LR_ORIGINAL: float = 0.01
LR_UNLEARN: float = 0.001
WEIGHT_DECAY: float = 0.0005
MOMENTUM: float = 0.937


# ==============================================================================
# Fuzzy Channel Erasure (FCE) Settings
# ==============================================================================
NECK_LAYER_INDEX: int = 0
ERASURE_STRENGTHS: List[float] = [0.3, 0.5, 0.7, 0.9]
DEFAULT_ERASURE_STRENGTH: float = 0.5
FUZZY_LOW_THRESHOLD: float = 0.2
FUZZY_HIGH_THRESHOLD: float = 0.8


# ==============================================================================
# Knowledge Distillation Settings
# ==============================================================================
ALPHA_DISTILL: float = 0.5
TEMPERATURE: float = 3.0


# ==============================================================================
# Relearnability / Leakage Evaluation (K-Shot) Settings
# ==============================================================================
K_SHOTS: List[int] = [5, 10, 20, 50]
RELEARN_EPOCHS: int = 10
RELEARN_LR: float = 0.001


# ==============================================================================
# Reproducibility & Device Auto-Detection
# ==============================================================================
SEED: int = 42


def get_device() -> str:
    """Auto-detect available computation device."""
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE: str = get_device()


def print_config() -> None:
    """Print a clean summary of current project settings."""
    print("=" * 70)
    print(" 'Forgetting on Purpose' Project Configuration")
    print("=" * 70)
    print(f"PROJECT_ROOT      : {PROJECT_ROOT}")
    print(f"DATA_DIR          : {DATA_DIR}")
    print(f"CHECKPOINTS_DIR   : {CHECKPOINTS_DIR}")
    print(f"RESULTS_DIR       : {RESULTS_DIR}")
    print(f"DEVICE            : {DEVICE}")
    print("-" * 70)
    print(f"DIOR Classes ({NUM_CLASSES})   : {DIOR_CLASSES}")
    print(f"Forget Class      : '{FORGET_CLASS}' (index {FORGET_CLASS_IDX})")
    print(f"Retain Classes ({NUM_RETAIN_CLASSES}): {RETAIN_CLASSES}")
    print("-" * 70)
    print(f"Model: {BASE_MODEL} | ImgSize: {IMG_SIZE} | Batch: {BATCH_SIZE}")
    print(f"Epochs - Original: {EPOCHS_ORIGINAL} | Retrain: {EPOCHS_RETRAIN} | Unlearn: {EPOCHS_UNLEARN}")
    print(f"FCE Erasure Strengths: {ERASURE_STRENGTHS}")
    print(f"Distillation: Alpha={ALPHA_DISTILL}, Temp={TEMPERATURE}")
    print(f"Seed: {SEED}")
    print("=" * 70)
