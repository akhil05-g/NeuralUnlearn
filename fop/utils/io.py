"""
fop.utils.io - File I/O utilities: directory management, checkpoint loading, JSON serialization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from fop.config import (
    PROJECT_ROOT, DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    LOGS_DIR, FEATURES_DIR, PLOTS_DIR,
)
from fop.utils.common import get_device

logger = logging.getLogger("fop.utils.io")


# ==============================================================================
# Directory Structure Management
# ==============================================================================
def ensure_dirs(base_root: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """Create the complete project directory structure required by all pipeline stages."""
    root = Path(base_root).resolve() if base_root else PROJECT_ROOT

    dirs_to_create: Dict[str, Path] = {
        "project_root": root,
        "data": root / "data",
        "dior_yolo": root / "data" / "dior_yolo",
        "images_train": root / "data" / "dior_yolo" / "images" / "train",
        "images_val": root / "data" / "dior_yolo" / "images" / "val",
        "images_test": root / "data" / "dior_yolo" / "images" / "test",
        "labels_train": root / "data" / "dior_yolo" / "labels" / "train",
        "labels_val": root / "data" / "dior_yolo" / "labels" / "val",
        "labels_test": root / "data" / "dior_yolo" / "labels" / "test",
        "dior_retain_yolo": root / "data" / "dior_retain_yolo",
        "retain_images_train": root / "data" / "dior_retain_yolo" / "images" / "train",
        "retain_images_val": root / "data" / "dior_retain_yolo" / "images" / "val",
        "retain_images_test": root / "data" / "dior_retain_yolo" / "images" / "test",
        "retain_labels_train": root / "data" / "dior_retain_yolo" / "labels" / "train",
        "retain_labels_val": root / "data" / "dior_retain_yolo" / "labels" / "val",
        "retain_labels_test": root / "data" / "dior_retain_yolo" / "labels" / "test",
        "checkpoints": root / "checkpoints",
        "results": root / "results",
        "logs": root / "logs",
        "features": root / "features",
        "plots": root / "plots",
    }

    created_count = 0
    for name, directory in dirs_to_create.items():
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_count += 1
        else:
            directory.mkdir(parents=True, exist_ok=True)

    logger.info("Verified %d directories (%d new) under: %s", len(dirs_to_create), created_count, root)
    return dirs_to_create


# ==============================================================================
# Checkpoint Management
# ==============================================================================
def load_checkpoint(path: Union[str, Path], device: Optional[Any] = None) -> Any:
    """Safely load a model checkpoint from disk (PyTorch or Ultralytics)."""
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    dev = get_device(str(device)) if isinstance(device, str) else (device or get_device())
    logger.info("Loading checkpoint from: %s on device: %s", checkpoint_path, dev)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        return checkpoint
    except Exception as exc:
        logger.warning("torch.load failed (%s). Trying Ultralytics YOLO...", exc)
        try:
            from ultralytics import YOLO
            return YOLO(str(checkpoint_path))
        except Exception as inner_exc:
            raise RuntimeError(f"Failed to load checkpoint: {inner_exc}") from inner_exc


# ==============================================================================
# Result Serialization (JSON)
# ==============================================================================
class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling NumPy types, PyTorch tensors, and Path objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if torch is not None and hasattr(torch, "Tensor") and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)


def save_results(results_dict: Dict[str, Any], path: Union[str, Path], indent: int = 4) -> Path:
    """Save experiment results dictionary to a JSON file."""
    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=indent, cls=SafeJSONEncoder, ensure_ascii=False)
    logger.info("Saved results to: %s", target_path)
    return target_path


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load experiment results from a JSON file."""
    target_path = Path(path).resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Results file not found: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded results from: %s", target_path)
    return data
