"""
fop.utils.common - Core utilities: seeding, device management, timing, logging, class lookups.
"""

from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import os
import random
import sys
import time

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from fop.config import (
    DIOR_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, SEED,
)

logger = logging.getLogger("fop.utils")


# ==============================================================================
# Reproducibility
# ==============================================================================
def set_seed(seed: int = SEED) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("Random seed set to %d (torch available, cuda: %s)", seed, torch.cuda.is_available())
    else:
        logger.info("Random seed set to %d (numpy/random only)", seed)


# ==============================================================================
# Device Management
# ==============================================================================
def get_device(preferred: Optional[str] = None):
    """Get the PyTorch compute device, with optional override."""
    if torch is None:
        return "cpu"
    if preferred is not None:
        target = preferred.lower().strip()
        if target.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested ('%s') but not available. Falling back to CPU.", preferred)
            return torch.device("cpu")
        return torch.device(target)

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        logger.info("Using GPU: %s (%d available)", torch.cuda.get_device_name(0), torch.cuda.device_count())
        return dev

    logger.info("Using CPU device.")
    return torch.device("cpu")


# ==============================================================================
# Class Index & Name Lookups
# ==============================================================================
def get_class_index(class_name: str) -> int:
    """Look up the integer class index for a given DIOR class name."""
    cleaned_name = class_name.strip().lower()
    if cleaned_name not in CLASS_TO_IDX:
        raise ValueError(f"Unknown class '{class_name}'. Must be one of: {list(CLASS_TO_IDX.keys())}")
    return CLASS_TO_IDX[cleaned_name]


def get_class_name(index: int) -> str:
    """Look up the class name for a given zero-based DIOR class index."""
    if index not in IDX_TO_CLASS:
        raise IndexError(f"Class index {index} out of range [0, {len(DIOR_CLASSES) - 1}].")
    return IDX_TO_CLASS[index]


# ==============================================================================
# Execution Timer
# ==============================================================================
class Timer(ContextDecorator):
    """Context manager and decorator for timing code blocks."""

    def __init__(self, description: str = "Operation", log_at_exit: bool = True) -> None:
        self.description = description
        self.log_at_exit = log_at_exit
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        if self.log_at_exit:
            logger.info("[%s] completed in %s", self.description, self.elapsed_str)

    @property
    def elapsed_str(self) -> str:
        if self.elapsed < 60:
            return f"{self.elapsed:.2f}s"
        minutes = int(self.elapsed // 60)
        seconds = self.elapsed % 60
        if minutes < 60:
            return f"{minutes}m {seconds:.1f}s"
        hours = int(minutes // 60)
        return f"{hours}h {minutes % 60}m {seconds:.0f}s"


# ==============================================================================
# Logging Configuration
# ==============================================================================
def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    name: str = "fop",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure project-wide logging with consistent formatting."""
    target_logger = logging.getLogger(name)
    target_logger.setLevel(level)

    if target_logger.handlers:
        target_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    target_logger.addHandler(stream_handler)

    if log_file is not None:
        file_path = Path(log_file).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        target_logger.addHandler(file_handler)

    return target_logger
