"""Shared helpers used across every pipeline stage.

Keeping configuration loading, logging and path resolution in one place means
each stage behaves consistently and the whole run is reproducible.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Project root = two levels up from this file (src/utils/common.py -> project).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the central YAML config as a dictionary."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return PROJECT_ROOT / rel_path


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs so every run produces identical output."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a logger that writes both to stdout and to the pipeline log file."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured -> reuse
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        log_path = resolve(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def ensure_dirs(*dirs: str) -> None:
    """Create each directory (config-relative) if it does not already exist."""
    for d in dirs:
        resolve(d).mkdir(parents=True, exist_ok=True)
