"""Path configuration for FairED.

Loads machine-specific data locations from ``configs/paths.yaml`` so that no
absolute paths are hardcoded in the package or notebooks. Copy
``configs/paths.example.yaml`` to ``configs/paths.yaml`` and fill in your local
data locations before running the pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Repository root = two levels up from this file (src/fair_ed/config.py).
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

# Expected keys in paths.yaml (see configs/paths.example.yaml).
REQUIRED_KEYS = (
    "visits_path",
    "meds_path",
    "pmh_path",
    "labs_path",
    "output_dir",
    "train_path",
    "test_path",
)


def load_paths(config_path: str | os.PathLike | None = None) -> dict:
    """Return the path configuration as a dict.

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML config. Defaults to
        ``configs/paths.yaml`` at the repository root, overridable with the
        ``FAIRED_PATHS`` environment variable.
    """
    if config_path is None:
        config_path = os.environ.get("FAIRED_PATHS", CONFIG_DIR / "paths.yaml")
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Copy configs/paths.example.yaml to configs/paths.yaml and fill in "
            "the locations of your local MIMIC-IV-ED / MC-MED data."
        )

    with open(config_path) as f:
        paths = yaml.safe_load(f) or {}

    missing = [k for k in REQUIRED_KEYS if k not in paths]
    if missing:
        raise KeyError(
            f"Missing required keys in {config_path}: {', '.join(missing)}. "
            "See configs/paths.example.yaml for the expected format."
        )
    return paths
