"""Loads ``config.yaml``; its ``paths.*`` entries are relative to ``REPO_ROOT``."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict:
    """A fresh dict parsed from config.yaml (callers may mutate it)."""
    return yaml.safe_load(CONFIG_PATH.read_text())
