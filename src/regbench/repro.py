"""Deterministic seeding and machine description."""
from __future__ import annotations

import hashlib
import os
import platform

import numpy as np


def stable_rng(*parts) -> np.random.Generator:
    """Generator seeded from sha256 of the '/'-joined parts.

    Python's ``hash()`` is salted per process; sha256 gives the same stream for
    the same key on every run and machine.
    """
    key = "/".join(str(p) for p in parts).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def cpu_info() -> str:
    """'<n_logical_cores>x <cpu model>', stored next to timing measurements."""
    return f"{os.cpu_count() or 0}x {_cpu_model()}"
