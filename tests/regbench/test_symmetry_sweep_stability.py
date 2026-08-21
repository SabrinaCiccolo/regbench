"""Reruns of the smoke-sized symmetry sweep stay within each other's Wilson intervals.

Open3D's FPFH/RANSAC/FGR are multithreaded and not order-deterministic, so
success-rate cells can move between reruns. Marked slow; skipped without the
ITODD templates.
"""
from __future__ import annotations

import pytest

from regbench.config import REPO_ROOT, load_config
from regbench.experiments.sweep_symmetry import run_sweep
from regbench.verify import cell_counts, intervals_overlap

pytestmark = pytest.mark.slow

TEMPLATE_DIR = REPO_ROOT / "data" / "template"
N_TRIALS_SMOKE = 3
N_RERUNS = 3
KEYS = ["symmetry_class", "severity"]


@pytest.mark.skipif(
    not all((TEMPLATE_DIR / f"itodd_{p}.ply").exists()
            for p in load_config()["symmetry_sweep"]["parts"]),
    reason="ITODD templates not built (regbench build-template itodd)")
def test_success_rate_cells_overlap_across_reruns():
    cfg = load_config()
    cfg["preprocess"]["voxel_divisor"] = 50
    baseline, _ = run_sweep(cfg, N_TRIALS_SMOKE)
    base = cell_counts(baseline, "success", KEYS)
    for rerun in range(N_RERUNS - 1):
        df, _ = run_sweep(cfg, N_TRIALS_SMOKE)
        for cell, counts in cell_counts(df, "success", KEYS).items():
            assert intervals_overlap(base[cell], counts), (rerun, cell, base[cell], counts)
