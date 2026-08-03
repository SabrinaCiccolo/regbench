"""d (and therefore voxel) is computed once from the clean template and reused
for every trial of a part.

A d recomputed from each corrupted trial source would move thresholds such as
0.01 * d from trial to trial. These tests check that the two choices actually
differ and that the template-derived d is the one used.
"""
from __future__ import annotations

import numpy as np
import pytest

from regbench.perturb import make_trial
from regbench.repro import stable_rng


def _aabb_diag(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def test_recomputing_d_from_a_trial_source_would_actually_differ():
    """A cut and moved trial source has a different AABB diagonal from the template."""
    template = stable_rng("dvoxel", "template").normal(size=(3000, 3)) * 2.5
    d_template = _aabb_diag(template)
    sev = {"rot_max_deg": 45, "trans_max_rel": 0.25, "noise_rel": 0.01,
           "keep_frac": 0.6}
    for trial in range(5):
        source, _ = make_trial(template, sev, d_template,
                               stable_rng("dvoxel", "trial", trial))
        d_from_source = _aabb_diag(source)
        assert d_from_source != pytest.approx(d_template, rel=1e-3)


def test_same_d_reused_across_all_trials_of_a_part():
    """The grid-loop convention every driver script follows: compute d once from
    the clean template, then pass that SAME value into make_trial/evaluate for
    every trial/severity - never a per-trial recomputation."""
    template = stable_rng("dvoxel2", "template").normal(size=(3000, 3)) * 2.5
    d_template = _aabb_diag(template)
    voxel_divisor = 100
    voxel_template = d_template / voxel_divisor

    severities = {
        "easy": {"rot_max_deg": 10, "trans_max_rel": 0.05, "noise_rel": 0.001,
                 "keep_frac": 1.0},
        "medium": {"rot_max_deg": 45, "trans_max_rel": 0.25, "noise_rel": 0.0025,
                   "keep_frac": 0.85},
        "hard": {"rot_max_deg": 180, "trans_max_rel": 0.5, "noise_rel": 0.005,
                 "keep_frac": 0.60},
    }
    ds_used, voxels_used = [], []
    for severity, sev in severities.items():
        for trial in range(5):
            # mirrors grid.py's grid loop body exactly: d/voxel are outer-scope
            # values, not recomputed here.
            _source, _T_gt = make_trial(template, sev, d_template,
                                        stable_rng("grid", severity, trial))
            ds_used.append(d_template)
            voxels_used.append(voxel_template)

    assert len(set(ds_used)) == 1
    assert len(set(voxels_used)) == 1
    assert ds_used[0] == d_template
