"""Distractor placement in clutter.py; the load_distractors check needs the ITODD templates."""
import numpy as np
import pytest

from regbench.config import REPO_ROOT
from regbench.experiments.clutter import load_distractors, scatter_distractors
from regbench.repro import stable_rng

TEMPLATE_DIR = REPO_ROOT / "data" / "template"
CCFG = {"template_dir": "data/template", "distractor_parts": ["cylinder", "washer"],
       "offset_rel": [0.6, 1.3], "distractor_points": 50}


@pytest.fixture
def distractors():
    """Two parts with the same point count, as load_distractors returns them."""
    rng = stable_rng("test-distractors")
    a = rng.normal(size=(CCFG["distractor_points"], 3))
    b = rng.normal(size=(CCFG["distractor_points"], 3)) * 0.5 + 5.0  # off-center
    return {"partA": a - a.mean(0), "partB": b - b.mean(0)}


def test_zero_distractors_is_empty(distractors):
    out = scatter_distractors(0, distractors, d=1.0, ccfg=CCFG,
                              rng=stable_rng("clutter-test", 0))
    assert out.shape == (0, 3)


def test_distractor_count_and_placement_distance(distractors):
    n, d = 3, 2.0
    lo, hi = CCFG["offset_rel"]
    part_size = next(iter(distractors.values())).shape[0]
    out = scatter_distractors(n, distractors, d=d, ccfg=CCFG,
                              rng=stable_rng("clutter-test", 1))
    assert out.shape == (n * part_size, 3)
    # each distractor chunk's centroid must land in [lo*d, hi*d] of the origin -
    # rotation is about the distractor's own (already-centered) centroid, so only
    # the translation offset should move it.
    for i in range(n):
        chunk = out[i * part_size:(i + 1) * part_size]
        dist = np.linalg.norm(chunk.mean(axis=0))
        assert lo * d - 1e-6 <= dist <= hi * d + 1e-6


def test_scatter_is_deterministic(distractors):
    a = scatter_distractors(2, distractors, d=1.0, ccfg=CCFG,
                            rng=stable_rng("clutter-test", 7))
    b = scatter_distractors(2, distractors, d=1.0, ccfg=CCFG,
                            rng=stable_rng("clutter-test", 7))
    assert np.array_equal(a, b)


@pytest.mark.skipif(not (TEMPLATE_DIR / "itodd_cylinder.ply").exists(),
                    reason="ITODD templates not built (regbench build-template itodd)")
def test_load_distractors_reads_all_parts_centered():
    ccfg = {"template_dir": "data/template",
           "distractor_parts": ["cylinder", "washer"], "distractor_points": 500}
    out = load_distractors(ccfg)
    assert set(out) == {"cylinder", "washer"}
    for pts in out.values():
        assert pts.shape == (500, 3)
        # centered on the FULL cloud's centroid before the 500-point subsample, so
        # the subsample's own mean is near but not exactly zero (sampling noise).
        assert np.linalg.norm(pts.mean(axis=0)) < 0.05 * np.linalg.norm(pts.std(axis=0))
