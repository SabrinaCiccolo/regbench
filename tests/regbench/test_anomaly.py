import numpy as np
import pytest

from regbench.anomaly import auroc, calibrate_tau, exclusion_mask, fp_tp_rates, \
    load_sidecar, point_distances, scan_score, sidecar_path, write_sidecar
from regbench.cloud_io import iter_industrial
from regbench.repro import stable_rng


def test_auroc_perfect_and_inverted():
    labels = np.array([0, 0, 0, 1, 1], dtype=bool)
    assert auroc(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), labels) == 1.0
    assert auroc(np.array([0.8, 0.9, 1.0, 0.1, 0.2]), labels) == 0.0


def test_auroc_matches_bruteforce_with_ties():
    rng = stable_rng("auroc", 0)
    scores = rng.integers(0, 10, size=200).astype(float)   # heavy ties
    labels = rng.random(200) < 0.3
    pos, neg = scores[labels], scores[~labels]
    brute = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert auroc(scores, labels) == pytest.approx(float(brute), abs=1e-12)


def test_scan_score_scale_free():
    dist = stable_rng("score", 0).random(5000)
    assert scan_score(2 * dist, 2 * 1.0, 0.995) == pytest.approx(
        scan_score(dist, 1.0, 0.995))


def test_tau_calibration_fpr_matches_quantile():
    pooled = stable_rng("tau", 0).random(100000)
    tau = calibrate_tau(pooled, 0.999)
    fpr, tpr = fp_tp_rates(pooled, tau)
    assert fpr == pytest.approx(0.001, rel=0.2)
    assert np.isnan(tpr)


def test_fp_counting_excludes_dilated_zone():
    # 1D line of points; defect = a block in the middle, hot = defect + its fringe.
    pts = np.stack([np.linspace(0, 1, 101), np.zeros(101), np.zeros(101)], axis=1)
    defect = (pts[:, 0] >= 0.45) & (pts[:, 0] <= 0.55)
    excl = exclusion_mask(pts, defect, radius=0.031)
    assert excl.sum() > defect.sum()                     # dilation grew the zone
    dist = np.where((pts[:, 0] >= 0.42) & (pts[:, 0] <= 0.58), 1.0, 0.0)
    fpr_excl, tpr = fp_tp_rates(dist, 0.5, defect, excl)
    fpr_raw, _ = fp_tp_rates(dist, 0.5, defect, defect)
    assert tpr == 1.0
    assert fpr_excl < fpr_raw                            # fringe not billed as FP
    assert fpr_excl == 0.0


def test_point_distances_zero_on_self():
    pts = stable_rng("pd", 0).normal(size=(300, 3))
    assert point_distances(pts, pts).max() == 0.0
    d = point_distances(pts + np.array([0.5, 0, 0]), pts)
    assert d.min() > 0.0


def test_sidecar_roundtrip_and_absence(tmp_path):
    ply = tmp_path / "scan.ply"
    mask = np.zeros(50, dtype=bool)
    mask[10:20] = True
    T = np.diag([1.0, 1.0, 1.0, 1.0])
    write_sidecar(ply, mask, T_gt=T, defect_type="bump")
    got = load_sidecar(ply)
    assert np.array_equal(got["defect_mask"], mask)
    assert np.array_equal(got["T_gt"], T)
    assert got["defect_type"] == "bump"
    assert load_sidecar(tmp_path / "other.ply") is None
    write_sidecar(tmp_path / "good.ply", np.zeros(5, dtype=bool))
    assert load_sidecar(tmp_path / "good.ply")["T_gt"] is None


def test_sidecars_invisible_to_iter_industrial(tmp_path):
    split = tmp_path / "PART" / "test_defect"
    split.mkdir(parents=True)
    (split / "000.ply").touch()
    write_sidecar(split / "000.ply", np.zeros(3, dtype=bool))
    assert sidecar_path(split / "000.ply").exists()
    found = list(iter_industrial(tmp_path, "PART", "test_defect"))
    assert found == [split / "000.ply"]
