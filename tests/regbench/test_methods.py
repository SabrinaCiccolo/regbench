import numpy as np
import pytest

from regbench.cloud_io import iter_industrial
from regbench.config import load_config
from regbench.methods import METHODS, prepare_target, register
from regbench.metrics import rre_deg, rte_rel
from regbench.perturb import make_trial
from regbench.repro import stable_rng

CFG = load_config()
REG = CFG["registration"]


@pytest.fixture(scope="module")
def blob():
    """Deterministic asymmetric surface (~4k points on a lumpy deformed sphere)."""
    rng = stable_rng("blob", 0)
    pts = rng.normal(size=(4000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    bumps = 1.0 + 0.3 * np.sin(3 * pts[:, 0]) + 0.2 * np.cos(5 * pts[:, 1] + 1.0) \
        + 0.15 * np.sin(7 * pts[:, 2])
    return pts * bumps[:, None]


@pytest.fixture(scope="module")
def target(blob):
    import open3d as o3d

    d = float(np.linalg.norm(blob.max(0) - blob.min(0)))
    voxel = d / 60
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(blob))
    pcd = pcd.voxel_down_sample(voxel)
    return prepare_target(pcd, voxel, REG), d


def test_registry_matches_config():
    assert set(CFG["methods"]) == set(METHODS)


@pytest.mark.parametrize("name", ["icp_pt2pt", "icp_pt2pl"])
def test_zero_perturbation_is_identity(name, target, blob):
    prep, d = target
    T = register(name, blob, prep, REG, trial_seed=0)
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
    assert rre_deg(T, np.eye(4)) < 0.5
    assert rte_rel(T, np.eye(4), d) < 0.001


def test_fpfh_ransac_recovers_easy_transform(target, blob):
    prep, d = target
    sev = {"rot_max_deg": 10, "trans_max_rel": 0.05, "noise_rel": 0.0005,
           "keep_frac": 1.0}
    source, T_gt = make_trial(blob, sev, d, stable_rng("easy", 0))
    T = register("fpfh_ransac", source, prep, REG, trial_seed=0)
    assert rre_deg(T, T_gt) < 2.0
    assert rte_rel(T, T_gt, d) < 0.01


def test_all_methods_return_valid_se3(target, blob):
    prep, d = target
    sev = {"rot_max_deg": 10, "trans_max_rel": 0.05, "noise_rel": 0.001,
           "keep_frac": 1.0}
    source, _ = make_trial(blob, sev, d, stable_rng("valid", 0))
    for name in CFG["methods"]:
        T = register(name, source, prep, REG, trial_seed=1)
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
        assert np.allclose(T[3], [0, 0, 0, 1])


def test_register_survives_degenerate_correspondences(target):
    """With almost no FPFH correspondences Open3D raises inside fgr; register()
    returns identity (a failed trial) instead."""
    prep, _ = target
    source = np.full((4, 3), 1e-3) + stable_rng("degenerate").normal(scale=1e-6, size=(4, 3))
    T = register("fgr", source, prep, REG, trial_seed=0)
    assert np.array_equal(T, np.eye(4))


def test_register_diag_tracks_success(target, blob):
    """return_diag=True must expose fitness/inlier_rmse/n_inliers, and a converged
    easy-severity trial must score higher fitness than a degenerate crashed one."""
    prep, d = target
    sev = {"rot_max_deg": 10, "trans_max_rel": 0.05, "noise_rel": 0.0005,
           "keep_frac": 1.0}
    source, _ = make_trial(blob, sev, d, stable_rng("diag-good", 0))
    T, diag = register("fpfh_ransac", source, prep, REG, trial_seed=0, return_diag=True)
    assert T.shape == (4, 4)
    assert set(diag) == {"fitness", "inlier_rmse", "n_inliers"}
    assert diag["fitness"] > 0.5
    assert diag["n_inliers"] > 0

    degenerate = np.full((4, 3), 1e-3) + stable_rng("degenerate").normal(
        scale=1e-6, size=(4, 3))
    _, bad_diag = register("fgr", degenerate, prep, REG, trial_seed=0,
                           return_diag=True)
    assert bad_diag["fitness"] < diag["fitness"]

    T_plain = register("fpfh_ransac", source, prep, REG, trial_seed=0)
    assert isinstance(T_plain, np.ndarray)


def test_iter_industrial_layout(tmp_path):
    for split in ("train_good", "test_good", "test_defect"):
        (tmp_path / "PART_A" / split).mkdir(parents=True)
    for name in ("b.ply", "a.ply", "c.pcd"):
        (tmp_path / "PART_A" / "train_good" / name).touch()
    got = [p.name for p in iter_industrial(tmp_path, "PART_A", "train_good")]
    assert got == ["a.ply", "b.ply", "c.pcd"]
    assert list(iter_industrial(tmp_path, "PART_A", "test_defect")) == []
    with pytest.raises(ValueError):
        list(iter_industrial(tmp_path, "PART_A", "nope"))
