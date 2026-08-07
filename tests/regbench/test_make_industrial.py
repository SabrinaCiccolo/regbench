import numpy as np
import open3d as o3d
import pytest

from regbench.anomaly import load_sidecar, point_distances, sidecar_path
from regbench.build.make_industrial import build_synthetic_part
from regbench.cloud_io import INDUSTRIAL_SPLITS, iter_industrial, load_cloud
from regbench.config import load_config
from regbench.repro import stable_rng

COUNTS = {"train_good": 2, "test_good": 2, "test_defect": 2}


def small_cfg() -> dict:
    cfg = load_config()
    cfg["industrial"]["synthetic"].update(
        n_train_good=2, n_test_good=2, n_test_defect=2, severity="tiny")
    # zero noise: lets the pose-convention test demand exact realignment
    cfg["severities"]["tiny"] = {"rot_max_deg": 30, "trans_max_rel": 0.2,
                                 "noise_rel": 0.0, "keep_frac": 0.9}
    return cfg


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("industrial")
    pts = stable_rng("sphere", 0).normal(size=(3000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    template = tmp / "tpl.ply"
    o3d.io.write_point_cloud(
        str(template), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    root = tmp / "root"
    build_synthetic_part("tpart", template, small_cfg(), root)
    return template, root


def test_layout_counts_and_sidecars(built):
    _, root = built
    for split in INDUSTRIAL_SPLITS:
        plys = list(iter_industrial(root, "tpart", split))
        assert len(plys) == COUNTS[split]
        assert all(sidecar_path(p).exists() for p in plys)


def test_true_pose_realigns_onto_template(built):
    _, root = built
    template = np.asarray(load_cloud(root / "tpart/train_good/000.ply").points)
    for split in ("train_good", "test_good"):
        ply = root / "tpart" / split / "001.ply"
        pts = np.asarray(load_cloud(ply).points)
        T = load_sidecar(ply)["T_gt"]
        aligned = pts @ T[:3, :3].T + T[:3, 3]
        # zero noise: every realigned point must sit ON a template point
        assert point_distances(aligned, template).max() < 1e-9


def test_mask_semantics(built):
    _, root = built
    for split in ("train_good", "test_good"):
        for ply in iter_industrial(root, "tpart", split):
            sc = load_sidecar(ply)
            assert sc["defect_mask"].sum() == 0 and sc["defect_type"] == ""
    for ply in iter_industrial(root, "tpart", "test_defect"):
        sc = load_sidecar(ply)
        n_defect = sc["defect_mask"].sum()
        assert 0 < n_defect < len(sc["defect_mask"])
        assert sc["defect_type"] == "bump"


def test_rebuild_bit_identical(built, tmp_path):
    template, root = built
    root2 = tmp_path / "root2"
    build_synthetic_part("tpart", template, small_cfg(), root2)
    for split in INDUSTRIAL_SPLITS:
        for ply in iter_industrial(root, "tpart", split):
            ply2 = root2 / "tpart" / split / ply.name
            assert ply.read_bytes() == ply2.read_bytes()
            a, b = load_sidecar(ply), load_sidecar(ply2)
            assert np.array_equal(a["defect_mask"], b["defect_mask"])
            assert np.array_equal(a["T_gt"], b["T_gt"])
