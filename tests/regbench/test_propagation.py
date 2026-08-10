import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from regbench.anomaly import fp_tp_rates, point_distances
from regbench.experiments.propagation import joint_offset_transform, offset_transform
from regbench.metrics import rre_deg, rte_rel
from regbench.repro import stable_rng


def test_offset_transform_pure_components():
    d = 2.5
    for i in range(10):
        T = offset_transform("rot", 7.0, d, stable_rng("off", "rot", i))
        assert rre_deg(T, np.eye(4)) == pytest.approx(7.0, abs=1e-9)
        assert rte_rel(T, np.eye(4), d) == 0.0
        T = offset_transform("trans", 0.02, d, stable_rng("off", "trans", i))
        assert rre_deg(T, np.eye(4)) == pytest.approx(0.0, abs=1e-6)
        assert rte_rel(T, np.eye(4), d) == pytest.approx(0.02, abs=1e-12)
    with pytest.raises(ValueError):
        offset_transform("both", 1.0, d, stable_rng("off", "bad"))
    # magnitude 0 must be exactly the identity (the sweep's clean anchor point)
    assert np.array_equal(offset_transform("rot", 0.0, d, stable_rng("off", 0)),
                          np.eye(4))


def test_joint_offset_transform_carries_both_components():
    d = 2.5
    for i in range(10):
        T = joint_offset_transform(7.0, 0.02, d, stable_rng("joint", i))
        assert rre_deg(T, np.eye(4)) == pytest.approx(7.0, abs=1e-9)
        assert rte_rel(T, np.eye(4), d) == pytest.approx(0.02, abs=1e-12)
    # zero magnitudes -> exactly identity, same convention as offset_transform
    assert np.array_equal(joint_offset_transform(0.0, 0.0, d, stable_rng("joint", 0)),
                          np.eye(4))
    # rotation axis and translation direction are independently drawn, not coupled
    T = joint_offset_transform(30.0, 0.1, 1.0, stable_rng("joint", "indep"))
    rot_axis = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    rot_axis /= np.linalg.norm(rot_axis)
    trans_dir = T[:3, 3] / np.linalg.norm(T[:3, 3])
    assert abs(np.dot(rot_axis, trans_dir)) < 0.99  # not parallel/antiparallel


def test_fpr_zero_at_true_pose_without_noise():
    template = stable_rng("fp", 0).normal(size=(500, 3))
    pts = template[::2].copy()                    # noise-free subset, true pose
    T = offset_transform("rot", 0.0, 1.0, stable_rng("fp", 1))
    aligned = pts @ T[:3, :3].T + T[:3, 3]
    dist = point_distances(aligned, template)
    mask = np.zeros(len(pts), dtype=bool)
    mask[:10] = True
    fpr, _ = fp_tp_rates(dist, tau=1e-12, defect_mask=mask, exclude_mask=mask)
    assert fpr == 0.0
