import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from regbench.metrics import evaluate, rre_deg, rte_rel
from regbench.perturb import make_trial
from regbench.repro import stable_rng


def T_from(R=np.eye(3), t=np.zeros(3)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def test_rre_identity_is_zero_and_finite():
    assert rre_deg(np.eye(4), np.eye(4)) == 0.0  # arccos clip: no NaN at theta -> 0


@pytest.mark.parametrize("theta", [1.0, 45.0, 179.0])
def test_rre_recovers_known_angle(theta):
    rng = stable_rng("test_rre", theta)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = Rotation.from_rotvec(np.deg2rad(theta) * axis).as_matrix()
    assert rre_deg(T_from(R), np.eye(4)) == pytest.approx(theta, abs=1e-9)


def test_rte_known_translation():
    assert rte_rel(T_from(t=np.array([3.0, 0.0, 4.0])), np.eye(4), d=10.0) \
        == pytest.approx(0.5)


def test_success_threshold_boundaries():
    thr = {"rre_deg": 2.0, "rte_rel": 0.01}
    ok = Rotation.from_rotvec(np.deg2rad(1.9) * np.array([0, 0, 1.0])).as_matrix()
    bad = Rotation.from_rotvec(np.deg2rad(2.1) * np.array([0, 0, 1.0])).as_matrix()
    assert evaluate(T_from(ok), np.eye(4), 1.0, thr)["success"]
    assert not evaluate(T_from(bad), np.eye(4), 1.0, thr)["success"]         # rre fails
    assert not evaluate(T_from(t=np.array([0.02, 0, 0])), np.eye(4), 1.0, thr)["success"]


def test_make_trial_convention_lock():
    """The perfect estimator T_gt scores rre~0, rte~0: kills the inverse-sign bug."""
    rng = stable_rng("convention", 0)
    target = rng.normal(size=(500, 3))
    sev = {"rot_max_deg": 45, "trans_max_rel": 0.25, "noise_rel": 0.0, "keep_frac": 1.0}
    source, T_gt = make_trial(target, sev, d=1.0, rng=stable_rng("convention", 1))
    realigned = source @ T_gt[:3, :3].T + T_gt[:3, 3]
    assert np.allclose(realigned, target, atol=1e-9)
    # arccos amplifies float error near cos=1: ~0 but not exactly 0
    assert rre_deg(T_gt, T_gt) == pytest.approx(0.0, abs=1e-4)
