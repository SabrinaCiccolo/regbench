import numpy as np
import pytest

from regbench.perturb import gauss_noise, inject_bump, plane_cut, random_rotation, \
    random_transform
from regbench.repro import stable_rng


def test_random_rotation_is_proper_and_bounded():
    for i in range(20):
        R = random_rotation(stable_rng("rot", i), max_angle_deg=45.0)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        assert angle <= 45.0 + 1e-9


def test_random_transform_translation_bounded():
    for i in range(20):
        T = random_transform(stable_rng("trans", i), 10.0, trans_max=0.5)
        assert np.linalg.norm(T[:3, 3]) <= 0.5 + 1e-12


def test_plane_cut_keeps_expected_count():
    pts = stable_rng("cut", 0).normal(size=(1000, 3))
    kept = plane_cut(pts, keep_frac=0.85, rng=stable_rng("cut", 1))
    assert abs(len(kept) - 850) <= 1
    assert len(plane_cut(pts, 1.0, stable_rng("cut", 2))) == 1000


def test_gauss_noise_std():
    pts = np.zeros((20000, 3))
    noisy = gauss_noise(pts, sigma=0.01, rng=stable_rng("noise", 0))
    assert noisy.std() == pytest.approx(0.01, rel=0.05)


def test_determinism_bit_identical():
    pts = stable_rng("det", 0).normal(size=(500, 3))
    a = gauss_noise(plane_cut(pts, 0.6, stable_rng("det", 1)), 0.01, stable_rng("det", 2))
    b = gauss_noise(plane_cut(pts, 0.6, stable_rng("det", 1)), 0.01, stable_rng("det", 2))
    assert np.array_equal(a, b)


def test_inject_bump_local_and_bounded():
    rng = stable_rng("bump", 0)
    pts = rng.normal(size=(2000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)   # unit sphere
    normals = pts.copy()                                 # outward normals
    out, mask = inject_bump(pts, normals, stable_rng("bump", 1),
                            radius=0.3, amplitude=0.05)
    moved = np.linalg.norm(out - pts, axis=1)
    assert mask.any() and not mask.all()
    assert np.all(moved[~mask] == 0.0)
    assert moved.max() == pytest.approx(0.05, rel=0.05)  # peak ~ amplitude at center
