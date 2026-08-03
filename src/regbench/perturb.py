"""Trial construction and synthetic defects on (N,3) point arrays.

``make_trial`` corrupts a copy of the target and moves it by T_pert; the
returned ground truth is ``T_gt = inv(T_pert)``, the transform a source-to-target
registration must estimate.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def random_rotation(rng: np.random.Generator, max_angle_deg: float) -> np.ndarray:
    """Rotation about a uniform axis on the sphere, angle uniform in [0, max]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0.0, max_angle_deg))
    return Rotation.from_rotvec(angle * axis).as_matrix()


def random_transform(rng: np.random.Generator, max_angle_deg: float,
                     trans_max: float) -> np.ndarray:
    """4x4 rigid transform; translation = uniform direction x uniform magnitude."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    T = np.eye(4)
    T[:3, :3] = random_rotation(rng, max_angle_deg)
    T[:3, 3] = direction * rng.uniform(0.0, trans_max)
    return T


def gauss_noise(pts: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Per-point isotropic Gaussian noise (sigma absolute; caller scales by d)."""
    return pts + rng.normal(0.0, sigma, size=pts.shape)


def plane_cut_mask(pts: np.ndarray, keep_frac: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Boolean keep-mask of the ``keep_frac`` side of a random half-plane."""
    if keep_frac >= 1.0:
        return np.ones(len(pts), dtype=bool)
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    proj = pts @ normal
    return proj <= np.quantile(proj, keep_frac)


def plane_cut(pts: np.ndarray, keep_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Keep the ``keep_frac`` side of a random half-plane (partial-overlap corruption)."""
    return pts[plane_cut_mask(pts, keep_frac, rng)].copy()


def make_trial(target_pts: np.ndarray, sev: dict, d: float,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build one trial source from the target: cut -> noise -> known rigid move.

    Returns ``(source_pts, T_gt)`` with ``T_gt = inv(T_pert)``: applying T_gt to the
    source aligns it back onto the target (up to noise and the cut).
    """
    pts = plane_cut(target_pts, sev["keep_frac"], rng)
    pts = gauss_noise(pts, sev["noise_rel"] * d, rng)
    T_pert = random_transform(rng, sev["rot_max_deg"], sev["trans_max_rel"] * d)
    pts = pts @ T_pert[:3, :3].T + T_pert[:3, 3]
    return pts, np.linalg.inv(T_pert)


def inject_bump(pts: np.ndarray, normals: np.ndarray, rng: np.random.Generator,
                radius: float, amplitude: float, sign: int = 1,
                ) -> tuple[np.ndarray, np.ndarray]:
    """Displace points near a random surface point along their normals.

    Raised-cosine falloff inside ``radius``, peak displacement ``amplitude``.
    Returns ``(pts', defect_mask)``.
    """
    center = pts[rng.integers(len(pts))]
    dist = np.linalg.norm(pts - center, axis=1)
    mask = dist < radius
    falloff = 0.5 * (1.0 + np.cos(np.pi * dist[mask] / radius))
    out = pts.copy()
    out[mask] += sign * amplitude * falloff[:, None] * normals[mask]
    return out, mask
