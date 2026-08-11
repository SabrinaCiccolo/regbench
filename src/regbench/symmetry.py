"""Symmetry-aware registration metrics and part-symmetry detection.

Rotation about a part's own symmetry axis is unobservable from geometry, so a
pose is scored against every task-equivalent ground truth
T_ref(theta) = S_theta @ T_gt, where S_theta rotates by theta about the symmetry
axis in the target frame. Also: a continuous symmetry score (rotational
self-overlap), discrete fold detection, branch consistency and Wilson intervals.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .metrics import rre_deg, rte_rel


def _axis_rotations(axis: np.ndarray, thetas_deg: np.ndarray) -> np.ndarray:
    """(K,3,3) stack of rotations by thetas about the unit ``axis``."""
    return Rotation.from_rotvec(np.deg2rad(thetas_deg)[:, None] * axis).as_matrix()


def fold_references(T_gt: np.ndarray, axis: np.ndarray, center: np.ndarray,
                    thetas_deg: np.ndarray) -> np.ndarray:
    """(K,4,4) stack of task-equivalent ground truths S_theta @ T_gt.

    S_theta rotates about the line (center, axis) in the target frame:
    S = [R_a, c - R_a c]. Applied on the left of T_gt (symmetry acts on the
    already-aligned cloud, which sits in the target frame).
    """
    R_a = _axis_rotations(axis, thetas_deg)                       # (K,3,3)
    refs = np.tile(np.eye(4), (len(thetas_deg), 1, 1))
    refs[:, :3, :3] = R_a @ T_gt[:3, :3]
    refs[:, :3, 3] = (R_a @ T_gt[:3, 3]) + center - R_a @ center
    return refs


def folded_evaluate(T_est: np.ndarray, T_gt: np.ndarray, d: float, thr: dict,
                    axis: np.ndarray | None, center: np.ndarray,
                    theta_step_deg: float = 0.25,
                    thetas_deg: np.ndarray | None = None) -> dict:
    """{rre_sym_deg, rte_sym, success_sym}: error modulo the symmetry group.

    ``axis=None`` is the trivial group (same as metrics.evaluate). Otherwise the
    errors are reported at the theta minimizing max(rre/thr_rre, rte/thr_rte), and
    success_sym is True if any theta passes both thresholds. ``thetas_deg`` sets
    an explicit theta set (a discrete C_n fold); the default is a dense grid with
    step ``theta_step_deg``.
    """
    if axis is None:
        rre = rre_deg(T_est, T_gt)
        rte = rte_rel(T_est, T_gt, d)
        return {"rre_sym_deg": rre, "rte_sym": rte,
                "success_sym": bool(rre < thr["rre_deg"] and rte < thr["rte_rel"]),
                "theta_deg": 0.0}

    thetas = (np.asarray(thetas_deg, dtype=float) if thetas_deg is not None
              else np.arange(0.0, 360.0, theta_step_deg))
    refs = fold_references(T_gt, axis, center, thetas)
    # rre over the stack: angle of R_est^T R_ref via the trace identity
    RtR = np.einsum("ji,kjl->kil", T_est[:3, :3], refs[:, :3, :3])
    cos = np.clip((np.trace(RtR, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rres = np.degrees(np.arccos(cos))
    rtes = np.linalg.norm(T_est[:3, 3] - refs[:, :3, 3], axis=1) / d
    worst = np.maximum(rres / thr["rre_deg"], rtes / thr["rte_rel"])
    k = int(np.argmin(worst))
    return {"rre_sym_deg": float(rres[k]), "rte_sym": float(rtes[k]),
            "success_sym": bool(np.any((rres < thr["rre_deg"])
                                       & (rtes < thr["rte_rel"]))),
            # the branch theta whose S_theta @ T_gt is nearest T_est
            "theta_deg": float(thetas[k])}


def rotate_about(pts: np.ndarray, axis: np.ndarray, center: np.ndarray,
                 theta_deg: float) -> np.ndarray:
    R = Rotation.from_rotvec(np.deg2rad(theta_deg) * axis).as_matrix()
    return (pts - center) @ R.T + center


def self_overlap(pts: np.ndarray, axis: np.ndarray, center: np.ndarray,
                 thetas_deg: np.ndarray, d: float) -> np.ndarray:
    """Self-overlap curve: for each theta, mean nearest-neighbour distance from the
    rotated cloud to the original, divided by d (~0 where the part maps onto itself)."""
    tree = cKDTree(pts)
    out = np.empty(len(thetas_deg))
    for i, theta in enumerate(thetas_deg):
        dist, _ = tree.query(rotate_about(pts, axis, center, theta), k=1)
        out[i] = dist.mean() / d
    return out


def detect_discrete_fold(pts: np.ndarray, axis: np.ndarray, center: np.ndarray,
                         d: float, n_max: int = 24, depth_frac: float = 0.35,
                         min_modulation: float = 1.6) -> tuple[int | None, dict]:
    """Largest discrete rotational order n (C_n, 2 <= n <= n_max) about
    (center, axis), or None.

    On the 1-degree self-overlap curve, n is accepted if the overlap at every
    nontrivial multiple of 360/n lies below floor + ``depth_frac`` * (median -
    floor). A curve with median/floor below ``min_modulation`` has no discrete
    structure (the continuous case) and returns None.

    ``info`` holds the curve and decision levels, plus ``fold_overlap_max`` (the
    worst overlap among the accepted multiples) when a fold is found.
    """
    thetas = np.arange(1.0, 360.0, 1.0)
    curve = self_overlap(pts, axis, center, thetas, d)
    floor = float(curve.min())
    med = float(np.median(curve))
    info = {"curve_thetas": thetas, "curve": curve, "floor": floor,
            "median": med, "threshold": floor + depth_frac * (med - floor)}
    if med < min_modulation * floor:
        return None, info
    for n in range(n_max, 1, -1):
        mult = np.arange(1, n) * (360.0 / n)
        vals = np.interp(mult, thetas, curve)
        if float(vals.max()) <= info["threshold"]:
            info["fold_overlap_max"] = float(vals.max())
            return n, info
    return None, info


PROBE_THETAS = np.array([45.0, 90.0, 135.0, 180.0])
SCORE_THETAS = np.arange(10.0, 181.0, 10.0)


def detect_symmetry_axis(pts: np.ndarray, d: float) -> tuple[np.ndarray, dict]:
    """Best rotational-symmetry axis among the 3 PCA axes (through the centroid).

    Returns (unit axis, info): the probe overlap of each candidate (mean
    self_overlap over PROBE_THETAS), the SCORE_THETAS curve of the winning axis,
    and ``sym_score`` = mean of that curve (low = symmetric).

    A symmetry of the whole object has its axis through the centroid along a PCA
    axis. A symmetric local feature off the centroid (e.g. a boss on a bracket) is
    not detected.
    """
    center = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)
    probes = [self_overlap(pts, Vt[i], center, PROBE_THETAS, d).mean()
              for i in range(3)]
    best = int(np.argmin(probes))
    axis = Vt[best]
    curve = self_overlap(pts, axis, center, SCORE_THETAS, d)
    return axis, {"center": center, "probe_overlaps": np.asarray(probes),
                  "curve_thetas": SCORE_THETAS, "curve": curve,
                  "sym_score": float(curve.mean()), "sym_score_min": float(curve.min())}


def branch_consistency(thetas_deg: np.ndarray, group_order: int | None = None,
                       bin_deg: float = 30.0) -> dict:
    """Do repeated trials reach the same symmetry branch?

    ``thetas_deg`` holds one ``folded_evaluate(...)["theta_deg"]`` per trial. They
    are binned into ``group_order`` bins for a discrete C_n, else into
    round(360/bin_deg) bins. Returns the Shannon entropy in bits, the entropy
    normalized by log2(n_bins), and the modal bin's frequency: entropy_norm ~ 0 is
    a consistent branch, ~ 1 a uniformly random one.
    """
    thetas = np.mod(np.asarray(thetas_deg, dtype=float), 360.0)
    n_bins = int(group_order) if group_order else max(1, round(360.0 / bin_deg))
    counts, _ = np.histogram(thetas, bins=np.linspace(0.0, 360.0, n_bins + 1))
    n = int(counts.sum())
    if n == 0:
        return {"entropy": float("nan"), "entropy_norm": float("nan"),
                "mode_frequency": float("nan"), "n_bins": n_bins, "n": 0}
    p = counts[counts > 0] / n
    entropy = float(-(p * np.log2(p)).sum())
    max_entropy = float(np.log2(n_bins)) if n_bins > 1 else 0.0
    return {"entropy": entropy,
            "entropy_norm": (entropy / max_entropy) if max_entropy > 0 else 0.0,
            "mode_frequency": float(counts.max() / n), "n_bins": n_bins, "n": n}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% at the default z)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
