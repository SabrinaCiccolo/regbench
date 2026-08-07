"""Template-difference anomaly scoring and ground-truth sidecars.

A test scan is registered onto the template and each point is scored by its
distance to the nearest template point. The scan score is a high quantile of
those distances divided by d; the per-point threshold tau is calibrated on good
scans only.

A sidecar ``<stem>.gt.npz`` next to a scan holds per-point defect labels (in the
scan's point order) and, for synthetic scans, the true pose T_gt (applying T_gt to
the scan aligns it onto the template). Scans without a sidecar are scored the same
way; only the ground-truth columns stay empty.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney; ties count 0.5). labels: 1 = anomalous."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc needs at least one positive and one negative")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def point_distances(pts: np.ndarray, template_pts: np.ndarray) -> np.ndarray:
    """Per-point nearest-neighbour distance to the template (vectorized cKDTree)."""
    return cKDTree(template_pts).query(pts, k=1)[0]


def scan_score(dist: np.ndarray, d: float, quantile: float) -> float:
    """Scan-level anomaly score: a robust high quantile of dist, as a fraction of d."""
    return float(np.quantile(dist, quantile) / d)


def calibrate_tau(pooled_dist: np.ndarray, quantile: float) -> float:
    """Point-level threshold from pooled GOOD-scan distances (defects never seen)."""
    return float(np.quantile(pooled_dist, quantile))


def exclusion_mask(pts: np.ndarray, defect_mask: np.ndarray,
                   radius: float) -> np.ndarray:
    """Dilated defect zone: points within ``radius`` of any true defect point.

    False positives are counted OUTSIDE this zone, so boundary points of a real
    defect are not billed as false alarms.
    """
    if defect_mask is None or not defect_mask.any():
        return np.zeros(len(pts), dtype=bool)
    near = cKDTree(pts[defect_mask]).query(pts, k=1)[0]
    return near <= radius


def fp_tp_rates(dist: np.ndarray, tau: float, defect_mask: np.ndarray | None = None,
                exclude_mask: np.ndarray | None = None) -> tuple[float, float]:
    """(FPR, TPR) of the per-point threshold ``dist > tau``.

    Without a defect mask every point is a potential false positive and TPR is nan.
    With one, FPR is measured outside the (optionally dilated) defect zone and TPR
    inside the true mask.
    """
    hot = dist > tau
    if defect_mask is None:
        return float(hot.mean()), float("nan")
    if exclude_mask is None:
        exclude_mask = defect_mask
    fp_zone = ~defect_mask & ~exclude_mask
    fpr = float(hot[fp_zone].mean()) if fp_zone.any() else float("nan")
    tpr = float(hot[defect_mask].mean()) if defect_mask.any() else float("nan")
    return fpr, tpr


# ---- GT sidecars -------------------------------------------------------------------

def sidecar_path(ply_path: Path | str) -> Path:
    return Path(ply_path).with_suffix(".gt.npz")


def write_sidecar(ply_path: Path | str, defect_mask: np.ndarray,
                  T_gt: np.ndarray | None = None, defect_type: str = "") -> Path:
    out = sidecar_path(ply_path)
    arrays = {"defect_mask": np.asarray(defect_mask, dtype=bool),
              "defect_type": np.array(defect_type)}
    if T_gt is not None:
        arrays["T_gt"] = np.asarray(T_gt, dtype=float)
    np.savez(out, **arrays)
    return out


def load_sidecar(ply_path: Path | str) -> dict | None:
    """{'defect_mask', 'T_gt' (or None), 'defect_type'} - or None if no sidecar."""
    path = sidecar_path(ply_path)
    if not path.exists():
        return None
    with np.load(path) as z:
        return {"defect_mask": z["defect_mask"],
                "T_gt": z["T_gt"] if "T_gt" in z else None,
                "defect_type": str(z["defect_type"])}
