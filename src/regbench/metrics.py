"""Registration error metrics: RRE in degrees, RTE as a fraction of the object size d."""
from __future__ import annotations

import numpy as np


def rre_deg(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """Relative rotation error in degrees: angle of R_est^T @ R_gt."""
    cos = (np.trace(T_est[:3, :3].T @ T_gt[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def rte_rel(T_est: np.ndarray, T_gt: np.ndarray, d: float) -> float:
    """Relative translation error: ||t_est - t_gt|| / d."""
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) / d)


def evaluate(T_est: np.ndarray, T_gt: np.ndarray, d: float, thr: dict) -> dict:
    """{rre_deg, rte, success} with success := both errors under the thresholds."""
    rre = rre_deg(T_est, T_gt)
    rte = rte_rel(T_est, T_gt, d)
    return {"rre_deg": rre, "rte": rte,
            "success": bool(rre < thr["rre_deg"] and rte < thr["rte_rel"])}
