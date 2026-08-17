"""BOP-style pose metrics (ADD-S, MSSD) on the stored symmetry-sweep poses.

- ADD-S (Hinterstoisser et al., 2012): mean nearest-neighbour distance between
  the template transformed by the estimate and by the ground truth.
- MSSD (Hodan et al., 2020): maximum distance between corresponding points,
  minimized over the part's symmetry group (identity, the detected C_n, or a
  2-degree grid for a continuous axis).

Both are divided by d and thresholded at 0.1 d, on a fixed 150-point subsample
of each preprocessed template. MSPD is not computed: it needs a camera model.
No registration is run.

    regbench bop

Outputs (tables_dir): bop_metrics.csv (sweep rows + add_s_rel, add_s_success,
mssd_rel, mssd_success) and bop_metrics.md.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from regbench.cloud_io import load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table
from regbench.repro import stable_rng
from regbench.symmetry import detect_discrete_fold, detect_symmetry_axis, \
    fold_references, wilson_ci

N_SUBSAMPLE = 150
ADD_THRESHOLD_REL = 0.1
MSSD_THRESHOLD_REL = 0.1
CONT_THETA_STEP_DEG = 2.0


def _T_from_row(row, prefix: str) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(
        [row[f"{prefix}_q{k}"] for k in "xyzw"]).as_matrix()
    T[:3, 3] = [row[f"{prefix}_t{k}"] for k in "xyz"]
    return T


def _transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def add_s(pts: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray, d: float) -> float:
    """Mean nearest-neighbour distance, est-transformed to gt-transformed, / d."""
    est_pts = _transform(pts, T_est)
    gt_pts = _transform(pts, T_gt)
    dist, _ = cKDTree(gt_pts).query(est_pts, k=1)
    return float(dist.mean() / d)


def mssd(pts: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray, d: float,
         axis: np.ndarray | None, center: np.ndarray,
         thetas_deg: np.ndarray | None) -> float:
    """Min over the symmetry group of the max corresponding-point distance, / d.

    ``axis=None`` is the trivial group (a single reference pose).
    """
    if axis is None:
        refs = T_gt[None, :, :]
    else:
        thetas = (thetas_deg if thetas_deg is not None
                  else np.arange(0.0, 360.0, CONT_THETA_STEP_DEG))
        refs = fold_references(T_gt, axis, center, thetas)
    est_pts = _transform(pts, T_est)  # (N, 3)
    # (K, N, 3): every reference pose applied to every point
    ref_pts = np.einsum("kij,nj->kni", refs[:, :3, :3], pts) + refs[:, None, :3, 3]
    dist = np.linalg.norm(ref_pts - est_pts[None, :, :], axis=2)  # (K, N)
    max_per_ref = dist.max(axis=1)  # (K,)
    return float(max_per_ref.min() / d)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser().parse_args(argv)  # no flags; enables `--help`
    cfg = load_config()
    scfg = cfg["symmetry_sweep"]
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    df = pd.read_csv(tables / "symmetry_sweep.csv")

    part_model: dict[str, dict] = {}
    for part, sym_class in scfg["parts"].items():
        template_path = REPO_ROOT / scfg["template_dir"] / f"itodd_{part}.ply"
        pcd, d, _ = preprocess_cloud(load_cloud(template_path), cfg)
        pts_full = np.asarray(pcd.points)
        axis, info = detect_symmetry_axis(pts_full, d)
        entry = {"d": d, "axis": None, "center": info["center"], "thetas": None,
                 "fold_order": 0 if sym_class == "axisymmetric" else 1}
        if sym_class != "axisymmetric":
            n, _ = detect_discrete_fold(pts_full, axis, info["center"], d)
            if n:
                entry["axis"] = axis
                entry["thetas"] = np.arange(n) * (360.0 / n)
                entry["fold_order"] = n
        else:
            entry["axis"] = axis
            entry["fold_order"] = 0

        rng = stable_rng("bop_metrics", part)
        n_sub = min(N_SUBSAMPLE, len(pts_full))
        idx = rng.choice(len(pts_full), size=n_sub, replace=False)
        entry["pts"] = pts_full[idx]
        part_model[part] = entry
        print(f"{part} ({sym_class}): fold_order={entry['fold_order']} "
              f"(0 = continuous axis, 1 = trivial), n_sub={n_sub}")

    add_rel, add_ok, mssd_rel_col, mssd_ok = [], [], [], []
    for row in df.itertuples(index=False):
        r = row._asdict()
        pm = part_model[r["part"]]
        T_est, T_gt = _T_from_row(r, "est"), _T_from_row(r, "gt")
        a = add_s(pm["pts"], T_est, T_gt, pm["d"])
        m = mssd(pm["pts"], T_est, T_gt, pm["d"], pm["axis"], pm["center"], pm["thetas"])
        add_rel.append(a)
        add_ok.append(a < ADD_THRESHOLD_REL)
        mssd_rel_col.append(m)
        mssd_ok.append(m < MSSD_THRESHOLD_REL)

    df["add_s_rel"] = add_rel
    df["add_s_success"] = add_ok
    df["mssd_rel"] = mssd_rel_col
    df["mssd_success"] = mssd_ok
    df.to_csv(tables / "bop_metrics.csv", index=False)

    order = [s for s in cfg["severities"] if s in set(df["severity"])]

    def rate_ci(sub: pd.DataFrame, col: str) -> str:
        k, n = int(sub[col].sum()), len(sub)
        if n == 0:
            return "-"
        lo, hi = wilson_ci(k, n)
        return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"

    def part_table(col: str) -> str:
        rows_md = []
        for (sym_class, part), g in df.groupby(["symmetry_class", "part"], sort=False):
            cells = [rate_ci(g[g["severity"] == s], col) for s in order]
            rows_md.append([f"{sym_class} / {part}", *cells])
        return format_table(["symmetry_class / part", *order], rows_md)

    # per-trial agreement with success_sym
    add_agree = int((df["add_s_success"] == df["success_sym"]).sum())
    mssd_agree = int((df["mssd_success"] == df["success_sym"]).sum())
    n = len(df)

    # rank agreement over (part, severity) cells, independent of the thresholds
    cell = df.groupby(["part", "severity"], sort=False).agg(
        success_sym=("success_sym", "mean"),
        add_s_success=("add_s_success", "mean"),
        mssd_success=("mssd_success", "mean")).reset_index()
    rho_add, p_add = spearmanr(cell["success_sym"], cell["add_s_success"])
    rho_mssd, p_mssd = spearmanr(cell["success_sym"], cell["mssd_success"])

    lines = [
        "# BOP-style metrics (ADD-S, MSSD) on the symmetry-sweep poses",
        "",
        f"ADD-S and MSSD on a fixed {N_SUBSAMPLE}-point subsample of each part's "
        "preprocessed template, thresholded at 0.1 x AABB diagonal; MSSD minimizes "
        "over the part's detected symmetry group. MSPD is not computed (it needs a "
        "camera model).",
        "",
        "## Success rates (%) by part: ADD-S < 0.1d",
        "", part_table("add_s_success"), "",
        "## Success rates (%) by part: MSSD < 0.1d (symmetry-aware)",
        "", part_table("mssd_success"), "",
        "## Agreement with the RRE/RTE symmetry-aware criterion (success_sym)",
        "",
        f"Per-trial success agreement, ADD-S vs success_sym: {add_agree}/{n} "
        f"({100 * add_agree / n:.1f}%).",
        f"Per-trial success agreement, MSSD vs success_sym: {mssd_agree}/{n} "
        f"({100 * mssd_agree / n:.1f}%).",
        "",
        f"Spearman rho of cell success rates across the {len(cell)} (part, severity) cells:",
        f"success_sym vs ADD-S: rho={rho_add:.3f}, p={p_add:.4f}. "
        f"success_sym vs MSSD: rho={rho_mssd:.3f}, p={p_mssd:.4f}.",
        "",
    ]
    (tables / "bop_metrics.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
