"""Offline re-score of the symmetry sweep with discrete C_n folds.

A part labeled asymmetric can still have a discrete rotational symmetry (the
star is C12). For each asymmetric-labeled part, symmetry.detect_discrete_fold
looks for a fold and the stored poses are re-scored over that C_n set;
axisymmetric parts keep the continuous fold. The plain and
continuous-fold columns are recomputed from the stored poses as a cross-check.
No registration is run.

    regbench rescore

Outputs (tables_dir): symmetry_sweep_rescored.csv (sweep rows + fold_order,
rre_dsym_deg, rte_dsym, success_dsym, theta_dsym_deg),
branch_consistency_discrete.csv, symmetry_success_rates_discrete.md.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from regbench.cloud_io import load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table
from regbench.symmetry import (branch_consistency, detect_discrete_fold,
                               detect_symmetry_axis, folded_evaluate, wilson_ci)


def _T_from_row(row, prefix: str) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(
        [row[f"{prefix}_q{k}"] for k in "xyzw"]).as_matrix()
    T[:3, 3] = [row[f"{prefix}_t{k}"] for k in "xyz"]
    return T


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser().parse_args(argv)  # no flags; enables `--help`
    cfg = load_config()
    scfg = cfg["symmetry_sweep"]
    thr = cfg["thresholds"]
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    df = pd.read_csv(tables / "symmetry_sweep.csv")

    # Per-part symmetry model, from the same preprocessed template the sweep used.
    part_fold: dict[str, dict] = {}
    diag_rows = []
    for part, sym_class in scfg["parts"].items():
        template_path = REPO_ROOT / scfg["template_dir"] / f"itodd_{part}.ply"
        pcd, d, _ = preprocess_cloud(load_cloud(template_path), cfg)
        pts = np.asarray(pcd.points)
        axis, info = detect_symmetry_axis(pts, d)
        entry = {"axis": axis, "center": info["center"], "thetas": None,
                 "fold_order": 0 if sym_class == "axisymmetric" else 1}
        diag = {"part": part, "class": sym_class, "order": "cont.",
                "floor": "", "median": "", "threshold": "", "worst_mult": ""}
        if sym_class != "axisymmetric":
            n, dinfo = detect_discrete_fold(pts, axis, info["center"], d)
            diag.update(order=(n if n else "-"),
                        floor=f"{dinfo['floor']:.4f}",
                        median=f"{dinfo['median']:.4f}",
                        threshold=f"{dinfo['threshold']:.4f}",
                        worst_mult=(f"{dinfo['fold_overlap_max']:.4f}"
                                    if n else "-"))
            if n:
                entry["thetas"] = np.arange(n) * (360.0 / n)
                entry["fold_order"] = n
            else:
                entry["axis"] = None  # trivial group: dsym = plain
        part_fold[part] = entry
        diag_rows.append(diag)
        print(f"{part} ({sym_class}): fold_order={entry['fold_order']}"
              f" (0 = continuous axis)")

    # Re-score every stored pose pair.
    plain_mismatch = cont_mismatch = 0
    out = {"fold_order": [], "rre_dsym_deg": [], "rte_dsym": [], "success_dsym": [],
           "theta_dsym_deg": []}
    for row in df.itertuples(index=False):
        r = row._asdict()
        pf = part_fold[r["part"]]
        T_est, T_gt = _T_from_row(r, "est"), _T_from_row(r, "gt")
        plain = folded_evaluate(T_est, T_gt, r["d"], thr, None, np.zeros(3))
        plain_mismatch += int(plain["success_sym"] != bool(r["success"]))
        # one call covers all three cases: axis=None -> plain, thetas=None ->
        # dense continuous grid, thetas set -> discrete C_n fold
        res = folded_evaluate(T_est, T_gt, r["d"], thr, pf["axis"],
                              pf["center"], thetas_deg=pf["thetas"])
        if pf["fold_order"] == 0:  # axisymmetric: cross-check vs stored column
            cont_mismatch += int(res["success_sym"] != bool(r["success_sym"]))
        out["fold_order"].append(pf["fold_order"])
        out["rre_dsym_deg"].append(res["rre_sym_deg"])
        out["rte_dsym"].append(res["rte_sym"])
        out["success_dsym"].append(res["success_sym"])
        out["theta_dsym_deg"].append(res["theta_deg"])
    for k, v in out.items():
        df[k] = v
    df.to_csv(tables / "symmetry_sweep_rescored.csv", index=False)
    print(f"cross-check: plain metric mismatches {plain_mismatch}/{len(df)}, "
          f"continuous-fold mismatches {cont_mismatch}")

    # branch consistency under each part's full detected group
    bc_rows = []
    for part, pf in part_fold.items():
        if pf["fold_order"] == 1:  # trivial group: no branch structure to measure
            continue
        group_order = pf["fold_order"] if pf["fold_order"] >= 2 else None
        sub = df[df["part"] == part]
        for severity, g in sub.groupby("severity", sort=False):
            bc_rows.append({"part": part, "fold_order": pf["fold_order"],
                            "severity": severity,
                            **branch_consistency(g["theta_dsym_deg"].to_numpy(),
                                                 group_order=group_order)})
    bc_df = pd.DataFrame(bc_rows)
    if len(bc_df):
        bc_df.to_csv(tables / "branch_consistency_discrete.csv", index=False)

    order = [s for s in cfg["severities"] if s in set(df["severity"])]

    def rate_ci(sub: pd.DataFrame, col: str) -> str:
        k, n = int(sub[col].sum()), len(sub)
        lo, hi = wilson_ci(k, n)
        return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"

    def part_table(col: str) -> str:
        rows_md = []
        for (sym_class, part), g in df.groupby(
                ["symmetry_class", "part"], sort=False):
            fold = {0: "cont.", 1: "-"}.get(part_fold[part]["fold_order"],
                                            f"C{part_fold[part]['fold_order']}")
            cells = [rate_ci(g[g["severity"] == s], col) for s in order]
            rows_md.append([f"{sym_class} / {part} / {fold}", *cells])
        return format_table(["symmetry_class / part / fold", *order], rows_md)

    diag_table = format_table(
        ["part", "class", "order", "curve floor", "curve median",
         "accept threshold", "worst multiple"],
        [[r["part"], r["class"], r["order"], r["floor"], r["median"],
          r["threshold"], r["worst_mult"]] for r in diag_rows])
    bc_table = format_table(
        ["part", "fold_order", "severity", "entropy_norm", "mode_frequency",
         "n_bins", "n"],
        [[r.part, r.fold_order, r.severity, f"{r.entropy_norm:.3f}",
          f"{r.mode_frequency:.3f}", r.n_bins, r.n] for r in bc_df.itertuples()])

    lines = ["# Discrete-fold re-score (offline, from symmetry_sweep.csv poses)",
             "",
             "success_dsym := success under the part's full detected symmetry "
             "group: discrete C_n fold",
             "for asymmetric-labeled parts with a detected discrete order, "
             "continuous axis fold for the",
             "axisymmetric class (unchanged from success_sym), plain metric "
             "otherwise. Same thresholds",
             f"(RRE < {thr['rre_deg']} deg AND RTE < {thr['rte_rel']} of d); "
             "95% Wilson intervals in brackets.",
             "",
             "## Detected discrete folds (asymmetric-labeled parts)",
             "",
             diag_table,
             "",
             "## Success rates (%) by part: plain metric",
             "", part_table("success"), "",
             "## Continuous-fold metric (success_sym, as swept)",
             "", part_table("success_sym"), "",
             "## Full-group metric (success_dsym, this re-score)",
             "", part_table("success_dsym"), "",
             "## Branch consistency under the full detected group (continuous or",
             "C_n): does the estimate keep reaching the SAME branch across",
             "trials, or an effectively random one? theta_dsym_deg = branch",
             "matched relative to T_gt (0 = true orientation); entropy_norm/",
             "mode_frequency summarize its spread (0/1 = fully consistent,",
             "1/1-of-n_bins = indistinguishable from a uniform-random branch",
             "draw).", "",
             bc_table,
             "",
             f"Cross-check against stored columns: plain mismatches "
             f"{plain_mismatch}/{len(df)}, continuous-fold mismatches "
             f"{cont_mismatch}.", ""]
    (tables / "symmetry_success_rates_discrete.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
