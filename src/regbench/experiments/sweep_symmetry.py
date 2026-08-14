"""Registration success versus part symmetry.

Runs the registration grid on each ITODD part in config `symmetry_sweep`,
labeled asymmetric or axisymmetric, and scores every pose with both the plain
and the symmetry-aware metric. Poses are stored so other metrics can be
computed offline (rescore.py, bop.py).

    regbench sweep-symmetry [--smoke] [--plot]

Outputs (tables_dir): symmetry_sweep.csv, symmetry_scores.csv,
branch_consistency.csv, symmetry_success_rates.md.
Figure (figures_dir): symmetry_sweep.png; --plot redraws it from the CSV.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from regbench import style
from regbench.cloud_io import load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table
from regbench.methods import prepare_target, register
from regbench.metrics import evaluate
from regbench.perturb import make_trial
from regbench.repro import stable_rng
from regbench.symmetry import branch_consistency, detect_symmetry_axis, folded_evaluate, \
    wilson_ci

FIELDS = ["part", "symmetry_class", "method", "severity", "trial", "rre_deg", "rte",
          "success", "rre_sym_deg", "rte_sym", "success_sym", "theta_deg", "template",
          "d", "voxel", "n_target", "n_source",
          "est_qx", "est_qy", "est_qz", "est_qw", "est_tx", "est_ty", "est_tz",
          "gt_qx", "gt_qy", "gt_qz", "gt_qw", "gt_tx", "gt_ty", "gt_tz"]


def _pose_cols(prefix: str, T: np.ndarray) -> dict:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # (x, y, z, w)
    return {f"{prefix}_q{k}": round(float(v), 9) for k, v in zip("xyzw", q)} | \
           {f"{prefix}_t{k}": round(float(v), 9) for k, v in zip("xyz", T[:3, 3])}


def run_sweep(cfg: dict, n_trials: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the sweep in memory: (one row per registration, one row per part)."""
    scfg = cfg["symmetry_sweep"]

    rows = []
    score_rows = []
    for part, sym_class in scfg["parts"].items():
        template_path = REPO_ROOT / scfg["template_dir"] / f"itodd_{part}.ply"
        target_pcd, d, voxel = preprocess_cloud(load_cloud(template_path), cfg)
        target = prepare_target(target_pcd, voxel, cfg["registration"])
        target_pts = np.asarray(target.pcd.points)
        # axis and score for every part; the fold applies to axisymmetric parts only
        axis, info = detect_symmetry_axis(target_pts, d)
        fold_axis = axis if sym_class == "axisymmetric" else None
        score_rows.append({
            "part": part, "symmetry_class": sym_class,
            "sym_score": round(info["sym_score"], 6),
            "sym_score_min": round(info["sym_score_min"], 6),
            "axis_x": round(float(axis[0]), 6), "axis_y": round(float(axis[1]), 6),
            "axis_z": round(float(axis[2]), 6),
            "probe_overlaps": "/".join(f"{v:.5f}" for v in info["probe_overlaps"]),
            "d": round(d, 6), "n_target": len(target_pts)})
        print(f"{part} ({sym_class}): n_target={len(target_pts)} d={d:.4f} "
              f"sym_score={info['sym_score']:.4f}")

        for severity, sev in cfg["severities"].items():
            for trial in range(n_trials):
                source_pts, T_gt = make_trial(
                    target_pts, sev, d, stable_rng("symmetry", part, severity, trial))
                for method in cfg["methods"]:
                    T_est = register(method, source_pts, target, cfg["registration"],
                                     trial_seed=trial)
                    rows.append({
                        "part": part, "symmetry_class": sym_class, "method": method,
                        "severity": severity, "trial": trial,
                        **evaluate(T_est, T_gt, d, cfg["thresholds"]),
                        **folded_evaluate(T_est, T_gt, d, cfg["thresholds"],
                                          fold_axis, info["center"]),
                        "template": template_path.name, "d": round(d, 6),
                        "voxel": round(voxel, 6), "n_target": len(target_pts),
                        "n_source": len(source_pts),
                        **_pose_cols("est", T_est), **_pose_cols("gt", T_gt),
                    })
        done = [r for r in rows if r["part"] == part]
        print(f"{part}: {len(done)} registrations done")

    return pd.DataFrame(rows, columns=FIELDS), pd.DataFrame(score_rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="n_trials=3, voxel d/50")
    ap.add_argument("--plot", action="store_true",
                    help="only redraw symmetry_sweep.png from symmetry_sweep.csv")
    args = ap.parse_args(argv)

    if args.plot:
        cfg = load_config()
        plot(cfg)
        return

    cfg = load_config()
    scfg = cfg["symmetry_sweep"]
    n_trials = 3 if args.smoke else scfg["n_trials"]
    if args.smoke:
        cfg["preprocess"]["voxel_divisor"] = 50

    df, scores = run_sweep(cfg, n_trials)
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "symmetry_sweep.csv", index=False)
    scores.to_csv(tables / "symmetry_scores.csv", index=False)

    order = list(cfg["severities"])

    def rate_ci(sub: pd.DataFrame, col: str) -> str:
        k, n = int(sub[col].sum()), len(sub)
        lo, hi = wilson_ci(k, n)
        return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"

    def ci_table(index_cols: list[str], col: str, index_label: str) -> str:
        idx_order = [tuple(t) for t in
                     df[index_cols].drop_duplicates().itertuples(index=False)]
        header = [index_label, *order]
        rows_md = []
        for idx in idx_order:
            mask = np.ones(len(df), dtype=bool)
            for c, v in zip(index_cols, idx):
                mask &= (df[c] == v).to_numpy()
            g = df[mask]
            cells = [rate_ci(g[g["severity"] == s], col) for s in order]
            rows_md.append([" / ".join(idx), *cells])
        return format_table(header, rows_md)

    score_by_part = scores.set_index("part")["sym_score"]
    succ_by_part = df.groupby("part", sort=False)["success"].mean()
    rho = succ_by_part.rank().corr(
        score_by_part.reindex(succ_by_part.index).rank())  # Spearman

    # branch consistency of the axisymmetric parts, pooled and per method
    axisym = df[df["symmetry_class"] == "axisymmetric"]
    bc_rows = []
    for (part, severity), g in axisym.groupby(["part", "severity"], sort=False):
        bc_rows.append({"part": part, "severity": severity, "method": "(all methods)",
                        **branch_consistency(g["theta_deg"].to_numpy())})
        for method, gm in g.groupby("method", sort=False):
            bc_rows.append({"part": part, "severity": severity, "method": method,
                            **branch_consistency(gm["theta_deg"].to_numpy())})
    bc_df = pd.DataFrame(bc_rows)
    bc_df.to_csv(tables / "branch_consistency.csv", index=False)
    bc_table = format_table(
        ["part", "severity", "method", "entropy_norm", "mode_frequency", "n_bins", "n"],
        [[r.part, r.severity, r.method, f"{r.entropy_norm:.3f}", f"{r.mode_frequency:.3f}",
          r.n_bins, r.n] for r in bc_df.itertuples()])

    lines = ["# Success rates (%) by symmetry class and severity", "",
             "Averaged over all 4 methods; 95% Wilson intervals in brackets.", "",
             "## Plain metric (success := RRE < "
             f"{cfg['thresholds']['rre_deg']} deg AND RTE < "
             f"{cfg['thresholds']['rte_rel']} of d)", "",
             ci_table(["symmetry_class"], "success", "symmetry_class"), "",
             "## Symmetry-aware metric (same thresholds, error modulo the part's",
             "rotation-symmetry axis for the axisymmetric class; asymmetric rows",
             "unchanged by construction)", "",
             ci_table(["symmetry_class"], "success_sym", "symmetry_class"), "",
             "## By part (plain | symmetry-aware)", "",
             ci_table(["symmetry_class", "part"], "success",
                      "symmetry_class / part"), "",
             ci_table(["symmetry_class", "part"], "success_sym",
                      "symmetry_class / part"), "",
             "## Continuous symmetry score (rotational self-overlap, low = symmetric)",
             "",
             format_table(["part", "class", "sym_score", "sym_score_min"],
                          [[r.part, r.symmetry_class, f"{r.sym_score:.4f}",
                            f"{r.sym_score_min:.4f}"] for r in scores.itertuples()]),
             "",
             f"Spearman rank correlation, sym_score vs overall per-part success "
             f"rate (plain, n=6 parts): rho = {rho:.3f}", "",
             "## Branch consistency (axisymmetric parts only): does the estimate",
             "keep reaching the SAME symmetry-group branch across independent",
             "trials, or an effectively random one? theta_deg is the branch",
             "folded_evaluate matched T_est to, relative to T_gt (0 = the true",
             "orientation); entropy_norm/mode_frequency summarize its spread over",
             "trials (0/1 = fully consistent, 1/1-of-n_bins = indistinguishable",
             "from a uniform-random branch draw).", "",
             bc_table, "",
             f"{n_trials} trials/severity/part", ""]
    (tables / "symmetry_success_rates.md").write_text("\n".join(lines))
    print("\n".join(lines))
    plot(cfg)


def plot(cfg: dict) -> None:
    """symmetry_sweep.png from symmetry_sweep.csv: success by class (bars) and by part (lines)."""
    csv = REPO_ROOT / cfg["paths"]["tables_dir"] / "symmetry_sweep.csv"
    if not csv.exists():
        raise SystemExit("run `regbench sweep-symmetry` first")
    df = pd.read_csv(csv)
    order = [s for s in cfg["severities"] if s in set(df["severity"])]
    class_colors = {"asymmetric": style.GREEN, "axisymmetric": style.RED}

    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.FIGSIZE_2PANEL)

    width = 0.2
    x = np.arange(len(order))
    has_sym = "success_sym" in df.columns
    for i, (sym_class, color) in enumerate(class_colors.items()):
        sub = df[df["symmetry_class"] == sym_class]
        rates = [sub[sub["severity"] == s]["success"].mean() * 100 for s in order]
        off = (2 * i - 1.5) * width if has_sym else (i - 0.5) * 0.35
        ax1.bar(x + off, rates, width if has_sym else 0.35,
                label=f"{sym_class} (plain)", color=color)
        if has_sym:
            rates_sym = [sub[sub["severity"] == s]["success_sym"].mean() * 100
                         for s in order]
            ax1.bar(x + off + width, rates_sym, width,
                    label=f"{sym_class} (sym-aware)", color=color,
                    alpha=0.45, hatch="//")
    style.style_success_axis(ax1, xlabel="severity preset", ylabel="success rate (%)",
                       title="By symmetry class (all methods pooled)",
                       xticks=(x, order))
    ax1.legend(fontsize=7)

    for part, g in df.groupby("part", sort=False):
        sym_class = g["symmetry_class"].iloc[0]
        rates = [g[g["severity"] == s]["success"].mean() * 100 for s in order]
        ls = "-" if sym_class == "asymmetric" else "--"
        ax2.plot(order, rates, label=part, color=class_colors[sym_class],
                 linestyle=ls, marker="o", linewidth=1.5, alpha=0.85)
    style.style_success_axis(ax2, xlabel="severity preset", title="By part")
    ax2.legend(fontsize=8)

    fig.suptitle("MVTec ITODD symmetry sweep: registration success vs part symmetry",
                y=1.02, fontsize=10, color=style.GREY)
    fig.tight_layout()
    out = REPO_ROOT / cfg["paths"]["figures_dir"] / "symmetry_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
