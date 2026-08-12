"""Registration under clutter (config `clutter`).

Only the target part is perturbed by the known T_gt; ``n_distractors`` other
ITODD parts, each with its own random pose, are added to the same source cloud.
Every method registers the combined cloud against the target template.

    regbench clutter [--smoke] [--plot]

Outputs (tables_dir): clutter_sweep.csv (one row per registration, with the
fitness diagnostics) and clutter_success_rates.md.
Figure (figures_dir): clutter_sweep.png; --plot redraws it from the CSV.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regbench import style
from regbench.cloud_io import load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table
from regbench.methods import prepare_target, register
from regbench.metrics import evaluate
from regbench.perturb import make_trial, random_rotation
from regbench.repro import stable_rng
from regbench.symmetry import wilson_ci

FIELDS = ["target_part", "n_distractors", "method", "severity", "trial", "rre_deg",
          "rte", "success", "fitness", "inlier_rmse", "n_inliers", "template", "d",
          "voxel", "n_target", "n_source"]


def load_distractors(ccfg: dict) -> dict[str, np.ndarray]:
    """part -> (distractor_points, 3) raw points, centered on their own centroid."""
    out = {}
    for part in ccfg["distractor_parts"]:
        pcd = load_cloud(REPO_ROOT / ccfg["template_dir"] / f"itodd_{part}.ply")
        pts = np.asarray(pcd.points)
        pts = pts - pts.mean(axis=0)
        rng = stable_rng("clutter-load", part)
        n = ccfg["distractor_points"]
        idx = rng.choice(len(pts), n, replace=False) if len(pts) > n \
            else np.arange(len(pts))
        out[part] = pts[idx]
    return out


def scatter_distractors(n: int, distractors: dict[str, np.ndarray], d: float,
                        ccfg: dict, rng: np.random.Generator) -> np.ndarray:
    """n distractor clouds, each randomly rotated and placed offset_rel * d from the origin."""
    if n == 0:
        return np.empty((0, 3))
    names = list(distractors)
    lo, hi = ccfg["offset_rel"]
    chunks = []
    for _ in range(n):
        pts = distractors[names[rng.integers(len(names))]]
        R = random_rotation(rng, 180.0)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        offset = direction * rng.uniform(lo, hi) * d
        chunks.append(pts @ R.T + offset)
    return np.concatenate(chunks, axis=0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="n_trials=3, voxel d/50")
    ap.add_argument("--plot", action="store_true",
                    help="only redraw clutter_sweep.png from clutter_sweep.csv")
    args = ap.parse_args(argv)

    cfg = load_config()
    ccfg = cfg["clutter"]

    if args.plot:
        csv = REPO_ROOT / cfg["paths"]["tables_dir"] / "clutter_sweep.csv"
        if not csv.exists():
            raise SystemExit("run `regbench clutter` first")
        plot(cfg, pd.read_csv(csv))
        return
    n_trials = 3 if args.smoke else ccfg["n_trials"]
    if args.smoke:
        cfg["preprocess"]["voxel_divisor"] = 50

    template_path = REPO_ROOT / ccfg["template_dir"] / f"itodd_{ccfg['target_part']}.ply"
    target_pcd, d, voxel = preprocess_cloud(load_cloud(template_path), cfg)
    target = prepare_target(target_pcd, voxel, cfg["registration"])
    target_pts = np.asarray(target.pcd.points)
    distractors = load_distractors(ccfg)
    sev = cfg["severities"][ccfg["severity"]]
    print(f"target={ccfg['target_part']} n_target={len(target_pts)} d={d:.4f} "
          f"distractors={list(distractors)}")

    rows = []
    for n_dist in ccfg["n_distractors"]:
        for trial in range(n_trials):
            source_pts, T_gt = make_trial(
                target_pts, sev, d, stable_rng("clutter", n_dist, trial))
            clutter_rng = stable_rng("clutter-scatter", n_dist, trial)
            clutter_pts = scatter_distractors(n_dist, distractors, d, ccfg, clutter_rng)
            scene_pts = np.concatenate([source_pts, clutter_pts], axis=0)
            for method in cfg["methods"]:
                T_est, diag = register(method, scene_pts, target, cfg["registration"],
                                       trial_seed=trial, return_diag=True)
                rows.append({
                    "target_part": ccfg["target_part"], "n_distractors": n_dist,
                    "method": method, "severity": ccfg["severity"], "trial": trial,
                    **evaluate(T_est, T_gt, d, cfg["thresholds"]),
                    "fitness": round(diag["fitness"], 6),
                    "inlier_rmse": "" if np.isnan(diag["inlier_rmse"])
                    else round(diag["inlier_rmse"], 6),
                    "n_inliers": diag["n_inliers"], "template": template_path.name,
                    "d": round(d, 6), "voxel": round(voxel, 6),
                    "n_target": len(target_pts), "n_source": len(scene_pts),
                })
        done = [r for r in rows if r["n_distractors"] == n_dist]
        print(f"n_distractors={n_dist}: {len(done)} registrations done")

    df = pd.DataFrame(rows, columns=FIELDS)
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "clutter_sweep.csv", index=False)

    def rate_ci(sub: pd.DataFrame) -> str:
        k, n = int(sub["success"].sum()), len(sub)
        lo, hi = wilson_ci(k, n)
        return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"

    n_dists = ccfg["n_distractors"]
    method_table = format_table(
        ["method", *[f"n={n}" for n in n_dists]],
        [[method, *(rate_ci(df[(df["method"] == method) & (df["n_distractors"] == n)])
                    for n in n_dists)] for method in cfg["methods"]])

    pooled_rows = []
    for n in n_dists:
        sub = df[df["n_distractors"] == n]
        k, ntot = int(sub["success"].sum()), len(sub)
        lo, hi = wilson_ci(k, ntot)
        ok_fit = sub.loc[sub["success"], "fitness"]
        bad_fit = sub.loc[~sub["success"], "fitness"]
        pooled_rows.append([n, f"{100 * k / ntot:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]",
                            f"{ok_fit.mean():.3f}", f"{bad_fit.mean():.3f}"])
    pooled_table = format_table(
        ["n_distractors", "success (%)", "mean fitness (successes)",
         "mean fitness (failures)"], pooled_rows)

    lines = ["# Clutter stress test: success rate (%) vs. distractor count", "",
             f"target = {ccfg['target_part']} | severity = {ccfg['severity']} | "
             "95% Wilson intervals in brackets", "",
             method_table,
             "", "## Pooled over methods", "",
             pooled_table,
             "", f"{n_trials} trials/(n_distractors x method)", ""]
    md = "\n".join(lines)
    (tables / "clutter_success_rates.md").write_text(md)
    print(md)
    plot(cfg, df)


def plot(cfg: dict, df: pd.DataFrame) -> None:
    style.apply()
    fig, ax = plt.subplots(figsize=style.FIGSIZE_1PANEL)
    n_dists = sorted(df["n_distractors"].unique())
    for method in cfg["methods"]:
        rates = [df[(df["method"] == method) & (df["n_distractors"] == n)]
                ["success"].mean() * 100 for n in n_dists]
        ax.plot(n_dists, rates, label=method, color=style.METHOD_COLORS[method],
               marker=style.METHOD_MARKERS[method], linewidth=2)
    style.style_success_axis(ax, xlabel="number of distractor objects in the scene",
                       ylabel="success rate (%)",
                       title=f"Registration under clutter ({df['target_part'].iloc[0]}, "
                             f"{df['severity'].iloc[0]} severity)",
                       xticks=n_dists)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = REPO_ROOT / cfg["paths"]["figures_dir"] / "clutter_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
