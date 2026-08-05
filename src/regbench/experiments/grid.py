"""Registration grid: every method on the same perturbed sources.

For each severity and trial one source is built from the template (seeded by
severity and trial) and all methods register it, so methods are compared on
identical inputs. Default: 4 methods x 3 severities x 50 trials.

    regbench grid [--smoke] [--template PATH --tag NAME]

Outputs (tables_dir): registration_grid.csv (one row per registration) and
success_rates.md.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from regbench.cloud_io import load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.methods import prepare_target, register
from regbench.metrics import evaluate
from regbench.mdtable import format_table
from regbench.perturb import make_trial
from regbench.repro import cpu_info, stable_rng

FIELDS = ["method", "severity", "trial", "rre_deg", "rte", "success", "time_s",
          "fitness", "inlier_rmse", "n_inliers", "template", "d", "voxel",
          "n_target", "n_source", "n_trials", "cpu"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="n_trials=3, voxel d/50")
    ap.add_argument("--template", default=None,
                    help="override paths.template (any .ply/.pcd under the repo root)")
    ap.add_argument("--tag", default="",
                    help="output filename suffix, e.g. --tag bunny writes "
                         "registration_grid_bunny.csv")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.smoke:
        cfg["n_trials"] = 3
        cfg["preprocess"]["voxel_divisor"] = 50
    if args.template:
        cfg["paths"]["template"] = args.template

    suffix = f"_{args.tag}" if args.tag else ""
    template_path = REPO_ROOT / cfg["paths"]["template"]
    target_pcd, d, voxel = preprocess_cloud(load_cloud(template_path), cfg)
    target = prepare_target(target_pcd, voxel, cfg["registration"])
    target_pts = np.asarray(target.pcd.points)
    cpu = cpu_info()
    print(f"template={template_path.name} n_target={len(target_pts)} "
          f"d={d:.4f} voxel={voxel:.5f} cpu={cpu}")

    rows = []
    for severity, sev in cfg["severities"].items():
        for trial in range(cfg["n_trials"]):
            source_pts, T_gt = make_trial(target_pts, sev, d,
                                          stable_rng("trial", severity, trial))
            for method in cfg["methods"]:
                t0 = time.perf_counter()
                T_est, diag = register(method, source_pts, target, cfg["registration"],
                                       trial_seed=trial, return_diag=True)
                elapsed = time.perf_counter() - t0
                rows.append({
                    "method": method, "severity": severity, "trial": trial,
                    **evaluate(T_est, T_gt, d, cfg["thresholds"]),
                    "time_s": round(elapsed, 4),
                    "fitness": round(diag["fitness"], 6),
                    "inlier_rmse": "" if np.isnan(diag["inlier_rmse"])
                    else round(diag["inlier_rmse"], 6),
                    "n_inliers": diag["n_inliers"], "template": template_path.name,
                    "d": round(d, 6), "voxel": round(voxel, 6),
                    "n_target": len(target_pts), "n_source": len(source_pts),
                    "n_trials": cfg["n_trials"], "cpu": cpu,
                })
        done = [r for r in rows if r["severity"] == severity]
        print(f"{severity}: {len(done)} registrations done")

    df = pd.DataFrame(rows, columns=FIELDS)
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / f"registration_grid{suffix}.csv", index=False)

    order = list(cfg["severities"])
    pivot = (df.pivot_table(index="method", columns="severity", values="success",
                            aggfunc="mean")[order] * 100).round(1)
    extras = df.groupby("method").agg(mean_rre_deg=("rre_deg", "mean"),
                                      mean_time_s=("time_s", "mean")).round(3)
    table = pivot.join(extras)
    header = ["method", *table.columns]
    rows_md = [[method, *(str(v) for v in row)] for method, row in table.iterrows()]
    md = ["# Success rates (%) by severity - " + template_path.name,
          "", format_table(header, rows_md), "",
          f"success := RRE < {cfg['thresholds']['rre_deg']} deg AND "
          f"RTE < {cfg['thresholds']['rte_rel']} (of d) | "
          f"{cfg['n_trials']} trials/severity", ""]
    (tables / f"success_rates{suffix}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
