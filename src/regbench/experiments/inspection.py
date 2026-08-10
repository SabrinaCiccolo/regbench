"""Template-difference inspection on the inspection datasets.

    regbench inspection [--smoke]   # register and score every test scan
    regbench inspection --plot      # redraw inspect_scores.png from the CSVs

Per part (config `inspect`) one train_good scan is the template; tau is
calibrated on the other registered train_good scans. Each test scan is
registered onto the template and scored by a high quantile of its
point-to-template distance. Sidecars add the true-pose error and the
point-level AUROC.

Outputs: inspect_{synthetic,mvtec}.csv and inspect_summary.md (tables_dir);
inspect/<part>/ heatmaps, <part>_gallery.png and inspect_scores.png
(figures_dir).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regbench import style
from regbench.anomaly import auroc, calibrate_tau, load_sidecar, point_distances, \
    scan_score
from regbench.cloud_io import iter_industrial, load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table
from regbench.methods import Prepared, prepare_target, register
from regbench.metrics import evaluate
from regbench.repro import stable_rng

FIELDS = ["part", "track", "split", "defect_type", "scan", "method", "n_points",
          "score", "tau", "rre_deg", "rte", "reg_success", "point_auroc", "time_s",
          "template", "d", "voxel", "score_quantile", "tau_quantile"]


def track_of(part: str) -> str:
    return "synthetic" if part.startswith("synth_") else "mvtec"


def load_scan(path: Path, cfg: dict) -> np.ndarray:
    """Scan points at full resolution; optional outlier removal breaks sidecar alignment."""
    pcd = load_cloud(path)
    if cfg["inspect"]["outlier_removal"]:
        pre = cfg["preprocess"]
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=pre["outlier_nb_neighbors"], std_ratio=pre["outlier_std_ratio"])
    return np.asarray(pcd.points)


def register_scan(pts: np.ndarray, target: Prepared, cfg: dict, seed_key: tuple,
                  ) -> tuple[np.ndarray, float]:
    seed = int(stable_rng(*seed_key).integers(2 ** 31))
    t0 = time.perf_counter()
    T_est = register(cfg["inspect"]["method"], pts, target, cfg["registration"], seed)
    return T_est, time.perf_counter() - t0


def heatmap_png(pts: np.ndarray, dist: np.ndarray, d: float, title: str, out: Path,
                rng: np.random.Generator, n_max: int, vmax: float) -> None:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    idx = rng.choice(len(pts), n_max, replace=False) if len(pts) > n_max \
        else np.arange(len(pts))
    sc = ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], s=2.0, c=dist[idx] / d,
                    cmap="viridis", vmin=0.0, vmax=vmax, rasterized=True)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02).set_label(
        "distance to template (fraction of d)")
    ax.set_title(title, fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def gallery_png(panels: list[tuple[str, np.ndarray, np.ndarray]], d: float,
                part: str, out: Path, rng: np.random.Generator, n_max: int,
                vmax: float) -> None:
    """Aligned heatmaps, good scans on the first row and defect scans on the second."""
    ncol = max(1, len(panels) // 2)
    fig = plt.figure(figsize=(3.2 * ncol, 7))
    for i, (title, pts, dist) in enumerate(panels):
        ax = fig.add_subplot(2, ncol, i + 1, projection="3d")
        idx = rng.choice(len(pts), n_max, replace=False) if len(pts) > n_max \
            else np.arange(len(pts))
        sc = ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], s=1.5, c=dist[idx] / d,
                        cmap="viridis", vmin=0.0, vmax=vmax, rasterized=True)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)
    fig.colorbar(sc, ax=fig.axes, shrink=0.5, pad=0.02).set_label(
        "distance to template (fraction of d)")
    fig.suptitle(f"{part}: aligned test scans, distance-to-template heatmaps")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_part(part: str, cfg: dict, smoke: bool, render: bool = True) -> list[dict]:
    icfg = cfg["inspect"]
    root = REPO_ROOT / cfg["paths"]["industrial_root"]
    template_ply = root / part / f"{icfg['template']}.ply"
    if not template_ply.exists():
        print(f"{part}: SKIPPED ({template_ply} missing - build it first)")
        return []
    template_pcd, d, voxel = preprocess_cloud(load_cloud(template_ply), cfg,
                                              recenter=False)
    target = prepare_target(template_pcd, voxel, cfg["registration"])
    template_pts = np.asarray(target.pcd.points)

    def cap(paths: list[Path]) -> list[Path]:
        return paths[:3] if smoke else paths

    # tau from the registered train_good scans, template excluded
    pooled = []
    calib = [p for p in iter_industrial(root, part, "train_good") if p != template_ply]
    for ply in cap(calib):
        pts = load_scan(ply, cfg)
        T_est, _ = register_scan(pts, target, cfg, ("inspect", part, "calib", ply.stem))
        aligned = pts @ T_est[:3, :3].T + T_est[:3, 3]
        pooled.append(point_distances(aligned, template_pts))
    tau = calibrate_tau(np.concatenate(pooled), icfg["tau_quantile"])

    if render:
        fig_dir = REPO_ROOT / cfg["paths"]["figures_dir"] / "inspect" / part
        fig_dir.mkdir(parents=True, exist_ok=True)
        rng_render = stable_rng("inspect-render", part)
        n_max = cfg["demo"]["render_points"]

    rows, panels, to_render = [], [], []
    for split in ("test_good", "test_defect"):
        for k, ply in enumerate(cap(list(iter_industrial(root, part, split)))):
            pts = load_scan(ply, cfg)
            T_est, elapsed = register_scan(pts, target, cfg,
                                           ("inspect", part, split, ply.stem))
            aligned = pts @ T_est[:3, :3].T + T_est[:3, 3]
            dist = point_distances(aligned, template_pts)
            sc = load_sidecar(ply)
            reg = {"rre_deg": "", "rte": "", "success": ""}
            point_auroc = ""
            defect_type = ""
            if sc is not None:
                defect_type = sc["defect_type"]
                if sc["T_gt"] is not None:
                    reg = evaluate(T_est, sc["T_gt"], d, cfg["thresholds"])
                mask = sc["defect_mask"]
                if len(mask) == len(dist) and mask.any() and not mask.all():
                    point_auroc = round(auroc(dist, mask), 4)
            rows.append({
                "part": part, "track": track_of(part), "split": split,
                "defect_type": defect_type, "scan": ply.stem,
                "method": icfg["method"], "n_points": len(pts),
                "score": round(scan_score(dist, d, icfg["score_quantile"]), 6),
                "tau": round(tau, 6), "rre_deg": reg["rre_deg"], "rte": reg["rte"],
                "reg_success": reg["success"], "point_auroc": point_auroc,
                "time_s": round(elapsed, 4), "template": icfg["template"],
                "d": round(d, 6), "voxel": round(voxel, 6),
                "score_quantile": icfg["score_quantile"],
                "tau_quantile": icfg["tau_quantile"],
            })
            if render and k < icfg["heatmaps_per_part"]:
                title = f"{split}/{ply.stem}" + (f" ({defect_type})" if defect_type else "")
                to_render.append((title, fig_dir / f"{split}_{ply.stem}.png",
                                  aligned, dist))
            if render and k < 4:
                panels.append((f"{split}/{ply.stem}", aligned, dist))
        print(f"{part}/{split}: {sum(r['split'] == split for r in rows)} scans scored")
    if render:
        # one colour scale per part, from the displayed distances
        vmax = float(np.quantile(np.concatenate([p[2] for p in panels]) / d, 0.995))
        for title, out, aligned, dist in to_render:
            heatmap_png(aligned, dist, d, title, out, rng_render, n_max, vmax)
        gallery_png(panels, d, part, REPO_ROOT / cfg["paths"]["figures_dir"] / "inspect"
                    / f"{part}_gallery.png", rng_render, n_max, vmax)
    return rows


def summary_md(df: pd.DataFrame) -> str:
    lines = ["# Inspection summary - scan-level template-difference detection", ""]
    rows_md = []
    for part, g in df.groupby("part", sort=False):
        y = (g["split"] == "test_defect").to_numpy()
        a = auroc(g["score"].to_numpy(float), y)
        pa = pd.to_numeric(g["point_auroc"], errors="coerce").dropna()
        rows_md.append([
            part, g["method"].iloc[0], f"{a:.3f}",
            f"{int((~y).sum())}/{int(y.sum())}",
            f"{g.loc[~y, 'score'].median():.4f}", f"{g.loc[y, 'score'].median():.4f}",
            f"{g['tau'].iloc[0]:.5f}",
            f"{pa.mean():.3f}" if len(pa) else "-"])
    lines.append(format_table(
        ["part", "method", "scan AUROC", "n_good/n_defect", "median good score",
         "median defect score", "tau", "mean point-AUROC (defect scans)"], rows_md))
    # per-defect-type AUROC where types exist (each type vs all good scans)
    typed = df[(df["split"] == "test_defect") & (df["defect_type"] != "")]
    if len(typed):
        lines += ["", "Per-defect-type scan AUROC (type vs all good):", ""]
        for (part, dtype), g in typed.groupby(["part", "defect_type"], sort=False):
            good = df[(df["part"] == part) & (df["split"] == "test_good")]
            scores = np.concatenate([good["score"].to_numpy(float),
                                     g["score"].to_numpy(float)])
            labels = np.r_[np.zeros(len(good), bool), np.ones(len(g), bool)]
            lines.append(f"- {part} / {dtype} (n={len(g)}): "
                         f"{auroc(scores, labels):.3f}")
    lines += ["", f"score = q{df['score_quantile'].iloc[0]} of point distance / d | "
                  f"tau = q{df['tau_quantile'].iloc[0]} of pooled registered "
                  f"train_good distances", ""]
    return "\n".join(lines)


def plot_scores(cfg: dict) -> None:
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    dfs = [pd.read_csv(p) for p in (tables / "inspect_synthetic.csv",
                                    tables / "inspect_mvtec.csv") if p.exists()]
    if not dfs:
        raise SystemExit("no inspect_*.csv yet - run `regbench inspection` first")
    df = pd.concat(dfs, ignore_index=True)
    style.apply()
    parts = list(dict.fromkeys(df["part"]))
    ncols = 5
    nrows = -(-len(parts) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(style.WIDTH_FULL, 2.5 * nrows),
                             sharey=False)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    for i, part in enumerate(parts):
        ax = axes[i // ncols, i % ncols]
        g = df[df["part"] == part]
        for x, (split, color) in enumerate([("test_good", style.NAVY),
                                            ("test_defect", style.ORANGE)]):
            s = g.loc[g["split"] == split, "score"].to_numpy(float)
            jitter = stable_rng("jitter", part, split).uniform(-0.12, 0.12, len(s))
            ax.scatter(np.full(len(s), x) + jitter, s, s=10, alpha=0.75, color=color)
        tau_rel = g["tau"].iloc[0] / g["d"].iloc[0]
        ax.axhline(tau_rel, color=style.CHANCE, ls="--", lw=1)
        y = (g["split"] == "test_defect").to_numpy()
        a = auroc(g["score"].to_numpy(float), y)
        # the row label carries the dataset, the title only the object
        short = part.replace("synth_itodd_", "").replace("synth_", "").replace("mvtec_", "")
        ax.set_title(f"{short}\nAUROC {a:.3f}", fontsize=7)
        ax.set_xticks([0, 1], ["good", "defect"], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        if i % ncols == 0:
            ax.set_ylabel("score / d", fontsize=7.5)
    for j in range(len(parts), nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()
    # synthetic parts come first (inspect_synthetic.csv is read first)
    n_synth = sum(1 for p in parts if p.startswith("synth"))
    row_labels = {0: f"synthetic ({n_synth})", 1: "MVTec 3D-AD (real)"}
    for row, label in row_labels.items():
        if row < nrows:
            axes[row, 0].annotate(label, xy=(0, 0.5), xycoords="axes fraction",
                                  xytext=(-42, 0), textcoords="offset points",
                                  rotation=90, va="center", ha="center", fontsize=7.5,
                                  color=style.GREY)
    handles = [plt.Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=style.NAVY, markersize=6, label="test good"),
               plt.Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=style.ORANGE, markersize=6, label="test defect"),
               plt.Line2D([0], [0], color=style.CHANCE, ls="--", lw=1,
                          label=r"$\tau/d$ (point alarm)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
              bbox_to_anchor=(0.5, -0.02 / nrows))
    fig.suptitle("Template-difference inspection: scan scores by split", fontsize=10)
    fig.tight_layout()
    out = REPO_ROOT / cfg["paths"]["figures_dir"] / "inspect_scores.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="only redraw inspect_scores.png from inspect_*.csv")
    ap.add_argument("--smoke", action="store_true",
                    help="3 scans per split, voxel d/50")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.plot:
        plot_scores(cfg)
        return
    if args.smoke:
        cfg["preprocess"]["voxel_divisor"] = 50
    style.apply()
    rows = []
    for part in cfg["inspect"]["parts"]:
        rows += run_part(part, cfg, args.smoke)
    if not rows:
        raise SystemExit("nothing ran - build datasets with `regbench build-industrial` first")
    df = pd.DataFrame(rows, columns=FIELDS)
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    for track, g in df.groupby("track", sort=False):
        out = tables / f"inspect_{track}.csv"
        g.to_csv(out, index=False)
        print(f"wrote {out} ({len(g)} rows)")
    md = summary_md(df)
    (tables / "inspect_summary.md").write_text(md)
    print(md)
    plot_scores(cfg)


if __name__ == "__main__":
    main()
