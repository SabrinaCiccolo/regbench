"""False inspection alarms as a function of alignment error.

    regbench propagation           # offset sweeps + registration runs
    regbench propagation --plot    # redraw fp_vs_error.png from the CSV

Each defective synthetic scan is aligned by its true pose and then moved by a
known offset: rotation-only and translation-only grids and a joint grid, with
several random axes/directions per magnitude. Real registrations by every
method are added for comparison. tau comes from true-pose-aligned train_good
scans; false positives are counted outside the dilated defect zone, the true
positive rate inside the defect.

Output: fp_vs_error.csv (tables_dir), fp_vs_error.png (figures_dir).
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
from regbench.anomaly import calibrate_tau, exclusion_mask, fp_tp_rates, load_sidecar, \
    point_distances
from regbench.cloud_io import iter_industrial, load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.methods import prepare_target, register
from regbench.metrics import evaluate
from regbench.repro import stable_rng

FIELDS = ["part", "scan", "mode", "magnitude", "rot_deg_offset", "trans_rel_offset",
          "method", "repeat", "rre_deg", "rte", "tau", "fpr", "tpr", "n_points",
          "n_defect_points", "d", "voxel", "tau_quantile", "exclusion_radius_mult"]


def offset_transform(mode: str, magnitude: float, d: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Rotation-only (magnitude deg) or translation-only (magnitude * d) offset."""
    T = np.eye(4)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    if mode == "rot":
        T[:3, :3] = Rotation.from_rotvec(np.deg2rad(magnitude) * v).as_matrix()
    elif mode == "trans":
        T[:3, 3] = magnitude * d * v
    else:
        raise ValueError(f"unknown offset mode {mode!r}")
    return T


def joint_offset_transform(rot_deg: float, trans_rel: float, d: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Rotation by ``rot_deg`` and translation by ``trans_rel * d``, random axis and direction."""
    T = np.eye(4)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    T[:3, :3] = Rotation.from_rotvec(np.deg2rad(rot_deg) * axis).as_matrix()
    T[:3, 3] = trans_rel * d * direction
    return T


def run(cfg: dict) -> None:
    pcfg = cfg["propagation"]
    part = pcfg["part"]
    root = REPO_ROOT / cfg["paths"]["industrial_root"]
    template_ply = root / part / f"{cfg['inspect']['template']}.ply"
    template_pcd, d, voxel = preprocess_cloud(load_cloud(template_ply), cfg,
                                              recenter=False)
    target = prepare_target(template_pcd, voxel, cfg["registration"])
    template_pts = np.asarray(target.pcd.points)

    # tau from true-pose-aligned train_good scans
    pooled = []
    for ply in iter_industrial(root, part, "train_good"):
        if ply == template_ply:
            continue
        pts = np.asarray(load_cloud(ply).points)
        T_gt = load_sidecar(ply)["T_gt"]
        aligned = pts @ T_gt[:3, :3].T + T_gt[:3, 3]
        pooled.append(point_distances(aligned, template_pts))
    tau = calibrate_tau(np.concatenate(pooled), pcfg["tau_quantile"])

    excl_radius = (pcfg["exclusion_radius_mult"]
                   * cfg["industrial"]["synthetic"]["defect"]["radius_rel"] * d)
    scans = list(iter_industrial(root, part, "test_defect"))
    rows = []

    def measure(pts, T, mask, excl, scan, mode, magnitude, repeat, *,
               rot_deg_offset=float("nan"), trans_rel_offset=float("nan"),
               method=""):
        aligned = pts @ T[:3, :3].T + T[:3, 3]
        dist = point_distances(aligned, template_pts)
        fpr, tpr = fp_tp_rates(dist, tau, mask, excl)
        err = evaluate(T, np.eye(4), d, cfg["thresholds"])  # vs perfect alignment
        rows.append({"part": part, "scan": scan, "mode": mode,
                     "magnitude": magnitude, "rot_deg_offset": rot_deg_offset,
                     "trans_rel_offset": trans_rel_offset, "method": method,
                     "repeat": repeat,
                     "rre_deg": round(err["rre_deg"], 4), "rte": round(err["rte"], 6),
                     "tau": round(tau, 6), "fpr": round(fpr, 6),
                     "tpr": round(tpr, 6), "n_points": len(pts),
                     "n_defect_points": int(mask.sum()), "d": round(d, 6),
                     "voxel": round(voxel, 6), "tau_quantile": pcfg["tau_quantile"],
                     "exclusion_radius_mult": pcfg["exclusion_radius_mult"]})

    # rotation-only and translation-only offsets from the true pose
    for ply in scans:
        pts = np.asarray(load_cloud(ply).points)
        sc = load_sidecar(ply)
        T_gt, mask = sc["T_gt"], sc["defect_mask"]
        pts_true = pts @ T_gt[:3, :3].T + T_gt[:3, 3]   # perfectly aligned once
        excl = exclusion_mask(pts_true, mask, excl_radius)
        for mode, grid in (("rot", pcfg["rot_deg"]), ("trans", pcfg["trans_rel"])):
            for magnitude in grid:
                for rep in range(pcfg["repeats"]):
                    T_off = offset_transform(
                        mode, magnitude, d,
                        stable_rng("prop", part, ply.stem, mode, magnitude, rep))
                    ro, to = (magnitude, 0.0) if mode == "rot" else (0.0, magnitude)
                    measure(pts_true, T_off, mask, excl, ply.stem, mode, magnitude,
                           rep, rot_deg_offset=ro, trans_rel_offset=to)
        print(f"{ply.stem}: single-axis sweep done ({len(rows)} rows total)")

    # joint rotation + translation offsets
    for ply in scans:
        pts = np.asarray(load_cloud(ply).points)
        sc = load_sidecar(ply)
        T_gt, mask = sc["T_gt"], sc["defect_mask"]
        pts_true = pts @ T_gt[:3, :3].T + T_gt[:3, 3]
        excl = exclusion_mask(pts_true, mask, excl_radius)
        for rot_mag in pcfg["joint_rot_deg"]:
            for trans_mag in pcfg["joint_trans_rel"]:
                for rep in range(pcfg["repeats"]):
                    T_off = joint_offset_transform(
                        rot_mag, trans_mag, d,
                        stable_rng("prop-joint", part, ply.stem, rot_mag,
                                  trans_mag, rep))
                    measure(pts_true, T_off, mask, excl, ply.stem, "joint",
                           float("nan"), rep, rot_deg_offset=rot_mag,
                           trans_rel_offset=trans_mag)
        print(f"{ply.stem}: joint sweep done ({len(rows)} rows total)")

    # residuals of real registrations, every method
    for method in cfg["methods"]:
        for ply in scans[:pcfg["n_registered"]]:
            pts = np.asarray(load_cloud(ply).points)
            sc = load_sidecar(ply)
            T_gt, mask = sc["T_gt"], sc["defect_mask"]
            seed = int(stable_rng("prop-reg", part, method, ply.stem).integers(2 ** 31))
            T_est = register(method, pts, target, cfg["registration"], seed)
            # residual after registration, expressed as an offset from the true pose
            T_resid = T_est @ np.linalg.inv(T_gt)
            pts_true = pts @ T_gt[:3, :3].T + T_gt[:3, 3]
            excl = exclusion_mask(pts_true, mask, excl_radius)
            measure(pts_true, T_resid, mask, excl, ply.stem, "registered",
                   float("nan"), 0, method=method)
        print(f"registered scatter: {pcfg['n_registered']} runs x {method}")

    df = pd.DataFrame(rows, columns=FIELDS)
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    out = tables / "fp_vs_error.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


def plot(cfg: dict) -> None:
    csv = REPO_ROOT / cfg["paths"]["tables_dir"] / "fp_vs_error.csv"
    if not csv.exists():
        raise SystemExit("run `regbench propagation` first")
    df = pd.read_csv(csv)
    thr = cfg["thresholds"]
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_FULL, 3.1))
    panels = [("rot", "rre_deg", axes[0], "rotation offset (deg)", thr["rre_deg"],
               "grid success threshold (2°)"),
              ("trans", "rte", axes[1], "translation offset (fraction of d)",
               thr["rte_rel"], "grid success threshold (0.01·d)")]
    reg = df[df["mode"] == "registered"]
    for mode, err_col, ax, xlabel, vline, vlabel in panels:
        sweep = df[df["mode"] == mode]
        agg = sweep.groupby("magnitude")[["fpr", "tpr"]]
        mag = agg.mean().index.to_numpy(float)
        ax.plot(mag, agg.mean()["fpr"], color=style.ORANGE, marker="D", lw=2,
                label="false-positive rate (outside defect)")
        q1 = agg.quantile(0.25)["fpr"]
        q3 = agg.quantile(0.75)["fpr"]
        ax.fill_between(mag, q1, q3, color=style.ORANGE, alpha=0.2, lw=0,
                        label="IQR across scans x repeats")
        ax.plot(mag, agg.mean()["tpr"], color=style.GREEN, marker=".", ls="--", lw=1.5,
                label="true-positive rate (defect zone)")
        # real registrations of every method
        for method in cfg["methods"]:
            sub = reg[reg["method"] == method]
            if len(sub):
                ax.scatter(sub[err_col], sub["fpr"], s=22,
                          color=style.METHOD_COLORS[method],
                          marker=style.METHOD_MARKERS[method], zorder=5,
                          label=f"actual {method} runs")
        ax.axvline(vline, color=style.CHANCE, ls=":", lw=1.2)
        ax.text(vline, 0.98, f" {vlabel}", rotation=90, va="top", fontsize=6.5,
                color=style.TEXT)
        ax.set_xscale("symlog", linthresh=mag[mag > 0].min())
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("rate (fraction of points)")
    handles, labels = axes[0].get_legend_handles_labels()

    # joint offset grid
    joint = df[df["mode"] == "joint"]
    ax3 = axes[2]
    if len(joint):
        piv = joint.pivot_table(index="rot_deg_offset", columns="trans_rel_offset",
                                values="fpr", aggfunc="mean")
        im = ax3.imshow(piv.to_numpy(), origin="lower", aspect="auto",
                        cmap="inferno", vmin=0, vmax=1)
        ax3.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns],
                       rotation=45, fontsize=8)
        ax3.set_yticks(range(len(piv.index)), [f"{r:g}" for r in piv.index],
                       fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iloc[i, j]
                if not np.isnan(v):
                    ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                             color="white" if v < 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax3, shrink=0.85, pad=0.02).set_label("mean FPR")
        ax3.set_xlabel("translation offset (fraction of d)", fontsize=7.5)
        ax3.set_ylabel("rotation offset (deg)", fontsize=7.5)
        ax3.set_title("joint rot+trans offset grid", fontsize=8.5)
    else:
        ax3.text(0.5, 0.5, "no joint-grid rows in this CSV\n"
                 "(rerun `regbench propagation`)", ha="center", va="center",
                 color=style.GREY)
        ax3.set_axis_off()

    part = df["part"].iloc[0]
    # leave room for the suptitle and the shared legend
    fig.tight_layout(rect=(0, 0.14, 1, 0.88))
    fig.suptitle(f"Registration error -> false inspection alarms ({part}, "
                 f"tau = q{df['tau_quantile'].iloc[0]} of good)", fontsize=9, y=0.97)
    fig.legend(handles, labels, fontsize=6.5, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    out = REPO_ROOT / cfg["paths"]["figures_dir"] / "fp_vs_error.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="only redraw fp_vs_error.png from fp_vs_error.csv")
    args = ap.parse_args(argv)
    cfg = load_config()
    if not args.plot:
        run(cfg)
    plot(cfg)


if __name__ == "__main__":
    main()
