"""Are the inspection track's registration failures symmetry branches?

Reads inspect_synthetic.csv; the templates are loaded only to detect discrete
folds. For each synthetic part, splits the test
scans by `reg_success` (plain pose metric) and compares the scan scores of the
two groups. For parts with a detected discrete fold, reports how far each
failed rotation error lies from the nearest multiple of the fold angle. A pose
on another symmetry branch superimposes the surfaces, so inspection does not
see it.

    regbench symmetry-check

Output (tables_dir): inspect_symmetry_check.md.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from regbench.cloud_io import load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table
from regbench.symmetry import detect_discrete_fold, detect_symmetry_axis

# synthetic inspection part -> the template it is made from
FOLD_TEMPLATES = {
    "synth_itodd_bracket_screw": "data/template/itodd_bracket_screw.ply",
    "synth_itodd_injection_pump": "data/template/itodd_injection_pump.ply",
    "synth_itodd_star": "data/template/itodd_star.ply",
    "synth_cable_gland": "data/template/cable_gland.ply",
    "synth_bunny": "data/template/bunny.ply",
}


def detect_fold(cfg: dict, template_rel: str) -> int | None:
    """The discrete rotational fold order of a template, or None if it has none."""
    pcd, d, _ = preprocess_cloud(load_cloud(REPO_ROOT / template_rel), cfg)
    pts = np.asarray(pcd.points)
    axis, info = detect_symmetry_axis(pts, d)
    order, _ = detect_discrete_fold(pts, axis, info["center"], d)
    return order


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser().parse_args(argv)  # no flags; enables `--help`
    cfg = load_config()
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    df = pd.read_csv(tables / "inspect_synthetic.csv")
    df = df[df["split"].isin(["test_good", "test_defect"])].copy()
    df["reg_success"] = df["reg_success"].astype(bool)

    folds = {}
    for part in sorted(df["part"].unique()):
        rel = FOLD_TEMPLATES.get(part)
        folds[part] = detect_fold(cfg, rel) if rel else None

    lines = ["# Inspection track: are the registration 'failures' symmetry branches?",
             "",
             "`reg_success` is the plain pose metric (RRE < "
             f"{cfg['thresholds']['rre_deg']} deg AND RTE < "
             f"{cfg['thresholds']['rte_rel']} of d). Scores are the pipeline's own scan",
             "scores (q0.995 of point distance / d), split by that column: if a failed",
             "pose landed on another branch of the part's rotational symmetry group the",
             "surfaces still coincide, so the two groups' scores should be",
             "indistinguishable and the scan AUROC should be unaffected.",
             ""]
    score_rows = []
    for part, g in df.groupby("part", sort=True):
        fold = folds.get(part)
        fold_s = f"C{fold}" if fold else "none"
        for split in ("test_good", "test_defect"):
            gs = g[g["split"] == split]
            bad, ok = gs[~gs["reg_success"]], gs[gs["reg_success"]]
            score_rows.append([
                part, fold_s, split, len(bad), len(ok),
                f"{bad['score'].median():.4f}" if len(bad) else "-",
                f"{ok['score'].median():.4f}" if len(ok) else "-"])
    lines.append(format_table(
        ["part", "fold", "split", "n failed", "n ok", "median score (failed)",
         "median score (ok)"], score_rows))

    lines += ["", "## Where the failed rotations actually landed", "",
              "For a part with a detected discrete fold C_n, `dist to n-fold` is the",
              "smallest angle between the trial's RRE and a multiple of 360/n degrees.",
              "A value near zero means the estimate reached an exact symmetry branch,",
              "not an arbitrary wrong pose.", ""]
    fold_rows = []
    for part, g in df.groupby("part", sort=True):
        bad = g[~g["reg_success"]]
        if not len(bad):
            continue
        fold = folds.get(part)
        rre = bad["rre_deg"].to_numpy()
        if fold:
            step = 360.0 / fold
            resid = np.minimum(rre % step, step - (rre % step))
            dist = f"{resid.max():.2f}"
        else:
            dist = "-"
        fold_rows.append([part, f"C{fold}" if fold else "none", len(bad),
                          f"{rre.min():.2f}-{rre.max():.2f}", dist])
    lines.append(format_table(
        ["part", "fold", "n failed", "RRE range (deg)", "max dist to n-fold (deg)"],
        fold_rows))

    lines += ["", "source: inspect_synthetic.csv", ""]
    out = tables / "inspect_symmetry_check.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
