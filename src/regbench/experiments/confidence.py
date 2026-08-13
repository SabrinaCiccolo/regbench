"""Does Open3D's fitness predict registration failure without ground truth?

Reads registration_grid.csv only. Reports the AUROC of fitness, inlier_rmse and
n_inliers at ranking failed trials above successful ones, and the failures
caught by the highest fitness threshold that flags no successful trial.

    regbench confidence

Outputs: confidence_signal.md (tables_dir), confidence_signal.png (figures_dir).
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regbench import style
from regbench.anomaly import auroc
from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table


def signal_auroc(df: pd.DataFrame, col: str, higher_is_worse: bool) -> tuple[float, int]:
    """AUROC of `col` at ranking FAILURES above successes (labels=1 -> failure)."""
    vals = df[col].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    fail = ~df.loc[mask, "success"].astype(bool).to_numpy()
    scores = vals[mask] if higher_is_worse else -vals[mask]
    if fail.sum() == 0 or (~fail).sum() == 0:
        return float("nan"), int(mask.sum())
    return auroc(scores, fail), int(mask.sum())


def zero_false_alarm_recall(df: pd.DataFrame) -> dict:
    """Recall of failures at the highest fitness threshold that flags no success."""
    succ = df["success"].astype(bool).to_numpy()
    fit = df["fitness"].to_numpy(dtype=float)
    thr = float(fit[succ].min())
    flagged = fit < thr
    fail = ~succ
    caught, total_fail = int((flagged & fail).sum()), int(fail.sum())
    missed = df[(~flagged) & fail]
    return {"threshold": thr, "caught": caught, "total_fail": total_fail,
           "recall": caught / total_fail if total_fail else float("nan"),
           "missed_rre": missed["rre_deg"].to_numpy()}


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser().parse_args(argv)  # no flags; enables `--help`
    cfg = load_config()
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    df = pd.read_csv(tables / "registration_grid.csv")

    signal_table = format_table(
        ["signal", "AUROC (all methods)", "n"],
        [[col, f"{a:.4f}", n] for col, worse in
         [("fitness", False), ("inlier_rmse", True), ("n_inliers", False)]
         for a, n in [signal_auroc(df, col, worse)]])
    method_table = format_table(
        ["method", "AUROC", "n"],
        [[method, f"{a:.4f}", n] for method, g in df.groupby("method", sort=False)
         for a, n in [signal_auroc(g, "fitness", False)]])
    lines = ["# Confidence signal: does fitness predict failure?", "",
             "AUROC of each diagnostic of the final ICP step at ranking failed",
             "trials above successful ones.", "",
             signal_table,
             "", "## fitness AUROC by method", "", method_table]

    op = zero_false_alarm_recall(df)
    lines += ["", "## Zero-false-alarm threshold", "",
             f"Threshold = {op['threshold']:.6f}, the lowest fitness of any successful",
             f"trial. `fitness < {op['threshold']:.4f}` flags {op['caught']}/"
             f"{op['total_fail']} ({100 * op['recall']:.1f}%) of the failed trials",
             "and no successful trial.", ""]
    if len(op["missed_rre"]):
        lines.append(f"RRE of the {len(op['missed_rre'])} missed failure(s): "
                     f"{', '.join(f'{v:.2f}' for v in op['missed_rre'])} deg.")
    else:
        lines.append("No failures were missed at this threshold.")
    lines += ["", "The threshold is calibrated on the single-object grid; fitness",
             "drops with clutter (clutter_success_rates.md), and a pose on another",
             "symmetry branch keeps a high fitness.", ""]
    md = "\n".join(lines)
    (tables / "confidence_signal.md").write_text(md)
    print(md)

    style.apply()
    fig, ax = plt.subplots(figsize=style.FIGSIZE_1PANEL)
    succ = df["success"].astype(bool)
    bins = np.linspace(0.0, 1.0, 41)
    # hatch separates the two outcomes without relying on red/green
    ax.hist(df.loc[~succ, "fitness"], bins=bins, color=style.RED, alpha=0.65,
           hatch="//", label=f"failure (n={int((~succ).sum())})")
    ax.hist(df.loc[succ, "fitness"], bins=bins, color=style.GREEN, alpha=0.65,
           label=f"success (n={int(succ.sum())})")
    ax.axvline(op["threshold"], color=style.TEXT, linestyle="--", linewidth=1)
    ax.text(op["threshold"], ax.get_ylim()[1] * 0.9,
           f" review-flag threshold ({op['threshold']:.4f})", fontsize=8, ha="right")
    ax.set_xlabel("fitness (inlier fraction, final ICP step)")
    ax.set_ylabel("trial count")
    ax.set_title("Registration confidence signal: fitness by outcome")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = REPO_ROOT / cfg["paths"]["figures_dir"] / "confidence_signal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
