"""Latency and accuracy-vs-speed from registration_grid.csv.

``time_s`` covers the source side of each registration; the template's
features are computed once beforehand. All methods run on CPU; the ``cpu``
column records the machine.

    regbench timing

Outputs: timing_summary.md (tables_dir), pareto.png (figures_dir).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regbench import style
from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table

SEVERITY_MARKERS = {"easy": "o", "medium": "s", "hard": "^"}


def summary_table(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    g = df.groupby(["method", "severity"])["time_s"]
    out = g.agg(mean_s="mean", median_s="median",
               p95_s=lambda s: np.quantile(s, 0.95)).round(4)
    out["throughput_hz"] = (1.0 / out["mean_s"]).round(2)
    out = out.reset_index()
    out["severity"] = pd.Categorical(out["severity"], order, ordered=True)
    return out.sort_values(["method", "severity"])


def write_summary_md(df: pd.DataFrame, table: pd.DataFrame, cpu: str, out: Path) -> str:
    rows_md = [[row["method"], row["severity"], f"{row['mean_s']:.4f}",
                f"{row['median_s']:.4f}", f"{row['p95_s']:.4f}",
                f"{row['throughput_hz']:.2f}"] for _, row in table.iterrows()]
    md_table = format_table(
        ["method", "severity", "mean_s", "median_s", "p95_s", "throughput_hz"], rows_md)
    lines = ["# Latency & throughput by method x severity", "",
             md_table, "",
             f"n = {df.groupby(['method', 'severity']).size().iloc[0]} "
             f"trials/cell | cpu = {cpu}", ""]
    md = "\n".join(lines)
    out.write_text(md)
    return md


def pareto_figure(df: pd.DataFrame, cfg: dict, out: Path) -> None:
    """Success rate vs mean time (log scale), one marker per (method, severity)."""
    style.apply()
    fig, ax = plt.subplots(figsize=style.FIGSIZE_1PANEL)
    order = [s for s in cfg["severities"] if s in set(df["severity"])]
    for method in cfg["methods"]:
        for severity in order:
            sub = df[(df["method"] == method) & (df["severity"] == severity)]
            if sub.empty:
                continue
            ax.scatter(sub["time_s"].mean(), sub["success"].mean() * 100,
                      color=style.METHOD_COLORS[method],
                      marker=SEVERITY_MARKERS[severity], s=70,
                      edgecolor="white", linewidth=0.6, zorder=3)
    method_handles = [plt.Line2D([0], [0], marker="o", color="none",
                                 markerfacecolor=style.METHOD_COLORS[m],
                                 markersize=8, label=m) for m in cfg["methods"]]
    severity_handles = [plt.Line2D([0], [0], marker=mk, color=style.GREY,
                                   linestyle="none", markersize=8, label=s)
                        for s, mk in SEVERITY_MARKERS.items() if s in order]
    leg1 = ax.legend(handles=method_handles, loc="lower left", fontsize=8,
                     title="method", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=severity_handles, loc="upper right", fontsize=8,
             title="severity", title_fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("mean registration time (s, log scale)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("Accuracy vs. speed")
    fig.suptitle(df["template"].iloc[0], y=1.02, fontsize=10, color=style.GREY)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(out)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser().parse_args(argv)  # no flags; enables `--help`
    cfg = load_config()
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    df = pd.read_csv(tables / "registration_grid.csv")
    order = [s for s in cfg["severities"] if s in set(df["severity"])]

    table = summary_table(df, order)
    md = write_summary_md(df, table, df["cpu"].iloc[0] if "cpu" in df else "unknown",
                          tables / "timing_summary.md")
    print(md)

    figures = REPO_ROOT / cfg["paths"]["figures_dir"]
    figures.mkdir(parents=True, exist_ok=True)
    pareto_figure(df, cfg, figures / "pareto.png")


if __name__ == "__main__":
    main()
