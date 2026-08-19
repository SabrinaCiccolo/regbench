"""Run-to-run spread of the inspection scan AUROC.

Reruns the inspection pipeline (registration and scoring, no figures) N times
with identical code, configuration and seeds; only Open3D's multithreaded
RANSAC can change the result. Reports mean, std and range of the scan AUROC per
part.

    regbench reproducibility run [--n N] [--parts P1,P2,...]
    regbench reproducibility report

Outputs (tables_dir): reproducibility.csv (one row per rep and part) and
reproducibility_report.md.
"""
from __future__ import annotations

import argparse

import pandas as pd

from regbench.anomaly import auroc
from regbench.config import REPO_ROOT, load_config
from regbench.experiments.inspection import run_part, track_of
from regbench.mdtable import format_table

DEFAULT_PARTS = ["mvtec_bagel", "synth_bunny"]
DEFAULT_N = 10

FIELDS = ["rep", "part", "track", "scan_auroc", "n_good", "n_defect"]


def part_auroc(part: str, cfg: dict) -> dict:
    rows = run_part(part, cfg, smoke=False, render=False)
    df = pd.DataFrame(rows)
    y = (df["split"] == "test_defect").to_numpy()
    a = auroc(df["score"].to_numpy(float), y)
    return {"track": track_of(part), "scan_auroc": round(a, 6),
            "n_good": int((~y).sum()), "n_defect": int(y.sum())}


def run(cfg: dict, parts: list[str], n: int) -> pd.DataFrame:
    rows = []
    for rep in range(n):
        for part in parts:
            info = part_auroc(part, cfg)
            rows.append({"rep": rep, "part": part, **info})
            print(f"rep {rep}: {part} scan_auroc={info['scan_auroc']:.4f}")
    return pd.DataFrame(rows, columns=FIELDS)


def report(df: pd.DataFrame) -> str:
    lines = ["# Scan AUROC across identical reruns", "",
             f"N = {df['rep'].nunique()} reruns of the inspection pipeline with "
             "identical code, configuration and seeds.", ""]
    table_rows = []
    for part, g in df.groupby("part", sort=False):
        a = g["scan_auroc"].to_numpy(float)
        table_rows.append([part, f"{a.mean():.4f}", f"{a.std(ddof=1):.4f}",
                           f"{a.min():.4f}", f"{a.max():.4f}",
                           f"{a.max() - a.min():.4f}"])
    lines.append(format_table(
        ["part", "mean AUROC", "std AUROC", "min", "max", "range"], table_rows))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--parts", default=",".join(DEFAULT_PARTS))
    args = ap.parse_args(argv)
    cfg = load_config()
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    csv = tables / "reproducibility.csv"

    if args.mode == "run":
        parts = args.parts.split(",")
        df = run(cfg, parts, args.n)
        df.to_csv(csv, index=False)
        print(f"wrote {csv} ({len(df)} rows)")
    else:
        if not csv.exists():
            raise SystemExit("run `regbench reproducibility run` first")
        df = pd.read_csv(csv)

    md = report(df)
    (tables / "reproducibility_report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
