"""Regenerate the committed results and compare them with the committed copies.

    regbench verify [--only PREFIX] [--scratch DIR] [-v]

Every artifact in INVENTORY is regenerated into a scratch directory (the
committed results/ tree is only read) and checked against the committed file:

  exact          identical text; CSVs equal to rtol 1e-9 without ``time_s``/``cpu``
  wilson:K,...   stochastic grids: for every success column, each cell grouped by
                 K must have a 95% Wilson interval overlapping the committed one
  derived:PATH   a rendering of a wilson artifact; passes when PATH passes
  partial:C=V    rows with C != V must be identical; rows with C == V only counted
  shape          values change on every run: same columns and row count
  figure         the PNG is regenerated and not empty

Before a command runs, its scratch directory is seeded with the committed CSVs,
so offline commands are checked against the inputs the committed tables were
built from. A row whose input data is missing is reported as SKIPPED. The exit
code is 1 if any row fails.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from regbench import config
from regbench.symmetry import wilson_ci

RESULTS = config.REPO_ROOT / "results" / "regbench"

_SWEEP = ("bracket_screw", "injection_pump", "star", "cylinder", "washer", "thread")
ITODD = tuple(f"data/template/itodd_{p}.ply" for p in _SWEEP)
BUNNY = ("data/template/bunny.ply",)
FOLD = ("data/template/cable_gland.ply", *BUNNY, *ITODD)
INSPECT = tuple(f"data/industrial/{p}" for p in config.load_config()["inspect"]["parts"])
PROPAGATION = ("data/industrial/synth_cable_gland",)
REPEATS = ("data/industrial/mvtec_bagel", "data/industrial/synth_bunny")

BUNNY_GRID = "grid --template data/template/bunny.ply --tag bunny"

# (artifact under results/regbench, regbench command, check, required inputs)
INVENTORY: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("tables/registration_grid.csv", "grid", "wilson:method,severity", ITODD[:1]),
    ("tables/success_rates.md", "grid", "derived:tables/registration_grid.csv", ITODD[:1]),
    ("tables/registration_grid_bunny.csv", BUNNY_GRID, "wilson:method,severity", BUNNY),
    ("tables/success_rates_bunny.md", BUNNY_GRID,
     "derived:tables/registration_grid_bunny.csv", BUNNY),
    ("tables/symmetry_sweep.csv", "sweep-symmetry", "wilson:symmetry_class,severity", ITODD),
    ("tables/symmetry_scores.csv", "sweep-symmetry", "exact", ITODD),
    ("tables/branch_consistency.csv", "sweep-symmetry",
     "derived:tables/symmetry_sweep.csv", ITODD),
    ("tables/symmetry_success_rates.md", "sweep-symmetry",
     "derived:tables/symmetry_sweep.csv", ITODD),
    ("figures/symmetry_sweep.png", "sweep-symmetry --plot", "figure", ()),
    ("tables/symmetry_sweep_rescored.csv", "rescore", "exact", ITODD),
    ("tables/branch_consistency_discrete.csv", "rescore", "exact", ITODD),
    ("tables/symmetry_success_rates_discrete.md", "rescore", "exact", ITODD),
    ("tables/bop_metrics.csv", "bop", "exact", ITODD),
    ("tables/bop_metrics.md", "bop", "exact", ITODD),
    ("tables/sym_regression.md", "regression", "exact", ()),
    ("tables/clutter_sweep.csv", "clutter", "wilson:method,n_distractors", ITODD),
    ("tables/clutter_success_rates.md", "clutter", "derived:tables/clutter_sweep.csv", ITODD),
    ("figures/clutter_sweep.png", "clutter --plot", "figure", ()),
    ("tables/confidence_signal.md", "confidence", "exact", ()),
    ("figures/confidence_signal.png", "confidence", "figure", ()),
    ("tables/timing_summary.md", "timing", "exact", ()),
    ("figures/pareto.png", "timing", "figure", ()),
    ("tables/fp_vs_error.csv", "propagation", "partial:mode=registered", PROPAGATION),
    ("figures/fp_vs_error.png", "propagation --plot", "figure", ()),
    ("tables/inspect_synthetic.csv", "inspection", "wilson:part,split", INSPECT),
    ("tables/inspect_mvtec.csv", "inspection", "wilson:part,split", INSPECT),
    ("tables/inspect_summary.md", "inspection", "derived:tables/inspect_synthetic.csv", INSPECT),
    ("figures/inspect_scores.png", "inspection --plot", "figure", ()),
    ("tables/inspect_symmetry_check.md", "symmetry-check", "exact", FOLD),
    ("tables/reproducibility.csv", "reproducibility run", "shape", REPEATS),
    ("tables/reproducibility_report.md", "reproducibility report", "exact", ()),
]

VOLATILE_COLUMNS = ("time_s", "cpu")
SUCCESS_COLUMNS = ("success", "success_sym", "reg_success")


# --- running a command into a scratch directory ------------------------------

def run_command(command: str, scratch: Path) -> None:
    """Run ``regbench <command>`` with tables_dir/figures_dir inside ``scratch``."""
    from regbench.cli import COMMANDS

    name, *argv = command.split()
    tables, figures = scratch / "tables", scratch / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for src in (RESULTS / "tables").glob("*.csv"):
        if not (tables / src.name).exists():
            shutil.copy2(src, tables / src.name)

    def load_config() -> dict:
        cfg = config.load_config()
        cfg["paths"]["tables_dir"] = str(tables)
        cfg["paths"]["figures_dir"] = str(figures)
        return cfg

    module = importlib.import_module(COMMANDS[name])
    original = module.load_config
    module.load_config = load_config
    try:
        module.main(argv)
    finally:
        module.load_config = original


# --- checks ----------------------------------------------------------------

def cell_counts(df: pd.DataFrame, column: str, keys: list[str]) -> dict[tuple, tuple[int, int]]:
    """{cell: (successes, trials)} for ``column`` grouped by ``keys``."""
    out = {}
    for cell, g in df.groupby(keys, sort=False):
        out[cell if isinstance(cell, tuple) else (cell,)] = (int(g[column].sum()), len(g))
    return out


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if the 95% Wilson intervals of two (k, n) counts overlap."""
    lo_a, hi_a = wilson_ci(*a)
    lo_b, hi_b = wilson_ci(*b)
    return hi_a >= lo_b and hi_b >= lo_a


def _drop_volatile(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in VOLATILE_COLUMNS if c in df.columns])


def check_exact(committed: Path, fresh: Path) -> tuple[bool, str]:
    if committed.suffix == ".csv":
        old, new = _drop_volatile(pd.read_csv(committed)), _drop_volatile(pd.read_csv(fresh))
        try:
            pd.testing.assert_frame_equal(old, new, check_exact=False, rtol=1e-9)
            return True, ""
        except AssertionError as exc:
            return False, str(exc).splitlines()[0]
    want, got = committed.read_text(), fresh.read_text()
    if want == got:
        return True, ""
    for i, (a, b) in enumerate(zip(want.splitlines(), got.splitlines()), 1):
        if a != b:
            return False, f"line {i}:\n    committed: {a}\n    fresh:     {b}"
    return False, "line count differs"


def check_wilson(committed: Path, fresh: Path, keys: list[str]) -> tuple[bool, str]:
    old, new = pd.read_csv(committed), pd.read_csv(fresh)
    columns = [c for c in SUCCESS_COLUMNS if c in old.columns and c in new.columns]
    if not columns:
        return False, "no success column"
    moved, outside, n_cells = 0, [], 0
    for column in columns:
        base, live = cell_counts(old, column, keys), cell_counts(new, column, keys)
        if set(base) != set(live):
            return False, f"{column}: cells differ"
        for cell, counts in base.items():
            n_cells += 1
            moved += counts != live[cell]
            if not intervals_overlap(counts, live[cell]):
                outside.append(f"{column} {'/'.join(map(str, cell))}: "
                               f"{counts[0]}/{counts[1]} vs {live[cell][0]}/{live[cell][1]}")
    if outside:
        return False, "non-overlapping intervals: " + "; ".join(outside)
    return True, f"{moved} of {n_cells} cells moved, all intervals overlap"


def check_partial(committed: Path, fresh: Path, spec: str) -> tuple[bool, str]:
    column, _, value = spec.partition("=")
    old, new = _drop_volatile(pd.read_csv(committed)), _drop_volatile(pd.read_csv(fresh))
    if (old[column] == value).sum() != (new[column] == value).sum():
        return False, f"number of {column}={value} rows changed"
    a = old[old[column] != value].reset_index(drop=True)
    b = new[new[column] != value].reset_index(drop=True)
    if a.equals(b):
        return True, f"{len(a)} rows identical"
    return False, f"{column}!={value} rows differ"


def check_shape(committed: Path, fresh: Path) -> tuple[bool, str]:
    old, new = pd.read_csv(committed), pd.read_csv(fresh)
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        return False, f"shape {old.shape} -> {new.shape}"
    return True, f"{len(old)} rows"


def check(kind: str, arg: str, committed: Path, fresh: Path) -> tuple[bool, str]:
    if kind == "exact":
        return check_exact(committed, fresh)
    if kind == "wilson":
        return check_wilson(committed, fresh, arg.split(","))
    if kind == "partial":
        return check_partial(committed, fresh, arg)
    if kind == "shape":
        return check_shape(committed, fresh)
    if kind == "figure":
        return fresh.stat().st_size > 0, ""
    raise ValueError(f"unknown check {kind!r}")


# --- driver ----------------------------------------------------------------

def verify(rows: list[tuple], scratch_root: Path, verbose: bool = False) -> int:
    results: dict[str, str] = {}
    errors: dict[str, str] = {}
    for artifact, command, spec, needs in rows:
        kind, _, arg = spec.partition(":")
        missing = [p for p in needs if not (config.REPO_ROOT / p).exists()]
        if missing:
            results[artifact] = "skipped"
            print(f"SKIPPED   {artifact} (missing {missing[0]}, see docs/reproducing.md)")
            continue
        if kind == "derived":
            results[artifact] = results.get(arg, "failed")
            print(f"{results[artifact].upper():<9} {artifact} (via {arg})")
            continue

        scratch = scratch_root / command.replace(" ", "_").replace("/", "_")
        if command not in errors and not scratch.exists():
            if verbose:
                print(f"          running: regbench {command}")
            try:
                run_command(command, scratch)
            except Exception as exc:  # noqa: BLE001
                errors[command] = f"{type(exc).__name__}: {exc}"
        fresh = scratch / artifact
        if command in errors:
            ok, note = False, f"`regbench {command}` raised {errors[command]}"
        elif not fresh.is_file():
            ok, note = False, f"`regbench {command}` wrote no {fresh.name}"
        else:
            ok, note = check(kind, arg, RESULTS / artifact, fresh)
        results[artifact] = "verified" if ok else "failed"
        print(f"{results[artifact].upper():<9} {artifact}")
        if note and (verbose or not ok):
            print(f"          {note}")

    counts = {s: sum(v == s for v in results.values()) for s in ("verified", "skipped", "failed")}
    print(f"\n{counts['verified']} verified, {counts['skipped']} skipped, "
          f"{counts['failed']} failed of {len(rows)}")
    return 1 if counts["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="regbench verify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="",
                    help="only artifacts whose path starts with this prefix (e.g. tables/bop)")
    ap.add_argument("--scratch", default=None,
                    help="regenerate into this directory and keep it")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    rows = [r for r in INVENTORY if r[0].startswith(args.only)]
    if not rows:
        raise SystemExit(f"no artifact starts with {args.only!r}")
    if args.scratch:
        scratch = Path(args.scratch).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        return verify(rows, scratch, args.verbose)
    with tempfile.TemporaryDirectory(prefix="regbench-verify-") as tmp:
        return verify(rows, Path(tmp), args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
