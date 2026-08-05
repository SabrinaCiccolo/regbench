"""Command-line entry point: ``regbench <command> [args...]``.

Each command forwards its remaining arguments to the module's own parser, so
``regbench <command> --help`` lists that command's flags.

    regbench grid --smoke
    regbench sweep-symmetry --plot
    regbench build-template itodd
    regbench verify
"""
from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS: dict[str, str] = {
    "grid": "regbench.experiments.grid",
    "sweep-symmetry": "regbench.experiments.sweep_symmetry",
    "rescore": "regbench.experiments.rescore",
    "bop": "regbench.experiments.bop",
    "regression": "regbench.experiments.regression",
    "clutter": "regbench.experiments.clutter",
    "confidence": "regbench.experiments.confidence",
    "timing": "regbench.experiments.timing",
    "inspection": "regbench.experiments.inspection",
    "propagation": "regbench.experiments.propagation",
    "symmetry-check": "regbench.experiments.symmetry_check",
    "reproducibility": "regbench.experiments.reproducibility",
    "build-template": "regbench.build.make_template",
    "build-industrial": "regbench.build.make_industrial",
    "verify": "regbench.verify",
}


def _top_level_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="regbench", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS),
                    help="which experiment/builder to run")
    ap.add_argument("args", nargs=argparse.REMAINDER,
                    help="forwarded verbatim to <command>'s own argparse "
                         "(e.g. --smoke, --plot, --n 5); try <command> --help")
    return ap


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = _top_level_parser()
    args = ap.parse_args(argv)
    module = importlib.import_module(COMMANDS[args.command])
    sys.exit(module.main(args.args))


if __name__ == "__main__":
    main()
