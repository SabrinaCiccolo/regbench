# Results index

Every file under `results/regbench/` and the command that produces it. See
[`reproducing.md`](reproducing.md) for data requirements and run order.

## Tables

| What it shows | Files (`tables/`) | Command |
|---|---|---|
| Success by method and severity, ITODD `bracket_screw` | `registration_grid.csv`, `success_rates.md` | `grid` |
| Same, Stanford Bunny | `registration_grid_bunny.csv`, `success_rates_bunny.md` | `grid --template data/template/bunny.ply --tag bunny` |
| Success by symmetry class and part, plain and symmetry-aware; symmetry scores; branch consistency | `symmetry_sweep.csv`, `symmetry_scores.csv`, `branch_consistency.csv`, `symmetry_success_rates.md` | `sweep-symmetry` |
| Discrete-fold re-score | `symmetry_sweep_rescored.csv`, `branch_consistency_discrete.csv`, `symmetry_success_rates_discrete.md` | `rescore` |
| ADD-S and MSSD | `bop_metrics.csv`, `bop_metrics.md` | `bop` |
| Logistic regression on the symmetry score | `sym_regression.md` | `regression` |
| Success vs number of distractors | `clutter_sweep.csv`, `clutter_success_rates.md` | `clutter` |
| Fitness as a failure signal | `confidence_signal.md` | `confidence` |
| Latency and throughput | `timing_summary.md` | `timing` |
| False alarms vs alignment error | `fp_vs_error.csv` | `propagation` |
| Template-difference inspection | `inspect_synthetic.csv`, `inspect_mvtec.csv`, `inspect_summary.md` | `inspection` |
| Inspection failures vs symmetry branches | `inspect_symmetry_check.md` | `symmetry-check` |
| Scan AUROC across identical reruns | `reproducibility.csv`, `reproducibility_report.md` | `reproducibility run`, `reproducibility report` |

## Figures

| What it shows | File (`figures/`) | Command |
|---|---|---|
| Success by symmetry class and part | `symmetry_sweep.png` | `sweep-symmetry --plot` |
| Success vs number of distractors | `clutter_sweep.png` | `clutter --plot` |
| Fitness of successful and failed trials | `confidence_signal.png` | `confidence` |
| Success vs mean registration time | `pareto.png` | `timing` |
| False alarms vs alignment error | `fp_vs_error.png` | `propagation --plot` |
| Inspection scan scores by part | `inspect_scores.png` | `inspection --plot` |

Commands are `regbench <command>`. `--plot` redraws a figure from its CSV
without running registrations. `regbench inspection` also writes per-scan
heatmaps to `figures/inspect/`, which are not committed.
