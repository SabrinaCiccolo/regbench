# Reproducing results

## Data

Nothing under `data/` is committed. Download:

| Dataset | Source | License | Expected location |
|---|---|---|---|
| MVTec ITODD, Base Package (CAD models) | https://www.mvtec.com/company/research/datasets/mvtec-itodd | CC BY-NC-SA 4.0 | `data/base_package/models/cad_models/*.ply` |
| MVTec 3D-AD | https://www.mvtec.com/company/research/datasets/mvtec-3d-ad | CC BY-NC-SA 4.0 | `data/mvtec3d/<class>/{train,test}/<type>/{xyz,gt}/` |
| Stanford Bunny | downloaded by Open3D (`open3d.data.BunnyMesh`) | see the Stanford 3D Scanning Repository | `data/open3d/` |

Then build the templates and the inspection datasets:

```
regbench build-template all     # bunny.ply, cable_gland.ply, itodd_<part>.ply in data/template/
regbench build-industrial all   # data/industrial/<part>/{train_good,test_good,test_defect}/
```

Both builders are deterministic.

## Experiments

Inputs: *ITODD* needs `data/template/itodd_*.ply`, *inspection* needs
`data/industrial/`, *committed* reads only CSVs in `results/regbench/tables/`.

| Experiment | Command | Inputs | Run time |
|---|---|---|---|
| Registration grid | `regbench grid` | ITODD | minutes |
| Registration grid, Bunny | `regbench grid --template data/template/bunny.ply --tag bunny` | Bunny | minutes |
| Symmetry sweep | `regbench sweep-symmetry` | ITODD | minutes |
| Discrete-fold re-score | `regbench rescore` | ITODD, sweep CSV | seconds |
| BOP metrics | `regbench bop` | ITODD, sweep CSV | seconds |
| Regression | `regbench regression` | committed | seconds |
| Clutter | `regbench clutter` | ITODD | minutes |
| Confidence signal | `regbench confidence` | committed | seconds |
| Timing summary | `regbench timing` | committed | seconds |
| Inspection | `regbench inspection` | inspection | minutes |
| Symmetry check | `regbench symmetry-check` | templates, inspection CSV | seconds |
| Propagation | `regbench propagation` | inspection | minutes |
| Reproducibility | `regbench reproducibility run`, then `report` | inspection | minutes |

Run times are orders of magnitude on a laptop CPU. `--smoke` (grid,
sweep-symmetry, clutter, inspection) runs a reduced configuration.

## Determinism

Every perturbation, subsample and seed comes from `repro.py::stable_rng`, so
the offline commands reproduce their tables exactly. Open3D's FPFH, RANSAC and
FGR are multithreaded and not order-deterministic: a rerun of the grids moves
individual trials, and `time_s` depends on the machine (`cpu` column).

## Verification

```
regbench verify
```

regenerates every committed artifact into a temporary directory and compares it
with the committed one: offline outputs must match exactly, stochastic grids
must have every success-rate cell within the 95% Wilson interval of the
committed cell. Rows whose input data is missing are reported as `SKIPPED`.
`regbench verify --only tables/bop` restricts the check to matching paths.
