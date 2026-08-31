# regbench

A benchmark for rigid point-cloud registration under part symmetry and
clutter. A template cloud is cut, noised and moved by a known rigid transform,
and four classical methods (point-to-point ICP, point-to-plane ICP,
FPFH + RANSAC, FGR) must recover the transform. For parts with rotational
symmetry, a plain rotation-error metric counts poses on another symmetry branch
as failures; regbench measures how large that effect is and how registration
error propagates into template-difference inspection.

The method is described in [`docs/method.md`](docs/method.md).

## Results

All tables are in [`results/regbench/tables/`](results/regbench/tables/); the
index is [`docs/provenance.md`](docs/provenance.md).

- **The plain metric under-reports symmetric parts.** On three axisymmetric
  ITODD parts at medium severity, success is 13.3% with the plain metric and
  37.9% with the symmetry-aware one, on the same poses; asymmetric parts are
  unchanged at 48.3%
  ([`symmetry_success_rates.md`](results/regbench/tables/symmetry_success_rates.md)).
- **Discrete symmetry matters too.** The `star` part, labeled asymmetric, has a
  detected twelve-fold symmetry; scoring it over that group raises its success
  from 53.8% to 75.0% (easy) and from 10.0% to 31.2% (hard)
  ([`symmetry_success_rates_discrete.md`](results/regbench/tables/symmetry_success_rates_discrete.md)).
- **Symmetry predicts failure.** In a trial-level logistic regression with
  severity and method as covariates, the continuous symmetry score has
  coefficient +0.99 (95% CI [+0.84, +1.18]); less symmetric parts succeed more
  often ([`sym_regression.md`](results/regbench/tables/sym_regression.md)).
  The BOP MSSD criterion agrees with the symmetry-aware metric on 88.4% of
  trials ([`bop_metrics.md`](results/regbench/tables/bop_metrics.md)).
- **Clutter.** With 0 to 4 distractor parts in the scene, pooled success drops
  from 70.0% to 40.0%; FPFH + RANSAC drops from 93.3% to 20.0%
  ([`clutter_success_rates.md`](results/regbench/tables/clutter_success_rates.md)).
- **Fitness flags failures.** Open3D's ICP fitness ranks failed trials above
  successful ones with AUROC 0.996; the lowest fitness of any successful trial,
  used as a threshold, flags 227 of 229 failures
  ([`confidence_signal.md`](results/regbench/tables/confidence_signal.md)).
- **Inspection.** Template-difference inspection separates defective scans
  with scan-level AUROC 0.95–1.00 on synthetic bumps and 0.40–0.59 on
  MVTec 3D-AD
  ([`inspect_summary.md`](results/regbench/tables/inspect_summary.md)). The
  mean false-alarm rate stays at or below 0.004 up to 2° or 0.01 d of alignment
  error and reaches 0.28 at 8° and 0.50 at 0.04 d
  ([`fp_vs_error.png`](results/regbench/figures/fp_vs_error.png)).

## Installation

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Usage

The input data (MVTec ITODD, MVTec 3D-AD, Stanford Bunny) is not included; see
[`docs/reproducing.md`](docs/reproducing.md) for where to get it.

```
regbench build-template all          # data/template/*.ply
regbench build-industrial all        # data/industrial/, inspection datasets

regbench grid
regbench sweep-symmetry
regbench rescore
regbench clutter
regbench inspection

regbench verify                      # regenerate results and compare
```

Every command and its outputs are listed in
[`docs/reproducing.md`](docs/reproducing.md).

## Layout

```
src/regbench/              perturbation, methods, metrics, symmetry, inspection scoring
src/regbench/experiments/  one module per experiment (regbench <command>)
src/regbench/build/        template and inspection-dataset builders
src/regbench/config.yaml   every parameter
results/regbench/          committed tables and figures
tests/regbench/            pytest suite
docs/                      method, results index, reproduction
```

## License

MIT, see [`LICENSE`](LICENSE). MVTec ITODD and MVTec 3D-AD are distributed by
MVTec under CC BY-NC-SA 4.0 and are not included in this repository.
