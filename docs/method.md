# Method

All parameters are in [`src/regbench/config.yaml`](../src/regbench/config.yaml).

## Perturb and recover

**Preprocessing** (`cloud_io.py::preprocess_cloud`). Statistical outlier removal
(20 neighbours, 2.0 std), recentring to the centroid, and voxel downsampling to
`d / 100`, where `d` is the diagonal of the axis-aligned bounding box of the
cleaned cloud. `d` and the voxel size are computed once per template and reused
for every trial.

**Trial** (`perturb.py::make_trial`). A copy of the template is cut by a random
half-plane that keeps a fraction `keep_frac` of the points, perturbed by
Gaussian noise of standard deviation `noise_rel · d`, and moved by a rigid
transform `T_pert` with rotation angle uniform in `[0, rot_max_deg]` about a
uniform axis and translation of uniform direction and magnitude in
`[0, trans_max_rel · d]`. The ground truth is `T_gt = T_pert⁻¹`.

| Severity | rot_max_deg | trans_max_rel | noise_rel | keep_frac |
|---|---|---|---|---|
| easy | 10 | 0.05 | 0.001 | 1.00 |
| medium | 45 | 0.25 | 0.0025 | 0.85 |
| hard | 180 | 0.5 | 0.005 | 0.60 |

Each trial is seeded from its key (e.g. severity and trial index) through
`repro.py::stable_rng`, and all methods register the same source.

## Registration methods

`methods.py`, on top of `open3d.pipelines.registration`. The source is
downsampled to the template's voxel size. Distances are multiples of the voxel
size `v`.

| Method | Coarse step | Refinement |
|---|---|---|
| `icp_pt2pt` | none (identity) | point-to-point ICP, max distance 2v |
| `icp_pt2pl` | none (identity) | point-to-plane ICP, max distance 2v |
| `fpfh_ransac` | FPFH (radius 5v) + RANSAC (3 points, distance 1.5v, edge-length check 0.9, 100k iterations) | point-to-plane ICP |
| `fgr` | FPFH + Fast Global Registration (distance 1.5v) | point-to-plane ICP |

Normals use radius 2v and 30 neighbours. The template's normals and features
are computed once; the reported time covers the source side only. If Open3D
raises (no usable correspondences), the trial is scored with the identity pose.

## Metrics

With `T_est = (R_est, t_est)` and `T_gt = (R_gt, t_gt)`:

$$\mathrm{RRE} = \arccos\frac{\operatorname{tr}(R_{est}^\top R_{gt}) - 1}{2}, \qquad
\mathrm{RTE} = \frac{\lVert t_{est} - t_{gt} \rVert}{d}$$

A trial succeeds if RRE < 2° and RTE < 0.01.

**Symmetry-aware metric** (`symmetry.py::folded_evaluate`). For a part with
symmetry axis `a` through `c`, let `S_θ` be the rotation by `θ` about that line,
`S_θ = [R_a(θ) | c − R_a(θ) c]`. Every `S_θ T_gt` is an equally valid ground truth.
The errors are reported at

$$\theta^* = \arg\min_\theta \max\left(\frac{\mathrm{RRE}_\theta}{2^\circ}, \frac{\mathrm{RTE}_\theta}{0.01}\right)$$

and the trial succeeds if some `θ` passes both thresholds. `θ` runs over a
0.25° grid for a continuous axis and over the multiples of `360°/n` for a
discrete `C_n` fold.

**Symmetry detection** (`symmetry.py`). The self-overlap of a cloud `P` at angle `θ` is

$$\sigma(\theta) = \frac{1}{|P|\,d} \sum_{p \in P} \min_{q \in P} \lVert S_\theta\, p - q \rVert .$$

The symmetry axis is the PCA axis, through the centroid, with the lowest mean
`σ` over 45°, 90°, 135° and 180°. `sym_score` is the mean of `σ` over
10°, 20°, …, 180° on that axis (low means symmetric). A discrete fold is the
largest `n ≤ 24` for which `σ(k · 360°/n)` lies below
`floor + 0.35 · (median − floor)` for every `k`, where floor and median are
taken over the 1° curve; curves with `median < 1.6 · floor` have no discrete
fold.

**Branch consistency** (`symmetry.py::branch_consistency`). The winning `θ*` of
repeated trials is binned (one bin per branch of `C_n`, or 30° bins for a
continuous axis); the normalized Shannon entropy is 0 when every trial reaches
the same branch and 1 when branches are uniformly random.

**Wilson intervals.** Success rates are reported with 95% Wilson score
intervals.

**BOP metrics** (`experiments/bop.py`). On a fixed 150-point subsample of each
template, ADD-S is the mean nearest-neighbour distance between the template
under `T_est` and under `T_gt`, and MSSD the maximum distance between
corresponding points minimized over the part's symmetry group (a 2° grid for a
continuous axis). Both are divided by `d` and pass below 0.1.

## Experiments

**Grid** (`grid`). 4 methods × 3 severities × 50 trials on one ITODD part
(`bracket_screw`) and on the Stanford Bunny.

**Symmetry sweep** (`sweep-symmetry`). 4 methods × 3 severities × 20 trials on
six ITODD parts: `bracket_screw`, `injection_pump` and `star` labeled
asymmetric, `cylinder`, `washer` and `thread` labeled axisymmetric. The
symmetry-aware metric uses the detected axis for the axisymmetric parts. Poses
are stored, so `rescore` (discrete folds of the asymmetric-labeled parts),
`bop` and `regression` run offline.

**Regression** (`regression`). Logistic regression of per-trial success on the
z-scored `sym_score`, severity and method, with a 1000-sample bootstrap
interval for the `sym_score` coefficient and McFadden pseudo-R² with and
without it.

**Clutter** (`clutter`). The `bracket_screw` source at medium severity plus 0,
1, 2 or 4 distractors. Each distractor is one of the other five ITODD parts,
subsampled to 3000 points, rotated at random and placed at `0.6–1.3 · d` from
the origin. 30 trials per distractor count and method.

**Confidence** (`confidence`). AUROC of the final ICP step's fitness,
inlier RMSE and inlier count at ranking failed grid trials above successful
ones.

**Inspection** (`inspection`). Each part has `train_good`, `test_good` and
`test_defect` scans (`build/make_industrial.py`):

- synthetic parts: scans made from a template at medium severity; defect scans
  carry a raised-cosine bump of radius 0.10 d and height 0.04 d along the
  normals (`perturb.py::inject_bump`), and every scan stores its true pose and
  defect mask;
- MVTec 3D-AD parts: organized scans with the background plane removed
  (RANSAC) and the largest DBSCAN cluster kept, with per-point labels from the
  2D ground-truth masks (`mvtec.py`).

The first `train_good` scan is the template. Every other scan is registered
onto it with `fpfh_ransac`; each point's score is its distance to the nearest
template point. The scan score is the 0.995 quantile of the point scores
divided by `d`; the point threshold τ is the 0.999 quantile of the pooled
scores of the registered `train_good` scans.

**Propagation** (`propagation`). Defect scans of the synthetic cable gland are
aligned by their true pose and moved by known offsets: rotation only
(0–16°), translation only (0–0.08 d) and a joint grid, three random
directions per magnitude. τ comes from the true-pose-aligned `train_good`
scans. The false-positive rate is counted outside the defect dilated by
1.5 × the bump radius, the true-positive rate inside the defect. Real
registrations by every method are added as points.

**Reproducibility** (`reproducibility`). The inspection pipeline is rerun ten
times with identical seeds on two parts; only Open3D's multithreaded RANSAC
changes the result.

## Equation-to-code map

| What | Implemented by |
|---|---|
| RRE, RTE | `metrics.py::rre_deg`, `metrics.py::rte_rel` |
| Plain success | `metrics.py::evaluate` |
| `S_θ T_gt` | `symmetry.py::fold_references` |
| Symmetry-aware success, `θ*` | `symmetry.py::folded_evaluate` |
| Self-overlap `σ(θ)` | `symmetry.py::self_overlap` |
| Symmetry axis, `sym_score` | `symmetry.py::detect_symmetry_axis` |
| Discrete fold | `symmetry.py::detect_discrete_fold` |
| Branch entropy | `symmetry.py::branch_consistency` |
| Wilson interval | `symmetry.py::wilson_ci` |
| ADD-S, MSSD | `experiments/bop.py::add_s`, `experiments/bop.py::mssd` |
| Scan score, τ | `anomaly.py::scan_score`, `anomaly.py::calibrate_tau` |
| False- and true-positive rates | `anomaly.py::exclusion_mask`, `anomaly.py::fp_tp_rates` |

Paths are relative to `src/regbench/`.
