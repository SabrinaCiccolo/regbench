# Confidence signal: does fitness predict failure?

AUROC of each diagnostic of the final ICP step at ranking failed
trials above successful ones.

| signal | AUROC (all methods) | n |
|---|---|---|
| fitness | 0.9955 | 600 |
| inlier_rmse | 0.8949 | 600 |
| n_inliers | 0.9753 | 600 |

## fitness AUROC by method

| method | AUROC | n |
|---|---|---|
| icp_pt2pt | 0.9900 | 150 |
| icp_pt2pl | 1.0000 | 150 |
| fpfh_ransac | 1.0000 | 150 |
| fgr | 1.0000 | 150 |

## Zero-false-alarm threshold

Threshold = 0.999676, the lowest fitness of any successful
trial. `fitness < 0.9997` flags 227/229 (99.1%) of the failed trials
and no successful trial.

RRE of the 2 missed failure(s): 0.93, 2.15 deg.

The threshold is calibrated on the single-object grid; fitness
drops with clutter (clutter_success_rates.md), and a pose on another
symmetry branch keeps a high fitness.
