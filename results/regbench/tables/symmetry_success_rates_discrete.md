# Discrete-fold re-score (offline, from symmetry_sweep.csv poses)

success_dsym := success under the part's full detected symmetry group: discrete C_n fold
for asymmetric-labeled parts with a detected discrete order, continuous axis fold for the
axisymmetric class (unchanged from success_sym), plain metric otherwise. Same thresholds
(RRE < 2.0 deg AND RTE < 0.01 of d); 95% Wilson intervals in brackets.

## Detected discrete folds (asymmetric-labeled parts)

| part | class | order | curve floor | curve median | accept threshold | worst multiple |
|---|---|---|---|---|---|---|
| bracket_screw | asymmetric | - | 0.0020 | 0.0600 | 0.0223 | - |
| injection_pump | asymmetric | - | 0.0022 | 0.0219 | 0.0091 | - |
| star | asymmetric | 12 | 0.0037 | 0.0083 | 0.0053 | 0.0049 |
| cylinder | axisymmetric | cont. |  |  |  |  |
| washer | axisymmetric | cont. |  |  |  |  |
| thread | axisymmetric | cont. |  |  |  |  |

## Success rates (%) by part: plain metric

| symmetry_class / part / fold | easy | medium | hard |
|---|---|---|---|
| asymmetric / bracket_screw / - | 95.0 [87.8, 98.0] | 61.2 [50.3, 71.2] | 15.0 [8.8, 24.4] |
| asymmetric / injection_pump / - | 100.0 [95.4, 100.0] | 60.0 [49.0, 70.0] | 3.8 [1.3, 10.5] |
| asymmetric / star / C12 | 53.8 [42.9, 64.3] | 23.8 [15.8, 34.1] | 10.0 [5.2, 18.5] |
| axisymmetric / cylinder / cont. | 30.0 [21.1, 40.8] | 5.0 [2.0, 12.2] | 1.2 [0.2, 6.7] |
| axisymmetric / washer / cont. | 22.5 [14.7, 32.8] | 2.5 [0.7, 8.7] | 1.2 [0.2, 6.7] |
| axisymmetric / thread / cont. | 88.8 [80.0, 94.0] | 32.5 [23.2, 43.4] | 6.2 [2.7, 13.8] |

## Continuous-fold metric (success_sym, as swept)

| symmetry_class / part / fold | easy | medium | hard |
|---|---|---|---|
| asymmetric / bracket_screw / - | 95.0 [87.8, 98.0] | 61.2 [50.3, 71.2] | 15.0 [8.8, 24.4] |
| asymmetric / injection_pump / - | 100.0 [95.4, 100.0] | 60.0 [49.0, 70.0] | 3.8 [1.3, 10.5] |
| asymmetric / star / C12 | 53.8 [42.9, 64.3] | 23.8 [15.8, 34.1] | 10.0 [5.2, 18.5] |
| axisymmetric / cylinder / cont. | 86.2 [77.0, 92.1] | 47.5 [36.9, 58.3] | 18.8 [11.7, 28.7] |
| axisymmetric / washer / cont. | 58.8 [47.8, 68.9] | 21.2 [13.7, 31.4] | 16.2 [9.7, 25.8] |
| axisymmetric / thread / cont. | 100.0 [95.4, 100.0] | 45.0 [34.6, 55.9] | 10.0 [5.2, 18.5] |

## Full-group metric (success_dsym, this re-score)

| symmetry_class / part / fold | easy | medium | hard |
|---|---|---|---|
| asymmetric / bracket_screw / - | 95.0 [87.8, 98.0] | 61.2 [50.3, 71.2] | 15.0 [8.8, 24.4] |
| asymmetric / injection_pump / - | 100.0 [95.4, 100.0] | 60.0 [49.0, 70.0] | 3.8 [1.3, 10.5] |
| asymmetric / star / C12 | 75.0 [64.5, 83.2] | 51.2 [40.5, 61.9] | 31.2 [22.2, 42.1] |
| axisymmetric / cylinder / cont. | 86.2 [77.0, 92.1] | 47.5 [36.9, 58.3] | 18.8 [11.7, 28.7] |
| axisymmetric / washer / cont. | 58.8 [47.8, 68.9] | 21.2 [13.7, 31.4] | 16.2 [9.7, 25.8] |
| axisymmetric / thread / cont. | 100.0 [95.4, 100.0] | 45.0 [34.6, 55.9] | 10.0 [5.2, 18.5] |

## Branch consistency under the full detected group (continuous or
C_n): does the estimate keep reaching the SAME branch across
trials, or an effectively random one? theta_dsym_deg = branch
matched relative to T_gt (0 = true orientation); entropy_norm/
mode_frequency summarize its spread (0/1 = fully consistent,
1/1-of-n_bins = indistinguishable from a uniform-random branch
draw).

| part | fold_order | severity | entropy_norm | mode_frequency | n_bins | n |
|---|---|---|---|---|---|---|
| star | 12 | easy | 0.552 | 0.650 | 12 | 80 |
| star | 12 | medium | 0.714 | 0.475 | 12 | 80 |
| star | 12 | hard | 0.923 | 0.225 | 12 | 80 |
| cylinder | 0 | easy | 0.634 | 0.463 | 12 | 80 |
| cylinder | 0 | medium | 0.620 | 0.450 | 12 | 80 |
| cylinder | 0 | hard | 0.904 | 0.250 | 12 | 80 |
| washer | 0 | easy | 0.690 | 0.350 | 12 | 80 |
| washer | 0 | medium | 0.761 | 0.325 | 12 | 80 |
| washer | 0 | hard | 0.901 | 0.212 | 12 | 80 |
| thread | 0 | easy | 0.174 | 0.912 | 12 | 80 |
| thread | 0 | medium | 0.489 | 0.650 | 12 | 80 |
| thread | 0 | hard | 0.868 | 0.250 | 12 | 80 |

Cross-check against stored columns: plain mismatches 0/1440, continuous-fold mismatches 0.
