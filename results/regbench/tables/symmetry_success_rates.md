# Success rates (%) by symmetry class and severity

Averaged over all 4 methods; 95% Wilson intervals in brackets.

## Plain metric (success := RRE < 2.0 deg AND RTE < 0.01 of d)

| symmetry_class | easy | medium | hard |
|---|---|---|---|
| asymmetric | 82.9 [77.6, 87.2] | 48.3 [42.1, 54.6] | 9.6 [6.5, 14.0] |
| axisymmetric | 47.1 [40.9, 53.4] | 13.3 [9.6, 18.2] | 2.9 [1.4, 5.9] |

## Symmetry-aware metric (same thresholds, error modulo the part's
rotation-symmetry axis for the axisymmetric class; asymmetric rows
unchanged by construction)

| symmetry_class | easy | medium | hard |
|---|---|---|---|
| asymmetric | 82.9 [77.6, 87.2] | 48.3 [42.1, 54.6] | 9.6 [6.5, 14.0] |
| axisymmetric | 81.7 [76.3, 86.1] | 37.9 [32.0, 44.2] | 15.0 [11.0, 20.1] |

## By part (plain | symmetry-aware)

| symmetry_class / part | easy | medium | hard |
|---|---|---|---|
| asymmetric / bracket_screw | 95.0 [87.8, 98.0] | 61.2 [50.3, 71.2] | 15.0 [8.8, 24.4] |
| asymmetric / injection_pump | 100.0 [95.4, 100.0] | 60.0 [49.0, 70.0] | 3.8 [1.3, 10.5] |
| asymmetric / star | 53.8 [42.9, 64.3] | 23.8 [15.8, 34.1] | 10.0 [5.2, 18.5] |
| axisymmetric / cylinder | 30.0 [21.1, 40.8] | 5.0 [2.0, 12.2] | 1.2 [0.2, 6.7] |
| axisymmetric / washer | 22.5 [14.7, 32.8] | 2.5 [0.7, 8.7] | 1.2 [0.2, 6.7] |
| axisymmetric / thread | 88.8 [80.0, 94.0] | 32.5 [23.2, 43.4] | 6.2 [2.7, 13.8] |

| symmetry_class / part | easy | medium | hard |
|---|---|---|---|
| asymmetric / bracket_screw | 95.0 [87.8, 98.0] | 61.2 [50.3, 71.2] | 15.0 [8.8, 24.4] |
| asymmetric / injection_pump | 100.0 [95.4, 100.0] | 60.0 [49.0, 70.0] | 3.8 [1.3, 10.5] |
| asymmetric / star | 53.8 [42.9, 64.3] | 23.8 [15.8, 34.1] | 10.0 [5.2, 18.5] |
| axisymmetric / cylinder | 86.2 [77.0, 92.1] | 47.5 [36.9, 58.3] | 18.8 [11.7, 28.7] |
| axisymmetric / washer | 58.8 [47.8, 68.9] | 21.2 [13.7, 31.4] | 16.2 [9.7, 25.8] |
| axisymmetric / thread | 100.0 [95.4, 100.0] | 45.0 [34.6, 55.9] | 10.0 [5.2, 18.5] |

## Continuous symmetry score (rotational self-overlap, low = symmetric)

| part | class | sym_score | sym_score_min |
|---|---|---|---|
| bracket_screw | asymmetric | 0.0533 | 0.0090 |
| injection_pump | asymmetric | 0.0213 | 0.0082 |
| star | asymmetric | 0.0078 | 0.0037 |
| cylinder | axisymmetric | 0.0049 | 0.0032 |
| washer | axisymmetric | 0.0039 | 0.0038 |
| thread | axisymmetric | 0.0079 | 0.0059 |

Spearman rank correlation, sym_score vs overall per-part success rate (plain, n=6 parts): rho = 1.000

## Branch consistency (axisymmetric parts only): does the estimate
keep reaching the SAME symmetry-group branch across independent
trials, or an effectively random one? theta_deg is the branch
folded_evaluate matched T_est to, relative to T_gt (0 = the true
orientation); entropy_norm/mode_frequency summarize its spread over
trials (0/1 = fully consistent, 1/1-of-n_bins = indistinguishable
from a uniform-random branch draw).

| part | severity | method | entropy_norm | mode_frequency | n_bins | n |
|---|---|---|---|---|---|---|
| cylinder | easy | (all methods) | 0.634 | 0.463 | 12 | 80 |
| cylinder | easy | icp_pt2pt | 0.226 | 0.750 | 12 | 20 |
| cylinder | easy | icp_pt2pl | 0.271 | 0.600 | 12 | 20 |
| cylinder | easy | fpfh_ransac | 0.744 | 0.250 | 12 | 20 |
| cylinder | easy | fgr | 0.552 | 0.500 | 12 | 20 |
| cylinder | medium | (all methods) | 0.620 | 0.450 | 12 | 80 |
| cylinder | medium | icp_pt2pt | 0.457 | 0.550 | 12 | 20 |
| cylinder | medium | icp_pt2pl | 0.246 | 0.700 | 12 | 20 |
| cylinder | medium | fpfh_ransac | 0.705 | 0.300 | 12 | 20 |
| cylinder | medium | fgr | 0.445 | 0.450 | 12 | 20 |
| cylinder | hard | (all methods) | 0.904 | 0.250 | 12 | 80 |
| cylinder | hard | icp_pt2pt | 0.745 | 0.300 | 12 | 20 |
| cylinder | hard | icp_pt2pl | 0.703 | 0.400 | 12 | 20 |
| cylinder | hard | fpfh_ransac | 0.866 | 0.250 | 12 | 20 |
| cylinder | hard | fgr | 0.745 | 0.300 | 12 | 20 |
| washer | easy | (all methods) | 0.690 | 0.350 | 12 | 80 |
| washer | easy | icp_pt2pt | 0.277 | 0.550 | 12 | 20 |
| washer | easy | icp_pt2pl | 0.279 | 0.500 | 12 | 20 |
| washer | easy | fpfh_ransac | 0.781 | 0.350 | 12 | 20 |
| washer | easy | fgr | 0.591 | 0.350 | 12 | 20 |
| washer | medium | (all methods) | 0.761 | 0.325 | 12 | 80 |
| washer | medium | icp_pt2pt | 0.478 | 0.450 | 12 | 20 |
| washer | medium | icp_pt2pl | 0.452 | 0.550 | 12 | 20 |
| washer | medium | fpfh_ransac | 0.832 | 0.200 | 12 | 20 |
| washer | medium | fgr | 0.627 | 0.400 | 12 | 20 |
| washer | hard | (all methods) | 0.901 | 0.212 | 12 | 80 |
| washer | hard | icp_pt2pt | 0.772 | 0.300 | 12 | 20 |
| washer | hard | icp_pt2pl | 0.772 | 0.300 | 12 | 20 |
| washer | hard | fpfh_ransac | 0.867 | 0.200 | 12 | 20 |
| washer | hard | fgr | 0.765 | 0.250 | 12 | 20 |
| thread | easy | (all methods) | 0.174 | 0.912 | 12 | 80 |
| thread | easy | icp_pt2pt | 0.131 | 0.900 | 12 | 20 |
| thread | easy | icp_pt2pl | -0.000 | 1.000 | 12 | 20 |
| thread | easy | fpfh_ransac | 0.360 | 0.750 | 12 | 20 |
| thread | easy | fgr | -0.000 | 1.000 | 12 | 20 |
| thread | medium | (all methods) | 0.489 | 0.650 | 12 | 80 |
| thread | medium | icp_pt2pt | 0.498 | 0.500 | 12 | 20 |
| thread | medium | icp_pt2pl | 0.277 | 0.750 | 12 | 20 |
| thread | medium | fpfh_ransac | 0.640 | 0.450 | 12 | 20 |
| thread | medium | fgr | 0.131 | 0.900 | 12 | 20 |
| thread | hard | (all methods) | 0.868 | 0.250 | 12 | 80 |
| thread | hard | icp_pt2pt | 0.743 | 0.350 | 12 | 20 |
| thread | hard | icp_pt2pl | 0.745 | 0.300 | 12 | 20 |
| thread | hard | fpfh_ransac | 0.783 | 0.300 | 12 | 20 |
| thread | hard | fgr | 0.838 | 0.250 | 12 | 20 |

20 trials/severity/part
