# BOP-style metrics (ADD-S, MSSD) on the symmetry-sweep poses

ADD-S and MSSD on a fixed 150-point subsample of each part's preprocessed template, thresholded at 0.1 x AABB diagonal; MSSD minimizes over the part's detected symmetry group. MSPD is not computed (it needs a camera model).

## Success rates (%) by part: ADD-S < 0.1d

| symmetry_class / part | easy | medium | hard |
|---|---|---|---|
| asymmetric / bracket_screw | 100.0 [95.4, 100.0] | 86.2 [77.0, 92.1] | 30.0 [21.1, 40.8] |
| asymmetric / injection_pump | 100.0 [95.4, 100.0] | 95.0 [87.8, 98.0] | 40.0 [30.0, 51.0] |
| asymmetric / star | 100.0 [95.4, 100.0] | 78.8 [68.6, 86.3] | 35.0 [25.5, 45.9] |
| axisymmetric / cylinder | 97.5 [91.3, 99.3] | 81.2 [71.3, 88.3] | 26.2 [17.9, 36.8] |
| axisymmetric / washer | 100.0 [95.4, 100.0] | 82.5 [72.7, 89.3] | 38.8 [28.8, 49.7] |
| axisymmetric / thread | 100.0 [95.4, 100.0] | 85.0 [75.6, 91.2] | 26.2 [17.9, 36.8] |

## Success rates (%) by part: MSSD < 0.1d (symmetry-aware)

| symmetry_class / part | easy | medium | hard |
|---|---|---|---|
| asymmetric / bracket_screw | 98.8 [93.3, 99.8] | 67.5 [56.6, 76.8] | 18.8 [11.7, 28.7] |
| asymmetric / injection_pump | 100.0 [95.4, 100.0] | 65.0 [54.1, 74.5] | 3.8 [1.3, 10.5] |
| asymmetric / star | 85.0 [75.6, 91.2] | 58.8 [47.8, 68.9] | 33.8 [24.3, 44.6] |
| axisymmetric / cylinder | 88.8 [80.0, 94.0] | 55.0 [44.1, 65.4] | 20.0 [12.7, 30.0] |
| axisymmetric / washer | 82.5 [72.7, 89.3] | 65.0 [54.1, 74.5] | 31.2 [22.2, 42.1] |
| axisymmetric / thread | 100.0 [95.4, 100.0] | 50.0 [39.3, 60.7] | 11.2 [6.0, 20.0] |

## Agreement with the RRE/RTE symmetry-aware criterion (success_sym)

Per-trial success agreement, ADD-S vs success_sym: 1027/1440 (71.3%).
Per-trial success agreement, MSSD vs success_sym: 1273/1440 (88.4%).

Spearman rho of cell success rates across the 18 (part, severity) cells:
success_sym vs ADD-S: rho=0.876, p=0.0000. success_sym vs MSSD: rho=0.939, p=0.0000.
