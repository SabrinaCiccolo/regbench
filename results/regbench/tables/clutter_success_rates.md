# Clutter stress test: success rate (%) vs. distractor count

target = bracket_screw | severity = medium | 95% Wilson intervals in brackets

| method | n=0 | n=1 | n=2 | n=4 |
|---|---|---|---|---|
| icp_pt2pt | 13.3 [5.3, 29.7] | 6.7 [1.8, 21.3] | 3.3 [0.6, 16.7] | 20.0 [9.5, 37.3] |
| icp_pt2pl | 76.7 [59.1, 88.2] | 70.0 [52.1, 83.3] | 70.0 [52.1, 83.3] | 80.0 [62.7, 90.5] |
| fpfh_ransac | 93.3 [78.7, 98.2] | 60.0 [42.3, 75.4] | 60.0 [42.3, 75.4] | 20.0 [9.5, 37.3] |
| fgr | 96.7 [83.3, 99.4] | 70.0 [52.1, 83.3] | 70.0 [52.1, 83.3] | 40.0 [24.6, 57.7] |

## Pooled over methods

| n_distractors | success (%) | mean fitness (successes) | mean fitness (failures) |
|---|---|---|---|
| 0 | 70.0 [61.3, 77.5] | 1.000 | 0.547 |
| 1 | 51.7 [42.8, 60.4] | 0.654 | 0.363 |
| 2 | 50.8 [42.0, 59.6] | 0.486 | 0.238 |
| 4 | 40.0 [31.7, 48.9] | 0.316 | 0.135 |

30 trials/(n_distractors x method)
