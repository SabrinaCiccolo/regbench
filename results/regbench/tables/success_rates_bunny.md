# Success rates (%) by severity - bunny.ply

| method | easy | medium | hard | mean_rre_deg | mean_time_s |
|---|---|---|---|---|---|
| fgr | 100.0 | 100.0 | 28.0 | 22.125 | 1.396 |
| fpfh_ransac | 100.0 | 100.0 | 60.0 | 17.727 | 1.39 |
| icp_pt2pl | 100.0 | 72.0 | 12.0 | 30.206 | 0.335 |
| icp_pt2pt | 100.0 | 14.0 | 6.0 | 33.811 | 0.28 |

success := RRE < 2.0 deg AND RTE < 0.01 (of d) | 50 trials/severity
