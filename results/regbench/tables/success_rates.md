# Success rates (%) by severity - itodd_bracket_screw.ply

| method | easy | medium | hard | mean_rre_deg | mean_time_s |
|---|---|---|---|---|---|
| fgr | 100.0 | 98.0 | 34.0 | 21.269 | 0.423 |
| fpfh_ransac | 100.0 | 88.0 | 26.0 | 42.235 | 0.632 |
| icp_pt2pl | 100.0 | 78.0 | 18.0 | 27.377 | 0.089 |
| icp_pt2pt | 92.0 | 8.0 | 0.0 | 32.627 | 0.077 |

success := RRE < 2.0 deg AND RTE < 0.01 (of d) | 50 trials/severity
