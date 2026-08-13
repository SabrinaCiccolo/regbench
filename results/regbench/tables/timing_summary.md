# Latency & throughput by method x severity

| method | severity | mean_s | median_s | p95_s | throughput_hz |
|---|---|---|---|---|---|
| fgr | easy | 0.4895 | 0.4641 | 0.6544 | 2.04 |
| fgr | medium | 0.4325 | 0.4214 | 0.5636 | 2.31 |
| fgr | hard | 0.3467 | 0.2532 | 0.7499 | 2.88 |
| fpfh_ransac | easy | 0.5962 | 0.5874 | 0.7160 | 1.68 |
| fpfh_ransac | medium | 0.5919 | 0.5593 | 0.7954 | 1.69 |
| fpfh_ransac | hard | 0.7087 | 0.6322 | 1.1831 | 1.41 |
| icp_pt2pl | easy | 0.0382 | 0.0244 | 0.0936 | 26.18 |
| icp_pt2pl | medium | 0.1424 | 0.1082 | 0.3640 | 7.02 |
| icp_pt2pl | hard | 0.0869 | 0.0308 | 0.5193 | 11.51 |
| icp_pt2pt | easy | 0.0521 | 0.0382 | 0.1483 | 19.19 |
| icp_pt2pt | medium | 0.1211 | 0.1182 | 0.2659 | 8.26 |
| icp_pt2pt | hard | 0.0565 | 0.0314 | 0.2520 | 17.70 |

n = 50 trials/cell | cpu = 8x 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
