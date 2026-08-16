# Inspection track: are the registration 'failures' symmetry branches?

`reg_success` is the plain pose metric (RRE < 2.0 deg AND RTE < 0.01 of d). Scores are the pipeline's own scan
scores (q0.995 of point distance / d), split by that column: if a failed
pose landed on another branch of the part's rotational symmetry group the
surfaces still coincide, so the two groups' scores should be
indistinguishable and the scan AUROC should be unaffected.

| part | fold | split | n failed | n ok | median score (failed) | median score (ok) |
|---|---|---|---|---|---|---|
| synth_bunny | none | test_good | 0 | 20 | - | 0.0137 |
| synth_bunny | none | test_defect | 0 | 20 | - | 0.0293 |
| synth_cable_gland | none | test_good | 1 | 19 | 0.2551 | 0.0140 |
| synth_cable_gland | none | test_defect | 0 | 20 | - | 0.0330 |
| synth_itodd_bracket_screw | none | test_good | 0 | 20 | - | 0.0259 |
| synth_itodd_bracket_screw | none | test_defect | 0 | 20 | - | 0.0366 |
| synth_itodd_injection_pump | none | test_good | 0 | 20 | - | 0.0101 |
| synth_itodd_injection_pump | none | test_defect | 0 | 20 | - | 0.0218 |
| synth_itodd_star | C12 | test_good | 7 | 13 | 0.0104 | 0.0086 |
| synth_itodd_star | C12 | test_defect | 11 | 9 | 0.0316 | 0.0318 |

## Where the failed rotations actually landed

For a part with a detected discrete fold C_n, `dist to n-fold` is the
smallest angle between the trial's RRE and a multiple of 360/n degrees.
A value near zero means the estimate reached an exact symmetry branch,
not an arbitrary wrong pose.

| part | fold | n failed | RRE range (deg) | max dist to n-fold (deg) |
|---|---|---|---|---|
| synth_cable_gland | none | 1 | 179.97-179.97 | - |
| synth_itodd_star | C12 | 18 | 29.90-90.02 | 0.21 |

source: inspect_synthetic.csv
