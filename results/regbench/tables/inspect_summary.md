# Inspection summary - scan-level template-difference detection

| part | method | scan AUROC | n_good/n_defect | median good score | median defect score | tau | mean point-AUROC (defect scans) |
|---|---|---|---|---|---|---|---|
| synth_cable_gland | fpfh_ransac | 0.950 | 20/20 | 0.0140 | 0.0330 | 0.00136 | 0.785 |
| synth_bunny | fpfh_ransac | 1.000 | 20/20 | 0.0137 | 0.0293 | 0.00578 | 0.810 |
| mvtec_cable_gland | fpfh_ransac | 0.453 | 21/65 | 0.1997 | 0.1326 | 0.02273 | 0.525 |
| mvtec_bagel | fpfh_ransac | 0.466 | 22/43 | 0.0712 | 0.0689 | 0.01883 | 0.608 |
| mvtec_dowel | fpfh_ransac | 0.512 | 26/52 | 0.0761 | 0.0778 | 0.01055 | 0.693 |
| mvtec_tire | fpfh_ransac | 0.399 | 25/54 | 0.0775 | 0.0638 | 0.01313 | 0.774 |
| mvtec_potato | fpfh_ransac | 0.591 | 22/46 | 0.4272 | 0.7433 | 1.14741 | 0.507 |
| synth_itodd_bracket_screw | fpfh_ransac | 0.995 | 20/20 | 0.0259 | 0.0366 | 0.00355 | 0.806 |
| synth_itodd_injection_pump | fpfh_ransac | 1.000 | 20/20 | 0.0101 | 0.0218 | 0.00200 | 0.775 |
| synth_itodd_star | fpfh_ransac | 1.000 | 20/20 | 0.0087 | 0.0317 | 0.00084 | 0.814 |

Per-defect-type scan AUROC (type vs all good):

- synth_cable_gland / bump (n=20): 0.950
- synth_bunny / bump (n=20): 1.000
- mvtec_cable_gland / bent (n=21): 0.390
- mvtec_cable_gland / hole (n=22): 0.587
- mvtec_cable_gland / thread (n=22): 0.381
- mvtec_bagel / crack (n=22): 0.486
- mvtec_bagel / hole (n=21): 0.446
- mvtec_dowel / bent (n=27): 0.510
- mvtec_dowel / cut (n=25): 0.514
- mvtec_tire / cut (n=27): 0.467
- mvtec_tire / hole (n=27): 0.332
- mvtec_potato / cut (n=23): 0.492
- mvtec_potato / hole (n=23): 0.690
- synth_itodd_bracket_screw / bump (n=20): 0.995
- synth_itodd_injection_pump / bump (n=20): 1.000
- synth_itodd_star / bump (n=20): 1.000

score = q0.995 of point distance / d | tau = q0.999 of pooled registered train_good distances
