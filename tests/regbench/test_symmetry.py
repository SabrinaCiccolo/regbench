import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from regbench.metrics import evaluate
from regbench.perturb import make_trial
from regbench.repro import stable_rng
from regbench.symmetry import (PROBE_THETAS, branch_consistency, detect_discrete_fold,
                               detect_symmetry_axis, fold_references, folded_evaluate,
                               self_overlap, wilson_ci)

THR = {"rre_deg": 2.0, "rte_rel": 0.01}


def cylinder_pts(n_ring=72, n_z=40, radius=0.3, height=1.0, center=None):
    """Dense open cylinder about the z axis (continuous rotational symmetry)."""
    theta = np.linspace(0.0, 2 * np.pi, n_ring, endpoint=False)
    z = np.linspace(-height / 2, height / 2, n_z)
    tt, zz = np.meshgrid(theta, z)
    pts = np.stack([radius * np.cos(tt).ravel(), radius * np.sin(tt).ravel(),
                    zz.ravel()], axis=1)
    if center is not None:
        pts = pts + center
    return pts


def T_from(R=np.eye(3), t=np.zeros(3)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def test_detect_axis_on_cylinder_off_center():
    center = np.array([0.5, -0.2, 0.3])
    pts = cylinder_pts(center=center)
    d = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axis, info = detect_symmetry_axis(pts, d)
    assert abs(axis[2]) > 0.999                      # z axis found (sign-free)
    assert info["sym_score"] < 0.005                 # near-perfect self-overlap
    assert np.allclose(info["center"], pts.mean(0))


def test_folded_forgives_azimuth_but_not_offaxis():
    """Azimuth spin about the symmetry axis: plain RRE = theta, folded ~ 0.
    The same spin about an off-axis direction stays a failure under both."""
    pts = cylinder_pts()
    d = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axis, info = detect_symmetry_axis(pts, d)
    rng = stable_rng("sym_fold", 0)
    sev = {"rot_max_deg": 45, "trans_max_rel": 0.25, "noise_rel": 0.0,
           "keep_frac": 1.0}
    _, T_gt = make_trial(pts, sev, d, rng)

    spin = Rotation.from_rotvec(np.deg2rad(90.0) * axis).as_matrix()
    T_est = T_from(spin @ T_gt[:3, :3],
                   spin @ T_gt[:3, 3] + info["center"] - spin @ info["center"])
    plain = evaluate(T_est, T_gt, d, THR)
    folded = folded_evaluate(T_est, T_gt, d, THR, axis, info["center"])
    assert plain["rre_deg"] == pytest.approx(90.0, abs=1e-6)
    assert not plain["success"]
    assert folded["rre_sym_deg"] < 0.3               # theta grid resolution
    assert folded["rte_sym"] < 1e-6
    assert folded["success_sym"]
    assert folded["theta_deg"] == pytest.approx(90.0, abs=0.3)

    off_axis = np.array([1.0, 0.0, 0.0]) if abs(axis[2]) > 0.9 else np.array(
        [0.0, 0.0, 1.0])
    spin_bad = Rotation.from_rotvec(np.deg2rad(90.0) * off_axis).as_matrix()
    T_bad = T_from(spin_bad @ T_gt[:3, :3], spin_bad @ T_gt[:3, 3])
    folded_bad = folded_evaluate(T_bad, T_gt, d, THR, axis, info["center"])
    assert folded_bad["rre_sym_deg"] > 10.0
    assert not folded_bad["success_sym"]


def test_folded_none_axis_equals_plain():
    rng = stable_rng("sym_plain", 0)
    pts = rng.normal(size=(300, 3))
    d = 1.0
    sev = {"rot_max_deg": 45, "trans_max_rel": 0.25, "noise_rel": 0.0,
           "keep_frac": 1.0}
    _, T_gt = make_trial(pts, sev, d, rng)
    spin = Rotation.from_rotvec(np.deg2rad(5.0) * np.array([0, 0, 1.0])).as_matrix()
    T_est = T_from(spin @ T_gt[:3, :3], T_gt[:3, 3])
    plain = evaluate(T_est, T_gt, d, THR)
    folded = folded_evaluate(T_est, T_gt, d, THR, None, np.zeros(3))
    assert folded["rre_sym_deg"] == pytest.approx(plain["rre_deg"])
    assert folded["rte_sym"] == pytest.approx(plain["rte"])
    assert folded["success_sym"] == plain["success"]


def test_fold_references_theta0_is_gt():
    T_gt = T_from(Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix(),
                  np.array([1.0, 2.0, 3.0]))
    refs = fold_references(T_gt, np.array([0.0, 0.0, 1.0]),
                           np.array([0.5, 0.5, 0.0]), np.array([0.0]))
    assert np.allclose(refs[0], T_gt)


def five_fold_pts(n_phi=360, n_z=30, r0=0.3, mod=0.3, height=1.0, jitter=0.002):
    """Prism whose cross-section radius is r0*(1 + mod*cos(5*phi)): exact C_5
    rotational symmetry, no continuous symmetry. Jitter breaks the sampling
    periodicity so overlap floors reflect noise, not grid coincidence."""
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    z = np.linspace(-height / 2, height / 2, n_z)
    pp, zz = np.meshgrid(phi, z)
    r = r0 * (1.0 + mod * np.cos(5 * pp))
    pts = np.stack([(r * np.cos(pp)).ravel(), (r * np.sin(pp)).ravel(),
                    zz.ravel()], axis=1)
    return pts + stable_rng("five_fold", 0).normal(scale=jitter, size=pts.shape)


def test_detect_discrete_fold_finds_c5():
    pts = five_fold_pts()
    d = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axis, info = detect_symmetry_axis(pts, d)
    assert abs(axis[2]) > 0.999
    n, dinfo = detect_discrete_fold(pts, axis, info["center"], d)
    assert n == 5
    assert dinfo["fold_overlap_max"] <= dinfo["threshold"]


def test_detect_discrete_fold_none_on_flat_and_asymmetric():
    # jittered cylinder: continuous symmetry = flat curve, no discrete structure
    pts = cylinder_pts()
    pts = pts + stable_rng("flat", 0).normal(scale=0.002, size=pts.shape)
    d = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axis, info = detect_symmetry_axis(pts, d)
    n, _ = detect_discrete_fold(pts, axis, info["center"], d)
    assert n is None
    # generic random cloud: modulated curve but no theta maps it onto itself
    pts = stable_rng("asym_cloud", 0).normal(size=(400, 3))
    d = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axis, info = detect_symmetry_axis(pts, d)
    n, _ = detect_discrete_fold(pts, axis, info["center"], d)
    assert n is None


def test_folded_evaluate_discrete_thetas():
    """C_5 fold forgives a 72-degree spin but not a 36-degree one (the dense
    continuous fold would wrongly forgive both)."""
    pts = five_fold_pts()
    d = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axis, info = detect_symmetry_axis(pts, d)
    rng = stable_rng("dsym_fold", 0)
    sev = {"rot_max_deg": 45, "trans_max_rel": 0.25, "noise_rel": 0.0,
           "keep_frac": 1.0}
    _, T_gt = make_trial(pts, sev, d, rng)
    thetas = np.arange(5) * 72.0

    def spun(theta_deg):
        spin = Rotation.from_rotvec(np.deg2rad(theta_deg) * axis).as_matrix()
        return T_from(spin @ T_gt[:3, :3], spin @ T_gt[:3, 3]
                      + info["center"] - spin @ info["center"])

    good = folded_evaluate(spun(72.0), T_gt, d, THR, axis, info["center"],
                           thetas_deg=thetas)
    assert good["rre_sym_deg"] < 1e-4 and good["success_sym"]
    bad = folded_evaluate(spun(36.0), T_gt, d, THR, axis, info["center"],
                          thetas_deg=thetas)
    assert bad["rre_sym_deg"] == pytest.approx(36.0, abs=1e-6)
    assert not bad["success_sym"]


def test_wilson_ci_bounds_and_width():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25
    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0 and 0.75 < lo < 1.0
    lo_small, hi_small = wilson_ci(10, 20)
    lo_big, hi_big = wilson_ci(120, 240)
    assert (hi_big - lo_big) < (hi_small - lo_small)   # more n = tighter


def test_folded_evaluate_theta_deg_trivial_group_is_zero():
    """axis=None (trivial group) reports theta_deg = 0."""
    folded = folded_evaluate(np.eye(4), np.eye(4), 1.0, THR, None, np.zeros(3))
    assert folded["theta_deg"] == 0.0


def test_folded_evaluate_discrete_thetas_theta_deg_matches_branch():
    """theta_deg must report exactly which of the explicit C_5 thetas matched -
    the same quantity detect_discrete_fold / rescore.py consumes."""
    pts = five_fold_pts()
    d = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axis, info = detect_symmetry_axis(pts, d)
    rng = stable_rng("dsym_theta", 0)
    sev = {"rot_max_deg": 45, "trans_max_rel": 0.25, "noise_rel": 0.0,
           "keep_frac": 1.0}
    _, T_gt = make_trial(pts, sev, d, rng)
    thetas = np.arange(5) * 72.0
    spin = Rotation.from_rotvec(np.deg2rad(144.0) * axis).as_matrix()
    T_est = T_from(spin @ T_gt[:3, :3],
                   spin @ T_gt[:3, 3] + info["center"] - spin @ info["center"])
    res = folded_evaluate(T_est, T_gt, d, THR, axis, info["center"],
                          thetas_deg=thetas)
    assert res["theta_deg"] == pytest.approx(144.0, abs=1e-6)


def test_branch_consistency_uniform_vs_concentrated():
    concentrated = np.zeros(30)                          # every trial: same branch
    uniform = np.linspace(0.0, 360.0, 30, endpoint=False)  # every trial: different

    bc_conc = branch_consistency(concentrated, group_order=6)
    assert bc_conc["mode_frequency"] == pytest.approx(1.0)
    assert bc_conc["entropy_norm"] == pytest.approx(0.0)

    bc_unif = branch_consistency(uniform, group_order=6)
    assert bc_unif["mode_frequency"] == pytest.approx(1.0 / 6, abs=0.05)
    assert bc_unif["entropy_norm"] > 0.9

    # continuous case (group_order=None): bin width falls back to bin_deg
    bc_cont = branch_consistency(concentrated, group_order=None, bin_deg=30.0)
    assert bc_cont["n_bins"] == 12
    assert bc_cont["mode_frequency"] == pytest.approx(1.0)


def test_branch_consistency_empty_is_nan():
    bc = branch_consistency(np.array([]), group_order=4)
    assert np.isnan(bc["entropy_norm"]) and np.isnan(bc["mode_frequency"])
    assert bc["n"] == 0


def test_detect_symmetry_axis_blind_spot_local_feature_off_centroid():
    """A symmetric boss on an asymmetric part is not detected: only the three PCA
    axes through the whole cloud's centroid are tested."""
    local_axis = np.array([0.0, 0.0, 1.0])
    local_center = np.array([0.6, 0.0, 0.0])
    cyl = cylinder_pts(n_ring=72, n_z=30, radius=0.15, height=0.6) + local_center

    # sanity: the boss IS genuinely symmetric about its own (off-centroid) axis.
    d_cyl = float(np.linalg.norm(cyl.max(0) - cyl.min(0)))
    true_overlap = self_overlap(cyl, local_axis, local_center, PROBE_THETAS, d_cyl)
    assert true_overlap.mean() < 0.01

    # an asymmetric bulk, rotated off-axis, dragging the whole part's centroid
    # and global PCA directions well away from the boss's own axis/center.
    rng = stable_rng("blind_spot", 0)
    blob = rng.normal(size=(6000, 3)) * np.array([1.2, 0.3, 0.2])
    raxis = np.array([0.3, 0.7, 0.6])
    raxis /= np.linalg.norm(raxis)
    blob = blob @ Rotation.from_rotvec(np.deg2rad(35.0) * raxis).as_matrix().T \
        + np.array([-1.6, 0.4, -0.3])

    pts = np.concatenate([cyl, blob], axis=0)
    d = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axis, info = detect_symmetry_axis(pts, d)

    # the whole-part centroid lands nowhere near the boss's own local center...
    assert np.linalg.norm(info["center"] - local_center) > 0.4
    # ...so detect_symmetry_axis cannot recover the boss's true local symmetry:
    # neither does the reported axis direction match it...
    angle_deg = np.degrees(np.arccos(
        np.clip(abs(np.dot(axis, local_axis)), -1.0, 1.0)))
    # ...nor does its self-overlap curve (evaluated about the WRONG, global
    # center) register anything close to the true local symmetry's near-zero
    # floor - at least one half of the blind spot always shows up.
    assert angle_deg > 15.0 or info["sym_score"] > 0.05
