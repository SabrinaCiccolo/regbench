"""Registration methods: four classical pipelines from ``open3d.pipelines.registration``.

The target's normals and FPFH features are computed once by ``prepare_target``;
everything a method needs on the source is computed inside ``register``, so the
measured time covers the source side only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration


@dataclass
class Prepared:
    """A target cloud with its offline-precomputed features."""

    pcd: o3d.geometry.PointCloud       # normals estimated
    fpfh: reg.Feature
    voxel: float


def _estimate_normals(pcd: o3d.geometry.PointCloud, voxel: float, cfg: dict) -> None:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=cfg["normal_radius_mult"] * voxel, max_nn=cfg["normal_max_nn"]))


def _fpfh(pcd: o3d.geometry.PointCloud, voxel: float, cfg: dict) -> reg.Feature:
    return reg.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(
        radius=cfg["fpfh_radius_mult"] * voxel, max_nn=cfg["fpfh_max_nn"]))


def prepare_target(pcd: o3d.geometry.PointCloud, voxel: float, cfg: dict) -> Prepared:
    """Normals + FPFH for the target, computed once per template."""
    _estimate_normals(pcd, voxel, cfg)
    return Prepared(pcd=pcd, fpfh=_fpfh(pcd, voxel, cfg), voxel=voxel)


def _icp_refine(source: o3d.geometry.PointCloud, target: Prepared, voxel: float,
                cfg: dict, init: np.ndarray) -> reg.RegistrationResult:
    return reg.registration_icp(
        source, target.pcd, cfg["icp_max_corr_mult"] * voxel, init,
        reg.TransformationEstimationPointToPlane())


def icp_pt2pt(source, target: Prepared, voxel: float, cfg: dict) -> reg.RegistrationResult:
    return reg.registration_icp(
        source, target.pcd, cfg["icp_max_corr_mult"] * voxel, np.eye(4),
        reg.TransformationEstimationPointToPoint())


def icp_pt2pl(source, target: Prepared, voxel: float, cfg: dict) -> reg.RegistrationResult:
    _estimate_normals(source, voxel, cfg)
    return _icp_refine(source, target, voxel, cfg, np.eye(4))


def fpfh_ransac(source, target: Prepared, voxel: float, cfg: dict) -> reg.RegistrationResult:
    _estimate_normals(source, voxel, cfg)
    source_fpfh = _fpfh(source, voxel, cfg)
    coarse = reg.registration_ransac_based_on_feature_matching(
        source, target.pcd, source_fpfh, target.fpfh, mutual_filter=True,
        max_correspondence_distance=cfg["ransac_dist_mult"] * voxel,
        estimation_method=reg.TransformationEstimationPointToPoint(False),
        ransac_n=cfg["ransac_n"],
        checkers=[reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  reg.CorrespondenceCheckerBasedOnDistance(
                      cfg["ransac_dist_mult"] * voxel)],
        criteria=reg.RANSACConvergenceCriteria(100000, 0.999))
    return _icp_refine(source, target, voxel, cfg, np.asarray(coarse.transformation))


def fgr(source, target: Prepared, voxel: float, cfg: dict) -> reg.RegistrationResult:
    _estimate_normals(source, voxel, cfg)
    source_fpfh = _fpfh(source, voxel, cfg)
    coarse = reg.registration_fgr_based_on_feature_matching(
        source, target.pcd, source_fpfh, target.fpfh,
        reg.FastGlobalRegistrationOption(
            maximum_correspondence_distance=cfg["ransac_dist_mult"] * voxel))
    return _icp_refine(source, target, voxel, cfg, np.asarray(coarse.transformation))


METHODS: dict[str, Callable] = {
    "icp_pt2pt": icp_pt2pt,
    "icp_pt2pl": icp_pt2pl,
    "fpfh_ransac": fpfh_ransac,
    "fgr": fgr,
}


NO_DIAG = {"fitness": 0.0, "inlier_rmse": float("nan"), "n_inliers": 0}


def register(name: str, source_pts: np.ndarray, target: Prepared, cfg: dict,
             trial_seed: int, return_diag: bool = False,
             ) -> np.ndarray | tuple[np.ndarray, dict]:
    """Run one method on raw source points; seeds Open3D's global rng (RANSAC/FGR).

    With ``return_diag=True`` also returns the final ICP step's fitness (inlier
    fraction), inlier_rmse and inlier count."""
    o3d.utility.random.seed(trial_seed)
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pts))
    source = source.voxel_down_sample(target.voxel)
    try:
        result = METHODS[name](source, target, target.voxel, cfg)
        T = np.asarray(result.transformation)
        diag = {"fitness": result.fitness, "inlier_rmse": result.inlier_rmse,
                "n_inliers": len(result.correspondence_set)}
    except RuntimeError as exc:
        # With almost no FPFH correspondences Open3D raises instead of returning
        # a transform; score it as a failed trial (identity pose).
        print(f"[register] {name} crashed ({exc}); scoring as a failed trial")
        T, diag = np.eye(4), dict(NO_DIAG)
    return (T, diag) if return_diag else T
