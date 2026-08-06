"""MVTec 3D-AD organized scans: TIFF/PNG loading and conversion to point clouds.

Selections are done on flat numpy masks so per-point defect labels stay aligned
with the points through the cleanup.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.image
import numpy as np
import open3d as o3d


def load_xyz_tiff(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """(H,W,3) float64 xyz image + (H,W) bool validity (nonzero, non-NaN pixels)."""
    import tifffile

    xyz = np.asarray(tifffile.imread(str(path)), dtype=np.float64)
    valid = ~np.all(xyz == 0, axis=2) & ~np.isnan(xyz).any(axis=2)
    return xyz, valid


def load_gt_mask(path: Path | str) -> np.ndarray:
    """(H,W) bool defect mask from a MVTec gt png (any nonzero channel = defect)."""
    img = matplotlib.image.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3].max(axis=2)
    return img > 0


def organized_to_cloud(xyz: np.ndarray, valid: np.ndarray,
                       gt_mask: np.ndarray | None = None, *,
                       plane_thresh_rel: float, dbscan_eps_rel: float,
                       dbscan_min_points: int, seed: int,
                       ) -> tuple[np.ndarray, np.ndarray | None]:
    """Organized scan -> (pts, labels): background plane removed (RANSAC),
    largest DBSCAN cluster kept.

    ``labels`` is the per-point defect mask (None without ``gt_mask``). ``seed``
    seeds Open3D's global rng, used by the plane RANSAC.
    """
    pts = xyz[valid]
    labels = gt_mask[valid] if gt_mask is not None else None

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    d = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    o3d.utility.random.seed(seed)
    _, plane_idx = pcd.segment_plane(distance_threshold=plane_thresh_rel * d,
                                     ransac_n=3, num_iterations=1000)
    keep = np.ones(len(pts), dtype=bool)
    keep[np.asarray(plane_idx, dtype=int)] = False
    pts, labels = pts[keep], (labels[keep] if labels is not None else None)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cluster = np.asarray(pcd.cluster_dbscan(eps=dbscan_eps_rel * d,
                                            min_points=dbscan_min_points))
    if (cluster >= 0).any():
        keep = cluster == np.bincount(cluster[cluster >= 0]).argmax()
        pts, labels = pts[keep], (labels[keep] if labels is not None else None)
    return pts, labels
