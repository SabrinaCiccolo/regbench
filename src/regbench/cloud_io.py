"""Point-cloud loading and preprocessing.

Inspection datasets use the layout

    <industrial_root>/<part>/{train_good,test_good,test_defect}/*.ply

which ``iter_industrial`` walks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import open3d as o3d

INDUSTRIAL_SPLITS = ("train_good", "test_good", "test_defect")


def load_cloud(path: Path | str) -> o3d.geometry.PointCloud:
    """Read a .ply/.pcd (anything Open3D parses) into a PointCloud; raise if empty."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"no points read from {path}")
    return pcd


def iter_industrial(root: Path | str, part: str, split: str) -> Iterator[Path]:
    """Yield sorted cloud paths for one (part, split) of the industrial layout."""
    if split not in INDUSTRIAL_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {INDUSTRIAL_SPLITS}")
    split_dir = Path(root) / part / split
    for pattern in ("*.ply", "*.pcd"):
        yield from sorted(split_dir.glob(pattern))


def preprocess_cloud(pcd: o3d.geometry.PointCloud, cfg: dict, *, recenter: bool = True,
                     ) -> tuple[o3d.geometry.PointCloud, float, float]:
    """Outlier removal -> recenter to centroid -> (downsampled cloud, d, voxel).

    d is the AABB diagonal of the cleaned cloud and voxel = d / voxel_divisor.
    Recentring makes sampled rotations pivot near the object. Inspection passes
    ``recenter=False`` so the template keeps the frame its sidecar poses refer to.
    """
    pre = cfg["preprocess"]
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=pre["outlier_nb_neighbors"], std_ratio=pre["outlier_std_ratio"])
    if recenter:
        pts = np.asarray(pcd.points)
        pcd.points = o3d.utility.Vector3dVector(pts - pts.mean(axis=0))
    aabb = pcd.get_axis_aligned_bounding_box()
    d = float(np.linalg.norm(aabb.get_extent()))
    voxel = d / pre["voxel_divisor"]
    return pcd.voxel_down_sample(voxel), d, voxel
