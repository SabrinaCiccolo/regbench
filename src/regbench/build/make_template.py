"""Write the template clouds to data/template/*.ply.

  bunny  -> bunny.ply            Stanford Bunny mesh vertices (open3d.data.BunnyMesh,
                                 downloaded once into data/open3d/)
  mvtec  -> cable_gland.ply      one MVTec 3D-AD xyz scan: background plane removed,
                                 largest DBSCAN cluster kept
  itodd  -> itodd_<part>.ply     MVTec ITODD CAD mesh, Poisson-disk sampled to
                                 30k points with a fixed Open3D seed

    regbench build-template [bunny|mvtec|itodd|all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import open3d as o3d

from regbench.config import REPO_ROOT, load_config

CFG = load_config()

MVTEC_SCAN = ("cable_gland", "train", "good", "000.tiff")
DBSCAN_EPS_REL = 0.02          # cluster radius as a fraction of the bbox diagonal
DBSCAN_MIN_POINTS = 30
PLANE_THRESH_REL = 0.01        # RANSAC plane distance, fraction of the bbox diagonal

ITODD_ROOT = "data/base_package/models/cad_models"   # MVTec ITODD Base Package
ITODD_N_POINTS = 30000
ITODD_PARTS_ASYMMETRIC = ["bracket_screw", "injection_pump", "star"]
ITODD_PARTS_AXISYMMETRIC = ["cylinder", "washer", "thread"]


def bunny_ply(out: Path) -> None:
    data_root = REPO_ROOT / CFG["paths"]["open3d_data"]
    data_root.mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh(data_root=str(data_root)).path)
    pcd = o3d.geometry.PointCloud(mesh.vertices)   # vertices: deterministic, no sampling rng
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"{out}: {len(pcd.points)} points")


def mvtec_ply(out: Path) -> None:
    from regbench.mvtec import load_xyz_tiff, organized_to_cloud

    cls, split, defect, name = MVTEC_SCAN
    tiff = REPO_ROOT / CFG["paths"]["mvtec_root"] / cls / split / defect / "xyz" / name
    xyz, valid = load_xyz_tiff(tiff)
    pts, _ = organized_to_cloud(xyz, valid, plane_thresh_rel=PLANE_THRESH_REL,
                                dbscan_eps_rel=DBSCAN_EPS_REL,
                                dbscan_min_points=DBSCAN_MIN_POINTS, seed=0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"{out}: {len(pcd.points)} points (from {int(valid.sum())} valid pixels)")


def itodd_ply(part: str, out: Path) -> None:
    mesh_path = REPO_ROOT / ITODD_ROOT / f"{part}.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"no vertices read from {mesh_path}")
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(0)          # sample_points_poisson_disk is seed-deterministic
    pcd = mesh.sample_points_poisson_disk(ITODD_N_POINTS)
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"{out}: {len(pcd.points)} points (from {part}.ply, "
          f"{len(mesh.vertices)} verts / {len(mesh.triangles)} tris)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                    choices=["bunny", "mvtec", "itodd", "all"])
    args = ap.parse_args(argv)
    which = args.which

    template_dir = REPO_ROOT / "data" / "template"
    template_dir.mkdir(parents=True, exist_ok=True)
    o3d.utility.random.seed(0)
    if which in ("bunny", "all"):
        try:
            bunny_ply(template_dir / "bunny.ply")
        except Exception as exc:                      # download can fail offline
            print(f"bunny failed ({exc})", file=sys.stderr)
            if which == "bunny":
                raise
    if which in ("mvtec", "all"):
        mvtec_ply(template_dir / "cable_gland.ply")
    if which in ("itodd", "all"):
        for part in ITODD_PARTS_ASYMMETRIC + ITODD_PARTS_AXISYMMETRIC:
            itodd_ply(part, template_dir / f"itodd_{part}.ply")


if __name__ == "__main__":
    main()
