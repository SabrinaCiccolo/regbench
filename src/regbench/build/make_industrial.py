"""Build the inspection datasets (config `industrial`).

    regbench build-industrial [synthetic|mvtec|all]

Writes <industrial_root>/<part>/{train_good,test_good,test_defect}/*.ply, each
scan with a <stem>.gt.npz sidecar (regbench/anomaly.py).

synthetic: scans made from a template by half-plane cut -> bump (defect scans
only) -> noise -> known rigid move, so each carries its true pose and defect mask.
mvtec: MVTec 3D-AD scans converted with regbench/mvtec.py, with per-point defect
labels from the 2D gt masks.

Both builders are deterministic (stable_rng seeds).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from regbench.anomaly import write_sidecar
from regbench.cloud_io import INDUSTRIAL_SPLITS, load_cloud, preprocess_cloud
from regbench.config import REPO_ROOT, load_config
from regbench.methods import prepare_target
from regbench.mvtec import load_gt_mask, load_xyz_tiff, organized_to_cloud
from regbench.perturb import gauss_noise, inject_bump, plane_cut_mask, random_transform
from regbench.repro import stable_rng


def write_scan(out_dir: Path, stem: str, pts: np.ndarray, defect_mask: np.ndarray,
               T_gt: np.ndarray | None, defect_type: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / f"{stem}.ply"
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(str(ply), pcd)
    write_sidecar(ply, defect_mask, T_gt=T_gt, defect_type=defect_type)


def build_synthetic_part(part: str, template_ply: Path, cfg: dict, root: Path) -> None:
    icfg = cfg["industrial"]["synthetic"]
    sev = cfg["severities"][icfg["severity"]]
    dcfg = icfg["defect"]
    template_pcd, d, voxel = preprocess_cloud(load_cloud(template_ply), cfg)
    target = prepare_target(template_pcd, voxel, cfg["registration"])  # for normals
    template = np.asarray(target.pcd.points)
    normals = np.asarray(target.pcd.normals)

    counts = {"train_good": icfg["n_train_good"], "test_good": icfg["n_test_good"],
              "test_defect": icfg["n_test_defect"]}
    for split, n in counts.items():
        for i in range(n):
            if split == "train_good" and i == 0:
                # the template scan: unperturbed, identity pose
                pts = template.copy()
                mask = np.zeros(len(pts), dtype=bool)
                T_gt: np.ndarray = np.eye(4)
                defect_type = ""
            else:
                rng = stable_rng("industrial", part, split, i)
                keep = plane_cut_mask(template, sev["keep_frac"], rng)
                pts = template[keep]
                mask = np.zeros(len(pts), dtype=bool)
                defect_type = ""
                if split == "test_defect":
                    pts, mask = inject_bump(pts, normals[keep], rng,
                                            radius=dcfg["radius_rel"] * d,
                                            amplitude=dcfg["amplitude_rel"] * d,
                                            sign=dcfg["sign"])
                    defect_type = "bump"
                pts = gauss_noise(pts, sev["noise_rel"] * d, rng)
                T_pert = random_transform(rng, sev["rot_max_deg"],
                                          sev["trans_max_rel"] * d)
                pts = pts @ T_pert[:3, :3].T + T_pert[:3, 3]
                T_gt = np.linalg.inv(T_pert)
            write_scan(root / part / split, f"{i:03d}", pts, mask, T_gt, defect_type)
        print(f"{part}/{split}: {n} scans")


def build_mvtec_part(cls: str, cfg: dict, root: Path) -> None:
    mcfg = cfg["industrial"]["mvtec"]
    mv = REPO_ROOT / cfg["paths"]["mvtec_root"] / cls
    part = f"mvtec_{cls}"

    def tiffs(*rel: str) -> list[Path]:
        return sorted((mv.joinpath(*rel) / "xyz").glob("*.tiff"))

    jobs: list[tuple[str, Path, Path | None, str]] = []
    for tiff in tiffs("train", "good")[:mcfg["n_train_good"]]:
        jobs.append(("train_good", tiff, None, ""))
    for tiff in tiffs("test", "good"):
        jobs.append(("test_good", tiff, None, ""))
    for dtype in mcfg["defect_types"][cls]:
        for tiff in tiffs("test", dtype):
            gt = tiff.parent.parent / "gt" / f"{tiff.stem}.png"
            jobs.append(("test_defect", tiff, gt, dtype))

    done: dict[str, int] = {s: 0 for s in INDUSTRIAL_SPLITS}
    for split, tiff, gt_path, dtype in jobs:
        xyz, valid = load_xyz_tiff(tiff)
        gt = load_gt_mask(gt_path) if gt_path is not None else None
        seed = int(stable_rng("mvtec", cls, split, dtype, tiff.stem).integers(2**31))
        pts, labels = organized_to_cloud(
            xyz, valid, gt, plane_thresh_rel=mcfg["plane_thresh_rel"],
            dbscan_eps_rel=mcfg["dbscan_eps_rel"],
            dbscan_min_points=mcfg["dbscan_min_points"], seed=seed)
        mask = labels if labels is not None else np.zeros(len(pts), dtype=bool)
        stem = f"{dtype}_{tiff.stem}" if dtype else tiff.stem
        write_scan(root / part / split, stem, pts, mask, None, dtype)
        done[split] += 1
    print(f"{part}: " + ", ".join(f"{s}={n}" for s, n in done.items()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                    choices=["synthetic", "mvtec", "all"])
    args = ap.parse_args(argv)
    which = args.which

    cfg = load_config()
    root = REPO_ROOT / cfg["paths"]["industrial_root"]
    if which in ("synthetic", "all"):
        for part, template in cfg["industrial"]["synthetic"]["parts"].items():
            build_synthetic_part(part, REPO_ROOT / template, cfg, root)
    if which in ("mvtec", "all"):
        for cls in cfg["industrial"]["mvtec"]["classes"]:
            build_mvtec_part(cls, cfg, root)


if __name__ == "__main__":
    main()
