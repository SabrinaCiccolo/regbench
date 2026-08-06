"""build-template itodd: deterministic sampling; the real-mesh test needs data/base_package/."""
import numpy as np
import open3d as o3d
import pytest

from regbench.build import make_template
from regbench.build.make_template import ITODD_ROOT, itodd_ply
from regbench.config import REPO_ROOT

ITODD_DATA = REPO_ROOT / ITODD_ROOT / "bracket_screw.ply"


def test_poisson_disk_sampling_is_deterministic():
    """Same seed, same Poisson-disk sample (small count: the sampler is slow)."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()

    o3d.utility.random.seed(0)
    a = np.asarray(mesh.sample_points_poisson_disk(200).points)
    o3d.utility.random.seed(0)
    b = np.asarray(mesh.sample_points_poisson_disk(200).points)

    assert a.shape == b.shape
    assert np.array_equal(a, b)


@pytest.mark.skipif(not ITODD_DATA.exists(),
                    reason="MVTec ITODD Base Package not downloaded "
                           "(data/base_package/, see docs/reproducing.md)")
def test_itodd_ply_point_count_and_rebuild_is_bit_identical(tmp_path, monkeypatch):
    # small point count: the full 30000 points take minutes to sample
    monkeypatch.setattr(make_template, "ITODD_N_POINTS", 300)
    out1, out2 = tmp_path / "a.ply", tmp_path / "b.ply"
    itodd_ply("bracket_screw", out1)
    itodd_ply("bracket_screw", out2)
    assert len(o3d.io.read_point_cloud(str(out1)).points) == 300
    assert out1.read_bytes() == out2.read_bytes()
