import numpy as np
import tifffile

from regbench.mvtec import load_gt_mask, load_xyz_tiff


def test_load_xyz_tiff_dtype_and_validity(tmp_path):
    xyz = np.zeros((4, 4, 3), dtype=np.float32)
    xyz[1, 1] = [1.0, 2.0, 3.0]
    xyz[2, 2] = [np.nan, 0, 0]
    path = tmp_path / "scan.tiff"
    tifffile.imwrite(path, xyz, photometric="minisblack")

    loaded, valid = load_xyz_tiff(path)
    assert loaded.dtype == np.float64
    assert valid.shape == (4, 4)
    assert valid[1, 1] and not valid[0, 0] and not valid[2, 2]


def test_load_gt_mask(tmp_path):
    import matplotlib.image

    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    path = tmp_path / "gt.png"
    matplotlib.image.imsave(path, mask, cmap="gray")

    loaded = load_gt_mask(path)
    assert loaded.dtype == bool
    assert loaded[2, 2]
    assert not loaded[0, 0]
