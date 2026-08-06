import numpy as np

from regbench.mvtec import organized_to_cloud

PARAMS = dict(plane_thresh_rel=0.02, dbscan_eps_rel=0.08, dbscan_min_points=5, seed=0)


def fake_scan(blob_invalid: int = 0):
    """40x40 organized scan: z=0 background plane + a raised 10x10 blob (the part)."""
    h = w = 40
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    z = np.zeros((h, w))
    blob = np.zeros((h, w), dtype=bool)
    blob[15:25, 15:25] = True
    z[blob] = 0.5 + 0.1 * np.hypot(x[blob] - 0.5, y[blob] - 0.5)
    xyz = np.stack([x, y, z], axis=2)
    valid = np.ones((h, w), dtype=bool)
    if blob_invalid:
        ii, jj = np.nonzero(blob)
        valid[ii[:blob_invalid], jj[:blob_invalid]] = False
    gt = blob.copy()
    gt[15:20, 15:25] = False        # only half the blob is "defective"
    return xyz, valid, blob, gt


def test_plane_removed_blob_labels_preserved():
    xyz, valid, blob, gt = fake_scan()
    pts, labels = organized_to_cloud(xyz, valid, gt, **PARAMS)
    assert len(pts) == blob.sum()                    # exactly the part survives
    assert np.all(pts[:, 2] > 0.4)                   # nothing from the plane
    assert labels.sum() == gt.sum()                  # labels rode along 1:1
    assert np.array_equal(labels, gt[valid & blob])  # in flattened pixel order
    assert organized_to_cloud(xyz, valid, None, **PARAMS)[1] is None


def test_invalid_pixels_dropped_labels_aligned():
    xyz, valid, blob, gt = fake_scan(blob_invalid=7)
    pts, labels = organized_to_cloud(xyz, valid, gt, **PARAMS)
    assert len(pts) == (blob & valid).sum()
    assert labels.sum() == (gt & valid).sum()
    assert np.array_equal(labels, gt[valid & blob])
