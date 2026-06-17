"""Pure-Python tests for atlas.viz (matplotlib Agg backend, no torch / no CUDA)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless

from atlas import viz  # noqa: E402
from atlas.viz._common import to_hwk  # noqa: E402

H, W, K, C = 12, 10, 4, 16


def _abundance_flat():
    rng = np.random.default_rng(0)
    return rng.random((H * W, K))


def _cube():
    rng = np.random.default_rng(1)
    return rng.random((H, W, C)).astype(np.float32)


def test_to_hwk_accepts_all_layouts():
    flat = _abundance_flat()
    assert to_hwk(flat, shape=(H, W)).shape == (H, W, K)
    hwk = flat.reshape(H, W, K)
    assert to_hwk(hwk).shape == (H, W, K)
    khw = hwk.transpose(2, 0, 1)
    assert to_hwk(khw).shape == (H, W, K)


def test_pseudo_rgb_normalized():
    img = viz.pseudo_rgb(_cube())
    assert img.shape == (H, W, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_helpers_return_fig_ax():
    cube = _cube()
    recon = np.clip(cube + 0.01, 0, 1)
    assert viz.show_pseudo_rgb(cube)[0] is not None
    assert viz.show_abundance_maps(_abundance_flat(), shape=(H, W))[0] is not None
    assert viz.plot_endmembers(np.random.default_rng(2).random((K, C)))[0] is not None
    assert viz.plot_spectra(cube, recon, n=2)[0] is not None
    for metric in ("mae", "rmse", "sam"):
        assert viz.show_error_map(cube, recon, metric=metric)[0] is not None


def test_rd_curve_single_and_multi():
    assert viz.plot_rd_curve([(0.10, 37.2), (0.06, 31.0)])[0] is not None
    multi = {
        "ATLAS": [{"bpppb": 0.10, "psnr": 37.2}, {"bpppb": 0.06, "psnr": 31.5}],
        "base": [{"bpppb": 0.10, "psnr": 36.9}, {"bpppb": 0.06, "psnr": 30.8}],
    }
    assert viz.plot_rd_curve(multi)[0] is not None
