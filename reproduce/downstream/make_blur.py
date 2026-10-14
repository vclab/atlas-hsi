"""Blur control for the downstream eval (paper Appendix G, WHU-Hi-HongHu only).

Applies a spatial Gaussian blur with sigma 1.5 pixels to the band-normalized original
cube (no compression, no blur across bands) and saves it as recons/whu_honghu/
blur_recon.npy. It measures how much a per-pixel classifier gains from spatial
smoothing alone, which is the effect behind JPEG2000's HongHu accuracy.

    python reproduce/downstream/make_blur.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from atlas.data import load_cube  # noqa: E402

SIGMA = 1.5


def main():
    cube = load_cube("whu_honghu", data_dir=str(REPO / "data" / "raw")).astype(np.float32)
    blurred = gaussian_filter(cube, sigma=(SIGMA, SIGMA, 0))
    out = HERE / "recons" / "whu_honghu" / "blur_recon.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, blurred.astype(np.float32))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
