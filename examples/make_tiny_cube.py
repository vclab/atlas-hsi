"""Generate ``tiny_cube.npy`` — a small synthetic low-rank HSI cube for the ATLAS quickstart.

The cube is built exactly the way ATLAS models one: a handful of smooth **endmember**
spectra mixed by spatially smooth **abundance** maps (``X = A @ E``), plus a little noise.
That gives it genuine low-rank structure, so NMF init and Gaussian splatting are
meaningful on it — unlike pure random noise.

Deterministic (fixed seed); rerun to regenerate the committed file:

    python examples/make_tiny_cube.py
"""
import os

import numpy as np

H, W, C, K = 48, 48, 20, 4          # 48x48 pixels, 20 bands, 4 endmembers
SEED = 0
OUT = os.path.join(os.path.dirname(__file__), "tiny_cube.npy")


def _smooth_endmembers(C, K, rng):
    """K smooth, non-negative spectra (Gaussian bumps at spread-out band centers)."""
    bands = np.arange(C)
    centers = np.linspace(0.1 * C, 0.9 * C, K)
    E = np.empty((K, C), dtype=np.float32)
    for i, c in enumerate(centers):
        width = rng.uniform(C / 6, C / 3)
        E[i] = np.exp(-0.5 * ((bands - c) / width) ** 2) + 0.05
    return E


def _smooth_abundances(H, W, K, rng):
    """K spatially smooth abundance maps (Gaussian blobs), softmax-normalized to sum to 1."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    maps = np.empty((H, W, K), dtype=np.float32)
    for i in range(K):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        sigma = rng.uniform(min(H, W) / 6, min(H, W) / 3)
        maps[:, :, i] = np.exp(-0.5 * (((yy - cy) ** 2 + (xx - cx) ** 2) / sigma ** 2))
    maps += 0.1  # floor so no pixel is purely one endmember
    maps = np.exp(maps) / np.exp(maps).sum(axis=2, keepdims=True)  # softmax over K
    return maps


def main():
    rng = np.random.default_rng(SEED)
    E = _smooth_endmembers(C, K, rng)               # (K, C)
    A = _smooth_abundances(H, W, K, rng)            # (H, W, K)
    X = A.reshape(-1, K) @ E                        # (H*W, C)
    X = X.reshape(H, W, C)
    X += rng.normal(0, 0.01, X.shape)               # mild sensor-like noise
    X = np.clip(X, 0, None).astype(np.float32)
    np.save(OUT, X)
    print(f"wrote {OUT}  shape={X.shape}  ({X.nbytes / 1024:.0f} KiB)  rank={K}")


if __name__ == "__main__":
    main()
