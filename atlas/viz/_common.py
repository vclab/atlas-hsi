"""Shared helpers for the viz module: matplotlib guard and array normalization."""
import numpy as np


def _mpl():
    """Import matplotlib.pyplot, with a friendly error if the [viz] extra is missing."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "ATLAS visualization requires matplotlib. Install the viz extra:\n"
            "    pip install \"atlas-hsi[viz]\""
        ) from e


def to_hwk(A, shape=None):
    """Normalize an abundance array to ``(H, W, k)``.

    Accepts:
      * ``(H, W, k)``      — returned as-is
      * ``(k, H, W)``      — channels-first, transposed
      * ``(H*W, k)``       — flattened (as stored by NMF init); needs ``shape=(H, W)``

    The ``k`` axis is identified as the smallest dimension (rank is ``<= ~32``,
    far below any spatial extent).
    """
    A = np.asarray(A)
    if A.ndim == 2:
        if shape is None:
            raise ValueError(
                f"abundance has flattened shape {A.shape}; pass shape=(H, W) to reshape it"
            )
        H, W = shape
        n, k = A.shape
        if n == H * W:
            return A.reshape(H, W, k)
        if k == H * W:  # (k, H*W)
            return A.T.reshape(H, W, A.shape[0])
        raise ValueError(f"abundance {A.shape} does not match shape {shape}")
    if A.ndim != 3:
        raise ValueError(f"expected a 2D or 3D abundance array, got shape {A.shape}")
    # 3D: put the smallest axis last.
    k_axis = int(np.argmin(A.shape))
    if k_axis == 2:
        return A
    return np.moveaxis(A, k_axis, 2)


def default_rgb_bands(C):
    """Pick three reasonable band indices (R, G, B) for a pseudo-RGB preview."""
    if C < 3:
        raise ValueError(f"need at least 3 bands for pseudo-RGB, got {C}")
    # Evenly spaced thirds — robust across very different band counts.
    return [C // 6, C // 2, (5 * C) // 6]
