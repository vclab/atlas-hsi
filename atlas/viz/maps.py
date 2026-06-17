"""Spatial visualizations: pseudo-RGB previews, abundance maps, error maps."""
import numpy as np

from ._common import _mpl, default_rgb_bands, to_hwk


def pseudo_rgb(cube, bands=None):
    """Build a per-channel-normalized pseudo-RGB image ``(H, W, 3)`` from a cube.

    ``cube`` is ``(H, W, C)``. ``bands`` is an optional ``[r, g, b]`` index list;
    if omitted, three evenly spaced bands are chosen.
    """
    cube = np.asarray(cube)
    if bands is None:
        bands = default_rgb_bands(cube.shape[2])
    rgb = np.stack([cube[:, :, b] for b in bands], axis=-1).astype(np.float32)
    mn = rgb.min(axis=(0, 1), keepdims=True)
    mx = rgb.max(axis=(0, 1), keepdims=True)
    return (rgb - mn) / (mx - mn + 1e-8)


def show_pseudo_rgb(cube, bands=None, ax=None, title=None, show=False):
    """Display a pseudo-RGB preview of a hyperspectral cube. Returns ``(fig, ax)``."""
    plt = _mpl()
    img = pseudo_rgb(cube, bands)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig, ax


def show_abundance_maps(A, shape=None, ncols=4, cmap="magma", titles=None,
                        suptitle=None, show=False):
    """Plot each abundance map in a grid (one panel per rank component).

    ``A`` may be ``(H, W, k)``, ``(k, H, W)``, or flattened ``(H*W, k)`` (the NMF
    init layout) — pass ``shape=(H, W)`` for the flattened case.
    """
    plt = _mpl()
    A = to_hwk(A, shape)
    k = A.shape[2]
    ncols = min(ncols, k)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i < k:
            ax.imshow(A[:, :, i], cmap=cmap)
            ax.set_title(titles[i] if titles else f"component {i}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def show_error_map(original, recon, metric="mae", cmap="inferno", ax=None,
                   title=None, colorbar=True, show=False):
    """Display a per-pixel reconstruction error map between two cubes.

    ``metric``:
      * ``"mae"`` — mean absolute error across bands (default)
      * ``"rmse"`` — per-pixel RMSE across bands
      * ``"sam"`` — per-pixel spectral angle (radians)
    """
    plt = _mpl()
    o = np.asarray(original, dtype=np.float64)
    r = np.asarray(recon, dtype=np.float64)
    if o.shape != r.shape:
        raise ValueError(f"shape mismatch: original {o.shape} vs recon {r.shape}")
    if metric == "mae":
        err = np.mean(np.abs(o - r), axis=-1)
        label = "mean abs error"
    elif metric == "rmse":
        err = np.sqrt(np.mean((o - r) ** 2, axis=-1))
        label = "RMSE"
    elif metric == "sam":
        dot = np.sum(o * r, axis=-1)
        cos = dot / (np.linalg.norm(o, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-8)
        err = np.arccos(np.clip(cos, -1, 1))
        label = "spectral angle (rad)"
    else:
        raise ValueError(f"unknown metric {metric!r} (use 'mae', 'rmse', or 'sam')")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    im = ax.imshow(err, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig, ax
