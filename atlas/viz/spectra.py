"""Spectral visualizations: per-pixel signatures and endmember spectra."""
import numpy as np

from ._common import _mpl


def plot_spectra(original, recon=None, coords=None, n=3, wavelengths=None,
                 ax=None, title=None, show=False, seed=0):
    """Plot spectral signatures (band vs reflectance) at one or more pixels.

    ``original`` and optional ``recon`` are ``(H, W, C)`` cubes. ``coords`` is a
    list of ``(row, col)`` pixels; if omitted, ``n`` pixels are sampled at random.
    When ``recon`` is given, each pixel's reconstructed spectrum is overlaid
    (dashed) on its original (solid) in a matching color.
    """
    plt = _mpl()
    o = np.asarray(original)
    H, W, C = o.shape
    if coords is None:
        rng = np.random.default_rng(seed)
        coords = [(int(rng.integers(H)), int(rng.integers(W))) for _ in range(n)]
    x = np.asarray(wavelengths) if wavelengths is not None else np.arange(C)
    xlabel = "wavelength" if wavelengths is not None else "band index"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (rr, cc) in enumerate(coords):
        color = cycle[i % len(cycle)]
        ax.plot(x, o[rr, cc, :], "-", color=color, label=f"orig ({rr},{cc})")
        if recon is not None:
            r = np.asarray(recon)
            ax.plot(x, r[rr, cc, :], "--", color=color, label=f"recon ({rr},{cc})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("reflectance")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig, ax


def plot_endmembers(E, wavelengths=None, ax=None, title="Endmember spectra",
                    show=False):
    """Plot NMF endmember spectra. ``E`` is ``(k, C)`` (rank by channels)."""
    plt = _mpl()
    E = np.asarray(E)
    if E.ndim != 2:
        raise ValueError(f"expected endmember matrix (k, C), got shape {E.shape}")
    # Orient so rows are endmembers (k is the smaller axis).
    if E.shape[0] > E.shape[1]:
        E = E.T
    k, C = E.shape
    x = np.asarray(wavelengths) if wavelengths is not None else np.arange(C)
    xlabel = "wavelength" if wavelengths is not None else "band index"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    for i in range(k):
        ax.plot(x, E[i], label=f"endmember {i}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("reflectance")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig, ax
