"""Rate-distortion curve plotting."""
import numpy as np

from ._common import _mpl


def _as_xy(points, ykey):
    """Coerce a series into sorted (x=bpppb, y) arrays.

    Accepts a list of ``(bpppb, y)`` tuples or a list of dicts with keys
    ``"bpppb"`` and ``ykey``.
    """
    xs, ys = [], []
    for p in points:
        if isinstance(p, dict):
            xs.append(p["bpppb"])
            ys.append(p[ykey])
        else:
            xs.append(p[0])
            ys.append(p[1])
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def plot_rd_curve(series, ykey="psnr", ax=None, title=None, show=False):
    """Plot one or more rate-distortion curves.

    ``series`` is either:
      * a list of points (``(bpppb, y)`` tuples or dicts) — a single curve, or
      * a dict ``{label: list_of_points}`` — multiple labeled curves.

    ``ykey`` selects the distortion metric when points are dicts (``"psnr"``,
    ``"ms_ssim"``, ``"sam"``, ...) and labels the y-axis. Points are sorted by
    rate, so they can be passed in any order.
    """
    plt = _mpl()
    if isinstance(series, dict):
        curves = series
    else:
        curves = {"ATLAS": series}

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure
    for label, points in curves.items():
        x, y = _as_xy(points, ykey)
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("rate (bpppb)")
    ax.set_ylabel(ykey.upper().replace("_", "-"))
    ax.grid(alpha=0.3)
    if len(curves) > 1:
        ax.legend()
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig, ax
