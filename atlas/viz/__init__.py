"""Lightweight visualization helpers for ATLAS (the ``[viz]`` extra).

These are pure-Python (numpy + matplotlib) and do **not** require the CUDA
``atlas_gsplat`` kernel — they operate on the numpy arrays you already have:
hyperspectral cubes ``(H, W, C)``, abundance maps, endmember spectra ``(k, C)``,
and rate–distortion measurements.

Install with::

    pip install "atlas-hsi[viz]"

Everything here is import-guarded: if matplotlib is missing, calling a helper
raises a clear message instead of failing at import time.
"""
from .curves import plot_rd_curve
from .maps import pseudo_rgb, show_abundance_maps, show_error_map, show_pseudo_rgb
from .spectra import plot_endmembers, plot_spectra

__all__ = [
    "pseudo_rgb",
    "show_pseudo_rgb",
    "show_abundance_maps",
    "show_error_map",
    "plot_spectra",
    "plot_endmembers",
    "plot_rd_curve",
]
