"""ATLAS — Adaptive Transfer of Low-rank Abundance Splatting.

Hyperspectral image compression via low-rank spectral Gaussian splatting.

The data utilities (`load_cube`, `to_tensor`, ...) import lightly. Training and the
model are imported lazily because they depend on the CUDA `atlas_gsplat` extension,
which is not required just to load data or inspect the package.
"""
__version__ = "0.1.0"

from .data import (
    load_abundance,
    load_cube,
    load_endmember,
    rank_for,
    to_tensor,
)

__all__ = [
    "__version__",
    "load_cube",
    "load_endmember",
    "load_abundance",
    "to_tensor",
    "rank_for",
    "train_nd",
    "GaussianImage_Cholesky_EA",
    "viz",
]


def __getattr__(name):
    # Lazy access to the CUDA-dependent symbols and the optional viz extra.
    if name == "train_nd":
        from .train import train_nd
        return train_nd
    if name == "GaussianImage_Cholesky_EA":
        from .model import GaussianImage_Cholesky_EA
        return GaussianImage_Cholesky_EA
    if name == "viz":
        import importlib
        return importlib.import_module(".viz", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
