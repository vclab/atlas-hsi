"""ATLAS's CUDA rasterization kernel: 2D Gaussian projection and N-channel splatting.

Forked from nerfstudio gsplat v0.1 (Apache-2.0, see LICENSE-APACHE) and trimmed to the
operations ATLAS uses. Internal to ATLAS; the public API is the ``atlas`` package.
"""
from .project_gaussians_2d import project_gaussians_2d
from .rasterize_sum import rasterize_gaussians_sum_general
from .utils import (
    bin_and_sort_gaussians,
    compute_cumulative_intersects,
    get_tile_bin_edges,
    map_gaussian_to_intersects,
)

__all__ = [
    "bin_and_sort_gaussians",
    "compute_cumulative_intersects",
    "get_tile_bin_edges",
    "map_gaussian_to_intersects",
    "project_gaussians_2d",
    "rasterize_gaussians_sum_general",
]
