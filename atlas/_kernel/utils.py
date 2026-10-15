"""Python bindings for binning and sorting gaussians"""

from typing import Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from . import cuda as _C


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
    )
    return (isect_ids, gaussian_ids)


def get_tile_bin_edges(
    num_intersects: int,
    isect_ids_sorted: Int[Tensor, "num_intersects 1"],
    tile_bounds: Tuple[int, int, int],
) -> Int[Tensor, "num_tiles 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    The result has one row per tile, so it is valid when there are fewer intersections than
    tiles. (The CUDA kernel inherited from gsplat v0.1 sized it by num_intersects, which read
    and wrote past the end of the buffer in that case.)

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): int32, shape (num_tiles, 2), contiguous; row t is the
          [start, end) range of intersections for tile t, and [0, 0] when the tile has
          none.
    """
    num_tiles = tile_bounds[0] * tile_bounds[1]
    tile_ids = (isect_ids_sorted.reshape(-1) >> 32).contiguous()  # sorted ascending
    tiles = torch.arange(num_tiles, device=tile_ids.device, dtype=tile_ids.dtype)
    # Each tile's run of intersections starts at the first index holding its id and ends one
    # past the last; a tile with no intersections gets start == end. Binary search needs no
    # host sync (unlike bincount) and cannot index outside the buffer.
    starts = torch.searchsorted(tile_ids, tiles, right=False)
    ends = torch.searchsorted(tile_ids, tiles, right=True)
    tile_bins = torch.stack([starts, ends], dim=1).to(torch.int32)
    tile_bins[starts == ends] = 0  # empty tiles as [0, 0], the convention of the CUDA kernel
    return tile_bins.contiguous()


def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
