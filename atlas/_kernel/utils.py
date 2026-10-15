"""Tile binning: list each Gaussian's tile overlaps, sort them by tile, index each run."""


import torch
from torch import Tensor

from . import cuda as _C


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Tensor,
    depths: Tensor,
    radii: Tensor,
    cum_tiles_hit: Tensor,
    tile_bounds: tuple[int, int, int],
) -> tuple[Tensor, Tensor]:
    """List every (tile, Gaussian) overlap as a sortable id. Not differentiable.

    Args:
        num_points: N, the number of Gaussians.
        num_intersects: total overlaps, ``cum_tiles_hit[-1]``.
        xys: (N, 2) float32 pixel-space centers.
        depths: (N,) float32 depths (0 for 2D Gaussians).
        radii: (N,) int32 pixel radii; 0 means the Gaussian is skipped.
        cum_tiles_hit: (N,) int32 running total of tiles hit; Gaussian i writes its
            overlaps starting at ``cum_tiles_hit[i - 1]``.
        tile_bounds: ``(tiles_x, tiles_y, 1)``.

    Returns:
        isect_ids: (num_intersects,) int64 with the tile index in the high 32 bits and the
            depth's float bits in the low 32, so sorting groups overlaps by tile.
        gaussian_ids: (num_intersects,) int32 index of the Gaussian for each overlap.
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
    isect_ids_sorted: Tensor,
    tile_bounds: tuple[int, int, int],
) -> Tensor:
    """Find each tile's run of overlaps in the sorted ids. Not differentiable.

    There is one row per tile, so the result stays valid when there are fewer overlaps than
    tiles. (The CUDA version inherited from gsplat v0.1 sized it by ``num_intersects`` and
    read and wrote past the end of the buffer in that case.)

    Args:
        num_intersects: total overlaps.
        isect_ids_sorted: (num_intersects,) int64 ids sorted ascending, so by tile.
        tile_bounds: ``(tiles_x, tiles_y, 1)``.

    Returns:
        tile_bins: (tiles_x * tiles_y, 2) int32, contiguous. Row t is the ``[start, end)``
            range of tile t's overlaps, and ``[0, 0]`` when the tile has none.
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
    num_tiles_hit: Tensor
) -> tuple[int, Tensor]:
    """Running total of tiles hit. Not differentiable.

    Args:
        num_tiles_hit: (N,) int32 number of tiles each Gaussian overlaps.

    Returns:
        num_intersects: int, the total number of overlaps.
        cum_tiles_hit: (N,) int32 inclusive cumulative sum of ``num_tiles_hit``.
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Tensor,
    depths: Tensor,
    radii: Tensor,
    cum_tiles_hit: Tensor,
    tile_bounds: tuple[int, int, int],
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    """List every tile overlap, sort the overlaps by tile, and index each tile's run.
    Not differentiable.

    Args:
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds: as for
            :func:`map_gaussian_to_intersects`.

    Returns:
        isect_ids_unsorted: (num_intersects,) int64 ids in Gaussian order.
        gaussian_ids_unsorted: (num_intersects,) int32 Gaussian index per overlap.
        isect_ids_sorted: (num_intersects,) int64 ids sorted ascending, so grouped by tile.
        gaussian_ids_sorted: (num_intersects,) int32 Gaussian indices in the sorted order.
        tile_bins: (tiles_x * tiles_y, 2) int32 ``[start, end)`` of each tile's run.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
