"""Slow, plain-PyTorch references for the CUDA kernel's tile binning, used by the tests."""
import torch

BLOCK = 16  # tile edge in pixels (BLOCK_X == BLOCK_Y in the kernel's config.h)


def random_gaussians(n, height, width, seed=0):
    """Project n random 2D Gaussians with the CUDA kernel; returns the binning inputs."""
    from atlas._kernel import project_gaussians_2d

    g = torch.Generator(device="cuda").manual_seed(seed)
    means = torch.rand(n, 2, device="cuda", generator=g) * 2 - 1
    chol = torch.rand(n, 3, device="cuda", generator=g) * 2 + 0.5
    tile_bounds = ((width + BLOCK - 1) // BLOCK, (height + BLOCK - 1) // BLOCK, 1)
    xys, depths, radii, _, tiles_hit = project_gaussians_2d(means, chol, height, width, tile_bounds)
    return xys, depths, radii, tiles_hit, tile_bounds


def get_tile_bbox(center, radius, tile_bounds):
    """Tile-space box of one Gaussian: inclusive min, exclusive max, clamped to the grid.
    Truncation toward zero matches the kernel's (int) casts."""
    tile_center = center / BLOCK
    tile_radius = float(radius) / BLOCK
    lo = (tile_center - tile_radius).to(torch.int32)
    hi = (tile_center + tile_radius).to(torch.int32) + 1
    clamp = lambda v, k: int(min(max(int(v), 0), tile_bounds[k]))  # noqa: E731
    return (clamp(lo[0], 0), clamp(lo[1], 1)), (clamp(hi[0], 0), clamp(hi[1], 1))


def map_gaussian_to_intersects(xys, radii, cum_tiles_hit, tile_bounds):
    """Every (tile, Gaussian) pair in the order the kernel writes them. 2D Gaussians have
    depth 0, so each intersection id is just ``tile << 32``. Slots the kernel never
    writes stay 0, as in its zero-initialized output."""
    total = int(cum_tiles_hit[-1])
    isect_ids = torch.zeros(total, dtype=torch.int64)
    gaussian_ids = torch.zeros(total, dtype=torch.int32)
    for g in range(len(radii)):
        if radii[g] <= 0:
            continue
        (x0, y0), (x1, y1) = get_tile_bbox(xys[g], radii[g], tile_bounds)
        k = 0 if g == 0 else int(cum_tiles_hit[g - 1])
        for ty in range(y0, y1):
            for tx in range(x0, x1):
                isect_ids[k] = (ty * tile_bounds[0] + tx) << 32
                gaussian_ids[k] = g
                k += 1
    return isect_ids, gaussian_ids


def get_tile_bin_edges(isect_ids_sorted, tile_bounds):
    """One [start, end) row per tile over the sorted intersections; empty tiles stay [0, 0]."""
    bins = torch.zeros((tile_bounds[0] * tile_bounds[1], 2), dtype=torch.int32)
    tiles = (isect_ids_sorted >> 32).tolist()
    for i, tile in enumerate(tiles):
        if i == 0 or tiles[i - 1] != tile:
            bins[tile, 0] = i
        bins[tile, 1] = i + 1
    return bins
