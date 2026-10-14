"""GPU regression test for tile binning with few Gaussians on a large image.

The rasterizer groups Gaussian-tile intersections by screen tile in ``tile_bins``, one
``[start, end)`` row per tile. The kernel inherited from gsplat v0.1 sized that buffer by
the number of intersections instead of the number of tiles, so whenever a frame had fewer
intersections than tiles (under about one Gaussian per 256 pixels), the last tiles read
and wrote past the end of the buffer. The backward pass then crashed with an illegal
memory access, which is how the quick self-test on Pavia Centre (1,200 Gaussians on
3,105 tiles) found it. Paper configurations never hit it: they start with 10 to 100
times more Gaussians than tiles.

Auto-skipped without a CUDA GPU + atlas_gsplat (see conftest.py).
"""
import harness
import pytest

H, W, N = 760, 760, 600  # 48 x 48 = 2,304 tiles, about 1,400 intersections


def _gaussians(torch, n, device):
    """n small Gaussians spread over the image, each covering a few pixels."""
    torch.manual_seed(0)
    means = (torch.rand(n, 2, device=device) * 2 - 1).requires_grad_()   # tanh space
    chol = torch.tensor([1.0, 0.0, 1.0], device=device).repeat(n, 1).requires_grad_()
    colors = torch.rand(n, 12, device=device).requires_grad_()
    opacity = torch.ones(n, 1, device=device)
    return means, chol, colors, opacity


@pytest.mark.gpu
def test_tile_bins_cover_every_tile():
    import torch

    harness._ensure_kernel_on_path()
    from atlas_gsplat.project_gaussians_2d import project_gaussians_2d
    from atlas_gsplat.utils import bin_and_sort_gaussians, compute_cumulative_intersects

    tile_bounds = ((W + 15) // 16, (H + 15) // 16, 1)
    num_tiles = tile_bounds[0] * tile_bounds[1]
    means, chol, _, _ = _gaussians(torch, N, "cuda")
    with torch.no_grad():
        xys, depths, radii, _, tiles_hit = project_gaussians_2d(means, chol, H, W, tile_bounds)
        num_isect, cum_hit = compute_cumulative_intersects(tiles_hit)
        assert num_isect < num_tiles, "test setup must have fewer intersections than tiles"
        *_, tile_bins = bin_and_sort_gaussians(
            N, num_isect, xys, depths, radii, cum_hit, tile_bounds)

    assert tile_bins.shape == (num_tiles, 2)
    starts, ends = tile_bins[:, 0], tile_bins[:, 1]
    assert bool((starts <= ends).all())
    assert int(ends.max()) <= num_isect
    assert int((ends - starts).sum()) == num_isect   # every intersection lands in one tile


@pytest.mark.gpu
def test_backward_with_fewer_intersections_than_tiles():
    import torch

    harness._ensure_kernel_on_path()
    from atlas_gsplat.project_gaussians_2d import project_gaussians_2d
    from atlas_gsplat.rasterize_sum import rasterize_gaussians_sum_general

    tile_bounds = ((W + 15) // 16, (H + 15) // 16, 1)
    means, chol, colors, opacity = _gaussians(torch, N, "cuda")
    xys, depths, radii, conics, tiles_hit = project_gaussians_2d(
        means, chol, H, W, tile_bounds)
    assert int(tiles_hit.sum()) < tile_bounds[0] * tile_bounds[1], \
        "test setup must have fewer intersections than tiles"
    out = rasterize_gaussians_sum_general(
        xys, depths, radii, conics, tiles_hit, colors, opacity, H, W, 16, 16,
        return_alpha=False)
    out.sum().backward()
    torch.cuda.synchronize()   # surfaces an asynchronous illegal memory access here

    assert out.shape == (H, W, 12)
    assert colors.grad is not None and bool(torch.isfinite(colors.grad).all())
