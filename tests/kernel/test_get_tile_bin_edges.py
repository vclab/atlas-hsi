"""Tile bins (one [start, end) row per tile) against the plain-PyTorch reference."""
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_tile_bins_match_reference():
    from _reference import get_tile_bin_edges as reference
    from _reference import random_gaussians

    from atlas._kernel import bin_and_sort_gaussians, compute_cumulative_intersects

    xys, depths, radii, tiles_hit, tile_bounds = random_gaussians(400, 240, 320)
    num_isect, cum_hit = compute_cumulative_intersects(tiles_hit)
    _, _, isect_sorted, _, tile_bins = bin_and_sort_gaussians(
        len(radii), num_isect, xys, depths, radii, cum_hit, tile_bounds)

    assert tile_bins.shape == (tile_bounds[0] * tile_bounds[1], 2)
    torch.testing.assert_close(tile_bins.cpu(), reference(isect_sorted.cpu(), tile_bounds))


@pytest.mark.gpu
def test_tile_bins_fewer_intersections_than_tiles():
    """Regression: 3 intersections on a 4 x 3 tile grid, two of them in the last tile."""
    from _reference import get_tile_bin_edges as reference

    from atlas._kernel import get_tile_bin_edges

    tile_bounds = (4, 3, 1)
    isect_ids_sorted = torch.tensor([2, 11, 11], dtype=torch.int64, device="cuda") << 32
    expected = torch.zeros((12, 2), dtype=torch.int32)
    expected[2] = torch.tensor([0, 1])
    expected[11] = torch.tensor([1, 3])

    tile_bins = get_tile_bin_edges(3, isect_ids_sorted, tile_bounds)
    torch.testing.assert_close(tile_bins.cpu(), expected)
    torch.testing.assert_close(reference(isect_ids_sorted.cpu(), tile_bounds), expected)
    assert tile_bins.is_contiguous()
