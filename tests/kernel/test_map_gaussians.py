"""map_gaussian_to_intersects (CUDA) against the plain-PyTorch reference."""
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_map_gaussian_to_intersects_matches_reference():
    from _reference import map_gaussian_to_intersects as reference
    from _reference import random_gaussians

    from atlas._kernel import compute_cumulative_intersects, map_gaussian_to_intersects

    xys, depths, radii, tiles_hit, tile_bounds = random_gaussians(300, 200, 150)  # non-square
    num_isect, cum_hit = compute_cumulative_intersects(tiles_hit)
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        len(radii), num_isect, xys, depths, radii, cum_hit, tile_bounds)
    ref_isect, ref_gaussian = reference(xys.cpu(), radii.cpu(), cum_hit.cpu(), tile_bounds)

    torch.testing.assert_close(isect_ids.cpu(), ref_isect)
    torch.testing.assert_close(gaussian_ids.cpu(), ref_gaussian)
