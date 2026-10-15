"""Project 2D Gaussians onto the image: pixel centers, radii, conics, and tiles hit."""


from torch import Tensor
from torch.autograd import Function

from . import cuda as _C


def project_gaussians_2d(
    means2d: Tensor,
    L_elements: Tensor,
    img_height: int,
    img_width: int,
    tile_bounds: tuple[int, int, int],
    clip_thresh: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project N 2D Gaussians onto the pixel grid. Differentiable in ``means2d`` and
    ``L_elements``.

    Args:
        means2d: (N, 2) float32 centers in normalized coordinates [-1, 1]; the pixel
            center is ``0.5 * W * (x + 1)``, likewise for y with H.
        L_elements: (N, 3) float32 Cholesky factor ``(l11, l21, l22)`` of each covariance,
            ``Sigma = L L^T`` with ``L = [[l11, 0], [l21, l22]]``.
        img_height: image height H in pixels.
        img_width: image width W in pixels.
        tile_bounds: ``(tiles_x, tiles_y, 1)``, the grid of 16 x 16 pixel tiles.
        clip_thresh: unused by the 2D kernel; kept so the call signature is unchanged.

    Returns:
        xys: (N, 2) float32 pixel-space centers.
        depths: (N,) float32, always 0 (2D Gaussians have no depth).
        radii: (N,) int32 pixel radius at 3 sigma; 0 means off-screen or degenerate, and
            such Gaussians are skipped by every later step.
        conics: (N, 3) float32 inverse covariance ``(a, b, c)`` of ``[[a, b], [b, c]]``.
        num_tiles_hit: (N,) int32 number of tiles each Gaussian overlaps.
    """
    return _ProjectGaussians2d.apply(
        means2d.contiguous(),
        L_elements.contiguous(),
        img_height,
        img_width,
        tile_bounds,
        clip_thresh,
    )

class _ProjectGaussians2d(Function):
    """Autograd function behind :func:`project_gaussians_2d`."""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        L_elements: Tensor,
        img_height: int,
        img_width: int,
        tile_bounds: tuple[int, int, int],
        clip_thresh: float = 0.01,
    ):
        num_points = means2d.shape[-2]

        (
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_2d_forward(
            num_points,
            means2d.contiguous(),
            L_elements.contiguous(),
            img_height,
            img_width,
            tile_bounds,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points

        # Save tensors.
        ctx.save_for_backward(
            means2d,
            L_elements,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit):
        (
            means2d,
            L_elements,
            radii,
            conics,
        ) = ctx.saved_tensors

        (v_cov2d, v_mean2d, v_L_elements) = _C.project_gaussians_2d_backward(
            ctx.num_points,
            means2d.contiguous(),
            L_elements.contiguous(),
            ctx.img_height,
            ctx.img_width,
            radii.contiguous(),
            conics.contiguous(),
            v_xys.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
        )

        # Return a gradient for each input.
        return (
            # means2d
            v_mean2d,
            # L_elements
            v_L_elements,
            # img_height
            None,
            # img_width
            None,
            # tile_bounds
            None,
            # clip_thresh
            None,
        )
