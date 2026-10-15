from typing import Callable


def _make_lazy_op(name: str) -> Callable:
    """``torch.ops.atlas.<name>``. Loading the kernel library registers the operators, so
    the first call triggers that load (and, without a prebuilt kernel, the first-use
    compile)."""
    def call_op(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        import torch

        from ._backend import _C  # noqa: F401  (loads the library)

        return getattr(torch.ops.atlas, name)(*args, **kwargs)

    return call_op


project_gaussians_2d_forward = _make_lazy_op("project_gaussians_2d_forward")
project_gaussians_2d_backward = _make_lazy_op("project_gaussians_2d_backward")
map_gaussian_to_intersects = _make_lazy_op("map_gaussian_to_intersects")
nd_rasterize_sum_forward_general = _make_lazy_op("nd_rasterize_sum_forward_general")
nd_rasterize_sum_backward_general = _make_lazy_op("nd_rasterize_sum_backward_general")
