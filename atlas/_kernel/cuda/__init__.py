from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


project_gaussians_2d_forward = _make_lazy_cuda_func("project_gaussians_2d_forward")
project_gaussians_2d_backward = _make_lazy_cuda_func("project_gaussians_2d_backward")
map_gaussian_to_intersects = _make_lazy_cuda_func("map_gaussian_to_intersects")
nd_rasterize_sum_forward_general = _make_lazy_cuda_func("nd_rasterize_sum_forward_general")
nd_rasterize_sum_backward_general = _make_lazy_cuda_func("nd_rasterize_sum_backward_general")
