#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions
    m.def("nd_rasterize_sum_forward_general", &nd_rasterize_forward_sum_general_tensor);
    m.def("nd_rasterize_sum_backward_general", &nd_rasterize_backward_sum_general_tensor);
    m.def("project_gaussians_2d_forward", &project_gaussians_2d_forward_tensor);
    m.def("project_gaussians_2d_backward", &project_gaussians_2d_backward_tensor);
    // utils
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
}
