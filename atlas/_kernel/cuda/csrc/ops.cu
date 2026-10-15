// Stable-ABI operator glue for the ATLAS CUDA kernel.
//
// Everything here uses only the LibTorch stable ABI, pinned to the torch 2.10 subset, so
// one compiled build loads on torch 2.10 and every later release. The operators register
// into PyTorch's dispatcher as torch.ops.atlas.<name> when the library loads.
#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))
#ifndef USE_CUDA
#define USE_CUDA  // exposes the CUDA part of torch's C shim (the current stream)
#endif

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>
#include <tuple>
#include <vector>

#include "backward.cuh"
#include "backward2d.cuh"
#include "config.h"
#include "forward.cuh"
#include "forward2d.cuh"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

namespace {

// The Python wrappers pass contiguous tensors; check rather than copy.
#define ATLAS_CHECK_INPUT(x)                                                   \
    STD_TORCH_CHECK(                                                           \
        (x).is_cuda() && (x).is_contiguous(),                                  \
        #x " must be a contiguous CUDA tensor"                                 \
    )

// The rasterizer keeps one register per channel, so it supports at most this many.
#define ATLAS_CHECK_CHANNELS(channels)                                         \
    STD_TORCH_CHECK(                                                           \
        (channels) >= 1 && (channels) <= MAX_REGISTER_CHANNELS,                \
        "the rasterizer supports 1 to ", MAX_REGISTER_CHANNELS,                \
        " channels (abundance rank), got ", (channels)                         \
    )

// Launch on torch's current stream for the tensor's device, not CUDA's default stream.
cudaStream_t current_stream(const Tensor &t) {
    void *stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream)
    );
    return static_cast<cudaStream_t>(stream);
}

dim3 to_dim3(const std::vector<int64_t> &v) {
    STD_TORCH_CHECK(v.size() == 3, "expected 3 entries, got ", v.size());
    return dim3(
        static_cast<unsigned>(v[0]),
        static_cast<unsigned>(v[1]),
        static_cast<unsigned>(v[2])
    );
}

// Zero-filled tensor of the given shape and dtype on the same device as `like`.
Tensor zeros(const Tensor &like, std::vector<int64_t> shape, ScalarType dtype) {
    return torch::stable::new_zeros(like, shape, dtype);
}

template <typename T> T *ptr(const Tensor &t) { return static_cast<T *>(t.data_ptr()); }

}  // namespace

// --- 2D projection -------------------------------------------------------------------

// Project N 2D Gaussians: pixel-space centers, depths (always 0 for 2D), integer radii,
// conics (inverse covariances), and how many screen tiles each one touches.
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> project_gaussians_2d_forward_op(
    int64_t num_points,
    Tensor means2d,
    Tensor L_elements,
    int64_t img_height,
    int64_t img_width,
    std::vector<int64_t> tile_bounds,
    double clip_thresh
) {
    ATLAS_CHECK_INPUT(means2d);
    ATLAS_CHECK_INPUT(L_elements);
    torch::stable::accelerator::DeviceGuard guard(means2d.get_device_index());

    // Zero-filled, not empty: the kernel returns early without writing for Gaussians that
    // land off-screen or have a degenerate covariance, and downstream code reads radius 0
    // and 0 tiles hit as "skip this Gaussian".
    Tensor xys = zeros(means2d, {num_points, 2}, ScalarType::Float);
    Tensor depths = zeros(means2d, {num_points}, ScalarType::Float);
    Tensor radii = zeros(means2d, {num_points}, ScalarType::Int);
    Tensor conics = zeros(means2d, {num_points, 3}, ScalarType::Float);
    Tensor num_tiles_hit = zeros(means2d, {num_points}, ScalarType::Int);

    project_gaussians_2d_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS,
        0,
        current_stream(means2d)>>>(
        static_cast<int>(num_points),
        ptr<const float2>(means2d),
        ptr<const float3>(L_elements),
        dim3(static_cast<unsigned>(img_width), static_cast<unsigned>(img_height), 1),
        to_dim3(tile_bounds),
        static_cast<float>(clip_thresh),
        ptr<float2>(xys),
        ptr<float>(depths),
        ptr<int>(radii),
        ptr<float3>(conics),
        ptr<int32_t>(num_tiles_hit)
    );
    return {xys, depths, radii, conics, num_tiles_hit};
}

// Gradients of the projection: d/d(cov2d), d/d(mean2d), d/d(Cholesky elements).
std::tuple<Tensor, Tensor, Tensor> project_gaussians_2d_backward_op(
    int64_t num_points,
    Tensor means2d,
    Tensor L_elements,
    int64_t img_height,
    int64_t img_width,
    Tensor radii,
    Tensor conics,
    Tensor v_xy,
    Tensor v_depth,
    Tensor v_conic
) {
    ATLAS_CHECK_INPUT(means2d);
    ATLAS_CHECK_INPUT(L_elements);
    ATLAS_CHECK_INPUT(radii);
    ATLAS_CHECK_INPUT(conics);
    ATLAS_CHECK_INPUT(v_xy);
    ATLAS_CHECK_INPUT(v_depth);
    ATLAS_CHECK_INPUT(v_conic);
    torch::stable::accelerator::DeviceGuard guard(means2d.get_device_index());

    // Zero-filled: Gaussians with radius 0 get no gradient (the kernel skips them).
    Tensor v_cov2d = zeros(means2d, {num_points, 3}, ScalarType::Float);
    Tensor v_mean2d = zeros(means2d, {num_points, 2}, ScalarType::Float);
    Tensor v_L_elements = zeros(means2d, {num_points, 3}, ScalarType::Float);

    project_gaussians_2d_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS,
        0,
        current_stream(means2d)>>>(
        static_cast<int>(num_points),
        ptr<const float2>(means2d),
        ptr<const float3>(L_elements),
        dim3(static_cast<unsigned>(img_width), static_cast<unsigned>(img_height), 1),
        ptr<const int>(radii),
        ptr<const float3>(conics),
        ptr<const float2>(v_xy),
        ptr<const float>(v_depth),
        ptr<const float3>(v_conic),
        ptr<float3>(v_cov2d),
        ptr<float2>(v_mean2d),
        ptr<float3>(v_L_elements)
    );
    return {v_cov2d, v_mean2d, v_L_elements};
}

// --- tile binning --------------------------------------------------------------------

// Every (tile, Gaussian) intersection: a sortable id (tile << 32 | depth bits) and the
// Gaussian it belongs to. Outputs start zeroed, as the kernel may leave slots unwritten.
std::tuple<Tensor, Tensor> map_gaussian_to_intersects_op(
    int64_t num_points,
    int64_t num_intersects,
    Tensor xys,
    Tensor depths,
    Tensor radii,
    Tensor cum_tiles_hit,
    std::vector<int64_t> tile_bounds
) {
    ATLAS_CHECK_INPUT(xys);
    ATLAS_CHECK_INPUT(depths);
    ATLAS_CHECK_INPUT(radii);
    ATLAS_CHECK_INPUT(cum_tiles_hit);
    torch::stable::accelerator::DeviceGuard guard(xys.get_device_index());

    Tensor isect_ids = zeros(xys, {num_intersects}, ScalarType::Long);
    Tensor gaussian_ids = zeros(xys, {num_intersects}, ScalarType::Int);

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS,
        0,
        current_stream(xys)>>>(
        static_cast<int>(num_points),
        ptr<const float2>(xys),
        ptr<const float>(depths),
        ptr<const int>(radii),
        ptr<const int32_t>(cum_tiles_hit),
        to_dim3(tile_bounds),
        ptr<int64_t>(isect_ids),
        ptr<int32_t>(gaussian_ids)
    );
    return {isect_ids, gaussian_ids};
}

// --- N-channel sum rasterizer --------------------------------------------------------

// Sum every Gaussian's coefficients into each pixel it covers, per channel. Also returns
// the final transmittance and the index of the last contributing Gaussian per pixel,
// which the backward pass needs.
std::tuple<Tensor, Tensor, Tensor> nd_rasterize_sum_forward_general_op(
    std::vector<int64_t> tile_bounds,
    std::vector<int64_t> block,
    std::vector<int64_t> img_size,
    Tensor gaussian_ids_sorted,
    Tensor tile_bins,
    Tensor xys,
    Tensor conics,
    Tensor colors,
    Tensor opacities,
    Tensor background
) {
    ATLAS_CHECK_INPUT(gaussian_ids_sorted);
    ATLAS_CHECK_INPUT(tile_bins);
    ATLAS_CHECK_INPUT(xys);
    ATLAS_CHECK_INPUT(conics);
    ATLAS_CHECK_INPUT(colors);
    ATLAS_CHECK_INPUT(opacities);
    ATLAS_CHECK_INPUT(background);
    STD_TORCH_CHECK(colors.dim() == 2, "colors must have 2 dimensions");
    const int64_t channels = colors.size(1);
    ATLAS_CHECK_CHANNELS(channels);
    torch::stable::accelerator::DeviceGuard guard(xys.get_device_index());

    const dim3 grid = to_dim3(tile_bounds);
    const dim3 threads = to_dim3(block);
    const dim3 size = to_dim3(img_size);  // (width, height, 1)
    const int64_t width = size.x, height = size.y;

    Tensor out_img = zeros(xys, {height, width, channels}, ScalarType::Float);
    Tensor final_Ts = zeros(xys, {height, width}, ScalarType::Float);
    Tensor final_idx = zeros(xys, {height, width}, ScalarType::Int);

    nd_rasterize_forward_sum_general<<<grid, threads, 0, current_stream(xys)>>>(
        grid,
        size,
        static_cast<unsigned>(channels),
        ptr<const int32_t>(gaussian_ids_sorted),
        ptr<const int2>(tile_bins),
        ptr<const float2>(xys),
        ptr<const float3>(conics),
        ptr<const float>(colors),
        ptr<const float>(opacities),
        ptr<float>(final_Ts),
        ptr<int>(final_idx),
        ptr<float>(out_img),
        ptr<const float>(background)
    );
    return {out_img, final_Ts, final_idx};
}

// Gradients of the rasterizer with respect to centers, conics, coefficients, opacities.
std::tuple<Tensor, Tensor, Tensor, Tensor> nd_rasterize_sum_backward_general_op(
    int64_t img_height,
    int64_t img_width,
    int64_t block_h,
    int64_t block_w,
    Tensor gaussian_ids_sorted,
    Tensor tile_bins,
    Tensor xys,
    Tensor conics,
    Tensor colors,
    Tensor opacities,
    Tensor background,
    Tensor final_Ts,
    Tensor final_idx,
    Tensor v_output,
    Tensor v_output_alpha
) {
    ATLAS_CHECK_INPUT(gaussian_ids_sorted);
    ATLAS_CHECK_INPUT(tile_bins);
    ATLAS_CHECK_INPUT(xys);
    ATLAS_CHECK_INPUT(conics);
    ATLAS_CHECK_INPUT(colors);
    ATLAS_CHECK_INPUT(opacities);
    ATLAS_CHECK_INPUT(background);
    ATLAS_CHECK_INPUT(final_Ts);
    ATLAS_CHECK_INPUT(final_idx);
    ATLAS_CHECK_INPUT(v_output);
    ATLAS_CHECK_INPUT(v_output_alpha);
    STD_TORCH_CHECK(
        xys.dim() == 2 && xys.size(1) == 2, "xys must have dimensions (num_points, 2)"
    );
    STD_TORCH_CHECK(colors.dim() == 2, "colors must have 2 dimensions");
    const int64_t num_points = xys.size(0);
    const int64_t channels = colors.size(1);
    ATLAS_CHECK_CHANNELS(channels);
    torch::stable::accelerator::DeviceGuard guard(xys.get_device_index());

    const dim3 grid(
        static_cast<unsigned>((img_width + block_w - 1) / block_w),
        static_cast<unsigned>((img_height + block_h - 1) / block_h),
        1
    );
    const dim3 threads(static_cast<unsigned>(block_w), static_cast<unsigned>(block_h), 1);
    const dim3 size(static_cast<unsigned>(img_width), static_cast<unsigned>(img_height), 1);

    Tensor v_xy = zeros(xys, {num_points, 2}, ScalarType::Float);
    Tensor v_conic = zeros(xys, {num_points, 3}, ScalarType::Float);
    Tensor v_colors = zeros(xys, {num_points, channels}, ScalarType::Float);
    Tensor v_opacity = zeros(xys, {num_points, 1}, ScalarType::Float);

    nd_rasterize_backward_sum_general_kernel<<<grid, threads, 0, current_stream(xys)>>>(
        grid,
        size,
        static_cast<unsigned>(channels),
        ptr<const int32_t>(gaussian_ids_sorted),
        ptr<const int2>(tile_bins),
        ptr<const float2>(xys),
        ptr<const float3>(conics),
        ptr<const float>(colors),
        ptr<const float>(opacities),
        ptr<const float>(background),
        ptr<const float>(final_Ts),
        ptr<const int>(final_idx),
        ptr<const float>(v_output),
        ptr<const float>(v_output_alpha),
        ptr<float2>(v_xy),
        ptr<float3>(v_conic),
        ptr<float>(v_colors),
        ptr<float>(v_opacity)
    );
    return {v_xy, v_conic, v_colors, v_opacity};
}

// --- registration --------------------------------------------------------------------

STABLE_TORCH_LIBRARY(atlas, m) {
    m.def(
        "project_gaussians_2d_forward(int num_points, Tensor means2d, Tensor L_elements, "
        "int img_height, int img_width, int[] tile_bounds, float clip_thresh) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    );
    m.def(
        "project_gaussians_2d_backward(int num_points, Tensor means2d, Tensor L_elements, "
        "int img_height, int img_width, Tensor radii, Tensor conics, Tensor v_xy, "
        "Tensor v_depth, Tensor v_conic) -> (Tensor, Tensor, Tensor)"
    );
    m.def(
        "map_gaussian_to_intersects(int num_points, int num_intersects, Tensor xys, "
        "Tensor depths, Tensor radii, Tensor cum_tiles_hit, int[] tile_bounds) "
        "-> (Tensor, Tensor)"
    );
    m.def(
        "nd_rasterize_sum_forward_general(int[] tile_bounds, int[] block, int[] img_size, "
        "Tensor gaussian_ids_sorted, Tensor tile_bins, Tensor xys, Tensor conics, "
        "Tensor colors, Tensor opacities, Tensor background) -> (Tensor, Tensor, Tensor)"
    );
    m.def(
        "nd_rasterize_sum_backward_general(int img_height, int img_width, int block_h, "
        "int block_w, Tensor gaussian_ids_sorted, Tensor tile_bins, Tensor xys, "
        "Tensor conics, Tensor colors, Tensor opacities, Tensor background, "
        "Tensor final_Ts, Tensor final_idx, Tensor v_output, Tensor v_output_alpha) "
        "-> (Tensor, Tensor, Tensor, Tensor)"
    );
}

STABLE_TORCH_LIBRARY_IMPL(atlas, CUDA, m) {
    m.impl("project_gaussians_2d_forward", TORCH_BOX(&project_gaussians_2d_forward_op));
    m.impl("project_gaussians_2d_backward", TORCH_BOX(&project_gaussians_2d_backward_op));
    m.impl("map_gaussian_to_intersects", TORCH_BOX(&map_gaussian_to_intersects_op));
    m.impl(
        "nd_rasterize_sum_forward_general", TORCH_BOX(&nd_rasterize_sum_forward_general_op)
    );
    m.impl(
        "nd_rasterize_sum_backward_general", TORCH_BOX(&nd_rasterize_sum_backward_general_op)
    );
}
