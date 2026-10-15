#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void nd_rasterize_backward_sum_general_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);
