#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);

__global__ void nd_rasterize_forward_sum_general(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float* __restrict__ out_img,
    const float* __restrict__ background
);
