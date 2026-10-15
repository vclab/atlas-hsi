#pragma once

#include "config.h"

#include <cuda_runtime.h>

inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    uint2 &bb_min,
    uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(
    const float2 pix_center,
    const float pix_radius,
    const dim3 tile_bounds,
    uint2 &tile_min,
    uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius / (float)BLOCK_X, pix_radius / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float3 &conic, float &radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

// compute vjp from df/d_conic to df/c_cov2d
inline __device__ void cov2d_to_conic_vjp(
    const float3 &conic, const float3 &v_conic, float3 &v_cov2d
) {
    // conic = inverse cov2d, so df/d_cov2d = -conic * df/d_conic * conic.
    // Both are symmetric 2x2 matrices. Same products, in the same order, as the
    // previous glm::mat2 version (column-major M[col][row]): M = (-X) G, then M X.
    const float a = -conic.x, b = -conic.y, c = -conic.z;   // -X
    const float p = v_conic.x, q = v_conic.y, r = v_conic.z; //  G
    const float m00 = a * p + b * q;
    const float m01 = b * p + c * q;
    const float m10 = a * q + b * r;
    const float m11 = b * q + c * r;
    const float x = conic.x, y = conic.y, z = conic.z;       //  X
    v_cov2d.x = m00 * x + m10 * y;                           // S[0][0]
    v_cov2d.y = (m00 * y + m10 * z) + (m01 * x + m11 * y);   // S[1][0] + S[0][1]
    v_cov2d.z = m01 * y + m11 * z;                           // S[1][1]
}
