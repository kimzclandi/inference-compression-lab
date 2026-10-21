// Residual-add/RMSNorm fusion. Reduction and rounding adapted from:
// MLX v0.29.3 mlx/backend/metal/kernels/rms_norm.metal
// Copyright © 2024 Apple Inc. MIT license: third_party/MLX-MIT.txt.
// The residual pair fusion and integration are project changes; RMSNorm and
// this SIMD reduction structure are not claimed as new algorithms.
// Inputs are row-contiguous T buffers; epsilon is a one-element FP32 buffer.
// Diagnostic variant: omit zero-fill and its barrier; mask inactive SIMD lanes.
// A threadgroup owns one row, each thread owns four consecutive elements.

const uint width = x_shape[x_ndim - 1];
const uint row = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = lid % 32;
const uint simd = lid / 32;
const uint first = lid * 4;
const size_t base = size_t(row) * width + first;
threadgroup float partial[32];
threadgroup float inverse_rms[1];
T values[4];
float sumsq = 0.0f;
for (uint i = 0; i < 4; ++i) {
    values[i] = T(0);
    if (first + i < width) {
        // Materialize the same T rounding as the native residual Add.
        values[i] = T(x[base + i] + residual[base + i]);
        h[base + i] = values[i];
        float value = float(values[i]);
        sumsq += value * value;
    }
}
sumsq = simd_sum(sumsq);
if (lane == 0) {
    partial[simd] = sumsq;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd == 0) {
    const uint groups = (width + 127) / 128;
    float part = lane < groups ? partial[lane] : 0.0f;
    sumsq = simd_sum(part);
    if (lane == 0) {
        inverse_rms[0] = metal::precise::rsqrt(sumsq / width + epsilon[0]);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0; i < 4; ++i) {
    if (first + i < width) {
        // MLX rounds normalization to T before multiplying the T weight.
        y[base + i] = weight[first + i] * T(values[i] * inverse_rms[0]);
    }
}
