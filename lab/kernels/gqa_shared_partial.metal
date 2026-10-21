// GQA decode specialization: share each K/V tile across seven query heads.
// Online stable softmax / split-K are existing algorithms, not a new Attention
// architecture. Compare MLX v0.29.3 sdpa_vector.h (Apple MIT, third_party/MLX-MIT.txt).
// This project writes a grouped tile schedule; native already implements online
// softmax and split-K. Logical shared loads do not measure DRAM traffic.
const uint tid = thread_position_in_threadgroup.x;
const uint lane = tid % 32;
const uint subgroup = tid / 32;
const uint part = threadgroup_position_in_grid.x;
const uint kv_head = threadgroup_position_in_grid.y;
const uint query_head = kv_head * 7 + subgroup;
const uint N = k_shape[2];
const uint parts = (N + 127) / 128;
const uint begin = part * 128;
const uint end = min(N, begin + 128);
threadgroup T ktile[32 * 64];
threadgroup T vtile[32 * 64];
float q0 = 0.0f, q1 = 0.0f;
if (subgroup < 7) {
    q0 = float(q[size_t(query_head) * q_strides[1] + (2 * lane) * q_strides[3]]) * 0.125f;
    q1 = float(q[size_t(query_head) * q_strides[1] + (2 * lane + 1) * q_strides[3]]) * 0.125f;
}
float maximum = -INFINITY;
float denominator = 0.0f;
float numerator0 = 0.0f, numerator1 = 0.0f;
for (uint tile_begin = begin; tile_begin < end; tile_begin += 32) {
    const uint count = min(uint(32), end - tile_begin);
    // All 256 threads cooperate in a single load of the shared K/V tile.
    for (uint index = tid; index < 32 * 64; index += 256) {
        const uint row = index / 64;
        const uint dim = index % 64;
        if (row < count) {
            ktile[index] = k[size_t(kv_head) * k_strides[1] +
                size_t(tile_begin + row) * k_strides[2] + dim * k_strides[3]];
            vtile[index] = v[size_t(kv_head) * v_strides[1] +
                size_t(tile_begin + row) * v_strides[2] + dim * v_strides[3]];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (subgroup < 7) {
        for (uint row = 0; row < count; ++row) {
            const uint index = row * 64 + 2 * lane;
            float score = q0 * float(ktile[index]) + q1 * float(ktile[index + 1]);
            score = simd_sum(score);
            const float next_maximum = max(maximum, score);
            const float rescale = metal::exp(maximum - next_maximum);
            const float probability = metal::exp(score - next_maximum);
            denominator = denominator * rescale + probability;
            numerator0 = numerator0 * rescale + probability * float(vtile[index]);
            numerator1 = numerator1 * rescale + probability * float(vtile[index + 1]);
            maximum = next_maximum;
        }
    }
    // The eighth SIMD group participates in barriers even though it owns no query.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (subgroup < 7) {
    const size_t base = (size_t(query_head) * parts + part) * 66;
    partial[base + 2 * lane] = numerator0;
    partial[base + 2 * lane + 1] = numerator1;
    if (lane == 0) {
        partial[base + 64] = maximum;
        partial[base + 65] = denominator;
    }
}
