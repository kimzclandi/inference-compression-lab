// Stable merge of per-partition (maximum, exponential sum, weighted numerator).
// Existing online-softmax algebra; no new algorithm claim.
const uint head = threadgroup_position_in_grid.x;
const uint lane = thread_position_in_threadgroup.x;
const uint parts = partial_shape[1];
const size_t head_base = size_t(head) * parts * 66;
float maximum = -INFINITY;
for (uint part = 0; part < parts; ++part) {
    maximum = max(maximum, partial[head_base + part * 66 + 64]);
}
float denominator = 0.0f;
float numerator0 = 0.0f, numerator1 = 0.0f;
for (uint part = 0; part < parts; ++part) {
    const size_t base = head_base + part * 66;
    const float factor = metal::fast::exp(partial[base + 64] - maximum);
    denominator += partial[base + 65] * factor;
    numerator0 += partial[base + 2 * lane] * factor;
    numerator1 += partial[base + 2 * lane + 1] * factor;
}
out[head * 64 + 2 * lane] = T(numerator0 / denominator);
fp32[head * 64 + 2 * lane] = numerator0 / denominator;
out[head * 64 + 2 * lane + 1] = T(numerator1 / denominator);
fp32[head * 64 + 2 * lane + 1] = numerator1 / denominator;
