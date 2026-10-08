// Diagnostic-only metadata: no Attention arithmetic or layout conversion.
const uint i = thread_position_in_grid.x;
if (i < 4) {
    meta[i] = int(q_shape[i]); meta[4+i] = int(q_strides[i]);
    meta[8+i] = int(k_shape[i]); meta[12+i] = int(k_strides[i]);
    meta[16+i] = int(v_shape[i]); meta[20+i] = int(v_strides[i]);
}
