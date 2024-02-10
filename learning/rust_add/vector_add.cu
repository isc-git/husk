extern "C" __global__ void
vector_add(float *a, float *b, size_t len, float* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  

    if (tid < len) {
        out[tid] = a[tid] + b[tid];
    }
}
