#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

const size_t N = 10000000;
const size_t BENCH = 1000;
const float MAX_ERR = 1e-6;

__global__ void
vector_add(float *a, float *b, size_t len, float* out) {
    for (int i = 0; i < len; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__ void
threaded_vector_add(float *a, float *b, size_t len, float* out) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    
    // every iteration, ith thread computes (stride * i) + i
    for (int i = index; i < N; i += stride) {
        out[i] = a[i] + b[i];
    }
}

__global__ void
blocked_vector_add(float *a, float *b, size_t len, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  

    if (i < N) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *d_a;
    float *b, *d_b;
    float *out, *d_out;

    // allocate host memory
    a = (float *) malloc(sizeof(*a) * N);
    b = (float *) malloc(sizeof(*b) * N);
    out = (float *) malloc(sizeof(*out) * N);

    // allocate device memory
    cudaMalloc((void**) &d_a, sizeof(*a) * N);
    cudaMalloc((void**) &d_b, sizeof(*b) * N);
    cudaMalloc((void**) &d_out, sizeof(*out) * N);

    // assign local memory
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // copy inputs to device
    cudaMemcpy(d_a, a, sizeof(*a) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(*b) * N, cudaMemcpyHostToDevice);

    // add on GPU
    vector_add<<<1,1>>>(d_a, d_b, N, d_out);

    // copy outputs to host
    cudaMemcpy(out, d_out, sizeof(*d_out) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    // add on GPU, with many threads
    threaded_vector_add<<<1,256>>>(d_a, d_b, N, d_out);

    // copy outputs to host
    cudaMemcpy(out, d_out, sizeof(*d_out) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    // add on GPU, with many blocks that have many threads
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size; 
    blocked_vector_add<<<num_blocks,block_size>>>(d_a, d_b, N, d_out);

    // copy outputs to host
    cudaMemcpy(out, d_out, sizeof(*d_out) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    // copy outputs to host
    cudaMemcpy(out, d_out, sizeof(*d_out) * N, cudaMemcpyDeviceToHost);

    /*
        PROFILING
    */
    printf("\nPROFILING\n");
    float start_time;
    float end_time;
    float elapsed;
    start_time = (float) clock()/CLOCKS_PER_SEC;
    for (int i = 0; i < BENCH; i++) {
        // copy inputs to device
        cudaMemcpy(d_a, a, sizeof(*a) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(*b) * N, cudaMemcpyHostToDevice);

        // add on GPU, with many blockst that have many threads
        blocked_vector_add<<<num_blocks,block_size>>>(d_a, d_b, N, d_out);

        // copy outputs to host
        cudaMemcpy(out, d_out, sizeof(*d_out) * N, cudaMemcpyDeviceToHost);
    }
    end_time = (float) clock()/CLOCKS_PER_SEC;
    elapsed = end_time - start_time;
    printf("%ld iterations of %ld numbers took %3.2f seconds\n", BENCH, N, elapsed);

    for (int i = 0; i < 10; i++) {
        printf("%f ", out[i]);
    }
    printf("...\n");
    for (int i = N - 10; i < N; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");

    free(a);
    free(b);
    free(out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    
    exit(EXIT_SUCCESS);
}
