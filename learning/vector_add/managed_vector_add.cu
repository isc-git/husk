#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

const size_t N = 10000000;
const size_t BENCH = 1000;
const float MAX_ERR = 1e-6;

__global__ void
blocked_vector_add(float *a, float *b, size_t len, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  

    if (i < len) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a;
    float *b;
    float *out;

    // allocate unified memory
    cudaMallocManaged(&a, sizeof(float) * N);
    cudaMallocManaged(&b, sizeof(float) * N);
    cudaMallocManaged(&out, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    /*
        PROFILING
    */
    printf("\nPROFILING\n");
    float start_time;
    float end_time;
    float elapsed;

    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size; 
    start_time = (float) clock()/CLOCKS_PER_SEC;
    for (int i = 0; i < BENCH; i++) {
        blocked_vector_add<<<num_blocks,block_size>>>(a, b, N, out);
    }
    cudaDeviceSynchronize();

    end_time = (float) clock()/CLOCKS_PER_SEC;
    elapsed = end_time - start_time;
    printf("%ld iterations of %ld numbers took %3.2f seconds\n", BENCH, N, elapsed);

    // print for sanity check
    for (int i = 0; i < 10; i++) {
        printf("%f ", out[i]);
    }
    printf("...\n");
    for (int i = N - 10; i < N; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");

    float total = 0.0f;
    for (int i = 0; i < N; i++) {
        total += out[i];
    }
    printf("total: %f\n", total);

    // check results formally
    for (int i = 0; i < N; i++) {
        assert(fabs(out[i] - 3.0f) < MAX_ERR);
    }

    // unified memoru freed via CUDA
    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    
    exit(EXIT_SUCCESS);
}
