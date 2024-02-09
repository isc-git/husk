#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <threads.h>

const size_t N = 10000000;
const size_t BENCH = 1000;
const size_t THREADS = 12;

typedef struct ThreadArgs {
    float *restrict a;
    float *restrict b;
    size_t len;
    float *restrict out;
} ThreadArgs;

int vector_add_multi_ent(void *args) {
    ThreadArgs *arg = (ThreadArgs*) args;
    for (int i = 0; i < arg->len; i++) {
        arg->out[i] = arg->a[i] + arg->b[i];
    }

    thrd_exit(EXIT_SUCCESS);
}

void vector_add(
    float *restrict a,
    float *restrict b,
    size_t len,
    float *restrict out
) {
    for (int i = 0; i < len; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    printf("processing %ld numbers\n", N);
    printf("\nSINGLE THREADED: \n");

    float *a;
    float *b;
    float *out;
    a = (float*) malloc(sizeof(*a) * N);
    b = (float*) malloc(sizeof(*b) * N);
    out = (float*) malloc(sizeof(*out) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    float start_time;
    float end_time;
    float elapsed;
    float elapsed_multi;

    // single threaded attempt
    start_time = (float) clock()/CLOCKS_PER_SEC;
    
    for (int i = 0; i < BENCH; i++) {
        vector_add(a, b, N, out);
    }

    end_time = (float) clock()/CLOCKS_PER_SEC;
    elapsed = end_time - start_time;
    printf("loop of %ld took %f seconds\n", BENCH, elapsed);

    // multi-threaded
    printf("\nMULTI-THREADED (%ld threads):\n", THREADS);
    start_time = (float) clock()/CLOCKS_PER_SEC;
    thrd_t threads[THREADS];
    ThreadArgs thread_args[THREADS];
    for (int i = 0; i < THREADS; i++) {

        size_t start = (N / THREADS) * i;
        size_t end = (N / THREADS) * (i + 1);

        if (end >= N) {
            end = N - 1;
        } else if (N - end <= THREADS) {
            end += N - end;
        }

        size_t len = end - start;
        printf(" %02d: %ld -> %ld: %ld\n", i, start, end, len);

        thread_args[i].a = &a[start];
        thread_args[i].b = &b[start];
        thread_args[i].len = len;
        thread_args[i].out = &out[start];
    }
    
    for (int outer = 0; outer < BENCH; outer++ ) {
        for (int i = 0; i < THREADS; i++) {
            int thread_res = thrd_create(&threads[i], vector_add_multi_ent, (void*) &thread_args[i]);
            if (thread_res != thrd_success) {
                fprintf(stderr, "failed to start thread %d: %d", i, thread_res);
                exit(thread_res);
            }
        }

        for (int i = 0; i < THREADS; i++) {
            int thread_res, local_res;
            thread_res = thrd_join(threads[i], &local_res);
            if (thread_res != thrd_success) {
                fprintf(stderr, "failed to start thread %d: %d", i, thread_res);
                exit(thread_res);
            }
        }
    }

    end_time = (float) clock()/CLOCKS_PER_SEC;
    // divided as time is in "CPU time"
    elapsed_multi = (end_time - start_time) / (float) THREADS;
    printf("loop of %ld took %f seconds\n", BENCH, elapsed);

    // compare the two
    printf("\nsingle/multi = %f\n", elapsed/elapsed_multi);

    // small print for sanity
    for (int i = 0; i < 10; i++) {
        printf("%3.1f ", out[i]);
    }
    printf("\n");

    for (int i = N - 10; i < N; i++) {
        printf("%3.1f ", out[i]);
    }
    printf("\n");


    exit(EXIT_SUCCESS);
}
