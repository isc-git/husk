# Sources
[Say Hello to CUDA]("https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/")

# Compiling C for best-effort
> cc -std=c17 -march=native -03 -pthread vector_add.c -o vector_add_c

## C perf
single threaded with 1000 loops of 10,000,000 float adds
- 4.6 seconds with -03
- 7.2 seconds with -02 and -01
- 31. seconds with -o0
