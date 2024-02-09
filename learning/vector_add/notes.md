# Sources
[Say Hello to CUDA]("https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/")

# Compiling C for best-effort
> cc -std=c17 -march=native -03 -pthread vector_add.c -o vector_add_c

## C perf
single threaded with 1000 loops of 10,000,000 float adds
- 4.6 seconds with -03
- 7.2 seconds with -02 and -01
- 31. seconds with -o0
multithreading resulted in ~10.3x speedup

# Profiling CUDA
using the `nsys` commandline utility.
> nsys profile <bin>
> nsys analyze <report name> --quiet
> nsys stats <report name> --quiet
trying to get appropriate number of threads/blocks failed for now.
`-O3` seemed to do more, maxed out at 30% GPU usage.
* Most of the time was spent in transferring data!
* On an integrated GPU device with unified memory,
    the performance was a magnitude better than the multithreaded result
