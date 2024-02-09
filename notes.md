# Nomenclature
- "Device": The GPU
- "Host": The CPU communicating with the GPU
- "Kernel": A function run on device
- "SMs": Streaming Multiprocessors

# CUDA Syntax Notes
```c
// indicate function is a kernel
__global__

// syntax for providing execution configuration.
// a kernel launch:
//   with a grid of B thread blocks
//   each thread block has T threads
example_kernel<<<B, T>>>();

// for all B in each loop, T threads compute simultaneously
```

# Compiling CUDA
> nvcc example.cu -o example

# CUDA
Basic memory management achieved via `cudaMalloc` and `cudaFree`.
Transfer data between host and device with `cudaMemcpy`.
- `threadIdx.x`, index of thread in block.
- `blockDim.x`, number of threads in block. "size of block"
- `blockIdx.x`, index of block with grid.
- `gridDim.x`, number of blocks int the grid (i think?) "size of grid"
Each SM consists of many parallel processors and can run concurrent blocks.
