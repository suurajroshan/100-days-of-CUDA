# Chapter 2 [PMPP](obsidian://open?vault=uni&file=100-days-of-cuda%2FPMPP-3rd-Edition.pdf)
NVCC -> CUDA C compiler. It processes a CUDA C program using the CUDA keywords to separate the host code and device code. The host code is straight ANSI C code, which is further compiled with the host's standard C/C++ compilers and is run as a traditional CPU process. The device code is marked with CUDA keywords for data parallel functions, called ***kernels***, and their associated helper functions and data structures. The device code is further complied by a runtime component of NVCC and executed on a GPU device. 

NVIDIA GTX1080 comes with up to 8 GB of DRAM, called **global memory**. Global memory and device memory refer to the same. 

`cudaMalloc()`
	allocates object in the device global memory
	Input:
		Address of a pointer to the allocated object
		Size of allocated object in terms of bytes (will be *n* times the size of a single precision floating number which is 4 bytes in most computers)
`cudaFree()`
	Frees object from **device global memory**
	Input:
		Pointer to freed object.
`cudaMemcpy()`
	Memory data transfer
	Input:
		Pointer to destination
		Pointer to source
		Number of bytes copied
		Type/direction of transfer

A kernel function specifies the code to be executed by all threads during a parallel phase. Since all these threads execute the same code, CUDA programming is an instance of the well-known *Single-Program Multiple-Data (SPMD)* parallel programming style. When a program’s host code launches a kernel, the CUDA run-time system generates a grid of threads that are organised into a two-level hierarchy. Each grid is organised as an array of thread blocks, which will be referred to as blocks. All blocks of a grid are of the same size; each block can contain up to 1024 threads.
The `__global__` keyword indicates that the function is a kernel and that it can be called from a host function to generate a grid of threads on a device.

| Function Declaration             | Executed on | Callable from |
|----------------------------------|------------|---------------|
| `__device__ float DeviceFunc()`  | Device     | Device        |
| `__global__ void KernelFunc()`   | Device     | Host          |
| `__host__ float HostFunc()`      | Host       | Host          |

