In accelerated applications data is allocated with `cudaMallocManaged()`-> CUDA function to allocate memory accessible by both CPU and GPUs. Memory allocated this way is called unified memory and is automatically migrated between the CPU and GPUs as needed.
Work on the GPU is asynchronous and CPU can work at the same time. CPU code can sync with the asynchronous GPU work, waiting for it to complete with `cudaDeviceSynchronize()`. 

`Thread`: The unit of execution for CUDA kernels
`block`: collection of threads 
`grid` : collection of blocks

Execution context -> special arguments given to CUDA kernels when launching using the `<<< ... >>>` syntax. It defines the number of blocks in the grid as well as the number of threads in each block.

GPU functions are called kernels. Kernels are launched with an execution configuration (this defines the number of blocks and threads in the grid).

`gridDim.x` -> CUDA variable available inside kernel that gives the number of blocks in the grid, 
`blockIdx.x` -> CUDA variable available inside executing kernel that gives the index to the thread to a data element  , 
`blockDim.x` -> CUDA variable available inside executing kernel that gives the index to the thread's block within the grid, 
`threadIdx.x` -> CUDA variable available inside executing kernel that gives the thread's block within the grid.
`threadIdx.x + blockIdx.x * blockDim.x` will map each thread to one element in the vector. Stride is calculated by `gridDim.x * blockDim.x` which is the number of threads in the grid.

# Accelerated Systems
Accelerated Systems (also called heterogeneous systems) are those composed of both CPUs and GPUs. 
`nvidia-smi` system management interface command line command.
`.cu` is the file extension for CUDA accelerated programs

```cpp
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}

int main()
{
  CPUFunction();

  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```
- The `__global__` keyword indicates that the following function will run on the GPU, and can be invoked **globally**, which in this context means either by the CPU, or, by the GPU.
- Often, code executed on the CPU is referred to as **host** code, and code running on the GPU is referred to as **device** code.
- Notice the return type `void`. It is required that functions defined with the `__global__` keyword return type `void`.
- Typically, when calling a function to run on the GPU, we call this function a **kernel**, which is **launched**.
- When launching a kernel, we must provide an **execution configuration**, which is done by using the `<<< ... >>>` syntax just prior to passing the kernel any expected arguments.
- At a high level, execution configuration allows programmers to specify the **thread hierarchy** for a kernel launch, which defines the number of thread groupings (called **blocks**), as well as how many **threads** to execute in each block.
- Unlike much C/C++ code, launching kernels is **asynchronous**: the CPU code will continue to execute _without waiting for the kernel launch to complete_.
- A call to `cudaDeviceSynchronize`, a function provided by the CUDA runtime, will cause the host (CPU) code to wait until the device (GPU) code completes, and only then resume execution on the CPU.
```bash
nvcc -arch=sm_70 -o hello-gpu 01-hello/01-hello-gpu.cu -run
```
example of the command to compile and execute `.cu` code.

`nvcc` is the NVIDIA CUDA Compiler which can compile CUDA accelerated applications both the host and the device code they contain. 
The `arch` flag indicates for which **architecture** the files must be compiled.
Providing the `run` flag will execute the successfully compiled binary.

## Launching parallel kernels
The execution configuration: ```<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>```

## Error handling
```cpp
cudaError_t err;
err = cudaMallocManaged(&a, N)                    // Assume the existence of `a` and `N`.

if (err != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
{
  printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}
```
Launching kernels which are defined to return `void`, do not return a value of type `cudaError_t`. To check for errors occurring at the last time of a kernel launch, for example if the launch configuration is erroneous, CUDA provides the `cudaGetLastError` function, which does return a value of type `cudaError_t`.
```cpp
/*
 * This launch should cause an error, but the kernel itself
 * cannot return it.
 */

someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
  printf("Error: %s\n", cudaGetErrorString(err));
}
```

Error handling function
```cpp
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

/*
 * The macro can be wrapped around any function returning
 * a value of type `cudaError_t`.
 */

  checkCuda( cudaDeviceSynchronize() )
}
```

## CUDA Streams
CUDA stream is a sequence of operations that are performed in order on the device. 