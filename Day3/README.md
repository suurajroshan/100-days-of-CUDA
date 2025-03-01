# Chapter 3 [PMPP](obsidian://open?vault=uni&file=100-days-of-cuda%2FPMPP-3rd-Edition.pdf)
Grid -> 3D array of blocks 
Block -> 2D array of threads

Creating a 1D grid that consists of 32 blocks each of which consists of 128 threads, the total number of threads is 128\*32 = 4096. The parameters that define the dimensions of grid and blocks are of type `dim3`.
```cpp
dim3 dimGrid(32,1,1);
dim3 dimBlock(128, 1,1);
vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

For one dimensional case, one can simply use:
```cpp
vecAddKernel<<<32,128>>>(...);
```

Maximum number of threads per block allowed are 1024.

## Synchronization
CUDA allows threads in the same block to coordinate their activities by using a barrier synchronization function ``__syncthreads()``
When a thread calls `__syncthreads()`, it will be held at the calling location until every thread in the block reaches the location. If present in code, `__syncthreads()` must be executed by all threads in a block or not be executed by all threads.
## Resource Assignment
Execution resources are organized into Streaming Multiprocessors (SMs). Multiple thread blocks can be assigned to each SM. For example consider a CUDA device that allows 8 blocks to be assigned to each SM, a shortage of one or more types of resources needed for simultaneous execution of 8 blocks, the CUDA runtime automatically reduces the number of blocks assigned to each SM until their combined resource usage falls below the limit. 
## Querying device properties

`cudaGetDeviceCount`: returns the number of available CUDA devices in the system.
`cudaGetDeviceProperties`: returns the properties of the device whose number is given as an argument. Example: 
```cpp
cudaDeviceProp dev_prop;
cudaGetDeviceProperties(&dev_prop, 0)
```

`dev_prop.maxThreadsPerBlock` indicates the maximal number of threads allowed in a block in the queried device.
`dev_prop.multiProcessorCount` returns the number of SMs
`dev_prop.clockRate` returns the clock frequency
`dev_prop.maxThreadsDim[0]`, `dev_prop.maxThreads Dim[1]`, and `dev_prop.maxThreadsDim[2]` returns the maximal number of threads allowed in each of `x`, `y`, and `z` dimension 
`dev_prop.maxGridSize[0]`, `dev_prop. maxGridSize[1]`, and `dev_prop.maxGridSize[2]` returns the maximal number of blocks allowed along each dimension of a grid
## Thread scheduling and latency tolerance 
A block assigned to an SM is further divided into 32 thread units called warps 
The size of warps is a property of a CUDA device, which is in the `warpSize` field of the device query variable
The warp is the unit of thread scheduling in SMs
An SM is designed to execute all threads in a warp following the Single Instruction, Multiple Data (SIMD) model which is at a given time, same instruction is applied to different sections in data with the instruction fetch/dispatch shared among the execution units in the SMs. These execution units are the hardware Streaming Processors (SPs) that execute instructions
Each SM can execute instructions for a small number of warps at any point in time.So why do we have so many warps in an SM when we can only execute a small subset of them at a given instant? When an instruction to be executed by a warp needs to wait for the result of a previously initiated long-latency operation, the warp is not selected for execution. Instead, another resident warp that is no longer waiting for results will be selected for execution. If more than one warp is ready for execution, a priority mechanism is used to select one for execution. This mechanism of filling the latency time of operations with work from other threads is often called “latency tolerance” or “latency hiding".