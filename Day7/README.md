# Chapter 4 [PMPP](obsidian://open?vault=uni&file=100-days-of-cuda%2FPMPP-3rd-Edition.pdf)
Assume a device D has 16,384 (16K) bytes of shared memory for thread blocks in each SM and each SM can accommodate 8 blocks. Now to reach this maximum, each block can use a maximum of 2KB of shared memory otherwise the number of blocks that can reside in each SM is reduced and hence the device cannot find more blocks to perform computations on when the there is idle time. 

Shared memory can become limiting factor. In the case of matrix multiplication ($C =A \cdot B$), for a tile size of $16 \times 16$, each block needs $16 × 16 × 4=1$K bytes for storage of $A$ assuming each element is a float type (4 bytes) and another 1KB for B. Then each block uses 2KB of shared memory. A 16KB shared memory allows 8 blocks to simultaneously reside in an SM. It is better if the number of blocks that can reside on the shared memory is equal to the maximum number of blocks allowed by the threading hardware in which case the shared memory would not be a limitation for the tile size. 

With the declaration,
```cpp
__shared__ As[TILE_WIDTH][TILE_WIDTH];
__shared__ Bs[TILE_WIDTH][TILE_WIDTH];
```
the size of the `TILE_WIDTH` cannot be changed without recompilation.

A declaration such as 
```cpp
extern __shared__ d_A[];
extern __shared__ d_b[];
```
and we can dynamically determine the amount of shared memory to be used according to the device query result and supply that as a configuration parameter to the kernel launch at runtime. 
```cpp
size_t size = calculate_approximate_SM_usage(dev_prop.sharedMemPerBlock, ...);
mareixMulKernel<<<dimGrid, dimBlock, size>>>(d_A, d_B, d_C, wA, wB);
```


# Chapter 5 [PMPP](obsidian://open?vault=uni&file=100-days-of-cuda%2FPMPP-3rd-Edition.pdf)
Threads in a warp execute the same instruction at any given point in time. When all threads in a warp execute a load instruction, the hardware detects whether they access consecutive global memory locations. That is, the most favourable access pattern is achieved when all threads in a warp access consecutive global memory locations. In this case, the hardware combines, or *coalesces*, all these accesses into a consolidated access to consecutive DRAM locations. 
	For example, for a given load instruction of a warp, if thread $0$ accesses global memory location $N$ , thread $1$ location $N+1$, thread $2$ location $N+2$, and so on, all these accesses will be coalesced, or combined into a single request for consecutive locations when accessing the DRAMs. Such coalesced access allows the DRAMs to deliver data as a burst.
In the case of matrix multiplication $C = A \cdot B$, and the computation of $C$ on the device is taken care by one thread for every index of the matrix. 
```cpp
for (int i=0; i < N; ++i){
	C[ty][tx] = A[ty][k] + B[k][tx];
}
```
looking at $A$ and $B$ independently suggests that the accesses to $A$ are not coalesced. $A[ty*N+k]$ the term $t_y * N$ is different for every different thread block which makes the access not coalesced. While for $B$ which is accessed by $B[k*N+tx]$, $k*N$ is not dependent on the `threadIdx` and hence would be same across all thread of a thread block meaning that the accesses are coalesced. 
Having the idea as to how the coalescing happens also suggests that the tiled matrix multiplication provides better performance as 
	- the memory loads are reduced due to the reuse of data in the shared memory 
	- remaining memory loads are coalesced so the DRAM bandwidth utilization is further improved
Although in the case of tiled matrix multiplication, `As` which is the `TILE_WIDTH x TILE_WIDTH` block of `A` is loaded in the shared memory, threads in a warp do not access consecutive locations. And this is not a problem since `As` is in shared memory, which does not require coalescing to achieve high-speed data access. 

