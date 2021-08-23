// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#define BLOCK_SIZE 128
#define N 4194304
// #define N 2048

__global__ void reduce0(int *g_idata, int *g_odata , int size)
{

    __shared__ int sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d\n" , i);
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1(int *g_idata, int *g_odata , int size)
{

    __shared__ int sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(int *g_idata, int *g_odata, int size)
{

    __shared__ int sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce3(int *g_idata, int *g_odata, int size)
{

    __shared__ int sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < size)
    {
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
        __syncthreads();
        // do reduction in shared mem
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        // write result for this block to global mem
        if (tid == 0)
            g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce4(int *g_idata, int *g_odata, int size)
{

    __shared__ int sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < size)
    {
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
        __syncthreads();
        // do reduction in shared mem
        for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid < 32)
        {
            sdata[tid] += sdata[tid + 32];
            __syncthreads();
            sdata[tid] += sdata[tid + 16];
            __syncthreads();
            sdata[tid] += sdata[tid + 8];
            __syncthreads();
            sdata[tid] += sdata[tid + 4];
            __syncthreads();
            sdata[tid] += sdata[tid + 2];
            __syncthreads();
            sdata[tid] += sdata[tid + 1];
        }

        // write result for this block to global mem
        if (tid == 0)
            g_odata[blockIdx.x] = sdata[0];
    }
}

#define KERNEL_NUM 5

void (*KERNELS[KERNEL_NUM])(int *, int * ,int) = {
    reduce0, reduce1, reduce2, reduce3, reduce4};

void constantInit(int *data, int size, int val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

void checkError(cudaError_t error, int line)
{

    if (error != cudaSuccess)
    {
        printf("### error occurred in line %d \n error : %s", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

float serial_reduction()
{

    clock_t beginn = clock();

    int *A = (int *)malloc(sizeof(int) * N);
    constantInit(A, N, 1);

    long int sum = 0;
    clock_t begin = clock();
    for (int i = 0; i < N; i++)
        sum += A[i];
    clock_t end = clock();
    float time_spent = ((float)(end - begin) / CLOCKS_PER_SEC) * 1000;

    printf("serial execution : %f ms\n", time_spent);

    clock_t endd = clock();
    float time_spentt = ((float)(endd - beginn) / CLOCKS_PER_SEC) * 1000;

    printf("total serial execution : %f ms\n", time_spentt);

    return time_spent;
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int reduction(int argc, char **argv, int n, int func_index)
{
    // Allocate host memory for matrices A and B
    unsigned int msize = n;
    unsigned int mem_size = sizeof(int) * msize;
    int *h_in = (int *)malloc(mem_size);

    constantInit(h_in, msize, 1);

    // Allocate device memory
    int *d_in;
    int *d_out;

    int grid_size = (n - 1) / BLOCK_SIZE + 1;
    if (func_index >= 3)
        grid_size /= 2;

    cudaError_t error;

    clock_t begin = clock();

    error = cudaMalloc((void **)&d_in, mem_size);
    checkError(error, __LINE__);

    int output_size = grid_size * sizeof(int);

    error = cudaMalloc((void **)&d_out,output_size );
    checkError(error, __LINE__);

    // copy host memory to device
    error = cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    checkError(error, __LINE__);

    float total_time = 0.0f;
    printf("grid size : %d block size : %d number of threads : %d \n", grid_size, BLOCK_SIZE, grid_size * BLOCK_SIZE);
    
    int stride = 1;
    int size = N;

    while (grid_size >= 1)
    {
        output_size = grid_size * sizeof(int);

        cudaEvent_t start;
        error = cudaEventCreate(&start);
        checkError(error, __LINE__);

        cudaEvent_t stop;
        error = cudaEventCreate(&stop);
        checkError(error, __LINE__);

        // Record the start event
        error = cudaEventRecord(start, NULL);
        checkError(error, __LINE__);

        dim3 threads(BLOCK_SIZE, 1, 1);
        dim3 grid(grid_size, 1, 1);
        
        KERNELS[func_index]<<<grid, threads>>>(d_in, d_out, size);

        error = cudaGetLastError();
        checkError(error, __LINE__);

        // Record the stop event
        error = cudaEventRecord(stop, NULL);
        checkError(error, __LINE__);

        // Wait for the stop event to complete
        error = cudaEventSynchronize(stop);
        checkError(error, __LINE__);

        float msecTotal = 0.0f;
        error = cudaEventElapsedTime(&msecTotal, start, stop);
        total_time += msecTotal;

        error = cudaEventElapsedTime(&msecTotal, start, stop);

        checkError(error, __LINE__);

        // Copy result from device to host
        grid_size /= BLOCK_SIZE;
        stride *= BLOCK_SIZE;
        size /= BLOCK_SIZE;
        cudaFree(d_in);
        d_in = d_out;
        error = cudaMalloc((void **)&d_out, output_size);
        checkError(error, __LINE__);
    }

    int *h_out = (int *)malloc(output_size);

    error = cudaMemcpy(h_out, d_in, output_size, cudaMemcpyDeviceToHost);
    checkError(error, __LINE__);

    int total_sum = 0;

    for(int i = 0 ; i < output_size / sizeof(int) ; i++)
        total_sum += h_out[i];


    printf("Elapsed time in msec = %f and bandwidth %f GB/s result = %d \n", total_time, mem_size / (total_time * 1e6), total_sum);

    // Clean up memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    clock_t end = clock();
    float time_spent = ((float)(end - begin) / CLOCKS_PER_SEC) * 1000;

    printf("execution + memory allocations : %f ms\n", time_spent);

    return EXIT_SUCCESS;
}

/**
* Program main
*/
int main(int argc, char **argv)
{
    printf("[Matrix Reduction Using CUDA] - Starting...\n");

    // By default, we use device 0
    int devID = 0;
    cudaSetDevice(devID);

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);
    checkError(error, __LINE__);

    error = cudaGetDeviceProperties(&deviceProp, devID);
    checkError(error, __LINE__);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    int n = N;

    printf("Array with size (%d)\n", n);

    // serial_reduction();

    for (size_t i = 0; i < KERNEL_NUM; i++)
    {
        printf("\n num implementation : %d \n", (int)i + 1);
        reduction(argc, argv, n, i);
    }

    return 0;
}
