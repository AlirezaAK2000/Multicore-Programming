
// nvcc -Xcompiler -fopenmp vector_add.cu -o vector_add for compiling

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>
#include <sys/time.h>

#include <stdio.h>

#define BLOCK_SIZE 256
#define STRIDE 1
#define VEC_SIZE 10000000

cudaError_t cuda_parallel_vector_add(int *c, int *a, int *b, int size);
void serial_vector_add(int *c, int *a, int *b, int size);
void omp_parallel_vector_add(int *c, int *a, int *b, int size);
int *fill_with_random(int size);
__global__ void addKernel(int *c, int *a, int *b, int *size, int *stride);
int *fill_with_zeros(int size);

int main()
{

    // using cuda

    int *a = fill_with_random(VEC_SIZE);
    int *b = fill_with_random(VEC_SIZE);
    int *c = fill_with_zeros(VEC_SIZE);

    // Add vectors in parallel.
    cudaError_t cudaStatus = cuda_parallel_vector_add(c, a, b, VEC_SIZE);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cuda_parallel_vector_add failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    free(a);
    free(b);
    free(c);

    // using openmp

    a = fill_with_random(VEC_SIZE);
    b = fill_with_random(VEC_SIZE);
    c = fill_with_zeros(VEC_SIZE);

    double start = omp_get_wtime();
    omp_parallel_vector_add(c, a, b, VEC_SIZE);
    double end = omp_get_wtime() - start;

    printf("time for openMP:  %f s \n", end);

    free(a);
    free(b);
    free(c);

    // serial code

    a = fill_with_random(VEC_SIZE);
    b = fill_with_random(VEC_SIZE);
    c = fill_with_zeros(VEC_SIZE);

    start = omp_get_wtime();
    serial_vector_add(c, a, b, VEC_SIZE);
    end = omp_get_wtime() - start;

    printf("time for serial:  %f s \n", end);

    free(a);
    free(b);
    free(c);

    return 0;
}

__global__ void addKernel(int *c, int *a, int *b, int *size, int *stride)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int start = i * (*stride);


    for (int j = start; j < start + *stride; j++)
        if(j < *size)    
            c[j] = a[j] + b[j];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t cuda_parallel_vector_add(int *c, int *a, int *b, int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int *dev_size = 0;
    int *dev_stride = 0;
    cudaError_t cudaStatus;

    int num_threads = BLOCK_SIZE / STRIDE + 1;
    int stride = STRIDE;

    dim3 DimGrid((size - 1 ) / num_threads + 1, 1, 1);
    dim3 DimBlock(num_threads, 1, 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&dev_stride, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&dev_size, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed !");
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed !");
    }

    cudaStatus = cudaMemcpy(dev_stride, &stride, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed !");
    }

    cudaStatus = cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed !");
    }
    double start = omp_get_wtime();
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<DimGrid, DimBlock>>>(dev_c, dev_a, dev_b, dev_size, dev_stride);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    double end = omp_get_wtime() - start;

    printf("time for CUDA:  %f s \n", end);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed !");
    }

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

// a
void serial_vector_add(int *c, int *a, int *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void omp_parallel_vector_add(int *c, int *a, int *b, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int *fill_with_random(int size)
{
    int *a = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++)
    {
        a[i] = rand() % 100;
    }

    return a;
}

int *fill_with_zeros(int size)
{
    int *a = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++)
    {
        a[i] = 0;
    }

    return a;
}