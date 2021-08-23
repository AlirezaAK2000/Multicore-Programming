
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



void print_info(unsigned int size);

__global__ void print_info_kernel()
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // int p = tid * bid;
    printf("‫‪Hello‬‬ ‫‪CUDA‬‬ ‫‪I’m‬‬ ‫‪a‬‬ ‫‪thread‬‬ %d ‫‪from‬‬ ‫‬‫‪block %d \n‬‬" , tid , bid);

}

int main()
{

    print_info(100);
    
    return 0;
}

void print_info(unsigned int size)
{

    cudaSetDevice(0);


    cudaError_t cudaStatus;
    print_info_kernel<<<4, size>>>();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
}
