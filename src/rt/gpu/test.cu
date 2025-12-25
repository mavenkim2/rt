#include <stdio.h>

__global__ void helloWorld() { printf("hello world\n"); }

int main()
{
    printf("Hello World from CPU!\n");

    // Launch the kernel on the GPU
    // <<<1, 5>>> means launch 1 block with 5 threads
    helloWorld<<<1, 5>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // CPU execution is asynchronous, so we must wait for the GPU to finish
    // before the program exits (otherwise the printfs might get cut off).
    cudaDeviceSynchronize();

    return 0;
}
