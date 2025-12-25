#include <stdio.h>
#include "helper_math.h"

#define PI             3.14159265358979323846f
#define MAX_COMPONENTS 32

struct VMM
{
    float kappas[MAX_COMPONENTS];
    float3 directions[MAX_COMPONENTS];
    float weights[MAX_COMPONENTS];

    uint32_t numComponents;
};

__global__ void helloWorld() { printf("hello world\n"); }

__global__ void InitializeVMMS(VMM *vmms, uint32_t numVMMs)
{
    uint32_t vmmIndex = threadIdx.x;
    if (vmmIndex >= numVMMs) return;
    const float gr               = 1.618033988749895f;
    const uint32_t numComponents = MAX_COMPONENTS / 2;

    VMM vmm;
    vmm.numComponents = numComponents;

    const float weight = 1.f / float(numComponents);
    const float kappa  = 5.f;
    uint32_t l         = numComponents - 1;
    for (uint32_t n = 0; n < MAX_COMPONENTS; n++)
    {
        float3 uniformDirection;
        if (n < l + 1)
        {
            float phi      = 2.0f * ((float)n / gr);
            float z        = 1.0f - ((2.0f * n + 1.0f) / float(l + 1));
            float sinTheta = sqrt(1.f - min(z * z, 1.f));

            // cos(theta) = z
            // sin(theta) = sin(arccos(z)) = sqrt(1 - z^2)
            float3 mu        = make_float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
            uniformDirection = mu;
        }
        else
        {
            uniformDirection = make_float3(0, 0, 1);
        }

        vmm.directions[n] = uniformDirection;
        if (n < numComponents)
        {
            vmm.kappas[n]  = kappa;
            vmm.weights[n] = weight;
        }
        else
        {
            vmm.kappas[n]  = 0.0f;
            vmm.weights[n] = 0.0f;
            // vmm.normalizations[i][j] = ONE_OVER_FOUR_PI;
            // vmm.eMinus2Kappa[i][j] = 1.0f;
            // vmm._meanCosines[i][j] = 0.0f;
        }
    }

    vmms[vmmIndex] = vmm;
    printf("%f %f %f\n", vmm.directions[0].x, vmm.directions[0].y, vmm.directions[0].z);
}

int main()
{
    printf("Hello World from CPU!\n");

    uint32_t numVMMs = 32;
    VMM *vmms;
    cudaMalloc(&vmms, sizeof(VMM) * numVMMs);
    InitializeVMMS<<<1, numVMMs>>>(vmms, numVMMs);
    // helloWorld<<<1, 5>>>();

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
