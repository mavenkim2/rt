#include "../common.hlsli"

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    // Compute chi squared estimate
    // (p - q)^2 / q
    // p^2/q - 2p + q
    // p = same as (eq. 21) in paper
    // q = evaluated component of vmm

    // NOTE: this differs from equation 22 in the paper, which evaluates only p^2/q

    float v = 0.f; // evaluation of vmm component
    float phi = 0.f; // mc estimate
    float V = 0.f; // evaluation of vmm
    float Li = 0.f; // sample weight
    float samplePdf = 0.f; // sample pdf

    float vLi = v * Li;
    float Vphi = V * phi;

    float chiSquared = vLi * Li / Sqr(Vphi);
    chiSquared -= 2.f * vLi / Vphi;
    chiSquared += v;
    chiSquared /= samplePdf;

    // Update chi squared estimate
}
