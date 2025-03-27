#include "sampling.hlsli"

#if 0
struct [raypayload] RayPayload 
{
    float3 pxOffset;
    float3 pyOffset;
    float3 dxOffset;
    float3 dyOffset;

    RNG rng : write(caller, closesthit) : read(caller, closesthit);
    float3 radiance : write(caller, miss, closesthit) : read(caller);
    float3 throughput : write(caller, closesthit) : read(caller, miss, closesthit);

    float3 pos : write(closesthit) : read(caller);
    float3 dir : write(closesthit) : read(caller);

    bool missed : write(caller, miss, closesthit) : read(caller);
};
#endif

struct RayPayload 
{
    float3 pxOffset;
    float3 pyOffset;
    float3 dxOffset;
    float3 dyOffset;

    RNG rng;
    float3 radiance;
    float3 throughput;

    float3 pos;
    float3 dir;

    bool missed;
};
