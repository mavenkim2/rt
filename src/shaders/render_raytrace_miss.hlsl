#include "common.hlsli"

[shader("miss")]
void main(inout RayPayload payload) 
{
    payload.radiance = float3(0, .5, 0);
}
