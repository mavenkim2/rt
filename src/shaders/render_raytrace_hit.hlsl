#include "common.hlsli"
[shader("closesthit")]
void main(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attr) 
{
    payload.radiance = float3(.5, 0, 0);
}
