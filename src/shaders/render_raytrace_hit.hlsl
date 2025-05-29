#include "payload.hlsli"

[shader("closesthit")]
void main(inout RayPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attr : SV_IntersectionAttributes) 
{
    payload.bary = attr.barycentrics;
    payload.rayT = RayTCurrent();
}
