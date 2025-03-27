#include "dgf_intersect.hlsli"

[shader("intersection")]
void main()
{
    uint primitiveIndex = PrimitiveIndex();

    float tHit = 0;
    uint kind = 0;
    float2 bary = 0;
    bool result = IntersectCluster(primitiveIndex, ObjectRayOrigin(), ObjectRayDirection(), 
                                   RayTMin(), RayTCurrent(), tHit, kind, bary);

    if (result) 
    {
        BuiltInTriangleIntersectionAttributes attr;
        attr.barycentrics = bary;
        ReportHit(tHit, kind, attr);
    }
}
