#include "ray_triangle_intersection.hlsli"
#include "dense_geometry.hlsli"

[shader("intersection")]
void main()
{
    uint primitiveIndex = PrimitiveIndex();
    
    uint blockIndex = primitiveIndex >> MAX_CLUSTER_TRIANGLES_BIT;
    uint triangleIndex = primitiveIndex & (MAX_CLUSTER_TRIANGLES - 1);
    
    DenseGeometry dg = GetDenseGeometryHeader(blockIndex);
    uint3 vids = dg.DecodeTriangle(triangleIndex);
    
    float3 p0 = dg.DecodePosition(vids[0]);
    float3 p1 = dg.DecodePosition(vids[1]);
    float3 p2 = dg.DecodePosition(vids[2]);
    
    float tHit;
    float2 bary;
    
    bool result = 
    RayTriangleIntersectionMollerTrumbore(ObjectRayOrigin(), ObjectRayDirection(), 
                                          p0, p1, p2, tHit, bary);
    
    result &= (RayTMin() < tHit && tHit <= RayTCurrent());
    
    if (result) 
    {
        BuiltInTriangleIntersectionAttributes attr;
        attr.barycentrics = bary;
        ReportHit(tHit, 0, attr);
    }
}
