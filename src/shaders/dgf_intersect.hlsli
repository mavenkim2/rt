#ifndef DGF_INTERSECT_HLSLI_
#define DGF_INTERSECT_HLSLI_

#include "ray_triangle_intersection.hlsli"
#include "dense_geometry.hlsli"

uint2 DecodeBlockAndTriangleIndex(uint primitiveIndex, uint hitKind)
{
#if LOG2_TRIANGLES_PER_LEAF == 0
    uint blockIndex = primitiveIndex >> MAX_CLUSTER_TRIANGLES_BIT;
    uint triangleIndex = primitiveIndex & (MAX_CLUSTER_TRIANGLES - 1);
#else
    uint blockIndex = primitiveIndex >> (MAX_CLUSTER_TRIANGLES_BIT - LOG2_TRIANGLES_PER_LEAF);
    uint triangleIndex = ((primitiveIndex & ((1u << (MAX_CLUSTER_TRIANGLES_BIT - LOG2_TRIANGLES_PER_LEAF)) - 1u)) << LOG2_TRIANGLES_PER_LEAF) + hitKind;
#endif
    return uint2(blockIndex, triangleIndex);
}



bool IntersectCluster(in uint instanceID, in uint primitiveIndex, in float3 o, in float3 d, in float tMin, 
                      in float tMax, out float tHit, out uint kind, out float2 bary, bool debug = false)
{
    uint2 blockTriangleIndices = DecodeBlockAndTriangleIndex(primitiveIndex, 0);
    uint blockIndex = blockTriangleIndices[0];
    uint triangleIndex = blockTriangleIndices[1];

    DenseGeometry dg = GetDenseGeometryHeader(instanceID, blockIndex, debug);

#if LOG2_TRIANGLES_PER_LEAF == 0
    uint3 vids = dg.DecodeTriangle(triangleIndex);
    
    float3 p0 = dg.DecodePosition(vids[0]);
    float3 p1 = dg.DecodePosition(vids[1]);
    float3 p2 = dg.DecodePosition(vids[2]);
    
    float tempHit;
    float2 tempBary;
    
    bool result = 
    RayTriangleIntersectionMollerTrumbore(o, d, p0, p1, p2, tempHit, tempBary);
    
    result &= (tMin < tHit && tHit <= tMax);

    tHit = result ? tempTHit : tHit;
    bary = result ? tempBary : bary;
    kind = 0;
#else
    bool result = false;
    tHit = tMax;
    for (uint i = triangleIndex; i < min(triangleIndex + TRIANGLES_PER_LEAF, dg.numTriangles); i++)
    {
        uint3 vids = dg.DecodeTriangle(i);
        
        float3 p0 = dg.DecodePosition(vids[0]);
        float3 p1 = dg.DecodePosition(vids[1]);
        float3 p2 = dg.DecodePosition(vids[2]);
        
        float tempTHit;
        float2 tempBary;
        
        bool tempResult = 
        RayTriangleIntersectionMollerTrumbore(o, d, p0, p1, p2, tempTHit, tempBary);
        
        tempResult &= (tMin < tempTHit && tempTHit <= tHit);

        result |= tempResult;
        tHit = tempResult ? tempTHit : tHit;
        bary = tempResult ? tempBary : bary;
        kind = tempResult ? i & (TRIANGLES_PER_LEAF - 1) : kind;
    }

#endif
    return result;
}

#endif
