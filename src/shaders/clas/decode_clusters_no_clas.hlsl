#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

RWStructuredBuffer<uint> indexBuffer : register(u0);
StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t1);
StructuredBuffer<uint> globals : register(t2);

[[vk::push_constant]] DecodePushConstant pc;
#define THREAD_GROUP_SIZE 32
[numthreads(THREAD_GROUP_SIZE, 1, 1)] 
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint id = dispatchThreadID.x;
    uint baseAddress = pc.baseAddress;

    if (id >= globals[GLOBALS_CLAS_COUNT_INDEX]) return;

    DecodeClusterData decodeClusterData = decodeClusterDatas[id];
    //uint pageIndex = decodeClusterData.pageIndex;
    uint clusterIndex = decodeClusterData.clusterIndex;
    uint indexBufferOffset = decodeClusterData.indexBufferOffset;
    uint vertexBufferOffset = decodeClusterData.vertexBufferOffset;

    DenseGeometry header = GetDenseGeometryHeader2(clusterIndex, baseAddress);

    // Decode triangle indices
    for (uint triangleIndex = 0; triangleIndex < header.numTriangles; triangleIndex++)
    {
        // write u8 indices
        uint3 triangleIndices = header.DecodeTriangle(triangleIndex);
        triangleIndices += vertexBufferOffset;

        indexBuffer[indexBufferOffset + triangleIndex * 3] = triangleIndices.x;
        indexBuffer[indexBufferOffset + triangleIndex * 3 + 1] = triangleIndices.y;
        indexBuffer[indexBufferOffset + triangleIndex * 3 + 2] = triangleIndices.z;
    }
}
