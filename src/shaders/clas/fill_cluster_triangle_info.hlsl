RWStructuredBuffer<BUILD_CLUSTERS_BOTTOM_LEVEL_INFO> clusterBottomLevelInfos : register(u4);

[numthreads(256, 1, 1)]
void main(uint3 groupID: SV_GroupID, uint3 dtID: SV_DispatchThreadID)
{
    // Each group is a BLAS
    if (dtID.x >= pc.numClusters) return;

    InterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX], 1);

    uint64_t indexBufferBaseAddress = ((pc.indexBufferBaseAddressHighBits << 32) | (pc.indexBufferBaseAddressLowBits));
    uint64_t vertexBufferBaseAddress = ((pc.vertexBufferBaseAddressHighBits << 32) | (pc.vertexBufferBaseAddressLowBits));

    BUILD_CLUSTER_TRIANGLE_INFO desc = (BuildClasDesc)0;
    desc.clusterId = dtID.x;
    desc.clusterFlags = 0;
    desc.triangleCount = header.numTriangles;
    desc.vertexCount = header.numVertices;
    desc.positionTruncateBitCount = 0;
    desc.indexFormat = 1; // VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV
    desc.opacityMicromapIndexFormat = 0;
    desc.baseGeometryIndexAndFlags = 0;
    desc.indexBufferStride = 0; // tightly packed
    desc.vertexBufferStride = 0;
    desc.geometryIndexAndFlagsBufferStride = 0;
    desc.opacityMicromapIndexBufferStride = 0;
    desc.indexBuffer = indexBufferBaseAddress + indexBufferOffset * 1;
    desc.vertexBuffer = vertexBufferBaseAddress + vertexBufferOffset * 12;
    desc.geometryIndexAndFlagsBuffer = 0;
    desc.opacityMicromapArray = 0;
    desc.opacityMicromapIndexBuffer = 0;

    buildClasDescs[groupID.x] = desc;
}
