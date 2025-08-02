#ifndef DENSE_GEOMETRY_HLSLI_
#define DENSE_GEOMETRY_HLSLI_

#include "bit_twiddling.hlsli"
#include "common.hlsli"
#include "../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../rt/shader_interop/as_shaderinterop.h"

ByteAddressBuffer clusterPageData : register(t8);

struct DenseGeometry 
{
    uint baseAddress;
    uint geoBaseAddress;
    uint shadBaseAddress;

    float4 lodBounds;
    float lodError;

    int3 anchor;
    uint2 octBase;

    uint3 posBitWidths;
    uint2 octBitWidths;
    uint indexBitWidth;
    uint numFaceIDBits;

    uint numTriangles;
    uint numVertices;

    uint normalOffset;
    uint faceIDOffset;
    uint indexOffset;
    uint ctrlOffset;
    uint firstBitOffset;
    int posPrecision;
    uint materialInfo;

    uint3 numPrevRestartsBeforeDwords;
    uint3 prevHighRestartBeforeDwords;
    int3 prevHighEdge1BeforeDwords;
    int3 prevHighEdge2BeforeDwords;

    uint flags;

    uint numBricks;
    uint brickOffset;

    bool debug;

    Brick DecodeBrick(uint brickIndex) 
    {
        Brick brick;
        return brick;
    }

    uint3 DecodeTriangle(uint triangleIndex)
    {
        enum 
        {
            Restart = 0,
            Edge1 = 1, 
            Edge2 = 2, 
            Backtrack = 3,
        };

        // per scene index Dense Geometry
        int3 indexAddress = int3(0, 1, 2);

        uint dwordIndex = triangleIndex >> 5;
        uint bitIndex = triangleIndex & 31u;

        uint bit = (1u << bitIndex);
        uint mask = (1u << bitIndex) - 1u;

        // x = restarts, y = edge, z = backtrack
        uint address = baseAddress + geoBaseAddress + ctrlOffset + dwordIndex * 12;
        uint2 vals = GetAlignedAddressAndBitOffset(address, 0);
        uint4 stripBitmasks = clusterPageData.Load4(vals[0]);
        stripBitmasks[0] = BitAlignU32(stripBitmasks[1], stripBitmasks[0], vals[1]);
        stripBitmasks[1] = BitAlignU32(stripBitmasks[2], stripBitmasks[1], vals[1]);
        stripBitmasks[2] = BitAlignU32(stripBitmasks[3], stripBitmasks[2], vals[1]);

        const uint restartBitmask = stripBitmasks[0];
        const uint edgeBitmask = stripBitmasks[1];
        const uint backtrackBitmask = stripBitmasks[2] & mask;

        // Count the number of strip restarts
        uint prevRestartsBeforeDwords = dwordIndex ? numPrevRestartsBeforeDwords[dwordIndex - 1] : 0u;
        uint numRestarts = countbits(restartBitmask & mask) + prevRestartsBeforeDwords;
        uint r = numRestarts;

        const uint isRestart = BitFieldExtractU32(stripBitmasks[0], 1, bitIndex);
        const uint isEdge1 = BitFieldExtractU32(stripBitmasks[1], 1, bitIndex);
        const uint isBacktrack = BitFieldExtractU32(stripBitmasks[2], 1, bitIndex);

        const uint isEdge1Bitmask = 0u - isEdge1;

        // Restart
        if (isRestart)
        {
            r++;
            indexAddress = uint3(2 * r + triangleIndex - 2, 2 * r + triangleIndex - 1, 2 * r + triangleIndex);
        }
        // Backtrack, edge1 or edge2
        else
        {
            indexAddress.z = 2 * r + triangleIndex;
            mask >>= isBacktrack;
            indexAddress.y = 2 * r + triangleIndex - 1 - isBacktrack;

            // ALL CASES
            // CURRENT IS RESTART: trivial case
            // CURRENT IS EDGE1, need to find what prev[1] is 
            // find the first previous triangle that is not an EDGE1, NOR a backtrack with a prev ctrl of EDGE2
            //       if it's restart, then 2 * r + prevTriangle - 1
            //       if it's edge2, then 2 * r + prevTriangle - 1
            //       if it's backtrack w/ prevctrl of EDGE1, then 2 * r + prevTriangle - 2
            // CURRENT IS EDGE2, need to find what prev[0] is 
            // find the first previous triangle that is not an EDGE2, NOR a backtrack with a prev ctrl of EDGE1
            //       if it's restart, then 2 * r + prevTriangle - 2
            //       if it's edge1, then 2 * r + prevTriangle - 1
            //       if it's backtrack w/ prevctrl of EDGE2, then 2 * r + prevTriangle - 2
            // CURRENT IS BACKTRACK, follow EDGE1 or EDGE2 rules based on opposite of prevCtrl. 
            // For example, if prevCtrl is EDGE1, then we follow EDGE2 rules

            int restartHighBit = firstbithigh(restartBitmask & mask);
            int otherEdgeHighBit = firstbithigh(~restartBitmask & ~backtrackBitmask & ((isEdge1Bitmask ^ ((backtrackBitmask >> 1) ^ edgeBitmask)) & mask));

            int prevRestartTriangle = restartHighBit == -1 ? (dwordIndex ? prevHighRestartBeforeDwords[dwordIndex - 1] : 0u) : restartHighBit + dwordIndex * 32;

            int3 prevHighEdge = isEdge1 ? prevHighEdge2BeforeDwords : prevHighEdge1BeforeDwords;
            int prevOtherEdgeTriangle = otherEdgeHighBit == -1 ? (dwordIndex ? prevHighEdge[dwordIndex - 1] : -1) : otherEdgeHighBit + dwordIndex * 32;

            int prevTriangle = max(prevRestartTriangle, prevOtherEdgeTriangle);
            uint isEdge1Restart = isEdge1 && (prevRestartTriangle == prevTriangle);

            uint increment = (prevOtherEdgeTriangle == prevTriangle || isEdge1Restart);
            indexAddress.x = 2 * r + prevTriangle - 2 + increment;

            indexAddress = isEdge1 ? indexAddress.yxz : indexAddress;
        }

        // Get the first bits mask
        uint2 baseAligned_bitOffset = GetAlignedAddressAndBitOffset(baseAddress + geoBaseAddress, firstBitOffset);
        uint alignedFirstBitsOffset = baseAligned_bitOffset[0];
        uint firstBitsOffset = baseAligned_bitOffset[1];

        uint4 firstBitMask[2];
        firstBitMask[0] = clusterPageData.Load4(alignedFirstBitsOffset);
        firstBitMask[1] = clusterPageData.Load4(alignedFirstBitsOffset + 16);

        firstBitMask[0][0] = BitAlignU32(firstBitMask[0][1], firstBitMask[0][0], firstBitsOffset);
        firstBitMask[0][1] = BitAlignU32(firstBitMask[0][2], firstBitMask[0][1], firstBitsOffset);
        firstBitMask[0][2] = BitAlignU32(firstBitMask[0][3], firstBitMask[0][2], firstBitsOffset);
        firstBitMask[0][3] = BitAlignU32(firstBitMask[1][0], firstBitMask[0][3], firstBitsOffset);

        firstBitMask[1][0] = BitAlignU32(firstBitMask[1][1], firstBitMask[1][0], firstBitsOffset);
        firstBitMask[1][1] = BitAlignU32(firstBitMask[1][2], firstBitMask[1][1], firstBitsOffset);
        firstBitMask[1][2] = BitAlignU32(firstBitMask[1][3], firstBitMask[1][2], firstBitsOffset);
        firstBitMask[1][3] >>= firstBitsOffset;

        uint4 numFirstBits[2];
        numFirstBits[0][0] = 0; 
        numFirstBits[0][1] = countbits(firstBitMask[0][0]);
        numFirstBits[0][2] = numFirstBits[0][1] + countbits(firstBitMask[0][1]);
        numFirstBits[0][3] = numFirstBits[0][2] + countbits(firstBitMask[0][2]);

        numFirstBits[1][0] = numFirstBits[0][3] + countbits(firstBitMask[0][3]); 
        numFirstBits[1][1] = numFirstBits[1][0] + countbits(firstBitMask[1][0]);  
        numFirstBits[1][2] = numFirstBits[1][1] + countbits(firstBitMask[1][1]);   
        numFirstBits[1][3] = numFirstBits[1][2] + countbits(firstBitMask[1][2]);

        uint3 vids;
        uint3 reuseIds = uint3(0, 0, 0);

        [[unroll]]
        for (int k = 0; k < 3; k++)
        {
            uint arrayIndex = indexAddress[k] >> 7u;
            uint dwordIndex = (indexAddress[k] >> 5u) & 0x3;
            uint bitIndex = indexAddress[k] & 31u;
            uint bit = 1u << bitIndex;
            uint mask = bit - 1u;
            uint vid = numFirstBits[arrayIndex][dwordIndex] + 
                        countbits(firstBitMask[arrayIndex][dwordIndex] & mask);

            if ((firstBitMask[arrayIndex][dwordIndex] & bit) == 0)
            {
                reuseIds[k] = indexAddress[k] - vid;
                vid = DecodeReuse(indexAddress[k] - vid);
            }
            vids[k] = vid;
        }

        return vids;
    }

    float3 DecodePosition(uint vertexIndex)
    {
        const uint bitsPerVertex = posBitWidths[0] + posBitWidths[1] + posBitWidths[2];
        const uint bitsOffset = vertexIndex * bitsPerVertex;

        uint2 vals = GetAlignedAddressAndBitOffset(baseAddress + geoBaseAddress, bitsOffset);
        uint3 data = clusterPageData.Load3(vals[0]);

        uint2 packed = uint2(BitAlignU32(data.y, data.x, vals[1]), 
                             BitAlignU32(data.z, data.y, vals[1]));

        int3 pos = int3(0, 0, 0);
        pos.x = BitFieldExtractU32(packed.x, posBitWidths.x, 0);
        packed.x = BitAlignU32(packed.y, packed.x, posBitWidths.x);

        packed.y >>= posBitWidths.x;
        pos.y = BitFieldExtractU32(packed.x, posBitWidths.y, 0);

        packed.x = BitAlignU32(packed.y, packed.x, posBitWidths.y);
        pos.z = BitFieldExtractU32(packed.x, posBitWidths.z, 0);
        
        const float scale = asfloat((127 - posPrecision) << 23);
        return (pos + anchor) * scale;
    }

    bool HasNormals()
    {
        const uint bitsPerNormal = octBitWidths[0] + octBitWidths[1];
        return (bitsPerNormal != 0);
    }

    float3 DecodeNormal(uint vertexIndex)
    {
        const uint bitsPerNormal = octBitWidths[0] + octBitWidths[1];

        const uint bitsOffset = bitsPerNormal * vertexIndex;

        uint2 vals = GetAlignedAddressAndBitOffset(baseAddress + shadBaseAddress + normalOffset, bitsOffset);
        uint2 data = clusterPageData.Load2(vals[0]);
        uint n = BitAlignU32(data.y, data.x, vals[1]);

        float nx = Dequantize<16>(BitFieldExtractU32(n, octBitWidths[0], 0) + octBase[0]) * 2 - 1;
        float ny = Dequantize<16>(BitFieldExtractU32(n, octBitWidths[1], octBitWidths[0]) + octBase[1]) * 2 - 1;

        return UnpackOctahedral(float2(nx, ny));
    }

    uint DecodeReuse(uint reuseIndex)
    {
        const uint bitsOffset = reuseIndex * indexBitWidth;
        uint2 vals = GetAlignedAddressAndBitOffset(baseAddress + geoBaseAddress + indexOffset, bitsOffset);
        uint2 result = clusterPageData.Load2(vals[0]);
        uint r = BitFieldExtractU32(BitAlignU32(result.y, result.x, vals[1]), indexBitWidth, 0);
        return r;
    }

    uint DecodeMaterialID(uint triangleIndex)
    {
        // Constant mode
        const uint constantBit = 0x80000000;
        if (materialInfo & constantBit)
        {
            return materialInfo & ~constantBit;
        }
        // Table mode
        else 
        {
            uint numHighOrderBits = BitFieldExtractU32(materialInfo, 5, 0) + 1;
            uint numEntries = BitFieldExtractU32(materialInfo, 5, 5);
            uint materialOffset = BitFieldExtractU32(materialInfo, 22, 10);

            uint numLowOrderBits = 32 - numHighOrderBits;

            uint entryBitWidth = firstbithigh(numEntries - 1) + 1;
            uint bitOffset = numHighOrderBits + numEntries * numLowOrderBits + triangleIndex * entryBitWidth;

            // First get the entry index
            uint2 offsets = GetAlignedAddressAndBitOffset(baseAddress + shadBaseAddress + materialOffset, bitOffset);
            uint2 result = clusterPageData.Load2(offsets[0]);
            uint entryIndex = BitFieldExtractU32(BitAlignU32(result.y, result.x, offsets[1]), entryBitWidth, 0);

            // Then get the entry
            // LSB
            bitOffset = numHighOrderBits + entryIndex * numLowOrderBits;
            offsets = GetAlignedAddressAndBitOffset(baseAddress + shadBaseAddress + materialOffset, bitOffset);
            result = clusterPageData.Load2(offsets[0]);
            uint entry = BitFieldExtractU32(BitAlignU32(result.y, result.x, offsets[1]), numLowOrderBits, 0);
            // MSB
            offsets = GetAlignedAddressAndBitOffset(baseAddress + shadBaseAddress + materialOffset, 0);
            result = clusterPageData.Load2(offsets[0]);
            uint msb = BitFieldExtractU32(BitAlignU32(result.y, result.x, offsets[1]), numHighOrderBits, 0);

            uint materialID = (msb << numLowOrderBits) | entry;

            return materialID;
        }
    }

    uint2 DecodeFaceIDAndRotateInfo(uint triangleIndex)
    {
        uint2 result = 0;
        if (numFaceIDBits)
        {
            uint2 offsets = GetAlignedAddressAndBitOffset(baseAddress + shadBaseAddress + faceIDOffset, 0);
            uint2 data = clusterPageData.Load2(offsets[0]);
            uint minFaceID = BitAlignU32(data.y, data.x, offsets[1]);
 
            offsets = GetAlignedAddressAndBitOffset(baseAddress + shadBaseAddress + faceIDOffset + 4, triangleIndex * (numFaceIDBits + 3u));
            data = clusterPageData.Load2(offsets[0]);
            uint packed = BitAlignU32(data.y, data.x, offsets[1]);
            uint faceIDDiff = BitFieldExtractU32(packed, numFaceIDBits, 0);
            uint rotateInfo = BitFieldExtractU32(packed, 3, numFaceIDBits);

            result.x = minFaceID + faceIDDiff;
            result.y = rotateInfo;
        }
        return result;
    }
};

// Taken from NaniteDataDecode.ush in Unreal Engine 5
DenseGeometry GetDenseGeometryHeader(uint4 packed[NUM_CLUSTER_HEADER_FLOAT4S], uint baseAddress, bool debug = false)
{
    DenseGeometry result;

    result.lodBounds.x = asfloat(packed[0].x);
    result.lodBounds.y = asfloat(packed[0].y);
    result.lodBounds.z = asfloat(packed[0].z);
    result.lodBounds.w = asfloat(packed[0].w);

    result.baseAddress = baseAddress;
    result.shadBaseAddress = packed[1].x;
    result.geoBaseAddress = packed[1].y;

    result.anchor[0] = BitFieldExtractI32((int)packed[1].z, ANCHOR_WIDTH, 0);
    result.numTriangles = BitFieldExtractU32(packed[1].z, 8, ANCHOR_WIDTH);

    result.anchor[1] = BitFieldExtractI32((int)packed[1].w, ANCHOR_WIDTH, 0);
    result.posBitWidths[0] = BitFieldExtractU32(packed[1].w, 5, ANCHOR_WIDTH);
    result.indexBitWidth = BitFieldExtractU32(packed[1].w, 3, ANCHOR_WIDTH + 5) + 1;

    result.anchor[2] = BitFieldExtractI32((int)packed[2].x, ANCHOR_WIDTH, 0);
    result.posPrecision = (int)BitFieldExtractU32(packed[2].x, 8, ANCHOR_WIDTH) + CLUSTER_MIN_PRECISION;
    
    result.numVertices     = BitFieldExtractU32(packed[2].y, 9, 0);
    uint reuseBufferLength = BitFieldExtractU32(packed[2].y, 8, 9);
    result.posBitWidths[1] = BitFieldExtractU32(packed[2].y, 5, 17);
    result.posBitWidths[2] = BitFieldExtractU32(packed[2].y, 5, 22);
    result.octBitWidths[0] = BitFieldExtractU32(packed[2].y, 5, 27);

    result.prevHighRestartBeforeDwords[1] = BitFieldExtractU32(packed[2].z, 6, 0);
    result.prevHighRestartBeforeDwords[2] = BitFieldExtractU32(packed[2].z, 7, 6);
    result.prevHighEdge1BeforeDwords[0] = BitFieldExtractI32(packed[2].z, 6, 13);
    result.prevHighEdge1BeforeDwords[1] = BitFieldExtractI32(packed[2].z, 7, 19);
    result.prevHighEdge2BeforeDwords[0] = BitFieldExtractI32(packed[2].z, 6, 26);

    result.prevHighRestartBeforeDwords[0] = BitFieldExtractU32(packed[2].w, 5, 0);
    result.prevHighEdge1BeforeDwords[2] = BitFieldExtractI32(packed[2].w, 8, 5);
    result.prevHighEdge2BeforeDwords[1] = BitFieldExtractI32(packed[2].w, 7, 13);
    result.prevHighEdge2BeforeDwords[2] = BitFieldExtractI32(packed[2].w, 8, 20);

    result.octBitWidths[1]                = BitFieldExtractU32(packed[3].x, 5, 0);
    result.numPrevRestartsBeforeDwords[0] = BitFieldExtractU32(packed[3].x, 6, 5);
    result.numPrevRestartsBeforeDwords[1] = BitFieldExtractU32(packed[3].x, 7, 11);
    result.numPrevRestartsBeforeDwords[2] = BitFieldExtractU32(packed[3].x, 8, 18);
    result.numFaceIDBits = BitFieldExtractU32(packed[3].x, 6, 26); 

    result.octBase[0] = BitFieldExtractU32(packed[3].y, 16, 0);
    result.octBase[1] = BitFieldExtractU32(packed[3].y, 16, 16);

    result.materialInfo = packed[3].z;

    result.lodError = asfloat(packed[3].w);

    result.flags = packed[4].x;

    // Size of vertex buffer and normal buffer
    const uint vertexBitWidth = result.posBitWidths[0] + result.posBitWidths[1] + result.posBitWidths[2];
    const uint octBitWidth = result.octBitWidths[0] + result.octBitWidths[1];

    result.normalOffset = 0;
    result.faceIDOffset = result.normalOffset + ((result.numVertices * octBitWidth + 7) >> 3);

    result.ctrlOffset = (result.numVertices * vertexBitWidth + 7) >> 3;
    result.indexOffset = result.ctrlOffset + 12 * ((result.numTriangles + 31u) >> 5u);
    result.firstBitOffset = (result.indexOffset << 3) + reuseBufferLength * result.indexBitWidth;

    result.debug = debug;

    return result;
}

DenseGeometry GetDenseGeometryHeader(uint basePageAddress, uint numClusters, uint clusterIndex)
{
    uint clusterHeaderSOAStride = numClusters * 16;
    uint baseOffset = basePageAddress + 4 + clusterIndex * 16;

    uint4 packedHeaderValues[NUM_CLUSTER_HEADER_FLOAT4S];
    for (int i = 0; i < NUM_CLUSTER_HEADER_FLOAT4S; i++)
    {
        packedHeaderValues[i] = clusterPageData.Load4(baseOffset + i * clusterHeaderSOAStride);
    }
    return GetDenseGeometryHeader(packedHeaderValues, basePageAddress);
}

uint GetClusterPageBaseAddress(uint pageIndex) 
{
    return pageIndex * CLUSTER_PAGE_SIZE;
}

uint GetNumClustersInPage(uint baseAddress)
{
    return clusterPageData.Load(baseAddress);
}

uint GetPageIndexFromClusterID(uint clusterID)
{
    return clusterID >> MAX_CLUSTERS_PER_PAGE_BITS;
}

uint GetClusterIndexFromClusterID(uint clusterID)
{
    return clusterID & (MAX_CLUSTERS_PER_PAGE - 1);
}

void GetBrickBounds(uint64_t bitMask, out uint3 minP, out uint3 maxP)
{
    minP.z = firstbitlow(bitMask) >> 4u;
    maxP.z = (firstbithigh(bitMask) >> 4u) + 1u;

    uint bits = (uint)bitMask | uint(bitMask >> 32u);
    bits |= bits >> 16u;
    minP.y = firstbitlow(bitMask) >> 2u;
    maxP.y = (firstbithigh((bitMask << 16u) >> 16u) >> 2u) + 1u;

    bits |= bits >> 8u;
    bits |= bits >> 4u;
    minP.x = firstbitlow(bitMask);
    maxP.x = (firstbithigh((bitMask << 28u) >> 28u)) + 1u;
}

#endif
