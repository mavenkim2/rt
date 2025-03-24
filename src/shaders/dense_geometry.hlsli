#include "../rt/shader_interop/dense_geometry_shaderinterop.h"
ByteAddressBuffer denseGeometryData : register(t5);
StructuredBuffer<PackedDenseGeometryHeader> denseGeometryHeaders : register(t6); 

// Taken from Platform.ush in Unreal Engine 5
uint BitAlignU32(uint high, uint low, uint shift)
{
	shift &= 31u;

	uint result = low >> shift;
	result |= shift > 0u ? (high << (32u - shift)) : 0u;
	return result;
}

uint BitFieldExtractU32(uint data, uint size, uint offset)
{
	size &= 31;
	offset &= 31;
	return (data >> offset) & ((1u << size) - 1u);
}

int BitFieldExtractI32(int data, uint size, uint offset)
{
	size &= 31u;
	offset &= 31u;
	const uint shift = (32u - size) & 31u;
	const int value = (data >> offset) & int((1u << size) - 1u);
	return (value << shift) >> shift;
}

uint BitFieldMaskU32(uint maskWidth, uint maskLocation)
{
	maskWidth &= 31u;
	maskLocation &= 31u;

	return ((1u << maskWidth) - 1u) << maskLocation;
}

uint BitFieldExtractAndAlignU32(inout uint2 data, uint size, uint offset)
{
    uint result = BitFieldExtractU32(data[0], size, offset);
    data[0] = BitAlignU32(data[1], data[0], offset + size);
    data[1] >>= (offset + size) & 31u;
    return result;
}

template <uint pow2>
uint AlignDownPow2(uint val)
{
    return val & ~(pow2 - 1);
}

uint2 GetAlignedAddressAndBitOffset(uint baseAddress, uint bitOffset)
{
    uint byteAligned = baseAddress + (bitOffset >> 3);
    uint aligned = AlignDownPow2<4>(byteAligned);
    uint newBitOffset = ((byteAligned & 3u) << 3) | (bitOffset & 0x7);

    return uint2(aligned, newBitOffset);
}

struct BitStreamReader 
{
    ByteAddressBuffer inputBuffer;
    uint4 buffer;

    uint alignedByteAddress;
    uint bitOffsetFromAddress;
    int bufferOffset;

    uint compileTimeMinBufferBits;
    uint compileTimeMinDwordBits;
    uint compileTimeMaxRemainingBits;

    template <uint compileTimeMaxBits>
    uint Read(uint numBits)
    {
        if (compileTimeMaxBits > compileTimeMinBufferBits)
        {
        	bitOffsetFromAddress += bufferOffset;	
        	uint address = alignedByteAddress + ((bitOffsetFromAddress >> 5) << 2);
        
        	uint4 data = inputBuffer.Load4(address);
        
        	// Shift bits down to align
        	buffer.x = BitAlignU32(data.y, data.x, bitOffsetFromAddress);
        	if (compileTimeMaxRemainingBits > 32) buffer.y = BitAlignU32(data.z, data.y, bitOffsetFromAddress);
        	if (compileTimeMaxRemainingBits > 64) buffer.z = BitAlignU32(data.w, data.z, bitOffsetFromAddress);
        	if (compileTimeMaxRemainingBits > 96) buffer.w = BitAlignU32(0, data.w,	bitOffsetFromAddress); 
        
        	bufferOffset = 0;
        
        	compileTimeMinDwordBits	= min(32, compileTimeMaxRemainingBits);
        	compileTimeMinBufferBits = min(97, compileTimeMaxRemainingBits);
        }
        else if (compileTimeMaxBits > compileTimeMinDwordBits)
        {
        	bitOffsetFromAddress += bufferOffset;
        
        	const bool bOffset32 = compileTimeMinDwordBits == 0 && bufferOffset == 32;
        	buffer.x = bOffset32 ? buffer.y : BitAlignU32(buffer.y, buffer.x, bufferOffset);
        	if (compileTimeMinBufferBits > 32) buffer.y	= bOffset32 ? buffer.z : BitAlignU32(buffer.z, buffer.y, bufferOffset);
        	if (compileTimeMinBufferBits > 64) buffer.z	= bOffset32 ? buffer.w : BitAlignU32(buffer.w, buffer.z, bufferOffset);
        	if (compileTimeMinBufferBits > 96) buffer.w	= bOffset32 ? 0u : BitAlignU32(0, buffer.w, bufferOffset);
        
        	bufferOffset = 0;
        
        	compileTimeMinDwordBits = min(32, compileTimeMaxRemainingBits);
        }
        
        const uint result = BitFieldExtractU32(buffer.x, numBits, bufferOffset);
        
        bufferOffset += numBits;
        compileTimeMinBufferBits    -= compileTimeMaxBits;
        compileTimeMinDwordBits     -= compileTimeMaxBits;
        compileTimeMaxRemainingBits -= compileTimeMaxBits;
        
        return result;
    }
};

BitStreamReader CreateBitStreamReader(ByteAddressBuffer inputBuffer, uint alignedStart, uint bitOffset, uint compileTimeMaxRemainingBits)
{
    BitStreamReader result;

    result.inputBuffer = inputBuffer;
    result.buffer = 0;
    result.alignedByteAddress = alignedStart;
    result.bitOffsetFromAddress = bitOffset;
    result.bufferOffset = 0;
    result.compileTimeMinBufferBits = 0;
    result.compileTimeMinDwordBits = 0;
    result.compileTimeMaxRemainingBits = compileTimeMaxRemainingBits;

    return result;
}

struct DenseGeometry 
{
    uint baseAddress;

    uint blockIndex;
    uint triangleIndex;

    int3 anchor;
    uint2 octBase;

    uint3 posBitWidths;
    uint2 octBitWidths;
    uint indexBitWidth;

    uint numTriangles;
    uint numVertices;

    uint normalOffset;
    uint indexOffset;
    uint ctrlOffset;
    uint firstBitOffset;
    int posPrecision;

    uint3 numPrevRestartsBeforeDwords;
    uint3 prevHighRestartBeforeDwords;
    int3 prevHighEdge1BeforeDwords;
    int3 prevHighEdge2BeforeDwords;

    uint3 DecodeTriangle(out float3 p[3], bool printDebug = false)
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
        uint2 vals = GetAlignedAddressAndBitOffset(baseAddress + ctrlOffset + dwordIndex * 12, 0);
        uint4 stripBitmasks = denseGeometryData.Load4(vals[0]);
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

            if (printDebug)
            {
                printf("block/tri index: %u %u\nnum restarts: %u\nis: %u %u %u\nprev triangle: %u %u %u\nrestart hb: %u\nother hb: %u\nnum prev restarts: %u %u %u\n", blockIndex, triangleIndex, numRestarts, isRestart, isEdge1, isBacktrack, prevRestartTriangle, prevOtherEdgeTriangle, prevTriangle, restartHighBit, otherEdgeHighBit, numPrevRestartsBeforeDwords[0], numPrevRestartsBeforeDwords[1], numPrevRestartsBeforeDwords[2]);
                printf("bitmask: %u %u %u, vals: %u %u\n", restartBitmask, edgeBitmask, backtrackBitmask, vals[0], vals[1]);
            }
        }
#if 0
        uint r = 1;
        int bt = 0;
        uint prevCtrl = Restart;

        uint2 ctrlBaseAligned_bitOffset = GetAlignedAddressAndBitOffset(baseAddress, ctrlBitOffset);
        uint alignedStart = ctrlBaseAligned_bitOffset[0];
        uint bitOffset = ctrlBaseAligned_bitOffset[1];

        BitStreamReader reader = CreateBitStreamReader(denseGeometryData, alignedStart, 
                                                       bitOffset,
                                                       MAX_CLUSTER_TRIANGLES * 2);
        for (int k = 1; k <= triangleIndex; k++)
        {
            uint ctrl = reader.Read<2>(2);
            int3 prev = indexAddress;
            switch (ctrl)
            {
                case Restart:
                {
                    r++;
                    indexAddress = uint3(2 * r + k - 2, 2 * r + k - 1, 2 * r + k);
                }
                break;
                case Edge1: 
                {
                    indexAddress = uint3(prev[2], prev[1], 2 * r + k);
                    bt = prev[0];
                }
                break;
                case Edge2: 
                {
                    indexAddress = uint3(prev[0], prev[2], 2 * r + k);
                    bt = prev[1];
                }
                break;
                case Backtrack: 
                {
                    indexAddress = prevCtrl == Edge1 ? uint3(bt, prev[0], 2 * r + k) 
                                                     : uint3(prev[1], bt, 2 * r + k);
                }
                break;
            }
            prevCtrl = ctrl;
        }
#endif

        // Get the first bits mask
        uint2 baseAligned_bitOffset = GetAlignedAddressAndBitOffset(baseAddress, firstBitOffset);
        uint alignedFirstBitsOffset = baseAligned_bitOffset[0];
        uint firstBitsOffset = baseAligned_bitOffset[1];

        uint4 firstBitMask[2];
        firstBitMask[0] = denseGeometryData.Load4(alignedFirstBitsOffset);
        firstBitMask[1] = denseGeometryData.Load4(alignedFirstBitsOffset + 16);

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
            p[k] = DecodePosition(vid);
        }
#if 1
        if (printDebug)
        {
            float3 n0 = DecodeNormal(vids[0]);
            float3 n1 = DecodeNormal(vids[1]);
            float3 n2 = DecodeNormal(vids[2]);

            printf("anchor: %i %i %i\nbit widths: %u %u %u %u\nnum tri: %u\nnum vert: %u\noffsets: index %u ctrl %u first %u\nprecision: %u\nblockindex: %u triIndex: %u\n, p: %f %f %f %f %f %f %f %f %f\nindex address: %u %u %u\nvids: %u %u %u\n, reuse: %u %u %u\n, octbase: %u %u, bitwidths: %u %u\nn: %f %f %f %f %f %f %f %f %f\n",
                anchor[0], anchor[1], anchor[2], posBitWidths[0], posBitWidths[1], posBitWidths[2], indexBitWidth,
                numTriangles, numVertices, indexOffset, ctrlOffset, firstBitOffset, posPrecision, 
                blockIndex, triangleIndex, 
                p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2],p[2][0], p[2][1], p[2][2], 
                indexAddress[0], indexAddress[1], indexAddress[2], vids[0], vids[1], vids[2], 
                reuseIds[0], reuseIds[1], reuseIds[2], octBase[0], octBase[1], octBitWidths[0], octBitWidths[1], n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, 
                n2.x, n2.y, n2.z);
        }
#endif

        return vids;
    }

    float3 DecodePosition(uint vertexIndex)
    {
        const uint bitsPerVertex = posBitWidths[0] + posBitWidths[1] + posBitWidths[2];
        const uint bitsOffset = vertexIndex * bitsPerVertex;

        uint2 vals = GetAlignedAddressAndBitOffset(baseAddress, bitsOffset);
        uint3 data = denseGeometryData.Load3(vals[0]);

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

    float3 DecodeNormal(uint vertexIndex)
    {
        const uint bitsPerNormal = octBitWidths[0] + octBitWidths[1];
        const uint bitsOffset = bitsPerNormal * vertexIndex;

        uint2 vals = GetAlignedAddressAndBitOffset(baseAddress + normalOffset, bitsOffset);
        uint2 data = denseGeometryData.Load2(vals[0]);
        uint n = BitAlignU32(data.y, data.x, vals[1]);

        float nx = Dequantize<16>(BitFieldExtractU32(n, octBitWidths[0], 0) + octBase[0]);
        float ny = Dequantize<16>(BitFieldExtractU32(n, octBitWidths[1], octBitWidths[0]) + octBase[1]);

        return UnpackOctahedral(float2(nx, ny));
    }

    uint DecodeReuse(uint reuseIndex)
    {
        const uint bitsOffset = reuseIndex * indexBitWidth;
        uint2 vals = GetAlignedAddressAndBitOffset(baseAddress + indexOffset, bitsOffset);
        uint2 result = denseGeometryData.Load2(vals[0]);
        uint r = BitFieldExtractU32(BitAlignU32(result.y, result.x, vals[1]), indexBitWidth, 0);
        return r;
    }

    void Print() 
    {
        printf("anchor: %i %i %i\nbit widths: %u %u %u %u\nnum tri: %u\nnum vert: %u, offsets: index %u ctrl %u first %u\nprecision: %u\n", 
            anchor[0], anchor[1], anchor[2], posBitWidths[0], posBitWidths[1], posBitWidths[2], indexBitWidth,
            numTriangles, numVertices, indexOffset, ctrlOffset, firstBitOffset, posPrecision);
    }

    void Print(uint2 cursor, float3 p[3], uint3 indexAddress, uint3 vids, uint3 reuseIds, AABB aabb) 
    {
        printf("cursor: %u %u\nanchor: %i %i %i\nbit widths: %u %u %u %u\nnum tri: %u\nnum vert: %u, offsets: index %u ctrl %u first %u\nprecision: %u\nblockindex: %u triIndex: %u\n, p: %f %f %f %f %f %f %f %f %f\nindex address: %u %u %u\nvids: %u %u %u\n, reuse: %u %u %u\naabb: %f %f %f %f %f %f\n", 
            cursor[0], cursor[1], anchor[0], anchor[1], anchor[2], posBitWidths[0], posBitWidths[1], posBitWidths[2], indexBitWidth,
            numTriangles, numVertices, indexOffset, ctrlOffset, firstBitOffset, posPrecision, 
            blockIndex, triangleIndex, 
            p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2],p[2][0], p[2][1], p[2][2], 
            indexAddress[0], indexAddress[1], indexAddress[2], vids[0], vids[1], vids[2], 
            reuseIds[0], reuseIds[1], reuseIds[2], aabb.minX, aabb.minY, aabb.minZ, aabb.maxX, aabb.maxY, aabb.maxZ);
    }
};

// Taken from NaniteDataDecode.ush in Unreal Engine 5
DenseGeometry GetDenseGeometryHeader(uint primitiveIndex)
{
    uint blockIndex = primitiveIndex >> MAX_CLUSTER_TRIANGLES_BIT;
    uint triangleIndex = primitiveIndex & (MAX_CLUSTER_TRIANGLES - 1);

    PackedDenseGeometryHeader packed = denseGeometryHeaders[blockIndex];

    DenseGeometry result;

    uint offset = 0;
    result.baseAddress = packed.a;

    result.blockIndex = blockIndex;
    result.triangleIndex = triangleIndex;

    result.anchor[0] = BitFieldExtractI32((int)packed.b, ANCHOR_WIDTH, 0);
    result.numTriangles = BitFieldExtractU32(packed.b, 8, ANCHOR_WIDTH);

    result.anchor[1] = BitFieldExtractI32((int)packed.c, ANCHOR_WIDTH, 0);
    result.posBitWidths[0] = BitFieldExtractU32(packed.c, 5, ANCHOR_WIDTH);
    result.indexBitWidth = BitFieldExtractU32(packed.c, 3, ANCHOR_WIDTH + 5) + 1;

    result.anchor[2] = BitFieldExtractI32((int)packed.d, ANCHOR_WIDTH, 0);
    result.posPrecision = (int)BitFieldExtractU32(packed.d, 8, ANCHOR_WIDTH) + CLUSTER_MIN_PRECISION;
    
    result.numVertices     = BitFieldExtractU32(packed.e, 9, 0);
    uint reuseBufferLength = BitFieldExtractU32(packed.e, 8, 9);
    result.posBitWidths[1] = BitFieldExtractU32(packed.e, 5, 17);
    result.posBitWidths[2] = BitFieldExtractU32(packed.e, 5, 22);
    result.octBitWidths[0] = BitFieldExtractU32(packed.e, 5, 27);

    result.prevHighRestartBeforeDwords[1] = BitFieldExtractU32(packed.f, 6, 0);
    result.prevHighRestartBeforeDwords[2] = BitFieldExtractU32(packed.f, 7, 6);
    result.prevHighEdge1BeforeDwords[0] = BitFieldExtractI32(packed.f, 6, 13);
    result.prevHighEdge1BeforeDwords[1] = BitFieldExtractI32(packed.f, 7, 19);
    result.prevHighEdge2BeforeDwords[0] = BitFieldExtractI32(packed.f, 6, 26);

    result.prevHighRestartBeforeDwords[0] = BitFieldExtractU32(packed.g, 5, 0);
    result.prevHighEdge1BeforeDwords[2] = BitFieldExtractI32(packed.g, 8, 5);
    result.prevHighEdge2BeforeDwords[1] = BitFieldExtractI32(packed.g, 7, 13);
    result.prevHighEdge2BeforeDwords[2] = BitFieldExtractI32(packed.g, 8, 20);

    result.octBitWidths[1]                = BitFieldExtractU32(packed.h, 5, 0);
    result.numPrevRestartsBeforeDwords[0] = BitFieldExtractU32(packed.h, 6, 5);
    result.numPrevRestartsBeforeDwords[1] = BitFieldExtractU32(packed.h, 7, 11);
    result.numPrevRestartsBeforeDwords[2] = BitFieldExtractU32(packed.h, 8, 18);

    result.octBase[0] = BitFieldExtractU32(packed.i, 16, 0);
    result.octBase[1] = BitFieldExtractU32(packed.i, 16, 16);

    // Size of vertex buffer and normal buffer
    const uint vertexBitWidth = result.posBitWidths[0] + result.posBitWidths[1] + result.posBitWidths[2];
    const uint octBitWidth = result.octBitWidths[0] + result.octBitWidths[1];

    result.normalOffset = (result.numVertices * vertexBitWidth + 7) >> 3;
    result.ctrlOffset = result.normalOffset + ((result.numVertices * octBitWidth + 7) >> 3);
    result.indexOffset = result.ctrlOffset + 12 * ((result.numTriangles + 31u) >> 5u);
    result.firstBitOffset = (result.indexOffset << 3) + reuseBufferLength * result.indexBitWidth;

    return result;
}
