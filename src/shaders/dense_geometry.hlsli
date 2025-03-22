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
    uint newBitOffset = ((byteAligned - aligned) << 3) + (bitOffset & 0x7);

    return uint2(aligned, newBitOffset);
}

struct DenseGeometry 
{
    uint baseAddress;

    int3 anchor;
    uint3 posBitWidths;
    uint indexBitWidth;
    uint numTriangles;
    uint numVertices;

    uint indexOffset;
    uint ctrlBitOffset;
    uint firstBitsOffset;
    int posPrecision;

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
            numTriangles, numVertices, indexOffset, ctrlBitOffset, firstBitsOffset, posPrecision);
    }

    void Print(uint2 cursor, uint blockIndex, uint triIndex, float3 p[3], uint3 indexAddress, uint3 vids, uint3 reuseIds, AABB aabb) 
    {
        printf("cursor: %u %u\nanchor: %i %i %i\nbit widths: %u %u %u %u\nnum tri: %u\nnum vert: %u, offsets: index %u ctrl %u first %u\nprecision: %u\nblockindex: %u triIndex: %u\n, p: %f %f %f %f %f %f %f %f %f\nindex address: %u %u %u\nvids: %u %u %u\n, reuse: %u %u %u\naabb: %f %f %f %f %f %f\n", 
            cursor[0], cursor[1], anchor[0], anchor[1], anchor[2], posBitWidths[0], posBitWidths[1], posBitWidths[2], indexBitWidth,
            numTriangles, numVertices, indexOffset, ctrlBitOffset, firstBitsOffset, posPrecision, 
            blockIndex, triIndex, 
            p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2],p[2][0], p[2][1], p[2][2], 
            indexAddress[0], indexAddress[1], indexAddress[2], vids[0], vids[1], vids[2], 
            reuseIds[0], reuseIds[1], reuseIds[2], aabb.minX, aabb.minY, aabb.minZ, aabb.maxX, aabb.maxY, aabb.maxZ);
    }
};

// Taken from NaniteDataDecode.ush in Unreal Engine 5
DenseGeometry GetDenseGeometryHeader(PackedDenseGeometryHeader packed)
{
    DenseGeometry result;

    uint offset = 0;
    result.baseAddress = packed.a;

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

    result.indexOffset = result.numVertices * (result.posBitWidths[0] + result.posBitWidths[1] + result.posBitWidths[2]);
    result.indexOffset = (result.indexOffset + 7) >> 3;
    result.ctrlBitOffset = (result.indexOffset << 3) + reuseBufferLength * result.indexBitWidth;
    result.firstBitsOffset = result.ctrlBitOffset + 2 * (result.numTriangles - 1);
    return result;
}
