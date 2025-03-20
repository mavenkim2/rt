#include "../rt/shader_interop/dense_geometry_shaderinterop.h"
ByteAddressBuffer denseGeometryData : register(t5);
StructuredBuffer<uint4> denseGeometryHeaders : register(t6); 

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

struct DenseGeometry 
{
    uint baseAddress;

    int3 anchor;
    uint3 posBitWidths;
    uint indexBitWidth;
    uint numTriangles;

    uint indexOffset;
    uint ctrlBitOffset;
    uint firstBitsOffset;
    int posPrecision;

    float3 DecodePosition(uint vertexIndex)
    {
        const uint bitsPerVertex = posBitWidths[0] + posBitWidths[1] + posBitWidths[2];
        const uint bitsOffset = vertexIndex * bitsPerVertex;
        uint3 data = denseGeometryData.Load3(baseAddress + ((bitsOffset >> 5) << 2));

        uint2 packed = uint2(BitAlignU32(data.y, data.x, bitsOffset),
                             BitAlignU32(data.z, data.y, bitsOffset));

        int3 pos;
        pos.x = BitFieldExtractU32(packed.x, posBitWidths.x, 0);
        packed.x = BitAlignU32(packed.y, packed.x, posBitWidths.x);

        packed.y >>= posBitWidths.x;
        pos.y = BitFieldExtractU32(packed.x, posBitWidths.y, 0);

        packed.x = BitAlignU32(packed.y, packed.x, posBitWidths.y);
        pos.z = BitFieldExtractU32(packed.x, posBitWidths.z, 0);
        
        const float scale = asfloat(asint(1.0) - (posPrecision << 23));
        return (pos + anchor) * scale;
    }

    uint DecodeReuse(uint reuseIndex)
    {
        const uint bitsOffset = reuseIndex * indexBitWidth;
        uint offset = indexOffset + ((bitsOffset >> 5) << 2);
        uint2 result = denseGeometryData.Load2(baseAddress + offset);
        return BitAlignU32(result.y, result.x, offset);
    }
};

// Taken from NaniteDataDecode.ush in Unreal Engine 5
DenseGeometry GetDenseGeometryHeader(uint4 packed)
{
    DenseGeometry result;

    uint offset = 0;
    result.anchor[0] = BitFieldExtractI32((int)packed.x, ANCHOR_WIDTH, 0);
    offset += ANCHOR_WIDTH;
    result.anchor[1] = BitFieldExtractI32((int)BitAlignU32(packed.y, packed.x, offset), ANCHOR_WIDTH, 0);
    offset += ANCHOR_WIDTH;
    result.anchor[2] = BitFieldExtractI32((int)BitAlignU32(packed.z, packed.y, offset), ANCHOR_WIDTH, 0);
    offset += ANCHOR_WIDTH;
    
    result.indexOffset = BitFieldExtractAndAlignU32(packed.zw, 8, offset);
    result.posBitWidths[0] = BitFieldExtractAndAlignU32(packed.zw, 5, 0);
    result.posBitWidths[1] = BitFieldExtractAndAlignU32(packed.zw, 5, 0);
    result.posBitWidths[2] = BitFieldExtractAndAlignU32(packed.zw, 5, 0);
    result.indexBitWidth = BitFieldExtractAndAlignU32(packed.zw, 8, 0);

    //result.numTriangles = ?;
    //result.numVertices = ?;
    uint numTriangles;
    result.ctrlBitOffset = BitFieldExtractAndAlignU32(packed.zw, 10, 0);
    result.firstBitsOffset = result.ctrlBitOffset + 2 * numTriangles;

    return result;
}
