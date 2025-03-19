ByteAddressBuffer denseGeometryData : register(t5);
static const uint denseGeometryHeaderSize = ?;

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

uint BitFieldExtractU32(uint data, uint size)
{
	size &= 31;
	return data & ((1u << size) - 1u);
}

struct DenseGeometry 
{
    int3 anchor;
    uint3 posBitWidths;
    uint indexBitWidth;

    uint baseAddress;
    uint indexOffset;
    int posPrecision;

    // Taken from NaniteDataDecode.ush in Unreal Engine 5
    float3 DecodePosition(uint vertexIndex)
    {
        const uint bitsPerVertex = posBitWidths[0] + posBitWidths[1] + posBitWidths[2];
        const uint bitsOffset = vertexIndex * bitsPerVertex;
        uint3 data = denseGeometryData.Load3(baseAddress + denseGeometryHeaderSize + (bitsOffset >> 5) << 2);

        uint2 packed = uint2(BitAlignU32(data.y, data.x, bitsOffset),
                             BitAlignU32(data.z, data.y, bitsOffset));

        int3 pos;
        pos.x = BitFieldExtractU32(packed.x, posBitWidths.x);
        packed.x = BitAlignU32(packed.y, packed.x, posBitWidths.x);

        packed.y >>= posBitWidths.x;
        pos.y = BitFieldExtractU32(packed.x, posBitWidths.y);

        packed.x = BitAlignU32(packed.y, packed.x, posBitWidths.y);
        pos.z = BitFieldExtractU32(packed.x, posBitWidths.z);
        
        const float scale = asfloat(asint(1.0) - (posPrecision << 23));
        return (pos + anchor) * scale;
    }
    uint DecodeReuse(uint reuseIndex)
    {
        const uint bitsOffset = reuseIndex * indexBitWidth;
        uint offset = ((indexOffset + bitsOffset >> 5) << 2);
        uint2 result = denseGeometryData.Load2(baseAddress + offset);
        return BitAlignU32(result.y, result.x, offset);
    }
};

