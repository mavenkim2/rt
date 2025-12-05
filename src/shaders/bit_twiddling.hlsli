#ifndef BIT_TWIDDLING_HLSLI_
#define BIT_TWIDDLING_HLSLI_

#include "../rt/shader_interop/bit_twiddling_shaderinterop.h"

uint BitAlignU32(uint high, uint low, uint shift)
{
	shift &= 31u;

	uint result = low >> shift;
	result |= shift > 0u ? (high << (32u - shift)) : 0u;
	return result;
}

#ifdef __SLANG__
__generic<uint pow2>
#else
template <uint pow2>
#endif
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

#define CAT(a, b) a##b
#define CONCAT(a, b) CAT(a, b)

#ifdef __SLANG__
#define TEMPLATE_UINT(name) __generic<uint name>
#else
#define TEMPLATE_UINT(name) template<uint name>
#endif

#define DefineBitStreamReader(name, inputBuffer) \
struct CONCAT(BitStreamReader_, name) \
{ \
    uint4 buffer; \
\
    uint alignedByteAddress; \
    uint bitOffsetFromAddress; \
    int bufferOffset; \
 \
    uint compileTimeMinBufferBits; \
    uint compileTimeMinDwordBits; \
    uint compileTimeMaxRemainingBits; \
 \
    TEMPLATE_UINT(compileTimeMaxBits) \
    uint Read(uint numBits) \
    { \
        if (compileTimeMaxBits > compileTimeMinBufferBits) \
        { \
        	bitOffsetFromAddress += bufferOffset;	 \
        	uint address = alignedByteAddress + ((bitOffsetFromAddress >> 5) << 2); \
         \
        	uint4 data = inputBuffer.Load4(address); \
         \
        	buffer.x = BitAlignU32(data.y, data.x, bitOffsetFromAddress); \
        	if (compileTimeMaxRemainingBits > 32) buffer.y = BitAlignU32(data.z, data.y, bitOffsetFromAddress); \
        	if (compileTimeMaxRemainingBits > 64) buffer.z = BitAlignU32(data.w, data.z, bitOffsetFromAddress); \
        	if (compileTimeMaxRemainingBits > 96) buffer.w = BitAlignU32(0, data.w,	bitOffsetFromAddress);  \
         \
        	bufferOffset = 0; \
         \
        	compileTimeMinDwordBits	= min(32, compileTimeMaxRemainingBits); \
        	compileTimeMinBufferBits = min(97, compileTimeMaxRemainingBits); \
        } \
        else if (compileTimeMaxBits > compileTimeMinDwordBits) \
        { \
        	bitOffsetFromAddress += bufferOffset; \
         \
        	const bool bOffset32 = compileTimeMinDwordBits == 0 && bufferOffset == 32; \
        	buffer.x = bOffset32 ? buffer.y : BitAlignU32(buffer.y, buffer.x, bufferOffset); \
        	if (compileTimeMinBufferBits > 32) buffer.y	= bOffset32 ? buffer.z : BitAlignU32(buffer.z, buffer.y, bufferOffset); \
        	if (compileTimeMinBufferBits > 64) buffer.z	= bOffset32 ? buffer.w : BitAlignU32(buffer.w, buffer.z, bufferOffset); \
        	if (compileTimeMinBufferBits > 96) buffer.w	= bOffset32 ? 0u : BitAlignU32(0, buffer.w, bufferOffset); \
         \
        	bufferOffset = 0; \
         \
        	compileTimeMinDwordBits = min(32, compileTimeMaxRemainingBits); \
        } \
         \
        const uint result = BitFieldExtractU32(buffer.x, numBits, bufferOffset); \
         \
        bufferOffset += numBits; \
        compileTimeMinBufferBits    -= compileTimeMaxBits; \
        compileTimeMinDwordBits     -= compileTimeMaxBits; \
        compileTimeMaxRemainingBits -= compileTimeMaxBits; \
         \
        return result; \
    } \
}; \
 \
CONCAT(BitStreamReader_, name) CONCAT(CreateBitStreamReader_, name)(uint alignedStart, uint bitOffset, uint compileTimeMaxRemainingBits) \
{ \
    CONCAT(BitStreamReader_, name) result; \
 \
    result.buffer = 0; \
    result.alignedByteAddress = alignedStart; \
    result.bitOffsetFromAddress = bitOffset; \
    result.bufferOffset = 0; \
    result.compileTimeMinBufferBits = 0; \
    result.compileTimeMinDwordBits = 0; \
    result.compileTimeMaxRemainingBits = compileTimeMaxRemainingBits; \
 \
    return result; \
}
#endif
