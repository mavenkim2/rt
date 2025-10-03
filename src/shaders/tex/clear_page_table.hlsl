RWTexture2D<uint> pageTable : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint width, height, levels;
    pageTable.GetDimensions(width, height);
    if (any(dtID.xy >= uint2(width, height))) return;

    pageTable[uint2(width, height)] = ~0u;
}
