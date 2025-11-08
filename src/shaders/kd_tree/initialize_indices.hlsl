RWStructuredBuffer<uint> indices : register(u0);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint index = dtID.x;
    indices[index] = index;
}
