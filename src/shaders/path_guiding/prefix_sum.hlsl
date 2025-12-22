StructuredBuffer<uint> sampleVMMIndices : register(t0);
RWStructuredBuffer<uint> vmmOffsets : register(u1);

struct Num 
{
    uint num;
};
[[vk::push_constant]] Num num;

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    for (uint i = 0; i < num.num; i++)
    {
        uint vmmIndex = sampleVMMIndices[i];
        vmmOffsets[vmmIndex]++;
    }

    uint prefixSum = 0;
    // TODO:
    uint numVMMs = 32;
    for (uint i = 0; i < numVMMs; i++)
    {
        uint value = vmmOffsets[i];
        vmmOffsets[i] = prefixSum;
        prefixSum += value;
    }
}
