StructuredBuffer<PathGuidingSample> samples : register(t0);

StructuredBuffer<VMM> vmms : register(t0);
StructuredBuffer<uint> vmmOffsets : register(t1);
StructuredBuffer<uint> vmmCounts : register(t2);
StructuredBuffer<Statistics> vmmStatistics : register(u3);

[numthreads(32, 1, 1)]
void main(uint3 dtID: SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
}
