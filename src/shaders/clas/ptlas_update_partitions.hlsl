RWStructuredBuffer<int> instanceIDFreeList : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x != 0) return;
    const int maxInstances = 1u << 21u;
    const int numPartitions = 16;
    const int numInstancesPerPartition = maxInstances / numPartitions;
    for (int partition = 0; partition < numPartitions; partition++)
    {
        uint freeListCountIndex = partition * (numInstancesPerPartition + 1u);
        if (instanceIDFreeList[freeListCountIndex] < 0)
        {
            instanceIDFreeList[freeListCountIndex] = 0;
        }
    }
}
