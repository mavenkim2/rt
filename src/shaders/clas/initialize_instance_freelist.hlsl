RWStructuredBuffer<int> instanceIDFreeList : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    int maxInstances = 1u << 21u;
    int numPartitions = 16;
    int numInstancesPerPartition = maxInstances / numPartitions;

    instanceIDFreeList[0] = maxInstances;
    for (int i = 0; i < numPartitions; i++)
    {
        instanceIDFreeList[i * (numInstancesPerPartition + 1u)] = numInstancesPerPartition;
        for (int j = 0; j < numInstancesPerPartition; j++)
        {
            instanceIDFreeList[i * (numInstancesPerPartition + 1u) + j + 1u] = j;
        }
    }
#if 0
    for (int i = 0; i < maxInstances; i++)
    {
        instanceIDFreeList[i + 1] = maxInstances - i - 1;
    }
#endif
}
