RWStructuredBuffer<int> instanceIDFreeList : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    int maxInstances = 1u << 24u;
    instanceIDFreeList[0] = maxInstances;
    for (int i = 0; i < maxInstances; i++)
    {
        instanceIDFreeList[i + 1] = maxInstances - i - 1;
    }
}
