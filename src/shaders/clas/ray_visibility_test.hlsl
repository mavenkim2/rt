StructuredBuffer<uint> offsets : register(t0);
StructuredBuffer<uint> data : register(t1);
RWStructuredBuffer<uint> buffer : register(u2);
RWStructuredBuffer<uint> globals : register(u3);

[numthreads(32, 1, 1)]
void main()
{
    float3 origin;
    float3 dir;
    float3 minP;
    float3 invGridSize;
    uint3 voxelDims;

    float3 diff = origin - minP;

    float3 p = diff * invGridSize;
    uint3 voxelStart = (diff * invGridSize) * dims;
    uint3 voxel = voxelStart;
    
    int3 step = 0;
    step.x = dir.x > 0.f ? 1 : -1;
    step.y = dir.y > 0.f ? 1 : -1;
    step.z = dir.z > 0.f ? 1 : -1;

    int3 checkStep = 0;
    checkStep.x = dir.x > 0.f ? 1 : 0;
    checkStep.y = dir.y > 0.f ? 1 : 0;
    checkStep.z = dir.z > 0.f ? 1 : 0;

    float3 invDir = rcp(dir);

    for (;;)
    {
        uint voxelIndex = (voxel.z * voxelDims.y + voxel.y) * voxelDims.x + voxel.x;
        for (uint i = offsets[voxelIndex]; i < offsets[voxelIndex + 1]; i++)
        {
            uint index;
            InterlockedAdd(globals[GLOBALS_DEBUG], 1, index);
            uint val = data[i];
            buffer[index] = val;
        }

        float3 nextT = (float3(voxel + checkStep) * invDims - p) * invDir;

        float t = nextT.x;
        uint dim = 0;
        for (int i = 1; i < 3; i++)
        {
            if (nextT[i] < t)
            {
                dim = i;
                t = nextT[i];
            }
        }

        voxel[dim] += step[dim];

        // make these visible somehow
    }
}
