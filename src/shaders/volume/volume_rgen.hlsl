#define PNANOVDB_HLSL
//#include "../../third_party/openvdb/nanovdb/nanovdb/PNanoVDB.h"

struct OctreeNode 
{
    float minValue;
    float maxValue;
    uint childIndex;
};

StructuredBuffer<OctreeNode> nodes : register(t0);
StructuredBuffer<int> parents : register(t1);

struct VolumeIterator
{
    float currentT;

    float3 rayO;
    float3 rayDir;

    float3 boundsMin;
    float3 boundsMax;

    int prev;
    int current;

    void Start()
    {
    }

    // internal node
    //      - if current ray pos is outside bounds, go to parent
    //      - find closest child that is in front of the ray
    //      - go to that child
    // leaf node
    //      - Next() terminates
    //      - return majorant and minorant

    void Next()
    {
        for (;;)
        {
            // TODO floating point precision
            float3 currentPos = rayO + currentT * rayDir;
            float3 center = (boundsMin + boundsMax) / 2.f;
            currentPos -= center;
            float3 extent = (boundsMax - boundsMin) / 2.f;
            int next;

            uint childIndex = nodes[current].childIndex;
            // go to parent
            if (childIndex == ~0u || any(currentPos > extent) || any(currentPos < -extent))
            {
                // go to parent
                if (current == 0)
                {
                    current = -1;
                    break;
                }
                int axisMask = (current - 1) & 0x7;
                next = parents[(current - 1) & ~0x7];

                boundsMin = float3((axisMask & 0x1) ? boundsMin.x - 2.f * extent.x : boundsMin.x,
                                   (axisMask & 0x2) ? boundsMin.y - 2.f * extent.y : boundsMin.y,
                                   (axisMask & 0x4) ? boundsMin.z - 2.f * extent.z : boundsMin.z);
                boundsMax = float3((axisMask & 0x1) ? boundsMax.x : boundsMax.x + 2.f * extent.x,
                                   (axisMask & 0x2) ? boundsMax.y : boundsMax.y + 2.f * extent.y,
                                   (axisMask & 0x4) ? boundsMax.z : boundsMax.z + 2.f * extent.z);
            }
            // go to child
            else 
            {
                uint closestChild = (currentPos.x >= 0.f) | ((currentPos.y >= 0.f) << 1) | ((currentPos.z >= 0.f) << 2);
                next = childIndex + closestChild;

                boundsMin = float3((closestChild & 0x1) ? center.x : boundsMin.x,
                                   (closestChild & 0x2) ? center.y : boundsMin.y,
                                   (closestChild & 0x4) ? center.z : boundsMin.z);
                boundsMax = float3((closestChild & 0x1) ? boundsMax.x : center.x,
                                   (closestChild & 0x2) ? boundsMax.y : center.y,
                                   (closestChild & 0x4) ? boundsMax.z : center.z);
            }

            prev = current;
            current = next;

            if (nodes[current].childIndex == ~0u) break;
        }
    }
};

[shader("raygeneration")]
void main()
{
    int stop = 5;
}

