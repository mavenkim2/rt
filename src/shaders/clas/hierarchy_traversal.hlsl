struct Queue 
{
    uint nodeReadOffset;
    uint nodeWriteOffset;

    uint leafReadOffset;
    uint leafWriteOffset;
};

// See if child should be visited
float2 TestNode(float4 lodBounds)
{
    // also candidate clusters/candidate nodes? 

    // Find length to cluster center
    float4 lodBounds;
    float3x4 objectToRender;

    float3 center = mul(objectToRender, float4(lodBounds.xyz, 1.f));
    float radius = ?;
    float distSqr = length2(center);

    // Find angle between vector to cluster center and view vector
    float z = dot(camera.forward, center);
    float x = distSqr - z * z;

    // Find angle between vector to cluster center and vector to tangent point on sphere
    float distTangentSqr = distSqr - radius * radius;

    float distTangent = sqrt(max(0.f, distTangentSqr));

    // Find cosine of the above angles subtracted/added
    float invDistSqr = rcp(distSqr);
    float cosSub = (z * distTangent + x * radius) * invDistSqr;
    float cosAdd = (z * distTangent - x * radius) * invDistSqr;

    // Clipping
    float depth = z - zNear;
    if (distSqr < 0.f || cosSub * distTangent < zNear)
    {
        float cosSubX = max(0.f, x - sqrt(radius * radius - depth * depth));
        cosSub = zNear / sqrt(cosSubX * cosSubX + zNear * zNear);
    }
    if (distSqr < 0.f || cosAdd * distTangent < zNear)
    {
        float cosAddX = x + sqrt(radius * radius - depth * depth));
        cosAdd = zNear / sqrt(cosAddX * cosAddX + zNear * zNear);
    }

    float minZ = max(z - radius, zNear);
    float maxZ = max(z + radius, zFar);

    return z + radius > zNear ? float2(minZ * cosAdd, maxZ * cosSub) : float2(0, 0);
}

#define MAX_NODES_PER_BATCH 16
#define PACKED_NODE_SIZE 12

groupshared uint groupNodeReadOffset;

globallycoherent RWStructuredBuffer<Queue> queue : register(u0);
globallycoherent RWByteAddressBuffer nodesBuffer : register(u1);

void WriteNode()
{
    nodesBuffer.Store(PACKED_NODE_SIZE
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    uint waveIndex = WaveGetLaneIndex();
    uint waveLaneCount = WaveGetLaneCount();
    uint nodesPerBatch = waveLaneCount;

    uint finishedThreads = nodesPerBatch;
    uint waveNodeReadOffset = 0;

    for (;;)
    {
        bool alreadyProcessed = false;

        // Pop node from queue
        {
            if (finishedThreads == nodesPerBatch)
            {
                alreadyProcessed = false;
                finishedThreads = 0;

                if (WaveIsFirstLane())
                {
                    InterlockedAdd(queue.nodeReadOffset, nodesPerBatch, waveNodeReadOffset);
                }
            }

            waveNodeReadOffset = WaveReadLaneFirst(waveNodeReadOffset);

            uint nodeIndex = waveNodeReadOffset + waveIndex;
            uint3 nodeData = nodesBuffer.Load3(nodeIndex * PACKED_NODE_SIZE);
            bool isReady = nodeData.x != ~0u && nodeData.y != ~0u && nodeData.z != ~0u;

            uint numToProcess = WaveActiveCountBits(isReady && !alreadyProcessed); 
            finishedThreads += numToProcess;
            bool processLeaves = WaveActiveAnyTrue(isReady && !alreadyProcessed);
            
            if (isReady && !alreadyProcessed)
            {
                // process node batch
                // get nodes somehow
                
                bool visible = false;
                uint numVisibleChildren = WaveActiveCountBits(visible);
                uint nodeWriteOffset;
                if (WaveIsFirstLane())
                {
                    InterlockedAdd(queue.nodeWriteOffset, numVisibleChildren, nodeWriteOffset);
                }
                nodeWriteOffset = WaveReadLaneFirst(writeOffset);

                DeviceMemoryBarrier();
            }
            alreadyProcessed |= isReady;

            if (!processLeaves) continue;
        }

        // Process a batch of leaves
        if (WaveIsFirstLane())
        {
            InterlockedAdd(queue.leafReadOffset, );
        }
    }
}

