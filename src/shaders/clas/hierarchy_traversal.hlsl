struct Queue 
{
    uint nodeReadOffset;
    uint nodeWriteOffset;

    uint leafReadOffset;
    uint leafWriteOffset;
};

struct HierarchyNode 
{
};

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

