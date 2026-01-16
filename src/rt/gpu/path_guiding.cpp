#include "path_guiding.h"

namespace rt
{

PathGuiding::PathGuiding(Device *device) : device(device), initialized(false)
{
    // TODO: hardcoded
    const uint32_t numSamples  = 1u << 22u;
    const uint32_t maxNumNodes = 1u << 16u;

    string path         = "../src/rt/gpu/kernel.cubin";
    ModuleHandle handle = device->RegisterModule(path);

    for (int kernel = 0; kernel < PATH_GUIDING_KERNEL_MAX; kernel++)
    {
        handles[kernel] =
            device->RegisterKernels((const char *)pathGuidingKernelNames[kernel].str, handle);
    }

    u32 numVMMs         = 32;
    u32 numBlocks       = numVMMs;
    const u32 blockSize = 256;

    gpuArena               = device->CreateArena(megabytes(512));
    Bounds3f *sampleBounds = device->Alloc<Bounds3f>(1, 4u);

    nodes                            = device->Alloc<KDTreeNode>(maxNumNodes, 4u);
    sampleStatistics                 = device->Alloc<SampleStatistics>(maxNumNodes, 8u);
    VMM *vmms                        = device->Alloc<VMM>(maxNumNodes, 4u);
    VMMStatistics *vmmStatistics     = device->Alloc<VMMStatistics>(maxNumNodes, 4);
    SplitStatistics *splitStatistics = device->Alloc<SplitStatistics>(maxNumNodes, 4);

    device->MemZero(nodes, sizeof(KDTreeNode) * maxNumNodes);
    device->MemZero(vmmStatistics, sizeof(VMMStatistics) * maxNumNodes);
    device->MemZero(splitStatistics, sizeof(SplitStatistics) * maxNumNodes);

    SampleData *samples = device->Alloc<SampleData>(numSamples, 4u);

    uint32_t *leafNodeIndices = device->Alloc<uint32_t>(maxNumNodes + 1, 4u);
    uint32_t *numLeafNodes    = leafNodeIndices;
    leafNodeIndices += 1;
    float *samplePosX = device->Alloc<float>(numSamples, 4u);
    float *samplePosY = device->Alloc<float>(numSamples, 4u);
    float *samplePosZ = device->Alloc<float>(numSamples, 4u);

    float *sampleDirX      = device->Alloc<float>(numSamples, 128u);
    float *sampleDirY      = device->Alloc<float>(numSamples, 4u);
    float *sampleDirZ      = device->Alloc<float>(numSamples, 4u);
    float *sampleWeights   = device->Alloc<float>(numSamples, 4u);
    float *samplePdfs      = device->Alloc<float>(numSamples, 4u);
    float *sampleDistances = device->Alloc<float>(numSamples, 4u);

    SOAFloat3 samplePositions  = {samplePosX, samplePosY, samplePosZ};
    SOAFloat3 sampleDirections = {sampleDirX, sampleDirY, sampleDirZ};

    LevelInfo *levelInfo = device->Alloc<LevelInfo>(1, 4u);

    VMMQueue *reduceQueue        = device->Alloc<VMMQueue>(1, 4u);
    VMMQueue *partitionQueue     = device->Alloc<VMMQueue>(1, 4u);
    VMMQueue *vmmQueue           = device->Alloc<VMMQueue>(1, 4u);
    WorkItem *reductionWorkItems = device->Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *partitionWorkItems = device->Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *vmmWorkItems       = device->Alloc<WorkItem>(numSamples, 4u);

    device->MemZero(vmmWorkItems, sizeof(WorkItem) * numSamples);
    device->MemZero(vmmQueue, sizeof(VMMQueue));

    VMMMapState *vmmMapStates = device->Alloc<VMMMapState>(maxNumNodes, 4u);

    float3 sceneMin = make_float3(0.f);
    float3 sceneMax = make_float3(100.f);

    const u32 numSampleBlocks = (numSamples + blockSize - 1) / blockSize;

    // device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_INITIALIZE_SAMPLES], numSampleBlocks,
    //                       blockSize, sampleStatistics, sampleBounds, levelInfo, nodes,
    //                       samples, sampleDirections, samplePositions, sampleIndices,
    //                       numSamples, sceneMin, sceneMax);

    cudaDeviceSynchronize();
}

void PathGuiding::Update()
{
    const uint32_t blockSize = 256;
    gpuArena->Clear();

    // TODO IMPORTANT: sync here

    const uint32_t numSamples = *numSamplesBuffer;

    // TODO: ?
    const uint32_t maxNumNodes = numSamples;

    if (numSamples == 0)
    {
        Print("No path guiding samples. Skipping update.\n");
        return;
    }

    float *samplePosX         = gpuArena->Alloc<float>(numSamples, 4u);
    float *samplePosY         = gpuArena->Alloc<float>(numSamples, 4u);
    float *samplePosZ         = gpuArena->Alloc<float>(numSamples, 4u);
    SOAFloat3 samplePositions = {samplePosX, samplePosY, samplePosZ};

    uint32_t *sampleIndices    = gpuArena->Alloc<uint32_t>(numSamples, 4u);
    uint32_t *newSampleIndices = gpuArena->Alloc<uint32_t>(numSamples, 4u);

    LevelInfo *levelInfo = gpuArena->Alloc<LevelInfo>(1, 4u);

    VMMQueue *reduceQueue        = gpuArena->Alloc<VMMQueue>(1, 4u);
    VMMQueue *partitionQueue     = gpuArena->Alloc<VMMQueue>(1, 4u);
    VMMQueue *vmmQueue           = gpuArena->Alloc<VMMQueue>(1, 4u);
    WorkItem *reductionWorkItems = gpuArena->Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *partitionWorkItems = gpuArena->Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *vmmWorkItems       = gpuArena->Alloc<WorkItem>(numSamples, 4u);

    // TODO: initial base level reduction
    const uint32_t numSampleBlocks = (numSamples + blockSize - 1) / blockSize;

    if (!initialized)
    {
        initialized = true;
        // TODO IMPORTANT: make sure the sample bounds are initialized properly
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_GET_SAMPLE_BOUNDS], numSampleBlocks,
                              blockSize, &sampleStatistics[0].bounds, samplePositions,
                              numSamplesBuffer);
    }

    // Update KD Tree
    for (uint32_t level = 0; level < MAX_TREE_DEPTH; level++)
    {
        const uint32_t maxNodesOnLevel = 1u << level;
        const uint32_t nodeBlocks      = (maxNodesOnLevel + WARP_SIZE - 1) / WARP_SIZE;

        const uint32_t nodeLargeBlocks = (maxNodesOnLevel + blockSize - 1) / blockSize;

        uint32_t *s0 = (level & 1) ? newSampleIndices : sampleIndices;
        uint32_t *s1 = (level & 1) ? sampleIndices : newSampleIndices;

        if (level > 0)
        {
            device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_GET_CHILD_NODE_OFFSET],
                                  nodeBlocks, WARP_SIZE, levelInfo, nodes, level);
        }

        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_BEGIN_LEVEL], 1, 1, levelInfo,
                              reduceQueue, partitionQueue);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_CALCULATE_CHILD_INDICES], 1, 1,
                              levelInfo, nodes);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_CREATE_WORK_ITEMS], nodeLargeBlocks,
                              blockSize, reduceQueue, reductionWorkItems, partitionQueue,
                              partitionWorkItems, nodes, levelInfo, sampleStatistics);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_CALCULATE_NODE_STATISTICS],
                              numSampleBlocks, blockSize, reduceQueue, reductionWorkItems,
                              nodes, sampleStatistics, samplePositions, s0);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_CALCULATE_SPLIT_LOCATIONS],
                              nodeBlocks, WARP_SIZE, levelInfo, nodes, sampleStatistics);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_BUILD_KDTREE], nodeLargeBlocks,
                              blockSize, partitionQueue, partitionWorkItems, levelInfo, nodes,
                              s0, s1, samplePositions);
    }
}

} // namespace rt
