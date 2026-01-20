#include "path_guiding.h"
#include "path_guiding_util.h"

namespace rt
{

PathGuiding::PathGuiding(Device *device) : device(device), initialized(false)
{
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

    nodes            = device->Alloc<KDTreeNode>(maxNumNodes, 4u);
    sampleStatistics = device->Alloc<SampleStatistics>(maxNumNodes, 8u);
    treeBuildState   = device->Alloc<KDTreeBuildState>(1, 4u);
    rootBounds       = device->Alloc<Bounds3f>(1, 4u);

    VMM *vmms                        = device->Alloc<VMM>(maxNumNodes, 4u);
    VMMStatistics *vmmStatistics     = device->Alloc<VMMStatistics>(maxNumNodes, 4);
    SplitStatistics *splitStatistics = device->Alloc<SplitStatistics>(maxNumNodes, 4);

    device->MemSet(nodes, 0xff, sizeof(KDTreeNode) * maxNumNodes);
    device->MemZero(vmmStatistics, sizeof(VMMStatistics) * maxNumNodes);
    device->MemZero(splitStatistics, sizeof(SplitStatistics) * maxNumNodes);
}

void PathGuiding::Update()
{
    const uint32_t blockSize       = 256;
    const uint32_t numSampleBlocks = (numSamples + blockSize - 1) / blockSize;

    gpuArena->Clear();

    // TODO: temp
    SampleData *samples = gpuArena->Alloc<SampleData>(numSamples, 4u);

    uint32_t *leafNodeIndices = gpuArena->Alloc<uint32_t>(maxNumNodes + 1, 4u);
    uint32_t *numLeafNodes    = leafNodeIndices;
    leafNodeIndices += 1;
    float *samplePosX = gpuArena->Alloc<float>(numSamples, 4u);
    float *samplePosY = gpuArena->Alloc<float>(numSamples, 4u);
    float *samplePosZ = gpuArena->Alloc<float>(numSamples, 4u);

    float *sampleDirX      = gpuArena->Alloc<float>(numSamples, 128u);
    float *sampleDirY      = gpuArena->Alloc<float>(numSamples, 4u);
    float *sampleDirZ      = gpuArena->Alloc<float>(numSamples, 4u);
    float *sampleWeights   = gpuArena->Alloc<float>(numSamples, 4u);
    float *samplePdfs      = gpuArena->Alloc<float>(numSamples, 4u);
    float *sampleDistances = gpuArena->Alloc<float>(numSamples, 4u);

    SOAFloat3 samplePositions  = {samplePosX, samplePosY, samplePosZ};
    SOAFloat3 sampleDirections = {sampleDirX, sampleDirY, sampleDirZ};

    uint32_t *sampleIndices    = gpuArena->Alloc<uint32_t>(numSamples, 4u);
    uint32_t *newSampleIndices = gpuArena->Alloc<uint32_t>(numSamples, 4u);

    uint32_t *buildNodeIndices     = gpuArena->Alloc<uint32_t>(numSamples, 4u);
    uint32_t *buildNextNodeIndices = gpuArena->Alloc<uint32_t>(numSamples, 4u);

    WorkItem *reductionWorkItems = gpuArena->Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *partitionWorkItems = gpuArena->Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *vmmWorkItems       = gpuArena->Alloc<WorkItem>(numSamples, 4u);

    device->MemZero(&buildNodeIndices[0], sizeof(buildNodeIndices[0]));
    device->MemZero(vmmWorkItems, sizeof(WorkItem) * numSamples);

    VMMMapState *vmmMapStates = gpuArena->Alloc<VMMMapState>(maxNumNodes, 4u);

    float3 sceneMin = make_float3(0.f);
    float3 sceneMax = make_float3(100.f);

    // TODO IMPORTANT: sync here
    // const uint32_t numSamples = *numSamplesBuffer;

    if (numSamples == 0)
    {
        Print("No path guiding samples. Skipping update.\n");
        return;
    }

    bool wasInit = initialized;
    if (!initialized)
    {
        initialized = true;
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_INITIALIZE_SAMPLES], numSampleBlocks,
                              blockSize, rootBounds, sampleStatistics, treeBuildState, nodes,
                              samples, sampleDirections, samplePositions, sampleIndices,
                              numSamples, sceneMin, sceneMax);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_GET_SAMPLE_BOUNDS], numSampleBlocks,
                              blockSize, rootBounds, samplePositions, numSamples);
        // numSamplesBuffer);
    }
    else
    {
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_UPDATE_START], numSampleBlocks,
                              blockSize, treeBuildState, samples, sampleDirections,
                              samplePositions, sampleIndices, numSamples, sceneMin, sceneMax);
    }

    // node indices

    // Update KD Tree
    // TODO: might have to block stride if there are too many nodes.
    for (uint32_t level = 0; level < MAX_TREE_DEPTH; level++)
    {
        const uint32_t maxNodesOnLevel = 1u << level;
        const uint32_t nodeBlocks      = (maxNodesOnLevel + WARP_SIZE - 1) / WARP_SIZE;

        uint32_t *s0 = (level & 1) ? newSampleIndices : sampleIndices;
        uint32_t *s1 = (level & 1) ? sampleIndices : newSampleIndices;

        uint32_t *nodeIndices     = (level & 1) ? buildNextNodeIndices : buildNodeIndices;
        uint32_t *nextNodeIndices = (level & 1) ? buildNodeIndices : buildNextNodeIndices;

        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_BEGIN_LEVEL], 1, 1, treeBuildState);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_CALCULATE_CHILD_INDICES], 1, 1,
                              nodes, treeBuildState, sampleStatistics, nodeIndices,
                              nextNodeIndices);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_CREATE_WORK_ITEMS], maxNodesOnLevel,
                              blockSize, treeBuildState, reductionWorkItems,
                              partitionWorkItems, nodes, sampleStatistics, nodeIndices);
        // NOTE: sample statistics are permanently stored as longlongs.
        // TODO: # thread blocks may need to be more for calculate node stats + buildkdtree
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_CALCULATE_NODE_STATISTICS], 1,
                              blockSize, treeBuildState, reductionWorkItems, nodes,
                              sampleStatistics, samplePositions, s0, rootBounds);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_CALCULATE_SPLIT_LOCATIONS],
                              nodeBlocks, WARP_SIZE, treeBuildState, nodes, sampleStatistics,
                              nodeIndices, rootBounds);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_BUILD_KDTREE], numSampleBlocks,
                              blockSize, treeBuildState, partitionWorkItems, nodes, s0, s1,
                              samplePositions);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_GET_CHILD_NODE_OFFSET], nodeBlocks,
                              WARP_SIZE, treeBuildState, nodes, level, nodeIndices);
    }
    if (wasInit)
    {
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_PRINT_STATS], 1, 1, treeBuildState,
                              sampleStatistics, nodes);
    }
    cudaDeviceSynchronize();

    // Update VMMs
    device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_FIND_LEAF_NODES], 1, 1, treeBuildState,
                          nodes, buildNodeIndices, sampleStatistics, nodes);

    VMMUpdateWorkItem *workItems0 = 0;
    VMMUpdateWorkItem *workItems1 = 0;

    // Splitting
    for (uint32_t iteration = 0; iteration < MAX_COMPONENTS / 2; iteration++)
    {
        VMMUpdateWorkItem *inputWorkItems  = iteration % 2 == 0 ? workItems0 : workItems1;
        VMMUpdateWorkItem *outputWorkItems = iteration % 2 == 0 ? workItems1 : workItems0;

        // reset statistics if isNew

        // update split statistics
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_UPDATE_SPLIT_STATISTICS],
                              uint32_t numBlocks, uint32_t blockSize, Args args...);
        // TODO: zero buildState->numVMMs before this kernel
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_SPLIT_COMPONENTS],
                              uint32_t numBlocks, uint32_t blockSize, Args args...);
        device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_UPDATE_MIXTURE], uint32_t numBlocks,
                              uint32_t blockSize, Args args...);

        // prepare for next loop iteration
    }

    // Merging
}

} // namespace rt
