#include "path_guiding.h"

namespace rt
{

PathGuiding::PathGuiding()
{
    // TODO: hardcoded
    string path         = "../src/rt/gpu/kernel.cubin";
    ModuleHandle handle = device->RegisterModule(path);

    for (int kernel = 0; kernel < PATH_GUIDING_KERNEL_MAX; kernel++)
    {
        handles[kernel] =
            device->RegisterKernels((const char *)pathGuidingKernelNames[handle].str, handle);
    }

    u32 numVMMs         = 32;
    u32 numBlocks       = numVMMs;
    const u32 blockSize = 256;

    // port my allocator
    device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_INITIALIZE_SAMPLES], numBlocks,
                          blockSize, numVMMs);

#if 0
    Bounds3f *sampleBounds = allocator.Alloc<Bounds3f>(1, 4u);

    KDTreeNode *nodes                  = allocator.Alloc<KDTreeNode>(maxNumNodes, 4u);
    SampleStatistics *sampleStatistics = allocator.Alloc<SampleStatistics>(maxNumNodes, 8u);
    VMM *vmms                          = allocator.Alloc<VMM>(maxNumNodes, 4u);
    VMMStatistics *vmmStatistics       = allocator.Alloc<VMMStatistics>(maxNumNodes, 4);
    SplitStatistics *splitStatistics   = allocator.Alloc<SplitStatistics>(maxNumNodes, 4);

    cudaMemset(nodes, 0, sizeof(KDTreeNode) * maxNumNodes);
    cudaMemset(vmmStatistics, 0, sizeof(VMMStatistics) * maxNumNodes);
    cudaMemset(splitStatistics, 0, sizeof(SplitStatistics) * maxNumNodes);

    SampleData *samples = allocator.Alloc<SampleData>(numSamples, 4u);

    uint32_t *leafNodeIndices = allocator.Alloc<uint32_t>(maxNumNodes + 1, 4u);
    uint32_t *numLeafNodes    = leafNodeIndices;
    leafNodeIndices += 1;
    float *samplePosX = allocator.Alloc<float>(numSamples, 4u);
    float *samplePosY = allocator.Alloc<float>(numSamples, 4u);
    float *samplePosZ = allocator.Alloc<float>(numSamples, 4u);

    float *sampleDirX      = allocator.Alloc<float>(numSamples, 128u);
    float *sampleDirY      = allocator.Alloc<float>(numSamples, 4u);
    float *sampleDirZ      = allocator.Alloc<float>(numSamples, 4u);
    float *sampleWeights   = allocator.Alloc<float>(numSamples, 4u);
    float *samplePdfs      = allocator.Alloc<float>(numSamples, 4u);
    float *sampleDistances = allocator.Alloc<float>(numSamples, 4u);

    SOAFloat3 samplePositions  = {samplePosX, samplePosY, samplePosZ};
    SOAFloat3 sampleDirections = {sampleDirX, sampleDirY, sampleDirZ};

    uint32_t *sampleIndices    = allocator.Alloc<uint32_t>(numSamples, 4u);
    uint32_t *newSampleIndices = allocator.Alloc<uint32_t>(numSamples, 4u);

    LevelInfo *levelInfo = allocator.Alloc<LevelInfo>(1, 4u);

    Queue *reduceQueue           = allocator.Alloc<Queue>(1, 4u);
    Queue *partitionQueue        = allocator.Alloc<Queue>(1, 4u);
    Queue *vmmQueue              = allocator.Alloc<Queue>(1, 4u);
    WorkItem *reductionWorkItems = allocator.Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *partitionWorkItems = allocator.Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *vmmWorkItems       = allocator.Alloc<WorkItem>(numSamples, 4u);

    cudaMemset(vmmWorkItems, 0, sizeof(WorkItem) * numSamples);
    cudaMemset(vmmQueue, 0, sizeof(Queue));

    VMMMapState *vmmMapStates = allocator.Alloc<VMMMapState>(maxNumNodes, 4u);

    float3 sceneMin = make_float3(0.f);
    float3 sceneMax = make_float3(100.f);

    printf("amount alloc: %u\n", allocator.offset);
#endif
}

} // namespace rt
