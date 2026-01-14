#include "path_guiding.h"

namespace rt
{

PathGuiding::PathGuiding(Device *device) : device(device)
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

    Bounds3f *sampleBounds = device->Alloc<Bounds3f>(1, 4u);

    KDTreeNode *nodes                  = device->Alloc<KDTreeNode>(maxNumNodes, 4u);
    SampleStatistics *sampleStatistics = device->Alloc<SampleStatistics>(maxNumNodes, 8u);
    VMM *vmms                          = device->Alloc<VMM>(maxNumNodes, 4u);
    VMMStatistics *vmmStatistics       = device->Alloc<VMMStatistics>(maxNumNodes, 4);
    SplitStatistics *splitStatistics   = device->Alloc<SplitStatistics>(maxNumNodes, 4);

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

    uint32_t *sampleIndices    = device->Alloc<uint32_t>(numSamples, 4u);
    uint32_t *newSampleIndices = device->Alloc<uint32_t>(numSamples, 4u);

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

    device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_INITIALIZE_SAMPLES], numSampleBlocks,
                          blockSize, sampleStatistics, sampleBounds, levelInfo, nodes, samples,
                          sampleDirections, samplePositions, sampleIndices, numSamples,
                          sceneMin, sceneMax);

    cudaDeviceSynchronize();
}

} // namespace rt
