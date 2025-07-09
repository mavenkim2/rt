#include "bit_packing.h"
#include "camera.h"
#include "dgfs.h"
#include "debug.h"
#include "graphics/vulkan.h"
#include "integrate.h"
#include "gpu_scene.h"
#include "math/simd_base.h"
#include "radix_sort.h"
#include "shader_interop/hierarchy_traversal_shaderinterop.h"
#include "string.h"
#include "memory.h"
#include "parallel.h"
#include "platform.h"
#include "shader_interop/as_shaderinterop.h"
#include "shader_interop/dense_geometry_shaderinterop.h"
#include "shader_interop/gpu_scene_shaderinterop.h"
#include "shader_interop/hit_shaderinterop.h"
#include "shader_interop/ray_shaderinterop.h"
#include "shader_interop/debug_shaderinterop.h"
#include "shader_interop/virtual_textures_shaderinterop.h"
#include "../third_party/nvapi/nvapi.h"
#include "nvapi.h"
#include "scene.h"
#include "win32.h"
#include "graphics/ptex.h"

namespace rt
{

void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          MaterialHashMap *materialHashMap, Mesh &mesh,
                          ChunkedLinkedList<Mesh, MemoryType_Shape> &shapes,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices,
                          ChunkedLinkedList<Light *, MemoryType_Light> &lights)
{
    PrimitiveIndices &primIndices = indices.AddBack();

    MaterialHandle materialHandle;
    LightHandle lightHandle;
    Texture *alphaTexture   = 0;
    DiffuseAreaLight *light = 0;

    // TODO: handle instanced mesh lights
    AffineSpace *transform = 0;

    // Check for material
    if (Advance(&tokenizer, "m "))
    {
        Assert(materialHashMap);
        string materialName      = ReadWord(&tokenizer);
        const MaterialNode *node = materialHashMap->Get(materialName);
        materialHandle           = node->handle;
    }

    if (Advance(&tokenizer, "transform "))
    {
        u32 transformIndex = ReadInt(&tokenizer);
        SkipToNextChar(&tokenizer);
    }

    // Check for area light
    if (Advance(&tokenizer, "a "))
    {
        ErrorExit(type == GeometryType::QuadMesh, "Only quad area lights supported for now\n");
        Assert(transform);

        DiffuseAreaLight *areaLight =
            ParseAreaLight(arena, &tokenizer, transform, sceneID, shapes.totalCount - 1);
        lightHandle      = LightHandle(LightClass::DiffuseAreaLight, lights.totalCount);
        lights.AddBack() = areaLight;
        light            = areaLight;
    }

    // Check for alpha
    if (Advance(&tokenizer, "alpha "))
    {
        Texture *alphaTexture = ParseTexture(arena, &tokenizer, directory);

        // TODO: this is also a hack: properly evaluate whether the alpha is
        // always 0
        if (lightHandle)
        {
            light->type = LightType::DeltaPosition;
        }
    }
    primIndices = PrimitiveIndices(lightHandle, materialHandle, alphaTexture);
}

// what's next?
// 1. restir di
// 3. clas
//      - are memory savings possible with this? it seems like not really, and that this
//      just speeds up rebuilds for dynamic/adaptively tessellated geometry. not really
//      what I need.
//      - on blackwell there's memory savings
//      - could not get it to run
// 4. actual bsdfs and brdfs
// 5. add other parts of the scene, with actual instancing
// 6. disney bsdf
// 7. recycle memory
// 8. textures
StaticArray<VkAabbPositionsKHR> CreateAABBForNTriangles(Arena *arena, ClusterBuilder &builder,
                                                        int numBlocks)
{
    u32 N          = TRIANGLES_PER_LEAF;
    u32 shift      = LOG2_TRIANGLES_PER_LEAF;
    u32 totalAabbs = MAX_CLUSTER_TRIANGLES >> shift;
    StaticArray<VkAabbPositionsKHR> aabbs(arena, totalAabbs * numBlocks);
    PrimRef *refs = builder.primRefs;
    for (int index = 0; index < builder.threadClusters.Length(); index++)
    {
        for (auto *node = builder.threadClusters[index].l.first; node != 0; node = node->next)
        {
            for (int i = 0; i < node->count; i++)
            {
                RecordAOSSplits &record = node->values[i];

                int numAabbs = (record.count + N - 1) >> shift;
                for (int j = 0; j < numAabbs; j++)
                {
                    VkAabbPositionsKHR aabb;
                    aabb.minX         = pos_inf;
                    aabb.minY         = pos_inf;
                    aabb.minZ         = pos_inf;
                    aabb.maxX         = neg_inf;
                    aabb.maxY         = neg_inf;
                    aabb.maxZ         = neg_inf;
                    int startRefIndex = record.start + N * j;
                    for (int subIndex = 0; subIndex < Min(N, record.count - startRefIndex);
                         subIndex++)
                    {
                        int refIndex = startRefIndex + subIndex;
                        aabb.minX    = Min(aabb.minX, -refs[refIndex].minX);
                        aabb.minY    = Min(aabb.minY, -refs[refIndex].minY);
                        aabb.minZ    = Min(aabb.minZ, -refs[refIndex].minZ);
                        aabb.maxX    = Max(aabb.maxX, refs[refIndex].maxX);
                        aabb.maxY    = Max(aabb.maxY, refs[refIndex].maxY);
                        aabb.maxZ    = Max(aabb.maxZ, refs[refIndex].maxZ);
                    }

                    aabbs.Push(aabb);
                }

                VkAabbPositionsKHR nullAabb = {};
                nullAabb.minX               = f32(NaN);
                for (int remaining = 0; remaining < totalAabbs - numAabbs; remaining++)
                {
                    aabbs.Push(nullAabb);
                }
            }
        }
    }
    return aabbs;
}

u32 GetNumTriangles(ClusterBuilder &builder)
{
    u32 count = 0;
    for (int index = 0; index < builder.threadClusters.Length(); index++)
    {
        for (auto *node = builder.threadClusters[index].l.first; node != 0; node = node->next)
        {
            for (int i = 0; i < node->count; i++)
            {
                RecordAOSSplits &record = node->values[i];
                count += record.count;
            }
        }
    }
    return count;
}

void Render(RenderParams2 *params, int numScenes, Image *envMap)
{
    ScenePrimitives **scenes = GetScenes();
    NvAPI_Status status      = NvAPI_Initialize();
    Assert(status == NVAPI_OK);
    // Compile shaders
    Shader fillBlasAddressArrayShader;
    Shader fillClusterBLASInfoShader;
    Shader fillInstanceShader;
    Shader getBlasAddressOffsetShader;
    Shader instanceCullingShader;
    Shader prepareIndirectShader;
    Shader hierarchyTraversalShader;

    Shader testShader;

    RayTracingShaderGroup groups[3];
    Arena *arena = params->arenas[GetThreadIndex()];
    {
        string raygenShaderName = "../src/shaders/render_raytrace_rgen.spv";
        string missShaderName   = "../src/shaders/render_raytrace_miss.spv";
        string hitShaderName    = "../src/shaders/render_raytrace_hit.spv";

        string rgenData = OS_ReadFile(arena, raygenShaderName);
        string missData = OS_ReadFile(arena, missShaderName);
        string hitData  = OS_ReadFile(arena, hitShaderName);

        Shader raygenShader = device->CreateShader(ShaderStage::Raygen, "raygen", rgenData);
        Shader missShader   = device->CreateShader(ShaderStage::Miss, "miss", missData);
        Shader hitShader    = device->CreateShader(ShaderStage::Hit, "hit", hitData);

        groups[0].shaders[0] = raygenShader;
        groups[0].numShaders = 1;
        groups[0].stage[0]   = ShaderStage::Raygen;

        groups[1].shaders[0] = missShader;
        groups[1].numShaders = 1;
        groups[1].stage[0]   = ShaderStage::Miss;

        groups[2].shaders[0] = hitShader;
        groups[2].stage[0]   = ShaderStage::Hit;
        groups[2].numShaders = 1;
        groups[2].type       = RayTracingShaderGroupType::Triangle;

        string fillBlasAddressArrayName = "../src/shaders/fill_blas_address_array.spv";
        string fillBlasAddressArrayData = OS_ReadFile(arena, fillBlasAddressArrayName);
        fillBlasAddressArrayShader      = device->CreateShader(
            ShaderStage::Compute, "fill blas address array", fillBlasAddressArrayData);

        string fillClusterBLASName     = "../src/shaders/fill_cluster_bottom_level_info.spv";
        string fillClusterBLASInfoData = OS_ReadFile(arena, fillClusterBLASName);
        fillClusterBLASInfoShader      = device->CreateShader(
            ShaderStage::Compute, "fill cluster bottom level info", fillClusterBLASInfoData);

        string fillInstanceShaderName = "../src/shaders/fill_instance_descs.spv";
        string fillInstanceShaderData = OS_ReadFile(arena, fillInstanceShaderName);
        fillInstanceShader = device->CreateShader(ShaderStage::Compute, "fill instance descs",
                                                  fillInstanceShaderData);

        string getBlasAddressOffsetName = "../src/shaders/get_blas_address_offset.spv";
        string getBlasAddressOffsetData = OS_ReadFile(arena, getBlasAddressOffsetName);
        getBlasAddressOffsetShader      = device->CreateShader(
            ShaderStage::Compute, "get blas address offset", getBlasAddressOffsetData);

        string instanceCullingShaderName = "../src/shaders/instance_culling.spv";
        string instanceCullingShaderData = OS_ReadFile(arena, instanceCullingShaderName);
        instanceCullingShader = device->CreateShader(ShaderStage::Compute, "instance culling",
                                                     instanceCullingShaderData);

        string prepareIndirectName = "../src/shaders/prepare_indirect.spv";
        string prepareIndirectData = OS_ReadFile(arena, prepareIndirectName);
        prepareIndirectShader = device->CreateShader(ShaderStage::Compute, "prepare indirect",
                                                     prepareIndirectData);

        string hierarchyTraversalName = "../src/shaders/hierarchy_traversal.spv";
        string hierarchyTraversalData = OS_ReadFile(arena, hierarchyTraversalName);
        hierarchyTraversalShader      = device->CreateShader(
            ShaderStage::Compute, "hierarchy traversal", hierarchyTraversalData);

        string testShaderName = "../src/shaders/test.spv";
        string testShaderData = OS_ReadFile(arena, testShaderName);
        testShader = device->CreateShader(ShaderStage::Compute, "test shader", testShaderData);
    }

    // Compile pipelines
    // fill blas address array
    DescriptorSetLayout fillBlasAddressArrayLayout = {};
    fillBlasAddressArrayLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(4, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(5, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline fillBlasAddressArrayPipeline =
        device->CreateComputePipeline(&fillBlasAddressArrayShader, &fillBlasAddressArrayLayout,
                                      0, "fill blas address array");

    // fill cluster bottom level info
    PushConstant fillClusterBottomLevelInfoPush;
    fillClusterBottomLevelInfoPush.offset = 0;
    fillClusterBottomLevelInfoPush.size   = sizeof(FillClusterBottomLevelInfoPushConstant);
    fillClusterBottomLevelInfoPush.stage  = ShaderStage::Compute;
    DescriptorSetLayout fillClusterBLASInfoLayout = {};
    fillClusterBLASInfoLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterBLASInfoLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterBLASInfoLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline fillClusterBLASInfoPipeline = device->CreateComputePipeline(
        &fillClusterBLASInfoShader, &fillClusterBLASInfoLayout,
        &fillClusterBottomLevelInfoPush, "fill cluster bottom level info");

    // fill instance descs
    DescriptorSetLayout fillInstanceLayout = {};
    fillInstanceLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    fillInstanceLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    fillInstanceLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    fillInstanceLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    fillInstanceLayout.AddBinding(4, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline fillInstancePipeline = device->CreateComputePipeline(
        &fillInstanceShader, &fillInstanceLayout, 0, "fill instance descs");

    // get blas address offset
    DescriptorSetLayout getBlasAddressOffsetLayout = {};
    getBlasAddressOffsetLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    getBlasAddressOffsetLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline getBlasAddressOffsetPipeline =
        device->CreateComputePipeline(&getBlasAddressOffsetShader, &getBlasAddressOffsetLayout,
                                      0, "get blas address offset");

    // prepare indirect
    DescriptorSetLayout prepareIndirectLayout = {};
    prepareIndirectLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline prepareIndirectPipeline = device->CreateComputePipeline(
        &prepareIndirectShader, &prepareIndirectLayout, 0, "prepare indirect");

    // instance culling
    PushConstant instanceCullingPush = {};
    instanceCullingPush.stage        = ShaderStage::Compute;
    instanceCullingPush.size         = 4;
    instanceCullingPush.offset       = 0;

    DescriptorSetLayout instanceCullingLayout = {};
    instanceCullingLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);
    instanceCullingLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);
    instanceCullingLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);
    instanceCullingLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);
    instanceCullingLayout.AddBinding(4, DescriptorType::StorageBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline instanceCullingPipeline =
        device->CreateComputePipeline(&instanceCullingShader, &instanceCullingLayout,
                                      &instanceCullingPush, "instance culling");

    // hierarchy traversal
    DescriptorSetLayout hierarchyTraversalLayout = {};
    hierarchyTraversalLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(1, DescriptorType::UniformBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(4, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(5, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(6, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(7, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(9, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                        DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(10, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline hierarchyTraversalPipeline = device->CreateComputePipeline(
        &hierarchyTraversalShader, &hierarchyTraversalLayout, 0, "hierarchy traversal");

    DescriptorSetLayout testLayout = {};
    testLayout.AddBinding(0, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    testLayout.AddBinding(1, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    testLayout.AddBinding((u32)RTBindings::ClusterPageData, DescriptorType::StorageBuffer,
                          VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline testPipeline =
        device->CreateComputePipeline(&testShader, &testLayout, 0, "test");

    Swapchain swapchain = device->CreateSwapchain(params->window, VK_FORMAT_R8G8B8A8_SRGB,
                                                  params->width, params->height);

    PushConstant pushConstant;
    pushConstant.stage  = ShaderStage::Raygen | ShaderStage::Miss;
    pushConstant.offset = 0;
    pushConstant.size   = sizeof(RayPushConstant);

    Semaphore submitSemaphore = device->CreateSemaphore();
    // Transfer data to GPU
    GPUScene gpuScene;
    gpuScene.cameraFromRaster = params->cameraFromRaster;
    gpuScene.renderFromCamera = params->renderFromCamera;
    gpuScene.cameraFromRender = params->cameraFromRender;
    gpuScene.lightFromRender  = params->lightFromRender;
    gpuScene.dxCamera         = params->dxCamera;
    gpuScene.lensRadius       = params->lensRadius;
    gpuScene.dyCamera         = params->dyCamera;
    gpuScene.focalLength      = params->focalLength;
    gpuScene.height           = params->height;
    gpuScene.fov              = params->fov;

    ShaderDebugInfo shaderDebug;

    CommandBuffer *transferCmd = device->BeginCommandBuffer(QueueType_Copy);

    ImageDesc gpuEnvMapDesc(ImageType::Type2D, envMap->width, envMap->height, 1, 1, 1,
                            VK_FORMAT_R8G8B8A8_SRGB, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_TILING_OPTIMAL);

    GPUImage gpuEnvMap = transferCmd->SubmitImage(envMap->contents, gpuEnvMapDesc).image;
    transferCmd->Barrier(&gpuEnvMap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transferCmd->FlushBarriers();

    submitSemaphore.signalValue = 1;
    transferCmd->SignalOutsideFrame(submitSemaphore);

    ImageDesc targetUavDesc(ImageType::Type2D, params->width, params->height, 1, 1, 1,
                            VK_FORMAT_R16G16B16A16_SFLOAT, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_IMAGE_TILING_OPTIMAL);
    GPUImage images[2] = {
        device->CreateImage(targetUavDesc),
        device->CreateImage(targetUavDesc),
    };

    // Create descriptor set layout and pipeline
    VkShaderStageFlags flags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                               VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                               VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    // VkShaderStageFlags flags   = VK_SHADER_STAGE_COMPUTE_BIT;
    DescriptorSetLayout layout = {};
    layout.pipelineLayout      = nullptr;
    layout.AddBinding((u32)RTBindings::Accel, DescriptorType::AccelerationStructure, flags);
    layout.AddBinding((u32)RTBindings::Image, DescriptorType::StorageImage, flags);
    layout.AddBinding((u32)RTBindings::Scene, DescriptorType::UniformBuffer, flags);
    layout.AddBinding((u32)RTBindings::GPUMaterial, DescriptorType::StorageBuffer, flags);
    layout.AddBinding((u32)RTBindings::PageTable, DescriptorType::SampledImage, flags);
    layout.AddBinding((u32)RTBindings::PhysicalPages, DescriptorType::SampledImage, flags);
    layout.AddBinding((u32)RTBindings::ShaderDebugInfo, DescriptorType::UniformBuffer, flags);
    layout.AddBinding((u32)RTBindings::ClusterPageData, DescriptorType::StorageBuffer, flags);
    layout.AddBinding((u32)RTBindings::PtexFaceData, DescriptorType::StorageBuffer, flags);
    layout.AddBinding((u32)RTBindings::Feedback, DescriptorType::StorageBuffer, flags);

    layout.AddImmutableSamplers();

    RayTracingState rts = device->CreateRayTracingPipeline(groups, ArrayLength(groups),
                                                           &pushConstant, &layout, 2, true);
    // VkPipeline pipeline = device->CreateComputePipeline(&shader, &layout, &pushConstant);
    // Build clusters
    ScratchArena sceneScratch;

    Bounds *bounds = PushArrayNoZero(sceneScratch.temp.arena, Bounds, numScenes);

    int maxDepth = 0;
    for (int i = 0; i < numScenes; i++)
    {
        maxDepth = Max(maxDepth, scenes[i]->depth.load(std::memory_order_acquire));
    }

    for (int depth = maxDepth; depth >= 0; depth--)
    {
        ParallelFor(0, numScenes, 1, [&](int jobID, int start, int count) {
            for (int i = start; i < start + count; i++)
            {
                if (scenes[i]->depth == depth)
                {
                    bounds[i] = GetSceneBounds(scenes[i]);
                }
            }
        });
    }

    Bounds sceneBounds;
    for (int i = 0; i < numScenes; i++)
    {
        sceneBounds.Extend(bounds[i]);
    }

    StaticArray<ScenePrimitives *> blasScenes(arena, numScenes);
    StaticArray<ScenePrimitives *> tlasScenes(arena, numScenes);

    for (int i = 0; i < numScenes; i++)
    {
        ScenePrimitives *scene = scenes[i];

        if (scene->geometryType == GeometryType::Instance)
        {
            tlasScenes.Push(scene);
            continue;
        }

        RecordAOSSplits record(neg_inf);

        Mesh *meshes = (Mesh *)scene->primitives;
        if (scene->geometryType == GeometryType::QuadMesh ||
            scene->geometryType == GeometryType::CatmullClark)
        {
            for (int j = 0; j < scene->numPrimitives; j++)
            {
                meshes[j] = ConvertQuadToTriangleMesh(sceneScratch.temp.arena, meshes[j]);
            }
            scene->geometryType = GeometryType::TriangleMesh;
        }
        blasScenes.Push(scene);
    }

    Scene *rootScene = GetScene();

    struct GPUTextureInfo
    {
        u8 *packedFaceData;
        int packedDataSize;
        int minLog2Dim;
        int numVirtualOffsetBits;
        int numFaceDimBits;
        int numFaceIDBits;
        u32 virtualAddressIndex;
        u32 faceDataOffset;
    };
    StaticArray<GPUTextureInfo> gpuTextureInfo(sceneScratch.temp.arena,
                                               rootScene->ptexTextures.size(),
                                               rootScene->ptexTextures.size());

    VirtualTextureManager virtualTextureManager(
        sceneScratch.temp.arena, 131072, 131072, PHYSICAL_POOL_NUM_PAGES_WIDE * PAGE_WIDTH,
        PHYSICAL_POOL_NUM_PAGES_WIDE * PAGE_WIDTH, 4, VK_FORMAT_BC1_RGB_UNORM_BLOCK);

    CommandBuffer *tileCmd          = device->BeginCommandBuffer(QueueType_Compute);
    Semaphore tileSubmitSemaphore   = device->CreateSemaphore();
    tileSubmitSemaphore.signalValue = 1;
    tileCmd->TransferWriteBarrier(&virtualTextureManager.gpuPhysicalPool);
    tileCmd->UAVBarrier(&virtualTextureManager.pageTable);
    tileCmd->FlushBarriers();
    virtualTextureManager.ClearTextures(tileCmd);

    struct TextureTempData
    {
        Tokenizer tokenizer;
        TextureMetadata metadata;
        string filename;
    };

    struct RequestHandle
    {
        u16 sortKey;
        int requestIndex;
        int ptexIndex;
    };

    StaticArray<TextureTempData> textureTempData(sceneScratch.temp.arena,
                                                 rootScene->ptexTextures.size(),
                                                 rootScene->ptexTextures.size());

    RequestHandle *handles = PushArrayNoZero(sceneScratch.temp.arena, RequestHandle,
                                             rootScene->ptexTextures.size());
    int numHandles         = 0;

    for (int i = 0; i < rootScene->ptexTextures.size(); i++)
    {
        PtexTexture &ptexTexture = rootScene->ptexTextures[i];
        string filename          = PushStr8F(sceneScratch.temp.arena, "%S.tiles",
                                             RemoveFileExtension(ptexTexture.filename));

        if (Contains(filename, "displacement")) continue;

        Tokenizer tokenizer;
        tokenizer.input  = OS_ReadFile(sceneScratch.temp.arena, filename);
        tokenizer.cursor = tokenizer.input.str;

        TextureMetadata fileHeader;
        GetPointerValue(&tokenizer, &fileHeader);

        textureTempData[numHandles].tokenizer = tokenizer;
        textureTempData[numHandles].metadata  = fileHeader;
        textureTempData[numHandles].filename  = filename;
        handles[numHandles].sortKey = SafeTruncateU32ToU16(fileHeader.virtualSqrtNumPages);
        handles[numHandles].requestIndex = numHandles;
        handles[numHandles].ptexIndex    = i;
        numHandles++;
    }

    SortHandles<RequestHandle, false>(handles, numHandles);

    u32 ptexFaceDataBytes = 0;
    for (int i = 0; i < numHandles; i++)
    {
        RequestHandle &handle        = handles[i];
        TextureTempData &textureTemp = textureTempData[handle.requestIndex];

        Tokenizer tokenizer        = textureTemp.tokenizer;
        TextureMetadata fileHeader = textureTemp.metadata;

        FaceMetadata2 *metaData = (FaceMetadata2 *)tokenizer.cursor;
        Advance(&tokenizer, sizeof(FaceMetadata2) * fileHeader.numFaces);

        auto array = StaticArray<FaceMetadata2>(sceneScratch.temp.arena, fileHeader.numFaces,
                                                fileHeader.numFaces);
        MemoryCopy(array.data, metaData, sizeof(FaceMetadata2) * fileHeader.numFaces);

        Vec3u *pinnedPages = (Vec3u *)tokenizer.cursor;
        Advance(&tokenizer, sizeof(Vec3u) * fileHeader.numPinnedPages);

        u32 allocIndex        = 0;
        Vec2u baseVirtualPage = virtualTextureManager.AllocateVirtualPages(
            sceneScratch.temp.arena, textureTemp.filename, fileHeader,
            Vec2u(fileHeader.virtualSqrtNumPages, fileHeader.virtualSqrtNumPages),
            tokenizer.cursor, allocIndex);

        virtualTextureManager.PinPages(tileCmd, allocIndex, pinnedPages,
                                       fileHeader.numPinnedPages);

        // Pack ptex face metadata into FaceData bitstream
        u32 maxVirtualOffset = neg_inf;

        int minLog2Dim = 16;
        int maxLog2Dim = 0;

        for (int faceIndex = 0; faceIndex < fileHeader.numFaces; faceIndex++)
        {
            maxVirtualOffset = Max(maxVirtualOffset, metaData[faceIndex].offsetX);
            maxVirtualOffset = Max(maxVirtualOffset, metaData[faceIndex].offsetY);

            minLog2Dim = Min(minLog2Dim, metaData[faceIndex].log2Width);
            maxLog2Dim = Max(maxLog2Dim, metaData[faceIndex].log2Width);
            minLog2Dim = Min(minLog2Dim, metaData[faceIndex].log2Height);
            maxLog2Dim = Max(maxLog2Dim, metaData[faceIndex].log2Height);
        }

        u32 numVirtualOffsetBits = Log2Int(maxVirtualOffset) + 1;
        u32 numFaceDimBits =
            (minLog2Dim == maxLog2Dim ? 0 : Log2Int(Max(maxLog2Dim - minLog2Dim, 1)) + 1);
        u32 numFaceIDBits = Log2Int(fileHeader.numFaces) + 1;

        u32 faceDataBitStreamBitSize =
            (2 * (numVirtualOffsetBits + numFaceDimBits) + 4 * numFaceIDBits + 9) *
            fileHeader.numFaces;
        u32 faceDataBitStreamSize = (faceDataBitStreamBitSize + 7) >> 3;

        u8 *faceDataStream    = PushArray(sceneScratch.temp.arena, u8, faceDataBitStreamSize);
        u32 faceDataBitOffset = 0;
        for (int faceIndex = 0; faceIndex < fileHeader.numFaces; faceIndex++)
        {
            FaceMetadata2 &faceMetadata = metaData[faceIndex];
            WriteBits((u32 *)faceDataStream, faceDataBitOffset, faceMetadata.offsetX,
                      numVirtualOffsetBits);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset, faceMetadata.offsetY,
                      numVirtualOffsetBits);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset,
                      faceMetadata.log2Width - minLog2Dim, numFaceDimBits);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset,
                      faceMetadata.log2Height - minLog2Dim, numFaceDimBits);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset,
                      faceMetadata.rotate & ((1u << 8u) - 1u), 8);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset, faceMetadata.rotate >> 31, 1);
            for (int edgeIndex = 0; edgeIndex < 4; edgeIndex++)
            {
                int neighborFace = faceMetadata.neighborFaces[edgeIndex];
                u32 writeNeighborFace =
                    neighborFace == -1 ? (~0u >> (32 - numFaceIDBits)) : (u32)neighborFace;
                WriteBits((u32 *)faceDataStream, faceDataBitOffset, writeNeighborFace,
                          numFaceIDBits);
            }
        }
        Assert(faceDataBitOffset == faceDataBitStreamBitSize);
        gpuTextureInfo[handle.ptexIndex].packedFaceData       = faceDataStream;
        gpuTextureInfo[handle.ptexIndex].packedDataSize       = faceDataBitStreamSize;
        gpuTextureInfo[handle.ptexIndex].minLog2Dim           = minLog2Dim;
        gpuTextureInfo[handle.ptexIndex].numVirtualOffsetBits = numVirtualOffsetBits;
        gpuTextureInfo[handle.ptexIndex].numFaceDimBits       = numFaceDimBits;
        gpuTextureInfo[handle.ptexIndex].numFaceIDBits        = numFaceIDBits;
        gpuTextureInfo[handle.ptexIndex].virtualAddressIndex  = allocIndex;
        gpuTextureInfo[handle.ptexIndex].faceDataOffset       = ptexFaceDataBytes;

        ptexFaceDataBytes += faceDataBitStreamSize;
    }

    u8 *faceDataByteBuffer = PushArrayNoZero(sceneScratch.temp.arena, u8, ptexFaceDataBytes);
    u32 ptexOffset         = 0;
    for (int i = 0; i < numHandles; i++)
    {
        RequestHandle &handle = handles[i];
        GPUTextureInfo &info  = gpuTextureInfo[handle.ptexIndex];
        Assert(info.faceDataOffset == ptexOffset);
        MemoryCopy(faceDataByteBuffer + ptexOffset, info.packedFaceData, info.packedDataSize);
        ptexOffset += info.packedDataSize;
    }

    // Populate GPU materials
    StaticArray<GPUMaterial> gpuMaterials(sceneScratch.temp.arena,
                                          rootScene->materials.Length());

    for (int i = 0; i < rootScene->materials.Length(); i++)
    {
        GPUMaterial material = rootScene->materials[i]->ConvertToGPU();
        int index            = rootScene->materials[i]->ptexReflectanceIndex;
        u32 materialID       = index;
        if (index != -1)
        {
            GPUTextureInfo &textureInfo = gpuTextureInfo[index];
            if (textureInfo.numVirtualOffsetBits != 0)
            {
                VirtualTextureManager::TextureInfo &texInfo =
                    virtualTextureManager.textureInfo[textureInfo.virtualAddressIndex];
                material.baseVirtualPage      = texInfo.baseVirtualPage;
                material.textureIndex         = textureInfo.virtualAddressIndex;
                material.minLog2Dim           = textureInfo.minLog2Dim;
                material.numVirtualOffsetBits = textureInfo.numVirtualOffsetBits;
                material.numFaceDimBits       = textureInfo.numFaceDimBits;
                material.numFaceIDBits        = textureInfo.numFaceIDBits;
                material.faceDataOffset       = textureInfo.faceDataOffset;
            }
            materialID |= (1u << 31u);
        }
        // TODO:
        if (material.eta != 0.f)
        {
            gpuMaterials.Push(material);
        }
    }
    GPUBuffer materialBuffer =
        tileCmd
            ->SubmitBuffer(gpuMaterials.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(GPUMaterial) * gpuMaterials.Length())
            .buffer;

    Assert(ptexOffset == ptexFaceDataBytes);
    GPUBuffer faceDataBuffer =
        tileCmd
            ->SubmitBuffer(faceDataByteBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           ptexFaceDataBytes)
            .buffer;

    tileCmd->SignalOutsideFrame(tileSubmitSemaphore);
    tileCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    tileCmd->Barrier(
        &virtualTextureManager.gpuPhysicalPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_READ_BIT);
    tileCmd->Barrier(
        &virtualTextureManager.pageTable, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_READ_BIT);
    tileCmd->FlushBarriers();
    device->SubmitCommandBuffer(tileCmd);

    const u32 maxPages              = 2048;
    const u32 maxNodes              = 16384;
    GPUBuffer clusterPageDataBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        maxPages * CLUSTER_PAGE_SIZE);
    GPUBuffer hierarchyNodeBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        maxNodes * sizeof(PackedHierarchyNode));
    CommandBuffer *dgfTransferCmd = device->BeginCommandBuffer(QueueType_Copy);

    u32 geoPageOffset       = 0;
    u32 hierarchyNodeOffset = 0;

    // fix the cracks
    // - would probably have to implement triangle distances
    // add other geometry
    // - makes sure code works with different instances
    // shading normals

    StaticArray<ClusterFileHeader> headers(sceneScratch.temp.arena, numScenes);
    for (int sceneIndex = 0; sceneIndex < numScenes; sceneIndex++)
    {
        ScenePrimitives *scene = scenes[sceneIndex];
        string filename        = scene->filename;
        string virtualGeoFilename =
            PushStr8F(sceneScratch.temp.arena, "%S%S.geo", params->directory,
                      RemoveFileExtension(filename));

        string clusterPageData = OS_ReadFile(sceneScratch.temp.arena, virtualGeoFilename);
        Tokenizer tokenizer;
        tokenizer.input  = clusterPageData;
        tokenizer.cursor = clusterPageData.str;

        ClusterFileHeader clusterFileHeader;
        GetPointerValue(&tokenizer, &clusterFileHeader);

        headers.Push(clusterFileHeader);
        Assert(clusterFileHeader.magic == CLUSTER_FILE_MAGIC);
        dgfTransferCmd->SubmitBuffer(
            &clusterPageDataBuffer, tokenizer.cursor, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            CLUSTER_PAGE_SIZE * clusterFileHeader.numPages, geoPageOffset);

        Advance(&tokenizer, clusterFileHeader.numPages * CLUSTER_PAGE_SIZE);
        PackedHierarchyNode *hierarchyNodes = (PackedHierarchyNode *)tokenizer.cursor;
        dgfTransferCmd->SubmitBuffer(
            &hierarchyNodeBuffer, hierarchyNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sizeof(PackedHierarchyNode) * clusterFileHeader.numNodes, hierarchyNodeOffset);

        geoPageOffset += clusterFileHeader.numPages * CLUSTER_PAGE_SIZE;
        hierarchyNodeOffset += clusterFileHeader.numNodes * sizeof(PackedHierarchyNode);
    }

    Semaphore sem   = device->CreateSemaphore();
    sem.signalValue = 1;
    dgfTransferCmd->SignalOutsideFrame(sem);

    device->SubmitCommandBuffer(dgfTransferCmd);

    CommandBuffer *allCommandBuffer = device->BeginCommandBuffer(QueueType_Graphics);
    allCommandBuffer->Wait(sem);
    Semaphore tlasSemaphore       = device->CreateSemaphore();
    GPUAccelerationStructure tlas = {};

    u32 numInstances     = 0;
    u32 numBlas          = blasScenes.Length();
    u32 maxWriteClusters = maxPages * MAX_CLUSTERS_PER_PAGE;
    if (tlasScenes.Length())
    {
        for (auto &tlas : tlasScenes)
        {
            numInstances += tlas->numPrimitives;
        }
    }
    else
    {
        numInstances = 1;
    }

    GPUBuffer indexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        megabytes(256));

    GPUBuffer vertexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        megabytes(320));
    GPUBuffer clasGlobalsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        sizeof(u32) * GLOBALS_SIZE);

    GPUBuffer decodeClusterDataBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(DecodeClusterData) * maxWriteClusters);
    GPUBuffer blasDataBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    sizeof(BLASData) * numBlas);
    GPUBuffer buildClusterTriangleInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_TRIANGLE_INFO) * maxWriteClusters);
    GPUBuffer buildClusterBottomLevelInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_BOTTOM_LEVEL_INFO) * numBlas);
    GPUBuffer clasPageInfoBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                        sizeof(CLASPageInfo) * maxPages);

    GPUBuffer debugHeaders = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(TestDenseGeometry) * maxWriteClusters);

    GPUBuffer blasClasAddressBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxWriteClusters);

    GPUBuffer blasAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * numBlas);
    GPUBuffer blasAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * numBlas);

    u32 clasBlasScratchSize, clasBlasAccelSize;
    device->GetClusterBLASBuildSizes(maxWriteClusters, maxWriteClusters, 1,
                                     clasBlasScratchSize, clasBlasAccelSize);
    GPUBuffer clasBlasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        clasBlasScratchSize);

    GPUBuffer clasBlasImplicitBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        clasBlasAccelSize);

    // Build the TLAS over BLAS
    StaticArray<GPUInstance> gpuInstances(sceneScratch.temp.arena, numInstances);
    GPUBuffer tlasBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        numInstances * sizeof(VkAccelerationStructureInstanceKHR));

    u32 tlasScratchSize, tlasAccelSize;
    device->GetTLASBuildSizes(numInstances, tlasScratchSize, tlasAccelSize);
    GPUBuffer tlasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        tlasScratchSize);
    GPUBuffer tlasAccelBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        tlasAccelSize);

    if (tlasScenes.Length() != 0)
    {
        Assert(tlasScenes.Length() == 1);
        for (int i = 0; i < tlasScenes.Length(); i++)
        {
            ScenePrimitives *scene = tlasScenes[i];
            Instance *instances    = (Instance *)scene->primitives;
            for (int instanceIndex = 0; instanceIndex < scene->numPrimitives; instanceIndex++)
            {
                GPUInstance gpuInstance;
                AffineSpace &transform =
                    scene->affineTransforms[instances[instanceIndex].transformIndex];

                for (int r = 0; r < 3; r++)
                {
                    for (int c = 0; c < 4; c++)
                    {
                        gpuInstance.renderFromObject[r][c] = transform[c][r];
                    }
                }
                gpuInstance.globalRootNodeOffset = 0;
                gpuInstances.Push(gpuInstance);
            }
        }
    }
    else
    {
        GPUInstance gpuInstance;
        AffineSpace &transform = params->renderFromWorld;
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                gpuInstance.renderFromObject[r][c] = transform[c][r];
            }
        }
        gpuInstance.globalRootNodeOffset = 0;
        gpuInstances.Push(gpuInstance);

        Assert(numInstances == 1);
        Assert(numScenes == 1);
    }

    TransferBuffer gpuInstancesBuffer =
        allCommandBuffer->SubmitBuffer(gpuInstances.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       sizeof(GPUInstance) * gpuInstances.Length());

    GPUBuffer visibleClustersBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        MAX_VISIBLE_CLUSTERS * sizeof(VisibleCluster));

    GPUBuffer queueBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sizeof(Queue));

    GPUBuffer workItemQueueBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(Vec4u) * (MAX_CANDIDATE_NODES + MAX_CANDIDATE_CLUSTERS));

    GPUBuffer debugLeafBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(Vec4u) * (MAX_CANDIDATE_NODES + MAX_CANDIDATE_CLUSTERS));

    u64 indexBufferAddress           = device->GetDeviceAddress(indexBuffer.buffer);
    u64 vertexBufferAddress          = device->GetDeviceAddress(vertexBuffer.buffer);
    u64 blasClasAddressBufferAddress = device->GetDeviceAddress(blasClasAddressBuffer.buffer);
    FillClusterTriangleInfoPushConstant fillPc;
    fillPc.indexBufferBaseAddressLowBits   = indexBufferAddress & 0xffffffff;
    fillPc.indexBufferBaseAddressHighBits  = (indexBufferAddress >> 32u) & 0xffffffff;
    fillPc.vertexBufferBaseAddressLowBits  = vertexBufferAddress & 0xffffffff;
    fillPc.vertexBufferBaseAddressHighBits = (vertexBufferAddress >> 32u) & 0xffffffff;

    FillClusterBottomLevelInfoPushConstant pc;
    pc.arrayBaseAddressLowBits  = blasClasAddressBufferAddress & (~0u);
    pc.arrayBaseAddressHighBits = blasClasAddressBufferAddress >> 32u;

    allCommandBuffer->ClearBuffer(&clasGlobalsBuffer);
    allCommandBuffer->ClearBuffer(&blasDataBuffer);
    allCommandBuffer->ClearBuffer(&indexBuffer);

    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    allCommandBuffer->FlushBarriers();

    // for (auto &header : headers)
    // {
    //     // TODO: need offsets
    //     // Write the BUILD_CLUSTERS_TRIANGLE_INFO descriptors on the GPU
    //     allCommandBuffer->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
    //                                    fillClusterTriangleInfoPipeline);
    //     DescriptorSet ds = fillClusterTriangleInfoLayout.CreateDescriptorSet();
    //     ds.Bind(&buildClusterTriangleInfoBuffer)
    //         .Bind(&decodeClusterDataBuffer)
    //         .Bind(&clasGlobalsBuffer)
    //         .Bind(&clasPageInfoBuffer)
    //         .Bind(&clusterPageDataBuffer);
    //
    //     allCommandBuffer->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
    //                                          fillClusterTriangleInfoLayout.pipelineLayout);
    //
    //     allCommandBuffer->PushConstants(&fillClusterTriangleInfoPush, &fillPc,
    //                                     fillClusterTriangleInfoLayout.pipelineLayout);
    //
    //     allCommandBuffer->Dispatch(header.numPages, 1, 1);
    //     allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                               VK_ACCESS_2_SHADER_WRITE_BIT,
    //                               VK_ACCESS_2_SHADER_READ_BIT);
    //     allCommandBuffer->Barrier(
    //         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
    //         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    //
    //     allCommandBuffer->FlushBarriers();
    // }
    //
    // {
    //     // Decode the clusters
    //     allCommandBuffer->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
    //                                    decodeDgfClustersPipeline);
    //     DescriptorSet ds = decodeDgfClustersLayout.CreateDescriptorSet();
    //     ds.Bind(&indexBuffer)
    //         .Bind(&vertexBuffer)
    //         .Bind(&decodeClusterDataBuffer)
    //         .Bind(&clasGlobalsBuffer)
    //         .Bind(&clusterPageDataBuffer);
    //
    //     allCommandBuffer->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
    //                                          decodeDgfClustersLayout.pipelineLayout);
    //     allCommandBuffer->DispatchIndirect(&clasGlobalsBuffer,
    //                                        sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX);
    //
    //     allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                               VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    //                               VK_ACCESS_2_SHADER_WRITE_BIT,
    //                               VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
    //                                   VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    //     allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                               VK_ACCESS_2_SHADER_WRITE_BIT,
    //                               VK_ACCESS_2_SHADER_READ_BIT);
    //     allCommandBuffer->FlushBarriers();
    // }

    const u32 maxNumTriangles = maxPages * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLES;
    const u32 maxNumVertices  = maxPages * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_VERTICES;
    const u32 maxNumClusters  = maxPages * MAX_CLUSTERS_PER_PAGE;
    // Build the CLAS
    GPUBuffer accelData = allCommandBuffer->BuildCLAS(
        &buildClusterTriangleInfoBuffer, &clusterAccelAddresses, &clusterAccelSizes,
        &clasGlobalsBuffer, sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, maxNumClusters,
        maxNumTriangles, maxNumVertices);

    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_READ_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    // VK_ACCESS_2_SHADER_WRITE_BIT);

    GPUBuffer clasAccelBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, megabytes(512));
    GPUBuffer compactedClasAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxWriteClusters);

    // TODO: don't need to pass in size info?
    {
        allCommandBuffer->CompactCLAS(&clusterAccelAddresses, &clasAccelBuffer,
                                      &compactedClasAddresses, 0, &clasGlobalsBuffer,
                                      sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, maxNumClusters,
                                      accelData.size, 0);
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                                  VK_ACCESS_2_SHADER_WRITE_BIT);
        allCommandBuffer->FlushBarriers();
    }

    // if (currentBuffer == 1)
    // {
    //     GPUBuffer readback = device->CreateBuffer(
    //         VK_BUFFER_USAGE_TRANSFER_DST_BIT, compactedSizes.size, MemoryUsage::GPU_TO_CPU);
    //     GPUBuffer readback2 = device->CreateBuffer(
    //         VK_BUFFER_USAGE_TRANSFER_DST_BIT, clusterAccelSizes.size,
    //         MemoryUsage::GPU_TO_CPU);
    //     Semaphore testSemaphore   = device->CreateSemaphore();
    //     testSemaphore.signalValue = 1;
    //     allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    //                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //                               VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    //                               VK_ACCESS_2_TRANSFER_READ_BIT);
    //     allCommandBuffer->FlushBarriers();
    //     allCommandBuffer->CopyBuffer(&readback, &compactedSizes);
    //     allCommandBuffer->CopyBuffer(&readback2, &clusterAccelSizes);
    //     allCommandBuffer->SignalOutsideFrame(testSemaphore);
    //
    //     device->SubmitCommandBuffer(allCommandBuffer);
    //     device->Wait(testSemaphore);
    //
    //     // BLASData *data = (BLASData *)readback.mappedPtr;
    //     u32 totalSize  = 0;
    //     u32 totalSize2 = 0;
    //     u32 *data      = (u32 *)readback.mappedPtr;
    //     u32 *data2     = (u32 *)readback2.mappedPtr;
    //
    //     for (int i = 0; i < 388469; i++)
    //     {
    //         totalSize += data[i];
    //         totalSize2 += data2[i];
    //     }
    //     printf("help me \n");
    //     int stop = 5;
    // }

    tlasSemaphore.signalValue = 1;
    allCommandBuffer->SignalOutsideFrame(tlasSemaphore);

    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    allCommandBuffer->FlushBarriers();
    device->SubmitCommandBuffer(allCommandBuffer);

    device->SubmitCommandBuffer(transferCmd);

    f32 frameDt = 1.f / 60.f;
    int envMapBindlessIndex;

    ViewCamera camera = {};
    camera.position   = params->pCamera;
    camera.forward    = Normalize(params->look - params->pCamera);
    camera.right      = Normalize(Cross(camera.forward, params->up));

    camera.pitch = ArcSin(camera.forward.y);
    camera.yaw   = -Atan2(camera.forward.z, camera.forward.x);

    Vec3f baseForward = camera.forward;
    Vec3f baseRight   = camera.right;

    TransferBuffer sceneTransferBuffers[2] = {
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene)),
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene))};

    TransferBuffer shaderDebugBuffers[2] = {
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene)),
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene))};

    bool mousePressed = false;
    OS_Key keys[4]    = {
        OS_Key_D,
        OS_Key_A,
        OS_Key_W,
        OS_Key_S,
    };
    int dir[4] = {};

    Vec3f cameraStart = params->pCamera;

    GPUBuffer counterBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u32));
    GPUBuffer nvapiBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 256);

    Semaphore frameSemaphore = device->CreateSemaphore();
    for (;;)
    {
        ScratchArena frameScratch;

        MSG message;
        while (PeekMessageW(&message, 0, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        f32 frameTime = OS_NowSeconds();

        ChunkedLinkedList<OS_Event> &events = OS_GetEvents();

        Vec2i dMouseP(0.f);

        for (auto *node = events.first; node != 0; node = node->next)
        {
            for (int i = 0; i < node->count; i++)
            {
                if (node->values[i].key == OS_Mouse_R)
                {
                    mousePressed = node->values[i].type == OS_EventType_KeyPressed;
                    dMouseP[0]   = node->values[i].mouseMoveX;
                    dMouseP[1]   = node->values[i].mouseMoveY;
                }
                else if (node->values[i].type == OS_EventType_MouseMove)
                {
                    dMouseP[0] = node->values[i].mouseMoveX;
                    dMouseP[1] = node->values[i].mouseMoveY;
                }
                for (int keyIndex = 0; keyIndex < ArrayLength(keys); keyIndex++)
                {
                    if (node->values[i].key == keys[keyIndex])
                    {
                        dir[keyIndex] = node->values[i].type == OS_EventType_KeyPressed;
                    }
                }
            }
        }

        if (!mousePressed)
        {
            dMouseP[0] = 0;
            dMouseP[1] = 0;
            MemoryZero(dir, sizeof(dir));
        }

        events.Clear();

        // Input
        {
            f32 speed = 5000.f;

            f32 rotationSpeed = 0.001f * PI;
            camera.RotateCamera(dMouseP, rotationSpeed);

            camera.position += (dir[2] - dir[3]) * camera.forward * speed * frameDt;
            camera.position += (dir[0] - dir[1]) * camera.right * speed * frameDt;

            camera.forward.x = Cos(camera.yaw) * Cos(camera.pitch);
            camera.forward.y = Sin(camera.pitch);
            camera.forward.z = -Sin(camera.yaw) * Cos(camera.pitch);
            camera.forward   = Normalize(camera.forward);

            camera.right = Normalize(Cross(camera.forward, params->up));

            LinearSpace3f axis;
            axis.e[2] = -camera.forward;
            f32 d = camera.forward.x * camera.forward.x + camera.forward.z * camera.forward.z;
            Assert(d != 0);
            if (d == 0)
            {
                axis.e[0][0] = 1.f;
                axis.e[0][1] = 0.f;
                axis.e[0][2] = 0.f;
            }
            else
            {
                f32 invSqrt  = 1 / Sqrt(d);
                axis.e[0][0] = -camera.forward.z * invSqrt;
                axis.e[0][1] = 0.f;
                axis.e[0][2] = camera.forward.x * invSqrt;
            }
            axis.e[1] = Cross(axis.e[2], axis.e[0]);
            axis      = Transpose(axis);

            AffineSpace cameraFromRender =
                AffineSpace(axis, Vec3f(0)) * Translate(cameraStart - camera.position);

            gpuScene.renderFromCamera = Inverse(cameraFromRender);
            gpuScene.cameraFromRender = cameraFromRender;
            OS_GetMousePos(params->window, shaderDebug.mousePos.x, shaderDebug.mousePos.y);
        }
        u32 dispatchDimX =
            (params->width + PATH_TRACE_NUM_THREADS_X - 1) / PATH_TRACE_NUM_THREADS_X;
        u32 dispatchDimY =
            (params->height + PATH_TRACE_NUM_THREADS_Y - 1) / PATH_TRACE_NUM_THREADS_Y;
        gpuScene.dispatchDimX = dispatchDimX;
        gpuScene.dispatchDimY = dispatchDimY;

        device->BeginFrame(true);

        u32 frame       = device->GetCurrentBuffer();
        GPUImage *image = &images[frame];
        string cmdBufferName =
            PushStr8F(frameScratch.temp.arena, "Graphics Cmd %u", device->frameCount);
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics, cmdBufferName);
        debugState.BeginFrame(cmd);

        if (device->frameCount == 0)
        {
            cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            cmd->Wait(submitSemaphore);
            cmd->Wait(tileSubmitSemaphore);
            device->Wait(tlasSemaphore);

            envMapBindlessIndex = device->BindlessIndex(&gpuEnvMap);

            device->DestroyBuffer(&accelData);
        }

        u32 currentBuffer = device->GetCurrentBuffer();

        MemoryCopy(sceneTransferBuffers[currentBuffer].mappedPtr, &gpuScene, sizeof(GPUScene));
        cmd->SubmitTransfer(&sceneTransferBuffers[currentBuffer]);
        MemoryCopy(shaderDebugBuffers[currentBuffer].mappedPtr, &shaderDebug,
                   sizeof(ShaderDebugInfo));
        cmd->SubmitTransfer(&shaderDebugBuffers[currentBuffer]);

        // Virtual texture system
        cmdBufferName = PushStr8F(frameScratch.temp.arena, "Virtual Texture Async Copy Cmd %u",
                                  device->frameCount);
        CommandBuffer *virtualTextureCopyCmd =
            device->BeginCommandBuffer(QueueType_Copy, cmdBufferName);
        cmdBufferName = PushStr8F(frameScratch.temp.arena, "Virtual Texture Transition Cmd %u",
                                  device->frameCount);
        CommandBuffer *transitionCmd =
            device->BeginCommandBuffer(QueueType_Graphics, cmdBufferName);
        virtualTextureManager.Update(cmd, virtualTextureCopyCmd, transitionCmd,
                                     QueueType_Graphics);
        device->SubmitCommandBuffer(transitionCmd);
        device->SubmitCommandBuffer(virtualTextureCopyCmd);

        cmd->ClearBuffer(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer);

        // Virtual geometry pass
        {
            cmd->ClearBuffer(&visibleClustersBuffer, ~0u);
            cmd->ClearBuffer(&workItemQueueBuffer, ~0u);
            cmd->ClearBuffer(&queueBuffer);
            cmd->ClearBuffer(&clasGlobalsBuffer);
            cmd->ClearBuffer(&blasDataBuffer);

            cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
            cmd->FlushBarriers();

            // Instance culling
            {
                NumPushConstant instanceCullingPushConstant;
                instanceCullingPushConstant.num = numInstances;
                device->BeginEvent(cmd, "Instance Culling");

                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, instanceCullingPipeline);
                DescriptorSet ds = instanceCullingLayout.CreateDescriptorSet();
                ds.Bind(&gpuInstancesBuffer.buffer)
                    .Bind(&clasGlobalsBuffer)
                    .Bind(&workItemQueueBuffer, 0, sizeof(Vec4u) * MAX_CANDIDATE_NODES)
                    .Bind(&queueBuffer)
                    .Bind(&blasDataBuffer);

                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        instanceCullingLayout.pipelineLayout);
                cmd->PushConstants(&instanceCullingPush, &instanceCullingPushConstant,
                                   instanceCullingLayout.pipelineLayout);
                cmd->Dispatch((numInstances + 63) / 64, 1, 1);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_ACCESS_2_SHADER_WRITE_BIT,
                             VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }

            // Hierarchy traversal
            {
                device->BeginEvent(cmd, "Hierarchy Traversal");

                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, hierarchyTraversalPipeline);
                DescriptorSet ds = hierarchyTraversalLayout.CreateDescriptorSet();
                ds.Bind(&queueBuffer)
                    .Bind(&sceneTransferBuffers[currentBuffer].buffer)
                    .Bind(&clasGlobalsBuffer)
                    .Bind(&workItemQueueBuffer, 0, MAX_CANDIDATE_NODES * sizeof(Vec4u))
                    .Bind(&workItemQueueBuffer, MAX_CANDIDATE_NODES * sizeof(Vec4u),
                          MAX_CANDIDATE_CLUSTERS * sizeof(Vec4u))
                    .Bind(&gpuInstancesBuffer.buffer)
                    .Bind(&hierarchyNodeBuffer)
                    .Bind(&visibleClustersBuffer)
                    .Bind(&blasDataBuffer)
                    .Bind(&clusterPageDataBuffer)
                    .Bind(&debugLeafBuffer);

                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        hierarchyTraversalLayout.pipelineLayout);
                cmd->Dispatch(1440, 1, 1);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }

#if 0
            {
                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, testPipeline);
                DescriptorSet ds = testLayout.CreateDescriptorSet();
                ds.Bind(&visibleClustersBuffer)
                    .Bind(&clasGlobalsBuffer)
                    .Bind(&clusterPageDataBuffer.buffer);
                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        testLayout.pipelineLayout);
                cmd->Dispatch(clusterFileHeader.numPages, 1, 1);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                cmd->FlushBarriers();
            }
#endif

            {
                // Prepare indirect args
                device->BeginEvent(cmd, "Prepare indirect");
                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, prepareIndirectPipeline);
                DescriptorSet ds = prepareIndirectLayout.CreateDescriptorSet();
                ds.Bind(&clasGlobalsBuffer);

                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        prepareIndirectLayout.pipelineLayout);
                cmd->Dispatch(1, 1, 1);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT,
                             VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }

            {
                device->BeginEvent(cmd, "Get BLAS Address Offset");
                // Calculate where clas addresses should be written for each blas
                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
                                  getBlasAddressOffsetPipeline);

                DescriptorSet ds = getBlasAddressOffsetLayout.CreateDescriptorSet();
                ds.Bind(&blasDataBuffer).Bind(&clasGlobalsBuffer);
                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        getBlasAddressOffsetLayout.pipelineLayout);

                cmd->DispatchIndirect(&clasGlobalsBuffer,
                                      sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }

            {
                // Write the clas addresses to a new buffer
                device->BeginEvent(cmd, "Fill BLAS Address Array");
                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
                                  fillBlasAddressArrayPipeline);
                DescriptorSet ds = fillBlasAddressArrayLayout.CreateDescriptorSet();
                ds.Bind(&clasGlobalsBuffer)
                    .Bind(&visibleClustersBuffer)
                    .Bind(&blasDataBuffer)
                    .Bind(&compactedClasAddresses) // clusterAccelAddresses)
                    .Bind(&blasClasAddressBuffer)
                    .Bind(&clasPageInfoBuffer);

                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        fillBlasAddressArrayLayout.pipelineLayout);
                cmd->DispatchIndirect(&clasGlobalsBuffer,
                                      sizeof(u32) * GLOBALS_CLAS_INDIRECT_X);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }

            {
                // Fill out the BUILD_CLUSTERS_BOTTOM_LEVEL_INFO descriptors
                device->BeginEvent(cmd, "Fill Cluster BLAS Info");
                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, fillClusterBLASInfoPipeline);

                DescriptorSet ds = fillClusterBLASInfoLayout.CreateDescriptorSet();
                ds.Bind(&blasDataBuffer)
                    .Bind(&buildClusterBottomLevelInfoBuffer)
                    .Bind(&clasGlobalsBuffer);
                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        fillClusterBLASInfoLayout.pipelineLayout);

                cmd->PushConstants(&fillClusterBottomLevelInfoPush, &pc,
                                   fillClusterBLASInfoLayout.pipelineLayout);
                cmd->DispatchIndirect(&clasGlobalsBuffer,
                                      sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);

                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_ACCESS_2_SHADER_WRITE_BIT,
                             VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }

            {
                // Build the BLASes
                device->BeginEvent(cmd, "Build BLAS");
                cmd->BuildClusterBLAS(&clasBlasImplicitBuffer, &clasBlasScratchBuffer,
                                      &buildClusterBottomLevelInfoBuffer, &blasAccelAddresses,
                                      &blasAccelSizes, &clasGlobalsBuffer,
                                      sizeof(u32) * GLOBALS_BLAS_COUNT_INDEX, maxWriteClusters,
                                      1);
                cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                             VK_ACCESS_2_SHADER_READ_BIT);
                cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                             VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }

            {
                // Prepare instance descriptors for TLAS build
                device->BeginEvent(cmd, "Prepare TLAS");
                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, fillInstancePipeline);

                DescriptorSet ds = fillInstanceLayout.CreateDescriptorSet();
                ds.Bind(&blasAccelAddresses)
                    .Bind(&clasGlobalsBuffer)
                    .Bind(&blasDataBuffer)
                    .Bind(&gpuInstancesBuffer.buffer)
                    .Bind(&tlasBuffer);
                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        fillInstanceLayout.pipelineLayout);

                cmd->DispatchIndirect(&clasGlobalsBuffer,
                                      sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_ACCESS_2_SHADER_WRITE_BIT,
                             VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
                cmd->FlushBarriers();

                device->EndEvent(cmd);
            }

            {
                // TODO: partitioned TLAS
                // Build the TLAS
                device->BeginEvent(cmd, "Build TLAS");
                tlas.as = cmd->BuildTLAS(&tlasAccelBuffer, &tlasScratchBuffer, &tlasBuffer, 1);
                cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                             VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                             VK_ACCESS_2_SHADER_READ_BIT);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }
        }

        RayPushConstant pc;
        pc.envMap   = envMapBindlessIndex;
        pc.frameNum = (u32)device->frameCount;
        pc.width    = envMap->width;
        pc.height   = envMap->height;

        VkPipelineStageFlags2 flags   = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
        cmd->Barrier(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer, flags,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->Barrier(&sceneTransferBuffers[currentBuffer].buffer, flags,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->Barrier(&shaderDebugBuffers[currentBuffer].buffer, flags,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->Barrier(image, VK_IMAGE_LAYOUT_GENERAL, flags, VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->FlushBarriers();
        cmd->BindPipeline(bindPoint, rts.pipeline);
        DescriptorSet descriptorSet = layout.CreateDescriptorSet();
        descriptorSet.Bind(&tlas.as)
            .Bind(image)
            .Bind(&sceneTransferBuffers[currentBuffer].buffer)
            .Bind(&materialBuffer)
            .Bind(&virtualTextureManager.pageTable)
            .Bind(&virtualTextureManager.gpuPhysicalPool)
            .Bind(&shaderDebugBuffers[currentBuffer].buffer)
            .Bind(&clusterPageDataBuffer)
            .Bind(&faceDataBuffer)
            .Bind(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer);

        cmd->BindDescriptorSets(bindPoint, &descriptorSet, rts.layout);
        cmd->PushConstants(&pushConstant, &pc, rts.layout);

        int beginIndex = TIMED_GPU_RANGE_BEGIN(cmd, "ray trace");
        // cmd->Dispatch(dispatchDimX, dispatchDimY, 1);
        cmd->TraceRays(&rts, params->width, params->height, 1);
        TIMED_RANGE_END(beginIndex);

        // Copy feedback from device to host
        CommandBuffer *transferCmd =
            device->BeginCommandBuffer(QueueType_Copy, "feedback copy cmd");
        transferCmd->WaitOn(cmd);
        // transferCmd->Barrier(VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        //                      VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        // transferCmd->FlushBarriers();
        transferCmd->CopyBuffer(
            &virtualTextureManager.feedbackBuffers[currentBuffer].stagingBuffer,
            &virtualTextureManager.feedbackBuffers[currentBuffer].buffer);
        device->SubmitCommandBuffer(transferCmd, true);

        debugState.EndFrame(cmd);
        device->CopyFrameBuffer(&swapchain, cmd, image);
        device->EndFrame(QueueFlag_Copy | QueueFlag_Graphics);

        debugState.PrintDebugRecords();

        // Wait until new update
        f32 endWorkFrameTime = OS_NowSeconds();
        f32 timeElapsed      = endWorkFrameTime - frameTime;

        if (timeElapsed < frameDt)
        {
            u32 msTimeToSleep = (u32)(1000.f * (frameDt - timeElapsed));
            if (msTimeToSleep > 0)
            {
                OS_Sleep(msTimeToSleep);
            }
        }

        while (timeElapsed < frameDt)
        {
            timeElapsed = OS_NowSeconds() - frameTime;
        }
    }
}
} // namespace rt
