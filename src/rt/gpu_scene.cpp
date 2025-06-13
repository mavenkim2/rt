#include "bit_packing.h"
#include "camera.h"
#include "dgfs.h"
#include "debug.h"
#include "graphics/vulkan.h"
#include "integrate.h"
#include "gpu_scene.h"
#include "math/simd_base.h"
#include "radix_sort.h"
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

// TODO: see if it's possible to group consecutive triangles in a strip into 1 procedural
// primitive (when the sah is small)
void BuildAllSceneBVHs(RenderParams2 *params, ScenePrimitives **scenes, int numScenes,
                       int maxDepth, Image *envMap)
{
    NvAPI_Status status = NvAPI_Initialize();
    Assert(status == NVAPI_OK);
    // Compile shaders
    Shader shader;

    Shader decodeDgfClustersShader;
    Shader fillBlasAddressArrayShader;
    Shader fillClusterBLASInfoShader;
    Shader fillClusterTriangleInfoShader;
    Shader fillInstanceShader;
    Shader getBlasAddressOffsetShader;

    RayTracingShaderGroup groups[3];
    Arena *arena = params->arenas[GetThreadIndex()];
    {
        string raygenShaderName       = "../src/shaders/render_raytrace_rgen.spv";
        string missShaderName         = "../src/shaders/render_raytrace_miss.spv";
        string hitShaderName          = "../src/shaders/render_raytrace_hit.spv";
        string intersectionShaderName = "../src/shaders/render_raytrace_dgf_intersect.spv";

        string rgenData      = OS_ReadFile(arena, raygenShaderName);
        string missData      = OS_ReadFile(arena, missShaderName);
        string hitData       = OS_ReadFile(arena, hitShaderName);
        string intersectData = OS_ReadFile(arena, intersectionShaderName);

        Shader raygenShader = device->CreateShader(ShaderStage::Raygen, "raygen", rgenData);
        Shader missShader   = device->CreateShader(ShaderStage::Miss, "miss", missData);
        Shader hitShader    = device->CreateShader(ShaderStage::Hit, "hit", hitData);
        Shader isectShader =
            device->CreateShader(ShaderStage::Intersect, "intersect", intersectData);

        groups[0].shaders[0] = raygenShader;
        groups[0].numShaders = 1;
        groups[0].stage[0]   = ShaderStage::Raygen;

        groups[1].shaders[0] = missShader;
        groups[1].numShaders = 1;
        groups[1].stage[0]   = ShaderStage::Miss;

        groups[2].shaders[0] = hitShader;
        groups[2].stage[0]   = ShaderStage::Hit;
#ifdef USE_PROCEDURAL_CLUSTER_INTERSECTION
        groups[2].shaders[1] = isectShader;
        groups[2].stage[1]   = ShaderStage::Intersect;
        groups[2].numShaders = 2;
        groups[2].type       = RayTracingShaderGroupType::Procedural;
#else
        groups[2].numShaders = 1;
        groups[2].type       = RayTracingShaderGroupType::Triangle;
#endif

        string shaderName = "../src/shaders/render_raytrace.spv";
        string shaderData = OS_ReadFile(arena, shaderName);
        shader = device->CreateShader(ShaderStage::Compute, "pathtrace", shaderData);

        string decodeDgfClustersName = "../src/shaders/decode_dgf_clusters.spv";
        string decodeDgfClustersData = OS_ReadFile(arena, decodeDgfClustersName);
        decodeDgfClustersShader      = device->CreateShader(
            ShaderStage::Compute, "decode dgf clusters", decodeDgfClustersData);

        string fillBlasAddressArrayName = "../src/shaders/fill_blas_address_array.spv";
        string fillBlasAddressArrayData = OS_ReadFile(arena, fillBlasAddressArrayName);
        fillBlasAddressArrayShader      = device->CreateShader(
            ShaderStage::Compute, "fill blas address array", fillBlasAddressArrayData);

        string fillClusterBLASName     = "../src/shaders/fill_cluster_bottom_level_info.spv";
        string fillClusterBLASInfoData = OS_ReadFile(arena, fillClusterBLASName);
        fillClusterBLASInfoShader      = device->CreateShader(
            ShaderStage::Compute, "fill cluster bottom level info", fillClusterBLASInfoData);

        string fillClusterTriangleInfoName = "../src/shaders/fill_cluster_triangle_info.spv";
        string fillClusterTriangleInfoData = OS_ReadFile(arena, fillClusterTriangleInfoName);
        fillClusterTriangleInfoShader      = device->CreateShader(
            ShaderStage::Compute, "fill cluster triangle info", fillClusterTriangleInfoData);

        string fillInstanceShaderName = "../src/shaders/fill_instance_descs.spv";
        string fillInstanceShaderData = OS_ReadFile(arena, fillInstanceShaderName);
        fillInstanceShader = device->CreateShader(ShaderStage::Compute, "fill instance descs",
                                                  fillInstanceShaderData);

        string getBlasAddressOffsetName = "../src/shaders/get_blas_address_offset.spv";
        string getBlasAddressOffsetData = OS_ReadFile(arena, getBlasAddressOffsetName);
        getBlasAddressOffsetShader      = device->CreateShader(
            ShaderStage::Compute, "get blas address offset", getBlasAddressOffsetData);
    }

    // Compile pipelines
#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
    // decode dgf clusters
    DescriptorSetLayout decodeDgfClustersLayout = {};
    PushConstant decodeDgfClustersPush;
    decodeDgfClustersPush.stage  = ShaderStage::Compute;
    decodeDgfClustersPush.size   = 4;
    decodeDgfClustersPush.offset = 0;

    PushConstant fillClusterBLASConstants;
    fillClusterBLASConstants.stage  = ShaderStage::Compute;
    fillClusterBLASConstants.size   = sizeof(FillClusterBottomLevelInfoPushConstant);
    fillClusterBLASConstants.offset = 0;

    decodeDgfClustersLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersLayout.AddBinding(
        (u32)RTBindings::DGFBytes, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersLayout.AddBinding((u32)RTBindings::DGFHeaders,
                                       DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline decodeDgfClustersPipeline = device->CreateComputePipeline(
        &decodeDgfClustersShader, &decodeDgfClustersLayout, &decodeDgfClustersPush);

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

    VkPipeline fillBlasAddressArrayPipeline = device->CreateComputePipeline(
        &fillBlasAddressArrayShader, &fillBlasAddressArrayLayout);

    // fill cluster bottom level info
    PushConstant fillClusterBLASInfoPush = {};
    fillClusterBLASInfoPush.stage        = ShaderStage::Compute;
    fillClusterBLASInfoPush.size         = sizeof(FillClusterBottomLevelInfoPushConstant);
    fillClusterBLASInfoPush.offset       = 0;

    DescriptorSetLayout fillClusterBLASInfoLayout = {};
    fillClusterBLASInfoLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterBLASInfoLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline fillClusterBLASInfoPipeline = device->CreateComputePipeline(
        &fillClusterBLASInfoShader, &fillClusterBLASInfoLayout, &fillClusterBLASInfoPush);

    // fill cluster triangle info
    PushConstant fillClusterTriangleInfoPush;
    fillClusterTriangleInfoPush.stage  = ShaderStage::Compute;
    fillClusterTriangleInfoPush.size   = sizeof(FillClusterTriangleInfoPushConstant);
    fillClusterTriangleInfoPush.offset = 0;

    DescriptorSetLayout fillClusterTriangleInfoLayout = {};
    fillClusterTriangleInfoLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline fillClusterTriangleInfoPipeline = device->CreateComputePipeline(
        &fillClusterTriangleInfoShader, &fillClusterTriangleInfoLayout,
        &fillClusterTriangleInfoPush);

    // fill instance descs
    PushConstant fillInstanceDescsPush;
    fillInstanceDescsPush.stage  = ShaderStage::Compute;
    fillInstanceDescsPush.size   = sizeof(FillInstanceDescsPushConstant);
    fillInstanceDescsPush.offset = 0;

    DescriptorSetLayout fillInstanceLayout = {};
    fillInstanceLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    fillInstanceLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline fillInstancePipeline = device->CreateComputePipeline(
        &fillInstanceShader, &fillInstanceLayout, &fillInstanceDescsPush);

    // get blas address offset
    PushConstant getBlasAddressOffsetPush = {};
    getBlasAddressOffsetPush.stage        = ShaderStage::Compute;
    getBlasAddressOffsetPush.size         = 4;
    getBlasAddressOffsetPush.offset       = 0;

    DescriptorSetLayout getBlasAddressOffsetLayout = {};
    getBlasAddressOffsetLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    getBlasAddressOffsetLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline getBlasAddressOffsetPipeline = device->CreateComputePipeline(
        &getBlasAddressOffsetShader, &getBlasAddressOffsetLayout, &getBlasAddressOffsetPush);

#endif

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
    gpuScene.lightFromRender  = params->lightFromRender;
    gpuScene.dxCamera         = params->dxCamera;
    gpuScene.lensRadius       = params->lensRadius;
    gpuScene.dyCamera         = params->dyCamera;
    gpuScene.focalLength      = params->focalLength;
    gpuScene.height           = params->height;
    gpuScene.fov              = params->fov;

    ShaderDebugInfo shaderDebug;

    RTBindingData bindingData;
    bindingData.materialIndex = 0;

    CommandBuffer *transferCmd = device->BeginCommandBuffer(QueueType_Copy);
    GPUBuffer bindingDataBuffer =
        transferCmd
            ->SubmitBuffer(&bindingData, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(RTBindingData))
            .buffer;

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
    int accelBindingIndex      = layout.AddBinding((u32)RTBindings::Accel,
                                                   DescriptorType::AccelerationStructure, flags);
    int imageBindingIndex =
        layout.AddBinding((u32)RTBindings::Image, DescriptorType::StorageImage, flags);

    int sceneBindingIndex =
        layout.AddBinding((u32)RTBindings::Scene, DescriptorType::UniformBuffer, flags);

    int bindingDataBindingIndex = layout.AddBinding((u32)RTBindings::RTBindingData,
                                                    DescriptorType::StorageBuffer, flags);

    int gpuMaterialBindingIndex =
        layout.AddBinding((u32)RTBindings::GPUMaterial, DescriptorType::StorageBuffer, flags);

    int pageTableBindingIndex =
        layout.AddBinding((u32)RTBindings::PageTable, DescriptorType::SampledImage, flags);

    int physicalPagesBindingIndex =
        layout.AddBinding((u32)RTBindings::PhysicalPages, DescriptorType::SampledImage, flags);

    int shaderDebugIndex = layout.AddBinding((u32)RTBindings::ShaderDebugInfo,
                                             DescriptorType::UniformBuffer, flags);

    int dgfHeaderIndex =
        layout.AddBinding((u32)RTBindings::DGFHeaders, DescriptorType::StorageBuffer, flags);
    int dgfBytesIndex =
        layout.AddBinding((u32)RTBindings::DGFBytes, DescriptorType::StorageBuffer, flags);

    int dgfInfoIndex =
        layout.AddBinding((u32)RTBindings::DGFInfo, DescriptorType::StorageBuffer, flags);
    int ptexFaceDataIndex =
        layout.AddBinding((u32)RTBindings::PtexFaceData, DescriptorType::StorageBuffer, flags);

    int feedbackBufferIndex =
        layout.AddBinding((u32)RTBindings::Feedback, DescriptorType::StorageBuffer, flags);

    // TODO: what is this?
    // int counterIndex = layout.AddBinding(8, DescriptorType::StorageBuffer, flags);
    // int nvApiIndex =
    //     layout.AddBinding(NVAPI_SLOT, DescriptorType::StorageBuffer, VK_SHADER_STAGE_ALL);

    layout.AddImmutableSamplers();

    RayTracingState rts = device->CreateRayTracingPipeline(groups, ArrayLength(groups),
                                                           &pushConstant, &layout, 2, true);
    // VkPipeline pipeline = device->CreateComputePipeline(&shader, &layout, &pushConstant);
    // Build clusters
    ScratchArena sceneScratch;

    Bounds *bounds = PushArrayNoZero(sceneScratch.temp.arena, Bounds, numScenes);

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

    struct BLASSceneGPUInfo
    {
        DenseGeometryBuildData data;
        GPUBuffer aabbBuffer;
        // GPUBuffer dgfBuffer;
        // GPUBuffer headerBuffer;
        u32 aabbLength;
        u8 *dgfGeoByteBuffer;
        u8 *dgfShadByteBuffer;

        u32 geoByteBufferLength;
        u32 shadingByteBufferLength;

        PackedDenseGeometryHeader *headers;
        u32 numHeaders;

        u32 blasNumVertices;

        u32 **firstUses;
        u32 **reuses;
        VkAabbPositionsKHR *positions;
        TriangleStripType **types;
        u32 **debugFaceIDs;
        u32 **debugIndices;
        u32 **debugRestartCount;
        u32 **debugRestartHighBit;
    };

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
        }
        gpuMaterials.Push(material);
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

    StaticArray<BLASSceneGPUInfo> blasSceneInfo(arena, blasScenes.Length());

    u32 numAabbs           = 0;
    u32 numTriangles       = 0;
    u32 numVertices        = 0;
    u32 dgfHeaderBytes     = 0;
    u32 dgfGeoBufferBytes  = 0;
    u32 dgfShadBufferBytes = 0;
    u32 totalNumHeaders    = 0;
    u32 totalNumClusters   = 0;

    CommandBuffer *dgfTransferCmd = device->BeginCommandBuffer(QueueType_Copy);
    for (int i = 0; i < blasScenes.Length(); i++)
    {
        ScenePrimitives *scene       = blasScenes[i];
        scene->semaphore             = device->CreateSemaphore();
        scene->semaphore.signalValue = 1;

        BLASSceneGPUInfo info;
        info.data.Init();
        DenseGeometryBuildData &data = info.data;

        RecordAOSSplits record;
        PrimRef *refs = ParallelGenerateMeshRefs<GeometryType::TriangleMesh>(
            sceneScratch.temp.arena, (Mesh *)scene->primitives, scene->numPrimitives, record,
            false);

        Mesh *meshes        = (Mesh *)scene->primitives;
        u32 blasNumVertices = 0;
        for (int meshIndex = 0; meshIndex < scene->numPrimitives; meshIndex++)
        {
            blasNumVertices += meshes[meshIndex].numVertices;
        }

        ClusterBuilder builder(arena, refs);
        builder.BuildClusters(record, true);

        builder.CreateDGFs(scene, &data, (Mesh *)scene->primitives, scene->numPrimitives,
                           sceneBounds);

        for (int j = 0; j < builder.threadClusters.size(); j++)
        {
            for (auto *node = builder.threadClusters[j].l.first; node != 0; node = node->next)
            {
                totalNumClusters += node->count;
            }
        }

        numTriangles += GetNumTriangles(builder);
        numVertices += blasNumVertices;
#ifdef USE_PROCEDURAL_CLUSTER_INTERSECTION
        // Convert primrefs to aabbs, submit to GPU

        StaticArray<VkAabbPositionsKHR> aabbs =
            CreateAABBForNTriangles(sceneScratch.temp.arena, builder, data.numBlocks);

        info.aabbLength = aabbs.Length();
        numAabbs += info.aabbLength;

        info.aabbBuffer =
            dgfTransferCmd
                ->SubmitBuffer(
                    aabbs.data,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    aabbs.Length() * sizeof(VkAabbPositionsKHR))
                .buffer;
        info.positions = aabbs.data;
#endif

        // Combine all blocks together
        info.geoByteBufferLength = data.geoByteBuffer.Length();
        dgfGeoBufferBytes += data.geoByteBuffer.Length();
        info.dgfGeoByteBuffer =
            PushArrayNoZero(sceneScratch.temp.arena, u8, data.geoByteBuffer.Length());
        data.geoByteBuffer.Flatten(info.dgfGeoByteBuffer);

        info.shadingByteBufferLength = data.shadingByteBuffer.Length();
        dgfShadBufferBytes += data.shadingByteBuffer.Length();
        info.dgfShadByteBuffer =
            PushArrayNoZero(sceneScratch.temp.arena, u8, data.shadingByteBuffer.Length());
        data.shadingByteBuffer.Flatten(info.dgfShadByteBuffer);

        // Upload headers
        totalNumHeaders += data.headers.Length();
        info.headers = PushArrayNoZero(sceneScratch.temp.arena, PackedDenseGeometryHeader,
                                       data.headers.Length());
        data.headers.Flatten(info.headers);
        info.numHeaders = data.headers.Length();

        // Debug
        info.types =
            PushArrayNoZero(sceneScratch.temp.arena, TriangleStripType *, data.numBlocks);
        // firstUses    = PushArrayNoZero(sceneScratch.temp.arena, u32 *,
        // data.numBlocks); reuses       = PushArrayNoZero(sceneScratch.temp.arena, u32 *,
        // data.numBlocks);
        info.debugFaceIDs = PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        info.debugIndices = PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        info.debugRestartHighBit =
            PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        info.debugRestartCount =
            PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        int c = 0;
        for (auto *node = data.types.first; node != 0; node = node->next)
        {
            info.types[c++] = node->values;
        }
        c = 0;
        for (auto *node = data.debugFaceIDs.first; node != 0; node = node->next)
        {
            info.debugFaceIDs[c++] = node->values;
        }
        // c = 0;
        // for (auto *node = data.firstUse.first; node != 0; node = node->next)
        // {
        //     firstUses[c++] = node->values;
        // }
        // c = 0;
        // for (auto *node = data.reuse.first; node != 0; node = node->next)
        // {
        //     reuses[c++] = node->values;
        // }
        c = 0;
        for (auto *node = data.debugIndices.first; node != 0; node = node->next)
        {
            info.debugIndices[c++] = node->values;
        }
        c = 0;
        for (auto *node = data.debugRestartHighBitPerDword.first; node != 0; node = node->next)
        {
            info.debugRestartHighBit[c++] = node->values;
        }
        c = 0;
        for (auto *node = data.debugRestartCountPerDword.first; node != 0; node = node->next)
        {
            info.debugRestartCount[c++] = node->values;
        }

        dgfTransferCmd->Signal(scene->semaphore);
        blasSceneInfo.Push(info);
        ReleaseArenaArray(builder.arenas);
    }

    u32 dgfGeoByteOffset  = 0;
    u32 dgfShadByteOffset = 0;
    u32 dgfHeaderOffset   = 0;
    u8 *dgfBytes =
        PushArrayNoZero(sceneScratch.temp.arena, u8, dgfGeoBufferBytes + dgfShadBufferBytes);
    PackedDenseGeometryHeader *headers =
        PushArrayNoZero(sceneScratch.temp.arena, PackedDenseGeometryHeader, totalNumHeaders);

    DGFGeometryInfo *geometryInfo =
        PushArrayNoZero(sceneScratch.temp.arena, DGFGeometryInfo, blasScenes.Length());

    for (int i = 0; i < blasScenes.Length(); i++)
    {
        BLASSceneGPUInfo &info = blasSceneInfo[i];
        MemoryCopy(dgfBytes + dgfGeoByteOffset, info.dgfGeoByteBuffer,
                   info.geoByteBufferLength);
        u32 byteOffset = dgfGeoByteOffset;
        dgfGeoByteOffset += info.geoByteBufferLength;

        MemoryCopy(dgfBytes + dgfGeoBufferBytes + dgfShadByteOffset, info.dgfShadByteBuffer,
                   info.shadingByteBufferLength);
        u32 shadByteOffset = dgfGeoBufferBytes + dgfShadByteOffset;
        dgfShadByteOffset += info.shadingByteBufferLength;

        for (int j = 0; j < info.numHeaders; j++)
        {
            info.headers[j].a += byteOffset;
            info.headers[j].z += shadByteOffset;
        }

        MemoryCopy(headers + dgfHeaderOffset, info.headers,
                   sizeof(PackedDenseGeometryHeader) * info.numHeaders);
        u32 headerOffset = dgfHeaderOffset;
        dgfHeaderOffset += info.numHeaders;

        geometryInfo[i].headerOffset = headerOffset;
    }

    Assert(dgfGeoByteOffset == dgfGeoBufferBytes);
    Assert(dgfShadByteOffset == dgfShadBufferBytes);
    Assert(totalNumHeaders == dgfHeaderOffset);
    GPUBuffer dgfHeaderBuffer =
        dgfTransferCmd
            ->SubmitBuffer(headers, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(PackedDenseGeometryHeader) * totalNumHeaders)
            .buffer;
    device->SetName(&dgfHeaderBuffer, "DGF Header Buffer");
    GPUBuffer dgfByteBuffer = dgfTransferCmd
                                  ->SubmitBuffer(dgfBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                 dgfGeoBufferBytes + dgfShadBufferBytes)
                                  .buffer;
    device->SetName(&dgfByteBuffer, "DGF Byte Buffer");

    GPUBuffer dgfGeometryInfoBuffer =
        dgfTransferCmd
            ->SubmitBuffer(geometryInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(DGFGeometryInfo) * blasScenes.Length())
            .buffer;
    device->SetName(&dgfGeometryInfoBuffer, "DGF Geometry Info Buffer");

    Semaphore sem   = device->CreateSemaphore();
    sem.signalValue = 1;

    dgfTransferCmd->SignalOutsideFrame(sem);

    dgfTransferCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    dgfTransferCmd->FlushBarriers();
    device->SubmitCommandBuffer(dgfTransferCmd);

    Print("num triangles: %u\nnum aabbs: %u\ndgf geo buffer bytes: %u\ndgf shad buffer bytes: "
          "%u\nptex face data bytes: %u\n",
          numTriangles, numAabbs, dgfGeoBufferBytes, dgfShadBufferBytes, ptexFaceDataBytes);

    CommandBuffer *allCommandBuffer = device->BeginCommandBuffer(QueueType_Graphics);
    Semaphore tlasSemaphore         = device->CreateSemaphore();
    GPUAccelerationStructure tlas   = {};

    u32 numInstances = 0;
    for (auto &tlas : tlasScenes)
    {
        numInstances += tlas->numPrimitives;
    }

#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
    GPUBuffer indexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        megabytes(256));
    GPUBuffer vertexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        megabytes(256));
    GPUBuffer clasGlobalsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * 4);

    GPUBuffer decodeClusterDataBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(DecodeClusterData) * totalNumClusters);
    GPUBuffer blasDataBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(BLASData) * 24);
    GPUBuffer buildClusterTriangleInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_TRIANGLE_INFO) * totalNumClusters);
    GPUBuffer buildClusterBottomLevelInfoBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             sizeof(BUILD_CLUSTERS_BOTTOM_LEVEL_INFO) * blasScenes.Length());

    // Group clusters into groups of 256
    u32 totalNumClusterPages = 0;
    for (int i = 0; i < blasScenes.Length(); i++)
    {
        BLASSceneGPUInfo &blasInfo = blasSceneInfo[i];
        u32 numPages =
            (blasInfo.numHeaders + MAX_CLUSTERS_PER_PAGE - 1) / MAX_CLUSTERS_PER_PAGE;
        totalNumClusterPages += numPages;
    }

    StaticArray<ClusterPageData> clusterPages(sceneScratch.temp.arena, totalNumClusterPages);
    u32 clusterOffset = 0;
    for (int blasIndex = 0; blasIndex < blasScenes.Length(); blasIndex++)
    {
        BLASSceneGPUInfo &blasInfo = blasSceneInfo[blasIndex];
        u32 numClusters            = blasInfo.numHeaders;
        u32 numPages      = (numClusters + MAX_CLUSTERS_PER_PAGE - 1) / MAX_CLUSTERS_PER_PAGE;
        u32 clusterOffset = 0;
        for (int pageIndex = 0; pageIndex < numPages; pageIndex++)
        {
            u32 pageNumClusters =
                Min(MAX_CLUSTERS_PER_PAGE, numClusters - pageIndex * MAX_CLUSTERS_PER_PAGE);
            ClusterPageData pageData;
            pageData.clusterStart = clusterOffset;
            pageData.clusterCount = pageNumClusters;
            pageData.blasIndex    = blasIndex;
            clusterPages.Push(pageData);

            clusterOffset += pageNumClusters;
        }
    }

    TransferBuffer clusterPagesBuffer =
        allCommandBuffer->SubmitBuffer(clusterPages.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       sizeof(ClusterPageData) * clusterPages.size());

    u64 indexBufferAddress  = device->GetDeviceAddress(indexBuffer.buffer);
    u64 vertexBufferAddress = device->GetDeviceAddress(vertexBuffer.buffer);

    FillClusterTriangleInfoPushConstant fillPc;
    fillPc.numClusters                     = totalNumHeaders;
    fillPc.indexBufferBaseAddressLowBits   = indexBufferAddress & 0xffffffff;
    fillPc.indexBufferBaseAddressHighBits  = (indexBufferAddress >> 32u) & 0xffffffff;
    fillPc.vertexBufferBaseAddressLowBits  = vertexBufferAddress & 0xffffffff;
    fillPc.vertexBufferBaseAddressHighBits = (vertexBufferAddress >> 32u) & 0xffffffff;

    {
        allCommandBuffer->ClearBuffer(&blasDataBuffer);
        allCommandBuffer->ClearBuffer(&clasGlobalsBuffer);
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        allCommandBuffer->FlushBarriers();

        // Write the BUILD_CLUSTERS_TRIANGLE_INFO descriptors on the GPU
        allCommandBuffer->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
                                       fillClusterTriangleInfoPipeline);
        DescriptorSet ds = fillClusterTriangleInfoLayout.CreateDescriptorSet();
        ds.Bind(&clusterPagesBuffer.buffer)
            .Bind(&buildClusterTriangleInfoBuffer)
            .Bind(&blasDataBuffer)
            .Bind(&decodeClusterDataBuffer)
            .Bind(&clasGlobalsBuffer);
        allCommandBuffer->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                             fillClusterTriangleInfoLayout.pipelineLayout);

        allCommandBuffer->PushConstants(&fillClusterTriangleInfoPush, &fillPc,
                                        fillClusterTriangleInfoLayout.pipelineLayout);

        allCommandBuffer->Dispatch(totalNumClusterPages, 1, 1);
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        allCommandBuffer->FlushBarriers();
    }

    {
        // Decode the clusters
        allCommandBuffer->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
                                       decodeDgfClustersPipeline);
        DescriptorSet ds = decodeDgfClustersLayout.CreateDescriptorSet();
        ds.Bind(&indexBuffer)
            .Bind(&vertexBuffer)
            .Bind(&decodeClusterDataBuffer)
            .Bind(&dgfByteBuffer)
            .Bind(&dgfHeaderBuffer);

        allCommandBuffer->PushConstants(&decodeDgfClustersPush, &fillPc,
                                        decodeDgfClustersLayout.pipelineLayout);
        allCommandBuffer->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                             decodeDgfClustersLayout.pipelineLayout);
        allCommandBuffer->Dispatch(totalNumClusters, 1, 1);

        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_ACCESS_2_SHADER_WRITE_BIT,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        allCommandBuffer->FlushBarriers();
    }

    GPUBuffer clusterAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * totalNumClusters);
    GPUBuffer clusterAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * totalNumClusters);
    {
        // Build the CLAS

        allCommandBuffer->BuildCLAS(&buildClusterTriangleInfoBuffer, &clusterAccelAddresses,
                                    &clusterAccelSizes, &clasGlobalsBuffer,
                                    sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, totalNumClusters,
                                    numTriangles, numVertices);
    }

    // TODO: Compact the CLAS over BLAS
    {
    }

    // Set up bottom level input
    u32 numBlas = blasSceneInfo.Length();

    GPUBuffer blasClasAddressBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * totalNumClusters);

    GPUBuffer blasAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizeof(u64) * numBlas);

    u64 blasClasAddressBufferAddress = device->GetDeviceAddress(blasClasAddressBuffer.buffer);
    FillClusterBottomLevelInfoPushConstant pc;
    pc.blasCount                = numBlas;
    pc.arrayBaseAddressLowBits  = blasClasAddressBufferAddress & (~0u);
    pc.arrayBaseAddressHighBits = blasClasAddressBufferAddress >> 32u;

    {
        // Calculate where clas addresses should be written for each blas
        allCommandBuffer->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
                                       getBlasAddressOffsetPipeline);

        DescriptorSet ds = getBlasAddressOffsetLayout.CreateDescriptorSet();
        ds.Bind(&blasDataBuffer).Bind(&clasGlobalsBuffer);
        allCommandBuffer->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                             getBlasAddressOffsetLayout.pipelineLayout);

        allCommandBuffer->PushConstants(&getBlasAddressOffsetPush, &pc,
                                        getBlasAddressOffsetLayout.pipelineLayout);

        allCommandBuffer->Dispatch((numBlas + 31) / 32, 1, 1);
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        allCommandBuffer->FlushBarriers();
    }

    // Write the clas addresses to a new buffer
    {
        allCommandBuffer->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
                                       fillBlasAddressArrayPipeline);
        DescriptorSet ds = fillBlasAddressArrayLayout.CreateDescriptorSet();
        ds.Bind(&clusterPagesBuffer.buffer)
            .Bind(&blasDataBuffer)
            .Bind(&clusterAccelAddresses)
            .Bind(&blasClasAddressBuffer);

        allCommandBuffer->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                             fillBlasAddressArrayLayout.pipelineLayout);
        allCommandBuffer->Dispatch((totalNumClusters + 31) / 32, 1, 1);
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        allCommandBuffer->FlushBarriers();
    }

    {
        // Fill out the BUILD_CLUSTERS_BOTTOM_LEVEL_INFO descriptors
        allCommandBuffer->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
                                       fillClusterBLASInfoPipeline);

        DescriptorSet ds = fillClusterBLASInfoLayout.CreateDescriptorSet();
        ds.Bind(&blasDataBuffer).Bind(&buildClusterBottomLevelInfoBuffer);
        allCommandBuffer->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                             fillClusterBLASInfoLayout.pipelineLayout);

        allCommandBuffer->PushConstants(&fillClusterBLASInfoPush, &pc,
                                        fillClusterBLASInfoLayout.pipelineLayout);

        allCommandBuffer->Dispatch(
            (blasScenes.Length() + FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE - 1) /
                FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE,
            1, 1);

        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_ACCESS_2_SHADER_WRITE_BIT,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        // allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        //                           VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        //                           VK_ACCESS_2_TRANSFER_WRITE_BIT,
        //                           VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        allCommandBuffer->FlushBarriers();
    }

    {
        // Build the BLASes
        allCommandBuffer->BuildClusterBLAS(
            &buildClusterBottomLevelInfoBuffer, &blasAccelAddresses, &clasGlobalsBuffer,
            sizeof(u32) * GLOBALS_BLAS_COUNT_INDEX, totalNumClusters);
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                  VK_ACCESS_2_SHADER_READ_BIT);
        allCommandBuffer->FlushBarriers();
    }

    // Build the TLAS over BLAS
    {
        GPUBuffer tlasBuffer;
        if (tlasScenes.Length() != 0)
        {
            Assert(0);
            // Assert(tlasScenes.Length() == 1);
            // for (int i = 0; i < tlasScenes.Length(); i++)
            // {
            //     ScenePrimitives *scene = tlasScenes[i];
            //     device->BeginEvent(allCommandBuffer, "TLAS Build");
            //     Instance *instances = (Instance *)scene->primitives;
            //     GPUAccelerationStructurePayload payload =
            //         allCommandBuffer->BuildTLAS(instances, scene->numPrimitives,
            //                                     scene->affineTransforms,
            //                                     scene->childScenes);
            //
            //     scene->gpuBVH = payload.as;
            //     tlas          = scene->gpuBVH;
            //
            //     tlasSemaphore.signalValue = 1;
            //     allCommandBuffer->SignalOutsideFrame(tlasSemaphore);
            // }
        }
        else
        {
            Assert(numScenes == 1);
            device->BeginEvent(allCommandBuffer, "TLAS Build");

            ScenePrimitives *scene     = blasScenes[0];
            Instance instance          = {};
            ScenePrimitives *baseScene = &GetScene()->scene;

            tlasBuffer =
                allCommandBuffer
                    ->CreateTLASInstances(&instance, 1, &params->renderFromWorld, &baseScene)
                    .buffer;

            tlasSemaphore.signalValue = 1;
            allCommandBuffer->SignalOutsideFrame(tlasSemaphore);
        }

        FillInstanceDescsPushConstant pc;
        pc.numInstances = numInstances;

        allCommandBuffer->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, fillInstancePipeline);

        DescriptorSet ds = fillInstanceLayout.CreateDescriptorSet();
        ds.Bind(&blasAccelAddresses).Bind(&tlasBuffer);
        allCommandBuffer->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                             fillInstanceLayout.pipelineLayout);

        allCommandBuffer->PushConstants(&fillInstanceDescsPush, &pc,
                                        fillInstanceLayout.pipelineLayout);
        allCommandBuffer->Dispatch((numInstances + 31) / 32, 1, 1);
    }

#else
    // TODO: new command buffers have to wait on ones from the previous depth
    // also this doesn't work if there's actually a nontrivial TLAS

    StaticArray<GPUAccelerationStructurePayload> uncompactedPayloads(sceneScratch.temp.arena,
                                                                     blasScenes.Length());
    u32 compactedBLASSize   = 0;
    u32 uncompactedBLASSize = 0;
    for (int i = 0; i < blasScenes.Length(); i++)
    {
        ScenePrimitives *scene = blasScenes[i];

        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics);
        device->BeginEvent(cmd, "BLAS Build");
        int bvhCount = 0;

        cmd->Wait(scene->semaphore);
        scene->semaphore.signalValue++;
        cmd->SignalOutsideFrame(scene->semaphore);

        cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
        cmd->FlushBarriers();

        GPUAccelerationStructurePayload payload =
            cmd->BuildCustomBLAS(&blasSceneInfo[i].aabbBuffer, blasSceneInfo[i].aabbLength);
        uncompactedPayloads.Push(payload);

        uncompactedBLASSize += payload.as.buffer.size;
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();

        // Compact
        QueryPool queryPool = cmd->GetCompactionSizes(&payload);
        device->SubmitCommandBuffer(cmd);
        device->EndEvent(cmd);

        allCommandBuffer->Wait(scene->semaphore);
        scene->semaphore.signalValue++;
        allCommandBuffer->Signal(scene->semaphore);

        scene->gpuBVH = allCommandBuffer->CompactAS(queryPool, &payload);
        compactedBLASSize += scene->gpuBVH.buffer.size;
        device->DestroyBuffer(&payload.scratch);
    }

    Print("compacted blas size: %u, uncompacted sze: %u\n", compactedBLASSize,
          uncompactedBLASSize);

    // Set the instance ids of each scene
    int runningTotal = 0;
    for (int i = 0; i < blasScenes.Length(); i++)
    {
        ScenePrimitives *scene = blasScenes[i];
        scene->gpuInstanceID   = runningTotal++;
    }

    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    if (tlasScenes.Length() != 0)
    {
        Assert(tlasScenes.Length() == 1);
        for (int i = 0; i < tlasScenes.Length(); i++)
        {
            ScenePrimitives *scene = tlasScenes[i];
            device->BeginEvent(allCommandBuffer, "TLAS Build");
            Instance *instances                     = (Instance *)scene->primitives;
            GPUAccelerationStructurePayload payload = allCommandBuffer->BuildTLAS(
                instances, scene->numPrimitives, scene->affineTransforms, scene->childScenes);

            scene->gpuBVH = payload.as;
            tlas          = scene->gpuBVH;

            tlasSemaphore.signalValue = 1;
            allCommandBuffer->SignalOutsideFrame(tlasSemaphore);
        }
    }
    else
    {
        Assert(numScenes == 1);
        device->BeginEvent(allCommandBuffer, "TLAS Build");

        ScenePrimitives *scene     = blasScenes[0];
        Instance instance          = {};
        ScenePrimitives *baseScene = &GetScene()->scene;

        GPUBuffer tlasBuffer =
            allCommandBuffer
                ->CreateTLASInstances(&instance, 1, &params->renderFromWorld, &baseScene)
                .buffer;
        tlas = allCommandBuffer->BuildTLAS(&tlasBuffer, 1).as;

        tlasSemaphore.signalValue = 1;
        allCommandBuffer->SignalOutsideFrame(tlasSemaphore);
    }
#endif
    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                              VK_ACCESS_2_SHADER_READ_BIT);
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
            OS_GetMousePos(params->window, shaderDebug.mousePos.x, shaderDebug.mousePos.y);
        }
        u32 dispatchDimX =
            (params->width + PATH_TRACE_NUM_THREADS_X - 1) / PATH_TRACE_NUM_THREADS_X;
        u32 dispatchDimY =
            (params->height + PATH_TRACE_NUM_THREADS_Y - 1) / PATH_TRACE_NUM_THREADS_Y;
        gpuScene.dispatchDimX = dispatchDimX;
        gpuScene.dispatchDimY = dispatchDimY;

        device->BeginFrame();

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

#ifdef USE_PROCEDURAL_CLUSTER_INTERSECTION
            for (int i = 0; i < uncompactedPayloads.Length(); i++)
            {
                auto &payload = uncompactedPayloads[i];
                device->DestroyAccelerationStructure(&payload.as);
            }
#endif
            for (int i = 0; i < blasSceneInfo.Length(); i++)
            {
                device->DestroyBuffer(&blasSceneInfo[i].aabbBuffer);
            }
            envMapBindlessIndex = device->BindlessIndex(&gpuEnvMap);
        }

        u32 currentBuffer = device->GetCurrentBuffer();

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

        RayPushConstant pc;
        pc.envMap   = envMapBindlessIndex;
        pc.frameNum = (u32)device->frameCount;
        pc.width    = envMap->width;
        pc.height   = envMap->height;

        MemoryCopy(sceneTransferBuffers[currentBuffer].mappedPtr, &gpuScene, sizeof(GPUScene));
        cmd->SubmitTransfer(&sceneTransferBuffers[currentBuffer]);
        MemoryCopy(shaderDebugBuffers[currentBuffer].mappedPtr, &shaderDebug,
                   sizeof(ShaderDebugInfo));
        cmd->SubmitTransfer(&shaderDebugBuffers[currentBuffer]);

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
        descriptorSet.Bind(accelBindingIndex, &tlas.as)
            .Bind(imageBindingIndex, image)
            .Bind(sceneBindingIndex, &sceneTransferBuffers[currentBuffer].buffer)
            .Bind(bindingDataBindingIndex, &bindingDataBuffer)
            .Bind(gpuMaterialBindingIndex, &materialBuffer)
            .Bind(pageTableBindingIndex, &virtualTextureManager.pageTable)
            .Bind(physicalPagesBindingIndex, &virtualTextureManager.gpuPhysicalPool)
            .Bind(dgfHeaderIndex, &dgfHeaderBuffer)
            .Bind(dgfBytesIndex, &dgfByteBuffer)
            .Bind(dgfInfoIndex, &dgfGeometryInfoBuffer)
            .Bind(ptexFaceDataIndex, &faceDataBuffer)
            .Bind(shaderDebugIndex, &shaderDebugBuffers[currentBuffer].buffer)
            .Bind(feedbackBufferIndex,
                  &virtualTextureManager.feedbackBuffers[currentBuffer].buffer);
        // .Bind(counterIndex, &counterBuffer)
        // .Bind(nvApiIndex, &nvapiBuffer);
        // .Bind(aabbIndex, &aabbBuffer);

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
