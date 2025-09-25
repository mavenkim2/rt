#include "bit_packing.h"
#include "camera.h"
#include "debug.h"
#include "graphics/render_graph.h"
#include "graphics/vulkan.h"
#include "integrate.h"
#include "gpu_scene.h"
#include "math/simd_base.h"
#include "radix_sort.h"
#include "random.h"
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
#include "scene/scene.h"
#include "win32.h"
#include "graphics/ptex.h"
#include "virtual_geometry/virtual_geometry_manager.h"

#include "../../third_party/streamline/include/sl.h"

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

static void FlattenInstances(ScenePrimitives *currentScene, const AffineSpace &transform,
                             Array<Instance> &instances, Array<AffineSpace> &transforms)
{
    Instance *sceneInstances = (Instance *)currentScene->primitives;
    for (u32 i = 0; i < currentScene->numPrimitives; i++)
    {
        Instance &instance = sceneInstances[i];
        AffineSpace newTransform =
            transform * currentScene->affineTransforms[instance.transformIndex];
        ScenePrimitives *nextScene = currentScene->childScenes[instance.id];

        if (nextScene->geometryType == GeometryType::Instance)
        {
            FlattenInstances(nextScene, newTransform, instances, transforms);
        }
        else
        {
            u32 resourceID          = nextScene->sceneIndex;
            Instance instance       = {};
            instance.transformIndex = transforms.Length();
            instance.id             = resourceID;

            transforms.Push(newTransform);
            instances.Push(instance);
        }
    }
}

// https://alextardif.com/TAA.html
static float Halton(uint32_t i, uint32_t b)
{
    float f = 1.0f;
    float r = 0.0f;

    while (i > 0)
    {
        f /= static_cast<float>(b);
        r = r + f * static_cast<float>(i % b);
        i = static_cast<uint32_t>(floorf(static_cast<float>(i) / static_cast<float>(b)));
    }

    return r;
}

void Render(RenderParams2 *params, int numScenes, Image *envMap)
{

    ScenePrimitives **scenes = GetScenes();
    NvAPI_Status status      = NvAPI_Initialize();
    Assert(status == NVAPI_OK);
    // Compile shaders
    Shader fillInstanceShader;
    Shader prepareIndirectShader;

    Shader testShader;
    Shader mvecShader;

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

        string fillInstanceShaderName = "../src/shaders/fill_instance_descs.spv";
        string fillInstanceShaderData = OS_ReadFile(arena, fillInstanceShaderName);
        fillInstanceShader = device->CreateShader(ShaderStage::Compute, "fill instance descs",
                                                  fillInstanceShaderData);

        string prepareIndirectName = "../src/shaders/prepare_indirect.spv";
        string prepareIndirectData = OS_ReadFile(arena, prepareIndirectName);
        prepareIndirectShader = device->CreateShader(ShaderStage::Compute, "prepare indirect",
                                                     prepareIndirectData);

        string testShaderName = "../src/shaders/test.spv";
        string testShaderData = OS_ReadFile(arena, testShaderName);
        testShader = device->CreateShader(ShaderStage::Compute, "test shader", testShaderData);

        string mvecShaderName = "../src/shaders/calculate_motion_vectors.spv";
        string mvecShaderData = OS_ReadFile(arena, mvecShaderName);
        mvecShader = device->CreateShader(ShaderStage::Compute, "mvec shader", mvecShaderData);
    }

    // Compile pipelines

    // fill instance descs
    DescriptorSetLayout fillInstanceLayout = {};
    for (int i = 0; i <= 5; i++)
    {
        fillInstanceLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                      VK_SHADER_STAGE_COMPUTE_BIT);
    }
    fillInstanceLayout.AddBinding(6, DescriptorType::UniformBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline fillInstancePipeline = device->CreateComputePipeline(
        &fillInstanceShader, &fillInstanceLayout, 0, "fill instance descs");

    // prepare indirect
    DescriptorSetLayout prepareIndirectLayout = {};
    prepareIndirectLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline prepareIndirectPipeline = device->CreateComputePipeline(
        &prepareIndirectShader, &prepareIndirectLayout, 0, "prepare indirect");

    DescriptorSetLayout testLayout = {};
    testLayout.AddBinding(0, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    testLayout.AddBinding(1, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    testLayout.AddBinding((u32)RTBindings::ClusterPageData, DescriptorType::StorageBuffer,
                          VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline testPipeline =
        device->CreateComputePipeline(&testShader, &testLayout, 0, "test");

    DescriptorSetLayout mvecLayout = {};
    mvecLayout.AddBinding(0, DescriptorType::SampledImage, VK_SHADER_STAGE_COMPUTE_BIT);
    mvecLayout.AddBinding(1, DescriptorType::StorageImage, VK_SHADER_STAGE_COMPUTE_BIT);
    mvecLayout.AddBinding(2, DescriptorType::UniformBuffer, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline mvecPipeline =
        device->CreateComputePipeline(&mvecShader, &mvecLayout, 0, "mvec");

    u32 targetWidth;
    u32 targetHeight;

    device->GetDLSSTargetDimensions(targetWidth, targetHeight);
    // targetWidth  = 1920;
    // targetHeight = 1080;

    Swapchain swapchain = device->CreateSwapchain(params->window, VK_FORMAT_R8G8B8A8_SRGB,
                                                  params->width, params->height);

    PushConstant pushConstant;
    pushConstant.stage  = ShaderStage::Raygen | ShaderStage::Miss;
    pushConstant.offset = 0;
    pushConstant.size   = sizeof(RayPushConstant);

    Semaphore submitSemaphore = device->CreateSemaphore();

    Mat4 rasterFromNDC = Scale(Vec3f(f32(targetWidth), -f32(targetHeight), 1.f)) *
                         Scale(Vec3f(1.f / 2.f, 1.f / 2.f, 1.f)) *
                         Translate(Vec3f(1.f, -1.f, 0.f));
    Mat4 rasterFromCamera = rasterFromNDC * params->NDCFromCamera;
    Mat4 cameraFromRaster = Inverse(rasterFromCamera);

    // Transfer data to GPU
    GPUScene gpuScene;
    gpuScene.clipFromRender   = params->NDCFromCamera * params->cameraFromRender;
    gpuScene.cameraFromRaster = cameraFromRaster;
    gpuScene.renderFromCamera = params->renderFromCamera;
    gpuScene.cameraFromRender = params->cameraFromRender;
    gpuScene.lightFromRender  = params->lightFromRender;
    gpuScene.lodScale         = 0.5f * params->NDCFromCamera[1][1] * params->height;
    gpuScene.dxCamera         = params->dxCamera;
    gpuScene.lensRadius       = params->lensRadius;
    gpuScene.dyCamera         = params->dyCamera;
    gpuScene.focalLength      = params->focalLength;
    gpuScene.height           = params->height;
    gpuScene.fov              = params->fov;
    gpuScene.p22              = rasterFromCamera[2][2];
    gpuScene.p23              = rasterFromCamera[3][2];

    ShaderDebugInfo shaderDebug;

    RenderGraph *graph = PushStructConstruct(arena, RenderGraph)();
    SetRenderGraph(graph);
    RenderGraph *rg = GetRenderGraph();

    CommandBuffer *transferCmd = device->BeginCommandBuffer(QueueType_Copy);

    ImageDesc gpuEnvMapDesc(ImageType::Type2D, envMap->width, envMap->height, 1, 1, 1,
                            VK_FORMAT_R8G8B8A8_SRGB, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_TILING_OPTIMAL);

    TransferBuffer gpuEnvMapTransfer =
        transferCmd->SubmitImage(envMap->contents, gpuEnvMapDesc);
    GPUImage gpuEnvMap = gpuEnvMapTransfer.image;
    ImageDesc depthBufferDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                              VK_FORMAT_R32_SFLOAT, MemoryUsage::GPU_ONLY,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle depthBuffer =
        rg->CreateImageResource("depth buffer image", depthBufferDesc);
    ImageDesc targetUavDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                            VK_FORMAT_R8G8B8A8_UNORM, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle image = rg->CreateImageResource("target image", targetUavDesc);

    ImageDesc imageOutDesc(ImageType::Type2D, params->width, params->height, 1, 1, 1,
                           VK_FORMAT_R8G8B8A8_UNORM, MemoryUsage::GPU_ONLY,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                           VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle imageOut = rg->CreateImageResource("image out", imageOutDesc);

    ImageDesc normalRoughnessDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                                  VK_FORMAT_R32G32B32A32_SFLOAT, MemoryUsage::GPU_ONLY,
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                  VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle normalRoughness =
        rg->CreateImageResource("normal roughness image", normalRoughnessDesc);

    ImageDesc diffuseAlbedoDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                                VK_FORMAT_R8G8B8A8_UNORM, MemoryUsage::GPU_ONLY,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle diffuseAlbedo =
        rg->CreateImageResource("diffuse albedo image", diffuseAlbedoDesc);

    ImageDesc specularAlbedoDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                                 VK_FORMAT_R8G8B8A8_UNORM, MemoryUsage::GPU_ONLY,
                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle specularAlbedo =
        rg->CreateImageResource("specular albedo image", specularAlbedoDesc);

    ImageDesc specularHitDistanceDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                                      VK_FORMAT_R32_SFLOAT, MemoryUsage::GPU_ONLY,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                      VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle specularHitDistance =
        rg->CreateImageResource("specular hit distance image", specularHitDistanceDesc);

    ImageDesc motionVectorDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                               VK_FORMAT_R32G32_SFLOAT, MemoryUsage::GPU_ONLY,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle motionVectorBuffer =
        rg->CreateImageResource("motion vector image", motionVectorDesc);

    transferCmd->Barrier(&gpuEnvMap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transferCmd->FlushBarriers();

    submitSemaphore.signalValue = 1;
    transferCmd->SignalOutsideFrame(submitSemaphore);
    device->SubmitCommandBuffer(transferCmd);
    device->Wait(submitSemaphore);
    device->DestroyBuffer(&gpuEnvMapTransfer.stagingBuffer);

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
    // layout.AddBinding((u32)RTBindings::Feedback, DescriptorType::StorageBuffer, flags);
    // layout.AddBinding(13, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(14, DescriptorType::StorageBuffer, flags);
    // layout.AddBinding(15, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(16, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(17, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(18, DescriptorType::StorageBuffer, flags);
    // layout.AddBinding(19, DescriptorType::StorageBuffer, flags);
    // layout.AddBinding(20, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(21, DescriptorType::StorageImage, flags);
    layout.AddBinding(22, DescriptorType::StorageImage, flags);
    layout.AddBinding(23, DescriptorType::StorageImage, flags);
    layout.AddBinding(24, DescriptorType::StorageImage, flags);
    layout.AddBinding(25, DescriptorType::StorageImage, flags);

    layout.AddImmutableSamplers();

    RayTracingState rts = device->CreateRayTracingPipeline(groups, ArrayLength(groups),
                                                           &pushConstant, &layout, 3, true);
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
                    bounds[i] = scenes[i]->GetSceneBounds();
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

        Mesh *meshes = (Mesh *)scene->primitives;
        Assert(scene->geometryType == GeometryType::TriangleMesh);
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
    // virtualTextureManager.ClearTextures(tileCmd);

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

#if 0
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
#endif

    u32 ptexFaceDataBytes  = 8;
    u8 *faceDataByteBuffer = PushArrayNoZero(sceneScratch.temp.arena, u8, ptexFaceDataBytes);
    u32 ptexOffset         = 0;
#if 0
    for (int i = 0; i < numHandles; i++)
    {
        RequestHandle &handle = handles[i];
        GPUTextureInfo &info  = gpuTextureInfo[handle.ptexIndex];
        Assert(info.faceDataOffset == ptexOffset);
        MemoryCopy(faceDataByteBuffer + ptexOffset, info.packedFaceData, info.packedDataSize);
        ptexOffset += info.packedDataSize;
    }
#endif

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
        gpuMaterials.Push(material);
    }
    GPUBuffer materialBuffer =
        tileCmd
            ->SubmitBuffer(gpuMaterials.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(GPUMaterial) * gpuMaterials.Length())
            .buffer;

    // Assert(ptexOffset == ptexFaceDataBytes);
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

    u32 numInstances = 0;
    u32 numBlas      = blasScenes.Length();
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

    // Virtual geometry initialization
    CommandBuffer *dgfTransferCmd = device->BeginCommandBuffer(QueueType_Compute);

    VirtualGeometryManager virtualGeometryManager(dgfTransferCmd, sceneScratch.temp.arena);
    StaticArray<AABB> blasSceneBounds(sceneScratch.temp.arena, numBlas);
    StaticArray<string> virtualGeoFilenames(sceneScratch.temp.arena, numBlas);
    HashIndex filenameHash(sceneScratch.temp.arena, NextPowerOfTwo(numBlas),
                           NextPowerOfTwo(numBlas));

    for (int sceneIndex = 0; sceneIndex < numBlas; sceneIndex++)
    {
        ScenePrimitives *scene    = blasScenes[sceneIndex];
        string virtualGeoFilename = scene->virtualGeoFilename;
        u32 hash                  = Hash(virtualGeoFilename);

        int resourceID = -1;
        for (int hashIndex = filenameHash.FirstInHash(hash); hashIndex != -1;
             hashIndex     = filenameHash.NextInHash(hashIndex))
        {
            if (virtualGeoFilenames[hashIndex] == virtualGeoFilename)
            {
                resourceID = hashIndex;
                break;
            }
        }
        if (resourceID != -1)
        {
            scene->sceneIndex = resourceID;
            continue;
        }

        scene->sceneIndex = virtualGeoFilenames.Length();
        virtualGeoFilenames.Push(virtualGeoFilename);
        filenameHash.AddInHash(hash, scene->sceneIndex);
        virtualGeometryManager.AddNewMesh(sceneScratch.temp.arena, dgfTransferCmd,
                                          virtualGeoFilename);

        auto &meshInfo =
            virtualGeometryManager.meshInfos[virtualGeometryManager.meshInfos.Length() - 1];
        Vec3f boundsMin = meshInfo.boundsMin;
        Vec3f boundsMax = meshInfo.boundsMax;

        AABB aabb;
        aabb.minX = boundsMin[0];
        aabb.minY = boundsMin[1];
        aabb.minZ = boundsMin[2];
        aabb.maxX = boundsMax[0];
        aabb.maxY = boundsMax[1];
        aabb.maxZ = boundsMax[2];
        blasSceneBounds.Push(aabb);
    }
    virtualGeometryManager.FinalizeResources(dgfTransferCmd);

    Semaphore sem   = device->CreateSemaphore();
    sem.signalValue = 1;
    dgfTransferCmd->SignalOutsideFrame(sem);
    dgfTransferCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_NONE);
    dgfTransferCmd->FlushBarriers();

    device->SubmitCommandBuffer(dgfTransferCmd);

    device->Wait(sem);

    CommandBuffer *allCommandBuffer = device->BeginCommandBuffer(QueueType_Graphics);
    allCommandBuffer->Wait(sem);
    Semaphore tlasSemaphore       = device->CreateSemaphore();
    GPUAccelerationStructure tlas = {};

    // Build the TLAS over BLAS
    StaticArray<Instance> instances;
    StaticArray<AffineSpace> instanceTransforms;

#if 0
    GPUBuffer tlasBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        virtualGeometryManager.maxInstances * sizeof(VkAccelerationStructureInstanceKHR));

    u32 tlasScratchSize, tlasAccelSize;
    device->GetTLASBuildSizes(1u << 21u, tlasScratchSize, tlasAccelSize);
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
#endif

    if (tlasScenes.Length() >= 1)
    {
        ScratchArena scratch(&sceneScratch.temp.arena, 1);
        ScenePrimitives *scene = tlasScenes[0];
        Array<Instance> flattenedInstances(scratch.temp.arena,
                                           scene->numPrimitives * tlasScenes.Length());
        Array<AffineSpace> flattenedTransforms(scratch.temp.arena,
                                               scene->numPrimitives * tlasScenes.Length());
        AffineSpace transform = AffineSpace::Identity();

        FlattenInstances(scene, transform, flattenedInstances, flattenedTransforms);
        instances = StaticArray<Instance>(sceneScratch.temp.arena, flattenedInstances.Length(),
                                          flattenedInstances.Length());
        instanceTransforms =
            StaticArray<AffineSpace>(sceneScratch.temp.arena, flattenedTransforms.Length(),
                                     flattenedTransforms.Length());
        MemoryCopy(instances.data, flattenedInstances.data,
                   sizeof(Instance) * instances.Length());
        MemoryCopy(instanceTransforms.data, flattenedTransforms.data,
                   sizeof(AffineSpace) * flattenedTransforms.Length());
    }
    else
    {
        instances            = StaticArray<Instance>(sceneScratch.temp.arena, 1);
        instanceTransforms   = StaticArray<AffineSpace>(sceneScratch.temp.arena, 1);
        Instance instance    = {};
        AffineSpace identity = AffineSpace::Identity();
        instances.Push(instance);
        instanceTransforms.Push(identity);
        Assert(numInstances == 1);
        Assert(numScenes == 1);
    }

    virtualGeometryManager.Test(sceneScratch.temp.arena, allCommandBuffer, instances,
                                instanceTransforms);
    // virtualGeometryManager.AllocateInstances(gpuInstances);

    // TransferBuffer gpuInstancesBuffer =
    //     allCommandBuffer->SubmitBuffer(gpuInstances.data,
    //     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    //                                    sizeof(GPUInstance) * gpuInstances.Length());

    // allCommandBuffer->SubmitBuffer(
    //     &virtualGeometryManager.instanceRefBuffer,
    //     virtualGeometryManager.newInstanceRefs.data, sizeof(InstanceRef) *
    //     virtualGeometryManager.newInstanceRefs.Length());

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

    device->Wait(tlasSemaphore);
    device->DestroyBuffer(&virtualGeometryManager.blasProxyScratchBuffer);
    device->DestroyBuffer(&virtualGeometryManager.mergedInstancesAABBBuffer);

    f32 frameDt = 1.f / 60.f;
    int envMapBindlessIndex;

    ViewCamera camera = {};
    // camera.pitch      = 0.0536022522f;
    // camera.yaw        = 1.15505993f;
    // camera.position   = Vec3f(5518.85205f, 1196.14368f, -11135.9834f);
    // camera.forward    = Vec3f(0.403283596f, 0.0535765812f, -0.913505197f);
    // camera.right      = Vec3f(0.911113858f, 0.0692767054f, 0.406290948f);

#if 1
    camera.position = Vec3f(0);
    // camera.position = Vec3f(4892.06055f, 767.444824f, -11801.2275f);
    // camera.position = Vec3f(5128.51562f, 1104.60583f, -6173.79395f);
    camera.forward = Normalize(params->look - params->pCamera);
    // camera.forward = Vec3f(-.290819466f, .091174677f, .9524323811f);
    camera.right = Normalize(Cross(camera.forward, params->up));

    camera.pitch = ArcSin(camera.forward.y);
    camera.yaw   = -Atan2(camera.forward.z, camera.forward.x);
#endif

    Vec3f baseForward = camera.forward;
    Vec3f baseRight   = camera.right;

    TransferBuffer sceneTransferBuffers[2] = {
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene)),
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene))};

    ResourceHandle sceneTransferBufferHandles[2] = {
        rg->RegisterExternalResource("scene buffer 0", &sceneTransferBuffers[0].buffer),
        rg->RegisterExternalResource("scene buffer 1", &sceneTransferBuffers[1].buffer),
    };

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

    Semaphore frameSemaphore = device->CreateSemaphore();

    // GPUBuffer readback  = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    //                                            virtualGeometryManager.clasGlobalsBuffer.size,
    //                                            MemoryUsage::GPU_TO_CPU);

    Semaphore transferSem   = device->CreateSemaphore();
    transferSem.signalValue = 1;

    u32 testCount = 0;
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

        Mat4 clipToPrevClip;
        Mat4 prevClipToClip;
        u32 numPhases     = 8 * u32(Ceil(Sqr(params->height / (f32)targetHeight)));
        f32 haltonSampleX = Halton((device->frameCount + 1) % numPhases, 2);
        f32 haltonSampleY = Halton((device->frameCount + 1) % numPhases, 3);
        float haltonX     = 2.f * haltonSampleX - 1.f;
        float haltonY     = 2.f * haltonSampleY - 1.f;
        float jitterX     = (.5f * haltonX / targetWidth);
        float jitterY     = (.5f * haltonY / targetHeight);

        float outJitterX = .5f * (haltonSampleX - 0.5f);
        float outJitterY = .5f * (0.5f - haltonSampleY);

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

            if (Length(camera.right) < 1e-8f)
            {
                // camera.right = Normalize(Cross
            }

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
                AffineSpace(axis, Vec3f(0)) * Translate(-camera.position);
            AffineSpace renderFromCamera = Inverse(cameraFromRender);

            Mat4 clipFromCamera = params->NDCFromCamera;
            clipFromCamera[2][0] -= jitterX;
            clipFromCamera[2][1] -= jitterY;

            clipToPrevClip = params->NDCFromCamera * Mat4(gpuScene.cameraFromRender) *
                             Mat4(renderFromCamera) * params->cameraFromClip;
            prevClipToClip = Inverse(clipToPrevClip);

            gpuScene.cameraP          = camera.position;
            gpuScene.renderFromCamera = renderFromCamera;
            gpuScene.cameraFromRender = cameraFromRender;
            gpuScene.prevClipFromClip = clipToPrevClip;
            gpuScene.clipFromRender   = clipFromCamera * Mat4(cameraFromRender);
            gpuScene.jitterX          = jitterX;
            gpuScene.jitterY          = jitterY;
            OS_GetMousePos(params->window, shaderDebug.mousePos.x, shaderDebug.mousePos.y);
        }
        u32 dispatchDimX =
            (targetWidth + PATH_TRACE_NUM_THREADS_X - 1) / PATH_TRACE_NUM_THREADS_X;
        u32 dispatchDimY =
            (targetHeight + PATH_TRACE_NUM_THREADS_Y - 1) / PATH_TRACE_NUM_THREADS_Y;
        gpuScene.dispatchDimX = dispatchDimX;
        gpuScene.dispatchDimY = dispatchDimY;

        // u32 *data = (u32 *)readback.mappedPtr;
        if (!device->BeginFrame(false))
        {
            Assert(0);
        }

        Print("frame: %u\n", device->frameCount);

        // for (int i = 0; i < 16; i++)
        // {
        //     Print("%u ", *((u32 *)readback2.mappedPtr + 131073 * i));
        // }
        // Print("\n");
        // Print("%u %u %u %u\n", data[GLOBALS_BLAS_BYTES],
        // data[GLOBALS_BLAS_CLAS_COUNT_INDEX],
        //       data[GLOBALS_VISIBLE_CLUSTER_COUNT_INDEX],
        //       data[GLOBALS_FREED_PARTITION_COUNT]);
        // Print("freed partition count: %u visible count: %u, writes: %u updates %u\n",
        //       data[GLOBALS_FREED_PARTITION_COUNT], data[GLOBALS_VISIBLE_PARTITION_COUNT],
        //       data[GLOBALS_PTLAS_WRITE_COUNT_INDEX],
        //       data[GLOBALS_PTLAS_UPDATE_COUNT_INDEX]);

        string cmdBufferName =
            PushStr8F(frameScratch.temp.arena, "Graphics Cmd %u", device->frameCount);
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics, cmdBufferName);

        rg->BeginFrame();
        debugState.BeginFrame(cmd);

        if (device->frameCount == 0)
        {
            device->Wait(tileSubmitSemaphore);
            device->Wait(tlasSemaphore);

            envMapBindlessIndex = device->BindlessIndex(&gpuEnvMap);
        }

        u32 currentBuffer   = device->frameCount & 1;
        u32 lastFrameBuffer = device->frameCount == 0 ? 0 : !currentBuffer;

        if (device->frameCount == 0)
        {
            CommandBuffer *computeCmd =
                device->BeginCommandBuffer(QueueType_Compute, "cmd populate readback");

            MemoryCopy(sceneTransferBuffers[currentBuffer].mappedPtr, &gpuScene,
                       sizeof(GPUScene));
            computeCmd->SubmitTransfer(&sceneTransferBuffers[currentBuffer]);
            MemoryCopy(shaderDebugBuffers[currentBuffer].mappedPtr, &shaderDebug,
                       sizeof(ShaderDebugInfo));
            computeCmd->SubmitTransfer(&shaderDebugBuffers[currentBuffer]);

            rg->StartPass(
                  6,
                  [queue                = virtualGeometryManager.queueBuffer,
                   clasGlobals          = virtualGeometryManager.clasGlobalsBuffer,
                   resourceBitVector    = virtualGeometryManager.resourceBitVector,
                   maxMinLodLevelBuffer = virtualGeometryManager.maxMinLodLevelBuffer,
                   candidateClusters    = virtualGeometryManager.candidateClusterBuffer,
                   candidateNodes =
                       virtualGeometryManager.candidateNodeBuffer](CommandBuffer *cmd) {
                      RenderGraph *rg = GetRenderGraph();
                      u32 offset, size;
                      GPUBuffer *buffer = rg->GetBuffer(candidateClusters, offset, size);
                      cmd->ClearBuffer(buffer, ~0u);

                      buffer = rg->GetBuffer(candidateNodes, offset, size);
                      cmd->ClearBuffer(buffer, ~0u);

                      buffer = rg->GetBuffer(queue, offset, size);
                      cmd->ClearBuffer(buffer, 0);

                      buffer = rg->GetBuffer(clasGlobals, offset, size);
                      cmd->ClearBuffer(buffer, 0);

                      buffer = rg->GetBuffer(resourceBitVector, offset, size);
                      cmd->ClearBuffer(buffer, 0);

                      buffer = rg->GetBuffer(maxMinLodLevelBuffer, offset, size);
                      cmd->ClearBuffer(buffer, 0);

                      cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
                      cmd->FlushBarriers();
                  })
                .AddHandle(virtualGeometryManager.resourceBitVector, ResourceUsageType::Write)
                .AddHandle(virtualGeometryManager.maxMinLodLevelBuffer,
                           ResourceUsageType::Write)
                .AddHandle(virtualGeometryManager.queueBuffer, ResourceUsageType::Write)
                .AddHandle(virtualGeometryManager.clasGlobalsBuffer, ResourceUsageType::Write)
                .AddHandle(virtualGeometryManager.candidateNodeBuffer,
                           ResourceUsageType::Write)
                .AddHandle(virtualGeometryManager.candidateClusterBuffer,
                           ResourceUsageType::Write);

            computeCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
            computeCmd->FlushBarriers();

            virtualGeometryManager.PrepareInstances(
                computeCmd, sceneTransferBufferHandles[currentBuffer], false);

            virtualGeometryManager.HierarchyTraversal(
                computeCmd, sceneTransferBufferHandles[currentBuffer]);

            rg->StartPass(2,
                          [&readback = virtualGeometryManager.readbackBuffer,
                           requests  = virtualGeometryManager.streamingRequestsBuffer](
                              CommandBuffer *cmd) {
                              u32 srcOffset, srcSize;
                              RenderGraph *rg = GetRenderGraph();
                              GPUBuffer *requestsBuffer =
                                  rg->GetBuffer(requests, srcOffset, srcSize);
                              BufferToBufferCopy copy;
                              copy.srcOffset = srcOffset;
                              copy.dstOffset = 0;
                              copy.size      = srcSize;
                              cmd->CopyBuffer(&readback, requestsBuffer, &copy, 1);
                              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                           VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_NONE);
                              cmd->FlushBarriers();
                          })
                .AddHandle(virtualGeometryManager.readbackBufferHandle,
                           ResourceUsageType::Write)
                .AddHandle(virtualGeometryManager.streamingRequestsBuffer,
                           ResourceUsageType::Read);

            rg->Compile();
            rg->Execute(computeCmd);
            rg->EndFrame();

            Semaphore sem   = device->CreateSemaphore();
            sem.signalValue = 1;
            computeCmd->SignalOutsideFrame(sem);
            device->SubmitCommandBuffer(computeCmd);
            device->Wait(sem);

            rg->BeginFrame();
        }
        else
        {
            MemoryCopy(sceneTransferBuffers[currentBuffer].mappedPtr, &gpuScene,
                       sizeof(GPUScene));
            cmd->SubmitTransfer(&sceneTransferBuffers[currentBuffer]);
            MemoryCopy(shaderDebugBuffers[currentBuffer].mappedPtr, &shaderDebug,
                       sizeof(ShaderDebugInfo));
            cmd->SubmitTransfer(&shaderDebugBuffers[currentBuffer]);

            // cmd->Barrier(&sceneTransferBuffers[currentBuffer].buffer, flags,
            //              VK_ACCESS_2_SHADER_WRITE_BIT);
            // cmd->Barrier(&shaderDebugBuffers[currentBuffer].buffer, flags,
            //              VK_ACCESS_2_SHADER_WRITE_BIT);
        }

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

        // cmd->ClearBuffer(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer);

        // Virtual geometry pass

        // TODO: ????
        // cmd->ClearBuffer(&virtualGeometryManager.workItemQueueBuffer, ~0u);
        // cmd->ClearBuffer(&virtualGeometryManager.visibleClustersBuffer, ~0u);
        // cmd->ClearBuffer(&virtualGeometryManager.partitionCountsBuffer);

        rg->StartPass(6,
                      [queue                = virtualGeometryManager.queueBuffer,
                       clasGlobals          = virtualGeometryManager.clasGlobalsBuffer,
                       resourceBitVector    = virtualGeometryManager.resourceBitVector,
                       maxMinLodLevelBuffer = virtualGeometryManager.maxMinLodLevelBuffer,
                       candidateClusters    = virtualGeometryManager.candidateClusterBuffer,
                       candidateNodes =
                           virtualGeometryManager.candidateNodeBuffer](CommandBuffer *cmd) {
                          RenderGraph *rg = GetRenderGraph();
                          u32 offset, size;
                          GPUBuffer *buffer = rg->GetBuffer(candidateClusters, offset, size);
                          cmd->ClearBuffer(buffer, ~0u);

                          buffer = rg->GetBuffer(candidateNodes, offset, size);
                          cmd->ClearBuffer(buffer, ~0u);

                          buffer = rg->GetBuffer(queue, offset, size);
                          cmd->ClearBuffer(buffer, 0);

                          buffer = rg->GetBuffer(clasGlobals, offset, size);
                          cmd->ClearBuffer(buffer, 0);

                          buffer = rg->GetBuffer(resourceBitVector, offset, size);
                          cmd->ClearBuffer(buffer, 0);

                          buffer = rg->GetBuffer(maxMinLodLevelBuffer, offset, size);
                          cmd->ClearBuffer(buffer, 0);

                          cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_2_SHADER_WRITE_BIT |
                                           VK_ACCESS_2_SHADER_READ_BIT);
                          cmd->FlushBarriers();
                      })
            .AddHandle(virtualGeometryManager.resourceBitVector, ResourceUsageType::Write)
            .AddHandle(virtualGeometryManager.maxMinLodLevelBuffer, ResourceUsageType::Write)
            .AddHandle(virtualGeometryManager.queueBuffer, ResourceUsageType::Write)
            .AddHandle(virtualGeometryManager.clasGlobalsBuffer, ResourceUsageType::Write)
            .AddHandle(virtualGeometryManager.candidateNodeBuffer, ResourceUsageType::Write)
            .AddHandle(virtualGeometryManager.candidateClusterBuffer,
                       ResourceUsageType::Write);

        // Streaming
        // bool test    = virtualGeometryManager.ProcessInstanceRequests(cmd);
        int cpuIndex = TIMED_CPU_RANGE_BEGIN();

        virtualGeometryManager.ProcessRequests(cmd, testCount);
        TIMED_RANGE_END(cpuIndex);

        // Instance culling
        {
            virtualGeometryManager.PrepareInstances(
                cmd, sceneTransferBufferHandles[currentBuffer], true);
        }

        // Hierarchy traversal
        {
            virtualGeometryManager.HierarchyTraversal(
                cmd, sceneTransferBufferHandles[currentBuffer]);
        }

        rg->StartPass(
              2,
              [&readback = virtualGeometryManager.readbackBuffer,
               requests = virtualGeometryManager.streamingRequestsBuffer](CommandBuffer *cmd) {
                  u32 srcOffset, srcSize;
                  RenderGraph *rg           = GetRenderGraph();
                  GPUBuffer *requestsBuffer = rg->GetBuffer(requests, srcOffset, srcSize);
                  BufferToBufferCopy copy;
                  copy.srcOffset = srcOffset;
                  copy.dstOffset = 0;
                  copy.size      = srcSize;
                  cmd->CopyBuffer(&readback, requestsBuffer, &copy, 1);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_NONE);
                  cmd->FlushBarriers();
              })
            .AddHandle(virtualGeometryManager.readbackBufferHandle, ResourceUsageType::Write)
            .AddHandle(virtualGeometryManager.streamingRequestsBuffer,
                       ResourceUsageType::Read);

        rg->StartComputePass(prepareIndirectPipeline, prepareIndirectLayout, 1,
                             [](CommandBuffer *cmd) {
                                 device->BeginEvent(cmd, "Prepare indirect");
                                 cmd->Dispatch(1, 1, 1);
                                 cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_ACCESS_2_SHADER_WRITE_BIT,
                                              VK_ACCESS_2_SHADER_READ_BIT);
                                 cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                              VK_ACCESS_2_SHADER_WRITE_BIT,
                                              VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
                                 cmd->FlushBarriers();
                                 device->EndEvent(cmd);
                             })
            .AddHandle(virtualGeometryManager.clasGlobalsBuffer, ResourceUsageType::RW);

        virtualGeometryManager.BuildClusterBLAS(cmd);

        virtualGeometryManager.BuildPTLAS(cmd);

        rg->StartPass(0, [](CommandBuffer *cmd) {
            cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                         VK_ACCESS_2_SHADER_READ_BIT);
            cmd->FlushBarriers();
        });

#if 0

            {
                // Prepare instance descriptors for TLAS build
                device->BeginEvent(cmd, "Prepare TLAS");
                cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, fillInstancePipeline);

                DescriptorSet ds = fillInstanceLayout.CreateDescriptorSet();
                ds.Bind(&virtualGeometryManager.blasAccelAddresses)
                    .Bind(&virtualGeometryManager.clasGlobalsBuffer)
                    .Bind(&virtualGeometryManager.blasDataBuffer)
                    .Bind(&gpuInstancesBuffer.buffer)
                    .Bind(&tlasBuffer)
                    .Bind(&virtualGeometryManager.voxelBlasInfosBuffer)
                    .Bind(&sceneTransferBuffers[currentBuffer].buffer);
                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                        fillInstanceLayout.pipelineLayout);

                cmd->DispatchIndirect(&virtualGeometryManager.clasGlobalsBuffer,
                                      sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_ACCESS_2_SHADER_WRITE_BIT,
                             VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
                cmd->FlushBarriers();

                device->EndEvent(cmd);
            }

            {
                // Build the TLAS
                device->BeginEvent(cmd, "Build TLAS");
                tlas.as = cmd->BuildTLAS(&tlasAccelBuffer, &tlasScratchBuffer, &tlasBuffer,
                                         1u << 21u); // virtualGeometryManager.maxInstances);
                cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                             VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                             VK_ACCESS_2_SHADER_READ_BIT);
                cmd->FlushBarriers();
                device->EndEvent(cmd);
            }
#endif

        RayPushConstant pc;
        pc.envMap   = envMapBindlessIndex;
        pc.frameNum = (u32)device->frameCount;
        pc.width    = envMap->width;
        pc.height   = envMap->height;

        VkPipelineStageFlags2 flags   = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
        // cmd->Barrier(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer, flags,
        //              VK_ACCESS_2_SHADER_WRITE_BIT);
        u64 ptlasAddress =
            device->GetDeviceAddress(virtualGeometryManager.tlasAccelBuffer.buffer);

        // TODO: not adding the handle of all of the barriers. maybe will need to if
        // automatic synchronization is done in the future
        rg->StartPass(
              7,
              [&layout, &ptlasAddress, &rts = rts, &pushConstant, &pc, bindPoint, targetWidth,
               targetHeight, &scene = sceneTransferBuffers[currentBuffer].buffer,
               &debug = shaderDebugBuffers[currentBuffer].buffer, &materialBuffer,
               &faceDataBuffer, &virtualTextureManager, &virtualGeometryManager,
               imageHandle = image, imageOutHandle = imageOut, depthBufferHandle = depthBuffer,
               normalRoughnessHandle = normalRoughness, diffuseAlbedoHandle = diffuseAlbedo,
               specularAlbedoHandle      = specularAlbedo,
               specularHitDistanceHandle = specularHitDistance](CommandBuffer *cmd) {
                  RenderGraph *rg               = GetRenderGraph();
                  GPUImage *image               = rg->GetImage(imageHandle);
                  GPUImage *imageOut            = rg->GetImage(imageOutHandle);
                  GPUImage *depthBuffer         = rg->GetImage(depthBufferHandle);
                  GPUImage *normalRoughness     = rg->GetImage(normalRoughnessHandle);
                  GPUImage *diffuseAlbedo       = rg->GetImage(diffuseAlbedoHandle);
                  GPUImage *specularAlbedo      = rg->GetImage(specularAlbedoHandle);
                  GPUImage *specularHitDistance = rg->GetImage(specularHitDistanceHandle);

                  cmd->Barrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                  cmd->Barrier(imageOut, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                  cmd->Barrier(depthBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                  cmd->Barrier(normalRoughness, VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                  cmd->Barrier(diffuseAlbedo, VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                  cmd->Barrier(specularAlbedo, VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                  cmd->Barrier(specularHitDistance, VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                  cmd->FlushBarriers();

                  cmd->BindPipeline(bindPoint, rts.pipeline);

                  cmd->StartBinding(bindPoint, rts.pipeline, &layout)
                      // .Bind(&tlas.as)
                      .Bind(&ptlasAddress)
                      .Bind(image)
                      .Bind(&scene)
                      .Bind(&materialBuffer)
                      .Bind(&virtualTextureManager.pageTable)
                      .Bind(&virtualTextureManager.gpuPhysicalPool)
                      .Bind(&debug)
                      .Bind(&virtualGeometryManager.clusterPageDataBuffer)
                      .Bind(&faceDataBuffer)
                      .Bind(&virtualGeometryManager.instancesBuffer)
                      .Bind(&virtualGeometryManager.resourceTruncatedEllipsoidsBuffer)
                      .Bind(&virtualGeometryManager.instanceTransformsBuffer)
                      .Bind(&virtualGeometryManager.partitionInfosBuffer)
                      // .Bind(&virtualGeometryManager.clasGlobalsBuffer)
                      .Bind(depthBuffer)
                      .Bind(normalRoughness)
                      .Bind(diffuseAlbedo)
                      .Bind(specularAlbedo)
                      .Bind(specularHitDistance)
                      // .Bind(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer);
                      .PushConstants(&pushConstant, &pc)
                      .End();

                  int beginIndex = TIMED_GPU_RANGE_BEGIN(cmd, "ray trace");
                  // cmd->Dispatch(dispatchDimX, dispatchDimY, 1);
                  cmd->TraceRays(&rts, targetWidth, targetHeight, 1);
                  TIMED_RANGE_END(beginIndex);
              })
            .AddHandle(image, ResourceUsageType::Write)
            .AddHandle(imageOut, ResourceUsageType::Write)
            .AddHandle(depthBuffer, ResourceUsageType::Write)
            .AddHandle(normalRoughness, ResourceUsageType::Write)
            .AddHandle(diffuseAlbedo, ResourceUsageType::Write)
            .AddHandle(specularAlbedo, ResourceUsageType::Write)
            .AddHandle(specularHitDistance, ResourceUsageType::Write);

        rg->StartComputePass(
              mvecPipeline, mvecLayout, 3,
              [depthBufferHandle = depthBuffer, motionVectorHandle = motionVectorBuffer,
               targetWidth, targetHeight](CommandBuffer *cmd) {
                  RenderGraph *rg              = GetRenderGraph();
                  GPUImage *motionVectorBuffer = rg->GetImage(motionVectorHandle);
                  GPUImage *depthBuffer        = rg->GetImage(depthBufferHandle);
                  device->BeginEvent(cmd, "Motion Vectors");
                  cmd->Barrier(motionVectorBuffer, VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_NONE,
                               VK_ACCESS_2_SHADER_WRITE_BIT);
                  cmd->Barrier(depthBuffer, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->FlushBarriers();
                  cmd->Dispatch((targetWidth + 7) / 8, (targetHeight + 7) / 8, 1);
                  device->EndEvent(cmd);
              })
            .AddHandle(depthBuffer, ResourceUsageType::Read)
            .AddHandle(motionVectorBuffer, ResourceUsageType::Write)
            .AddHandle(sceneTransferBufferHandles[currentBuffer], ResourceUsageType::Read);

        // cmd->Barrier(&image, VK_IMAGE_LAYOUT_GENERAL,
        //              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        //              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        //              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        //              VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        // cmd->Barrier(&image, VK_IMAGE_LAYOUT_GENERAL,
        //              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        //              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        //              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        //              VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        rg->StartPass(
              8,
              [&gpuScene, &params, &clipToPrevClip, &prevClipToClip, &camera, outJitterX,
               outJitterY, imageHandle = image, imageOutHandle = imageOut,
               depthBufferHandle = depthBuffer, normalRoughnessHandle = normalRoughness,
               diffuseAlbedoHandle = diffuseAlbedo, specularAlbedoHandle = specularAlbedo,
               specularHitDistanceHandle = specularHitDistance,
               motionVectorsBufferHandle = motionVectorBuffer](CommandBuffer *cmd) {
                  RenderGraph *rg               = GetRenderGraph();
                  GPUImage *image               = rg->GetImage(imageHandle);
                  GPUImage *imageOut            = rg->GetImage(imageOutHandle);
                  GPUImage *depthBuffer         = rg->GetImage(depthBufferHandle);
                  GPUImage *normalRoughness     = rg->GetImage(normalRoughnessHandle);
                  GPUImage *diffuseAlbedo       = rg->GetImage(diffuseAlbedoHandle);
                  GPUImage *specularAlbedo      = rg->GetImage(specularAlbedoHandle);
                  GPUImage *specularHitDistance = rg->GetImage(specularHitDistanceHandle);
                  GPUImage *motionVectorsBuffer = rg->GetImage(motionVectorsBufferHandle);

                  cmd->Barrier(image, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->Barrier(normalRoughness, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->Barrier(motionVectorsBuffer, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->Barrier(diffuseAlbedo, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->Barrier(specularAlbedo, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->Barrier(specularHitDistance, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->FlushBarriers();

                  DLSSTargets targets = device->InitializeDLSSTargets(
                      image, diffuseAlbedo, specularAlbedo, normalRoughness,
                      motionVectorsBuffer, depthBuffer, specularHitDistance, imageOut);
                  device->BeginEvent(cmd, "DLSS RR");
                  cmd->DLSS(targets, gpuScene.cameraFromRender, gpuScene.renderFromCamera,
                            params->NDCFromCamera, params->cameraFromClip, clipToPrevClip,
                            prevClipToClip, camera.position, params->up, camera.forward,
                            camera.right, params->fov, params->aspectRatio,
                            Vec2f(outJitterX, outJitterY));
                  device->EndEvent(cmd);
              })
            .AddHandle(image, ResourceUsageType::Read)
            .AddHandle(diffuseAlbedo, ResourceUsageType::Read)
            .AddHandle(specularAlbedo, ResourceUsageType::Read)
            .AddHandle(normalRoughness, ResourceUsageType::Read)
            .AddHandle(motionVectorBuffer, ResourceUsageType::Read)
            .AddHandle(depthBuffer, ResourceUsageType::Read)
            .AddHandle(specularHitDistance, ResourceUsageType::Read)
            .AddHandle(imageOut, ResourceUsageType::Write);

#if 0
        // Copy feedback from device to host
        CommandBuffer *transferCmd =
            device->BeginCommandBuffer(QueueType_Copy, "feedback copy cmd");
        cmd->Signal(transferSem);
        transferCmd->Wait(transferSem);
        transferSem.signalValue++;
        // transferCmd->CopyBuffer(
        //     &virtualTextureManager.feedbackBuffers[currentBuffer].stagingBuffer,
        //     &virtualTextureManager.feedbackBuffers[currentBuffer].buffer);
        transferCmd->CopyBuffer(&virtualGeometryManager.readbackBuffer,
                                &virtualGeometryManager.streamingRequestsBuffer);
        // transferCmd->CopyBuffer(&virtualGeometryManager.partitionReadbackBuffer,
        //                         &virtualGeometryManager.partitionCountsBuffer);
        device->SubmitCommandBuffer(transferCmd, true);
#endif

        debugState.EndFrame(cmd);

        rg->StartPass(1, [&swapchain, imageOutHandle = imageOut](CommandBuffer *cmd) {
              RenderGraph *rg    = GetRenderGraph();
              GPUImage *imageOut = rg->GetImage(imageOutHandle);

              device->CopyFrameBuffer(&swapchain, cmd, imageOut);
          }).AddHandle(imageOut, ResourceUsageType::Read);

        rg->Compile();
        rg->Execute(cmd);
        rg->EndFrame();

        device->EndFrame(QueueFlag_Graphics);

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
