#include "integrate.h"
#include "gpu_scene.h"
#include "shader_interop/miss_shaderinterop.h"
#include "shader_interop/gpu_scene_shaderinterop.h"
#include "scene.h"
#include "vulkan.h"
#include "win32.h"

namespace rt
{

SceneShapeParse StartSceneShapeParse()
{
    SceneShapeParse result;
    result.buffer                = device->BeginCommandBuffer(QueueType_Copy);
    result.semaphore             = device->CreateGraphicsSemaphore();
    result.semaphore.signalValue = 1;
    return result;
}

void EndSceneShapeParse(ScenePrimitives *scene, SceneShapeParse *parse)
{
    parse->buffer->Signal(parse->semaphore);
    scene->semaphore = parse->semaphore;
    device->SubmitCommandBuffer(parse->buffer);
}

GPUMesh CopyMesh(SceneShapeParse *parse, Arena *arena, Mesh &mesh)
{
    size_t vertexSize  = sizeof(mesh.p[0]) * mesh.numVertices;
    size_t indicesSize = sizeof(mesh.indices[0]) * mesh.numIndices;
    size_t totalSize   = vertexSize + sizeof(mesh.indices[0]) * mesh.numIndices;

    TransferBuffer transferBuffer = device->GetStagingBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, totalSize);

    u64 deviceAddress = device->GetDeviceAddress(transferBuffer.buffer.buffer);

    u8 *ptr = (u8 *)transferBuffer.mappedPtr;
    Assert(ptr);
    MemoryCopy(ptr, mesh.p, vertexSize);
    ptr += vertexSize;
    MemoryCopy(ptr, mesh.indices, indicesSize);

    parse->buffer->SubmitTransfer(&transferBuffer);

    GPUMesh result;
    result.vertexAddress = deviceAddress;
    result.indexAddress  = deviceAddress + vertexSize;
    result.numIndices    = mesh.numIndices;
    result.numVertices   = mesh.numVertices;
    result.numFaces      = mesh.numFaces;

    return result;
}

void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          MaterialHashMap *materialHashMap, GPUMesh &mesh,
                          ChunkedLinkedList<GPUMesh, MemoryType_Shape> &shapes,
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
// 2. dgfs
// 3. clas
// 4. actual bsdfs and brdfs
// 5. environment map miss
// 6. add other parts of the scene
// 7. ability to move around
// 8. recycle memory

void BuildAllSceneBVHs(RenderParams2 *params, ScenePrimitives **scenes, int numScenes,
                       int maxDepth, Image *envMap)
{
    // Compile shaders
    string raygenShaderName = "../src/shaders/render_raytrace_rgen.spv";
    string missShaderName   = "../src/shaders/render_raytrace_miss.spv";
    string hitShaderName    = "../src/shaders/render_raytrace_hit.spv";

    Arena *arena    = params->arenas[0];
    string rgenData = OS_ReadFile(arena, raygenShaderName);
    string missData = OS_ReadFile(arena, missShaderName);
    string hitData  = OS_ReadFile(arena, hitShaderName);

    Swapchain swapchain = device->CreateSwapchain(params->window, VK_FORMAT_B8G8R8A8_UNORM,
                                                  params->width, params->height);

    Shader *rayShaders[RST_Max];
    int counts[RST_Max] = {};
    Shader raygenShaders[1];
    raygenShaders[counts[RST_Raygen]++] =
        device->CreateShader(ShaderStage::Raygen, "raygen", rgenData);
    Shader missShaders[1];
    missShaders[counts[RST_Miss]++] =
        device->CreateShader(ShaderStage::Miss, "miss", missData);
    Shader hitShaders[1];
    hitShaders[counts[RST_ClosestHit]++] =
        device->CreateShader(ShaderStage::Hit, "hit", hitData);

    rayShaders[RST_Raygen]     = raygenShaders;
    rayShaders[RST_Miss]       = missShaders;
    rayShaders[RST_ClosestHit] = hitShaders;

    PushConstant pushConstant;
    pushConstant.stage  = ShaderStage::Miss;
    pushConstant.offset = 0;
    pushConstant.size   = sizeof(MissPushConstant);

    GPUScene gpuScene;
    gpuScene.cameraFromRaster = params->cameraFromRaster;
    gpuScene.renderFromCamera = params->renderFromCamera;
    gpuScene.dxCamera         = params->dxCamera;
    gpuScene.dyCamera         = params->dyCamera;
    gpuScene.lensRadius       = params->lensRadius;
    gpuScene.focalLength      = params->focalLength;

    CommandBuffer *transferCmd    = device->BeginCommandBuffer(QueueType_Copy);
    GPUBuffer sceneTransferBuffer = transferCmd->SubmitBuffer(
        &gpuScene, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene));

    GPUImage gpuEnvMap = transferCmd->SubmitImage(envMap->contents, VK_IMAGE_USAGE_SAMPLED_BIT,
                                                  VK_FORMAT_R8G8B8_SRGB, VK_IMAGE_TYPE_2D,
                                                  envMap->width, envMap->height);
    device->SubmitCommandBuffer(transferCmd);
    // TODO: need to sync this transfer with the start of ray tracing
    // and bind envmap and sampler properly
    // with bindless

    GPUImage images[2] = {
        device->CreateImage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TYPE_2D, params->width,
                            params->height),
        device->CreateImage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TYPE_2D, params->width,
                            params->height),
    };

    DescriptorSetLayout layout = {};
    int accelBindingIndex      = layout.AddBinding((u32)RTBindings::Accel,
                                                   VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    int imageBindingIndex =
        layout.AddBinding((u32)RTBindings::Image, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                          VK_SHADER_STAGE_RAYGEN_BIT_KHR);

    int sceneBindingIndex =
        layout.AddBinding((u32)RTBindings::Scene, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          VK_SHADER_STAGE_RAYGEN_BIT_KHR);

    RayTracingState rts = device->CreateRayTracingPipeline(rayShaders, counts, &pushConstant,
                                                           &layout, params->maxDepth);

    for (int depth = maxDepth; depth >= 0; depth--)
    {
        ScratchArena scratch;
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics);
        device->BeginEvent(cmd, "BLAS Build");
        int bvhCount = 0;

        Semaphore semaphore   = device->CreateGraphicsSemaphore();
        semaphore.signalValue = 1;
        cmd->Signal(semaphore);

        QueryPool query = device->CreateQuery(QueryType_CompactSize, numScenes);

        GPUAccelerationStructure **as =
            PushArrayNoZero(scratch.temp.arena, GPUAccelerationStructure *, numScenes);

        for (int i = 0; i < numScenes; i++)
        {
            ScenePrimitives *scene = scenes[i];
            cmd->Wait(scene->semaphore);
            if (scene->depth.load(std::memory_order_acquire) == depth)
            {
                switch (scene->geometryType)
                {
                    case GeometryType::TriangleMesh:
                    {
                        GPUMesh *meshes = (GPUMesh *)scene->primitives;
                        scene->gpuBVH   = cmd->BuildBLAS(meshes, scene->numPrimitives);
                        as[bvhCount++]  = &scene->gpuBVH;
                    }
                    break;
                    case GeometryType::Instance:
                    {
                        Instance *instances = (Instance *)scene->primitives;
                        scene->gpuBVH =
                            cmd->BuildTLAS(instances, scene->numPrimitives,
                                           scene->affineTransforms, scene->childScenes);
                        as[bvhCount++] = &scene->gpuBVH;
                    }
                    break;
                    default: Assert(0);
                }
            }
        }

        if (bvhCount)
        {
            QueryPool queryPool = device->GetCompactionSizes(cmd, as, bvhCount);
            device->SubmitCommandBuffer(cmd);
            device->EndEvent(cmd);

            CommandBuffer *compactCmd = device->BeginCommandBuffer(QueueType_Graphics);
            compactCmd->Wait(semaphore);
            device->CompactBLASes(compactCmd, queryPool, as, bvhCount);
            device->SubmitCommandBuffer(compactCmd);
        }

        // TODO: need to sync here
        if (maxDepth == 0)
        {
            Assert(numScenes == 1);
            CommandBuffer *tlasCmd     = device->BeginCommandBuffer(QueueType_Graphics);
            Instance instance          = {};
            ScenePrimitives *baseScene = &GetScene()->scene;
            tlasCmd->BuildTLAS(&instance, 1, &params->renderFromWorld, &baseScene);
        }
    }

    GPUAccelerationStructure &tlas = GetScene()->scene.gpuBVH;
    f32 frameDt                    = 1.f / 60.f;
    for (;;)
    {
        MSG message;
        while (PeekMessageW(&message, 0, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        f32 frameTime = OS_NowSeconds();

        MissPushConstant pc;
        pc.envMap = 0;
        pc.width  = envMap->width;
        pc.height = envMap->height;

        device->BeginFrame();
        u32 frame          = device->GetCurrentBuffer();
        GPUImage *image    = &images[frame];
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics);
        cmd->Barrier(image, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->FlushBarriers();
        cmd->BindPipeline(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rts.pipeline);
        DescriptorSet descriptorSet = layout.CreateDescriptorSet();
        descriptorSet.Bind(accelBindingIndex, &tlas.as)
            .Bind(imageBindingIndex, image)
            .Bind(sceneBindingIndex, &sceneTransferBuffer);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, &descriptorSet,
                                rts.layout);
        cmd->PushConstants(&pushConstant, &pc, rts.layout);
        cmd->TraceRays(&rts, params->width, params->height, 1);
        device->CopyFrameBuffer(&swapchain, cmd, image);
        device->EndFrame();

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
