#include "camera.h"
#include "dgfs.h"
#include "integrate.h"
#include "gpu_scene.h"
#include "math/simd_base.h"
#include "memory.h"
#include "parallel.h"
#include "platform.h"
#include "shader_interop/dense_geometry_shaderinterop.h"
#include "shader_interop/gpu_scene_shaderinterop.h"
#include "shader_interop/hit_shaderinterop.h"
#include "shader_interop/ray_shaderinterop.h"
#include "shader_interop/debug_shaderinterop.h"
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

#if 0
GPUMesh ProcessMesh(SceneShapeParse *parse, Arena *arena, Mesh &mesh)
{
    ScratchArena scratch;

    size_t vertexSize  = sizeof(mesh.p[0]) * mesh.numVertices;
    size_t indicesSize = sizeof(mesh.indices[0]) * mesh.numIndices;
    size_t normalSize  = 0;
    size_t totalSize   = vertexSize + indicesSize;

    u32 *octNormals = 0;
    if (mesh.n)
    {
        octNormals = PushArrayNoZero(scratch.temp.arena, u32, mesh.numVertices);
        for (int i = 0; i < mesh.numVertices; i++)
        {
            octNormals[i] = EncodeOctahedral(mesh.n[i]);
        }
        normalSize = sizeof(u32) * mesh.numVertices;
        totalSize += normalSize;
    }

    u64 alignment = device->GetMinAlignment(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    TransferBuffer transferBuffer = device->GetStagingBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        totalSize + 2 * alignment);

    u64 deviceAddress = device->GetDeviceAddress(transferBuffer.buffer.buffer);

    GPUMesh result = {};

    uintptr_t ptr = (uintptr_t)transferBuffer.mappedPtr;
    Assert(ptr);
    MemoryCopy((void *)ptr, mesh.p, vertexSize);
    ptr += vertexSize;
    ptr                = AlignPow2(ptr, alignment);
    result.indexOffset = ptr - (uintptr_t)transferBuffer.mappedPtr;
    MemoryCopy((void *)ptr, mesh.indices, indicesSize);
    ptr += indicesSize;
    if (normalSize)
    {
        ptr                 = AlignPow2(ptr, alignment);
        result.normalOffset = ptr - (uintptr_t)transferBuffer.mappedPtr;
        MemoryCopy((void *)ptr, octNormals, normalSize);
    }

    parse->buffer->SubmitTransfer(&transferBuffer);

    result.buffer        = transferBuffer.buffer;
    result.deviceAddress = deviceAddress;
    result.vertexOffset  = 0;
    result.vertexSize    = vertexSize;
    result.vertexStride  = sizeof(Vec3f);
    result.indexSize     = indicesSize;
    result.indexStride   = sizeof(u32);
    result.normalSize    = normalSize;
    result.normalStride  = sizeof(u32);
    result.numIndices    = mesh.numIndices;
    result.numVertices   = mesh.numVertices;
    result.numFaces      = mesh.numFaces;

    return result;
}
#endif

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
// 2. dgf attributes, geom IDs for differentiating materials
// 3. clas
//      - are memory savings possible with this? it seems like not really, and that this
//      just speeds up rebuilds for dynamic/adaptively tessellated geometry. not really
//      what I need.
//      - on blackwell there's memory savings
// 4. actual bsdfs and brdfs
// 5. add other parts of the scene, with actual instancing
// 6. disney bsdf
// 7. recycle memory
// 8. textures

void BuildAllSceneBVHs(RenderParams2 *params, ScenePrimitives **scenes, int numScenes,
                       int maxDepth, Image *envMap)
{
    // Build clusters
    // TODO: make sure to NOT make gpumeshes
    ScratchArena sceneScratch;
    Arena *arena   = ArenaAlloc();
    Arena **arenas = GetArenaArray(arena);

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

    CommandBuffer *dgfTransferCmd = device->BeginCommandBuffer(QueueType_Copy);

    GPUBuffer aabbBuffer;
    GPUBuffer dgfBuffer;
    GPUBuffer headerBuffer;
    u32 aabbLength;
    DenseGeometryBuildData data;
    data.Init();
    u8 *dgfByteBuffer;
    PackedDenseGeometryHeader *headers;
#if 0
    u32 **firstUses;
    u32 **reuses;
    VkAabbPositionsKHR *positions;
#endif
    TriangleStripType **types;
    u32 **debugIndices;

    u32 **debugRestartCount;
    u32 **debugRestartHighBit;

    for (int i = 0; i < numScenes; i++)
    {
        ScenePrimitives *scene       = scenes[i];
        scene->semaphore             = device->CreateGraphicsSemaphore();
        scene->semaphore.signalValue = 1;
        if (scene->geometryType == GeometryType::Instance) continue;

        RecordAOSSplits record(neg_inf);

        PrimRef *refs = ParallelGenerateMeshRefs<GeometryType::TriangleMesh>(
            sceneScratch.temp.arena, scenes[i], record, false);

        ClusterBuilder builder(arena, scene, refs);
        builder.BuildClusters(record, true);
        builder.CreateDGFs(&data, (Mesh *)scene->primitives, scene->numPrimitives,
                           sceneBounds);

        // Convert primrefs to aabbs, submit to GPU
        StaticArray<VkAabbPositionsKHR> aabbs(sceneScratch.temp.arena,
                                              MAX_CLUSTER_TRIANGLES * data.numBlocks);

        // Debug
        // positions = aabbs.data;
        types = PushArrayNoZero(sceneScratch.temp.arena, TriangleStripType *, data.numBlocks);
        // firstUses    = PushArrayNoZero(sceneScratch.temp.arena, u32 *,
        // data.numBlocks); reuses       = PushArrayNoZero(sceneScratch.temp.arena, u32 *,
        // data.numBlocks);
        debugIndices        = PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        debugRestartHighBit = PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        debugRestartCount   = PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);

        u32 pos = 0;
        for (int i = 0; i < builder.threadClusters.Length(); i++)
        {
            for (auto *node = builder.threadClusters[i].l.first; node != 0; node = node->next)
            {
                for (int i = 0; i < node->count; i++)
                {
                    u32 numTriangles        = 0;
                    RecordAOSSplits &record = node->values[i];

                    for (int refIndex = record.start; refIndex < record.start + record.count;
                         refIndex++)
                    {
                        VkAabbPositionsKHR aabb;
                        aabb.minX = -refs[refIndex].minX;
                        aabb.minY = -refs[refIndex].minY;
                        aabb.minZ = -refs[refIndex].minZ;
                        aabb.maxX = refs[refIndex].maxX;
                        aabb.maxY = refs[refIndex].maxY;
                        aabb.maxZ = refs[refIndex].maxZ;
                        aabbs.Push(aabb);
                    }
                    VkAabbPositionsKHR nullAabb = {};
                    nullAabb.minX               = f32(NaN);
                    for (int remaining = record.count; remaining < MAX_CLUSTER_TRIANGLES;
                         remaining++)
                    {
                        aabbs.Push(nullAabb);
                    }
                }
            }
        }

        aabbLength = aabbs.Length();
        aabbBuffer = dgfTransferCmd->SubmitBuffer(
            aabbs.data,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            aabbs.Length() * sizeof(VkAabbPositionsKHR));

        // Combine all blocks together
        dgfByteBuffer = PushArrayNoZero(sceneScratch.temp.arena, u8, data.byteBuffer.Length());
        data.byteBuffer.Flatten(dgfByteBuffer);
        dgfBuffer = dgfTransferCmd->SubmitBuffer(
            dgfByteBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, data.byteBuffer.Length());

        // Upload headers
        headers = PushArrayNoZero(sceneScratch.temp.arena, PackedDenseGeometryHeader,
                                  data.headers.Length());
        data.headers.Flatten(headers);
        headerBuffer = dgfTransferCmd->SubmitBuffer(
            headers, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sizeof(PackedDenseGeometryHeader) * data.headers.Length());

        // Debug
        int c = 0;
        for (auto *node = data.types.first; node != 0; node = node->next)
        {
            types[c++] = node->values;
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
            debugIndices[c++] = node->values;
        }
        c = 0;
        for (auto *node = data.debugRestartHighBitPerDword.first; node != 0; node = node->next)
        {
            debugRestartHighBit[c++] = node->values;
        }
        c = 0;
        for (auto *node = data.debugRestartCountPerDword.first; node != 0; node = node->next)
        {
            debugRestartCount[c++] = node->values;
        }

        dgfTransferCmd->Signal(scene->semaphore);

        // Send offsets buffer and dense geometry info
        ReleaseArenaArray(builder.arenas);
    }
    device->SetName(&dgfBuffer, "DGF Buffer");
    device->SubmitCommandBuffer(dgfTransferCmd);

    // Set the instance ids of each scene
    int runningTotal = 0;
    for (int i = 0; i < numScenes; i++)
    {
        ScenePrimitives *scene = scenes[i];
        if (scene->geometryType != GeometryType::Instance)
        {
            scene->gpuInstanceID = runningTotal;

            // GPUMesh *gpuMeshes = (GPUMesh *)scene->primitives;
            for (int primIndex = 0; primIndex < scene->numPrimitives; primIndex++)
            {
#if 0
                GPUMesh &gpuMesh     = gpuMeshes[primIndex];
                int bindlessVertices = device->BindlessStorageIndex(
                    &gpuMesh.buffer, gpuMesh.vertexOffset, gpuMesh.vertexSize);
                Assert(bindlessVertices == 3 * runningTotal);
                int bindlessIndices = device->BindlessStorageIndex(
                    &gpuMesh.buffer, gpuMesh.indexOffset, gpuMesh.indexSize);
                Assert(bindlessIndices == 3 * runningTotal + 1);
                int bindlessNormals = device->BindlessStorageIndex(
                    &gpuMesh.buffer, gpuMesh.normalOffset, gpuMesh.normalSize);
                Assert(bindlessNormals == 3 * runningTotal + 2);
#endif
                runningTotal++;
            }
        }
        else
        {
            scene->gpuInstanceID = scene->sceneIndex;
        }
    }

    Swapchain swapchain = device->CreateSwapchain(params->window, VK_FORMAT_R8G8B8A8_SRGB,
                                                  params->width, params->height);

    // Compile shaders
    Shader shader;
    {
        string shaderName = "../src/shaders/render_raytrace.spv";
        Arena *arena      = params->arenas[0];
        string shaderData = OS_ReadFile(arena, shaderName);
        shader = device->CreateShader(ShaderStage::Compute, "pathtrace", shaderData);
    }

    PushConstant pushConstant;
    pushConstant.stage  = ShaderStage::Compute;
    pushConstant.offset = 0;
    pushConstant.size   = sizeof(RayPushConstant);

    Semaphore submitSemaphore = device->CreateGraphicsSemaphore();
    // Transfer data to GPU
    GPUScene gpuScene;
    gpuScene.cameraFromRaster = params->cameraFromRaster;
    gpuScene.renderFromCamera = params->renderFromCamera;
    gpuScene.lightFromRender  = params->lightFromRender;
    gpuScene.dxCamera         = params->dxCamera;
    gpuScene.lensRadius       = params->lensRadius;
    gpuScene.dyCamera         = params->dyCamera;
    gpuScene.focalLength      = params->focalLength;

    ShaderDebugInfo shaderDebug;

    GPUMaterial gpuMaterial;
    gpuMaterial.eta = 1.1;

    RTBindingData bindingData;
    bindingData.materialIndex = 0;

    CommandBuffer *transferCmd  = device->BeginCommandBuffer(QueueType_Copy);
    GPUBuffer bindingDataBuffer = transferCmd->SubmitBuffer(
        &bindingData, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(RTBindingData));
    GPUBuffer materialBuffer = transferCmd->SubmitBuffer(
        &gpuMaterial, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(GPUMaterial));

    GPUImage gpuEnvMap = transferCmd->SubmitImage(envMap->contents, VK_IMAGE_USAGE_SAMPLED_BIT,
                                                  VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TYPE_2D,
                                                  envMap->width, envMap->height);
    transferCmd->Barrier(&gpuEnvMap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_READ_BIT);

    submitSemaphore.signalValue = 1;
    transferCmd->Signal(submitSemaphore);
    device->SubmitCommandBuffer(transferCmd);

    GPUImage images[2] = {
        device->CreateImage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TYPE_2D, params->width,
                            params->height),
        device->CreateImage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TYPE_2D, params->width,
                            params->height),
    };

    // Create descriptor set layout and pipeline
    DescriptorSetLayout layout = {};
    int accelBindingIndex      = layout.AddBinding((u32)RTBindings::Accel,
                                                   VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                                                   VK_SHADER_STAGE_COMPUTE_BIT);
    int imageBindingIndex      = layout.AddBinding(
        (u32)RTBindings::Image, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT);

    int sceneBindingIndex =
        layout.AddBinding((u32)RTBindings::Scene, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          VK_SHADER_STAGE_COMPUTE_BIT);

    int bindingDataBindingIndex =
        layout.AddBinding((u32)RTBindings::RTBindingData, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          VK_SHADER_STAGE_COMPUTE_BIT);

    int gpuMaterialBindingIndex =
        layout.AddBinding((u32)RTBindings::GPUMaterial, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          VK_SHADER_STAGE_COMPUTE_BIT);

    int denseGeometryBufferIndex =
        layout.AddBinding((u32)RTBindings::DenseGeometryData,
                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);

    int packedDenseGeometryHeaderBufferIndex =
        layout.AddBinding((u32)RTBindings::PackedDenseGeometryHeaders,
                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);

    int shaderDebugIndex =
        layout.AddBinding((u32)RTBindings::ShaderDebugInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          VK_SHADER_STAGE_COMPUTE_BIT);

    int aabbIndex =
        layout.AddBinding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);

    layout.AddImmutableSamplers();

    VkPipeline pipeline = device->CreateComputePipeline(&shader, &layout, &pushConstant);

    Semaphore tlasSemaphore = device->CreateGraphicsSemaphore();

    GPUAccelerationStructure tlas = {};
    // TODO: new command buffers have to wait on ones from the previous depth
    // also this doesn't work if there's actually a nontrivial TLAS
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
                        // scene->gpuBVH   = cmd->BuildBLAS(meshes, scene->numPrimitives);
                        scene->gpuBVH  = cmd->BuildCustomBLAS(&aabbBuffer, aabbLength);
                        as[bvhCount++] = &scene->gpuBVH;
                    }
                    break;
                    case GeometryType::Instance:
                    {
                        Assert(0);
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
            semaphore.signalValue = 2;
            compactCmd->Signal(semaphore);
            compactCmd->CompactAS(queryPool, as, bvhCount);
            device->SubmitCommandBuffer(compactCmd);
        }

        if (maxDepth == 0)
        {
            Assert(numScenes == 1);
            CommandBuffer *tlasCmd = device->BeginCommandBuffer(QueueType_Graphics);
            tlasCmd->Wait(semaphore);
            tlasSemaphore.signalValue = 1;
            tlasCmd->Signal(tlasSemaphore);
            Instance instance          = {};
            ScenePrimitives *baseScene = &GetScene()->scene;
            tlas = tlasCmd->BuildTLAS(&instance, 1, &params->renderFromWorld, &baseScene);
            device->SubmitCommandBuffer(tlasCmd);
        }
    }

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

    for (;;)
    {
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

        device->BeginFrame();
        u32 frame          = device->GetCurrentBuffer();
        GPUImage *image    = &images[frame];
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics);

        if (device->frameCount == 0)
        {
            cmd->Wait(submitSemaphore);
            cmd->Wait(tlasSemaphore);
            envMapBindlessIndex = device->BindlessIndex(&gpuEnvMap);
        }

        RayPushConstant pc;
        pc.envMap   = envMapBindlessIndex;
        pc.frameNum = (u32)device->frameCount;
        pc.width    = envMap->width;
        pc.height   = envMap->height;

        u32 currentBuffer = device->GetCurrentBuffer();
        MemoryCopy(sceneTransferBuffers[currentBuffer].mappedPtr, &gpuScene, sizeof(GPUScene));
        cmd->SubmitTransfer(&sceneTransferBuffers[currentBuffer]);
        MemoryCopy(shaderDebugBuffers[currentBuffer].mappedPtr, &shaderDebug,
                   sizeof(ShaderDebugInfo));
        cmd->SubmitTransfer(&shaderDebugBuffers[currentBuffer]);

        cmd->Barrier(&sceneTransferBuffers[currentBuffer].buffer,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->Barrier(&shaderDebugBuffers[currentBuffer].buffer,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->Barrier(image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->FlushBarriers();
        cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        DescriptorSet descriptorSet = layout.CreateDescriptorSet();
        descriptorSet.Bind(accelBindingIndex, &tlas.as)
            .Bind(imageBindingIndex, image)
            .Bind(sceneBindingIndex, &sceneTransferBuffers[currentBuffer].buffer)
            .Bind(bindingDataBindingIndex, &bindingDataBuffer)
            .Bind(gpuMaterialBindingIndex, &materialBuffer)
            .Bind(denseGeometryBufferIndex, &dgfBuffer)
            .Bind(packedDenseGeometryHeaderBufferIndex, &headerBuffer)
            .Bind(shaderDebugIndex, &shaderDebugBuffers[currentBuffer].buffer)
            .Bind(aabbIndex, &aabbBuffer);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &descriptorSet,
                                layout.pipelineLayout);

        cmd->PushConstants(&pushConstant, &pc, layout.pipelineLayout);
        cmd->Dispatch(
            (params->width + PATH_TRACE_NUM_THREADS_X - 1) / PATH_TRACE_NUM_THREADS_X,
            (params->height + PATH_TRACE_NUM_THREADS_Y - 1) / PATH_TRACE_NUM_THREADS_Y, 1);
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
