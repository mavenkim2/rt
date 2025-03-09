#include "gpu_scene.h"
#include "scene.h"
#include "vulkan.h"
#include "win32.h"

namespace rt
{

SceneShapeParse StartSceneShapeParse()
{
    SceneShapeParse result = SceneShapeParse{
        .buffer    = device->BeginCommandBuffer(QueueType_Copy),
        .semaphore = device->CreateGraphicsSemaphore(),
    };
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

    GPUMesh result = {
        .vertexAddress = deviceAddress,
        .indexAddress  = deviceAddress + vertexSize,
        .numIndices    = mesh.numIndices,
        .numVertices   = mesh.numVertices,
        .numFaces      = mesh.numFaces,
    };

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

void BuildAllSceneBVHs(Arena **arenas, ScenePrimitives **scenes, int numScenes, int maxDepth,
                       const Mat4 &NDCFromCamera, const Mat4 &cameraFromRender,
                       int screenHeight)
{
    // Compile shaders
    string raygenShaderName = "../src/shaders/spirv/render_raytrace_rgen.spv";
    Arena *arena            = arenas[0];
    string data             = OS_ReadFile(arena, raygenShaderName);

    Shader *rayShaders[RST_Max];
    int counts[RST_Max] = {};
    Shader raygenShaders[1];
    raygenShaders[counts[RST_Raygen]++] =
        device->CreateShader(ShaderStage::Raygen, "raygen", data);
    Shader missShaders[1];

    rayShaders[RST_Raygen] = raygenShaders;

    RayTracingState rts = device->CreateRayTracingPipeline(rayShaders, counts, maxDepth);

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
                        scene->gpuBVH  = device->CreateBLAS(cmd, meshes, scene->numPrimitives);
                        as[bvhCount++] = &scene->gpuBVH;
                    }
                    break;
                    default: Assert(0);
                }
            }
        }

        QueryPool queryPool = device->GetCompactionSizes(cmd, as, bvhCount);
        device->SubmitCommandBuffer(cmd);
        device->EndEvent(cmd);

        CommandBuffer *compactCmd = device->BeginCommandBuffer(QueueType_Graphics);
        compactCmd->Wait(semaphore);
        device->CompactBLASes(compactCmd, queryPool, as, bvhCount);
        device->SubmitCommandBuffer(compactCmd);

        device->BindAccelerationStructure(rts.set, as[0]->as);
    }

    // vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, state.VkPipeline
    // pipeline); vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
    //
    //                         VkPipelineLayout layout, uint32_t firstSet,
    //                         uint32_t descriptorSetCount,
    //                         const VkDescriptorSet *pDescriptorSets,
    //                         uint32_t dynamicOffsetCount, const uint32_t *pDynamicOffsets);
}
} // namespace rt
