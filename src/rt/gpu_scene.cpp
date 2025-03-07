#include "gpu_scene.h"
#include "scene.h"
#include "vulkan.h"

namespace rt
{

SceneShapeParse StartSceneShapeParse()
{
    return SceneShapeParse{.buffer = device->BeginTransfers()};
}

void EndSceneShapeParse(SceneShapeParse *parse) { parse->buffer.SubmitToQueue(); }

GPUMesh CopyMesh(SceneShapeParse *parse, Arena *arena, Mesh &mesh)
{
    size_t vertexSize  = sizeof(mesh.p[0]) * mesh.numVertices;
    size_t indicesSize = sizeof(mesh.indices[0]) * mesh.numIndices;
    size_t totalSize   = vertexSize + sizeof(mesh.indices[0]) * mesh.numIndices;

    TransferBuffer transferBuffer = device->GetStagingBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, totalSize);

    u64 deviceAddress = device->GetDeviceAddress(transferBuffer.buffer.buffer);

    u8 *ptr = (u8 *)transferBuffer.mappedPtr;
    MemoryCopy(ptr, mesh.p, vertexSize);
    ptr += vertexSize;
    MemoryCopy(ptr, mesh.indices, indicesSize);

    parse->buffer.SubmitTransfer(&transferBuffer);

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
                          HashMap<MaterialNode> *materialHashMap, GPUMesh &mesh,
                          ChunkedLinkedList<GPUMesh, MemoryType_Shape> &shapes,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices,
                          ChunkedLinkedList<Light *, MemoryType_Light> &lights)
{
    return;
}

void BuildAllSceneBVHs(Arena **arenas, ScenePrimitives **scenes, int numScenes, int maxDepth,
                       const Mat4 &NDCFromCamera, const Mat4 &cameraFromRender,
                       int screenHeight)
{
    CommandBuffer cmd = device->BeginCommandBuffer(QueueType_Graphics);
    device->BeginEvent(&cmd, "BLAS Build");

    for (int depth = maxDepth; depth >= 0; depth--)
    {
        for (int i = 0; i < numScenes; i++)
        {
            ScenePrimitives *scene = scenes[i];
            if (scene->depth.load(std::memory_order_acquire) == depth)
            {
                switch (scene->geometryType)
                {
                    case GeometryType::TriangleMesh:
                    {
                        GPUMesh *meshes = (GPUMesh *)scene->primitives;
                        scene->gpuBVH = device->CreateBLAS(&cmd, meshes, scene->numPrimitives);
                    }
                    break;
                    default: Assert(0);
                }
            }
        }
    }

    device->EndEvent(&cmd);
}
} // namespace rt
