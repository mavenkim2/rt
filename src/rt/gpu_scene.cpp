#include "vulkan.h"

namespace rt
{

GPUMesh CopyMesh(CommandBuffer *buffer, Arena *arena, Mesh &mesh)
{
    GPUMesh result;
    GPUBuffer &vertexBuffer = result.vertexBuffer;
    buffer.desc.size =

        device->CreateBuffer(&vertexBuffer, buffer.desc, mesh.p);

    GPUBuffer &indexBuffer = result.indexBuffer;
    indexBuffer.desc.size =

        device->CreateBuffer(&indexBuffer, indexBuffer.desc, mesh.indices);

    size_t vertexSize  = sizeof(mesh.p[0]) * mesh.numVertices;
    size_t indicesSize = sizeof(mesh.indices[0]) * mesh.numIndices;
    size_t totalSize   = vertexSize + sizeof(mesh.indices[0]) * mesh.numIndices;

    int numBuffers = 2;

    device->TransferData(totalSize, numBuffers, [&](void *ptr, u32 *alignment) {
        MemoryCopy(ptr, mesh.p, vertexSize);

        uintptr_t indexOutPtr =
            ((uintptr_t)ptr + vertexSize + alignment - 1) & ~(alignment - 1);

        MemoryCopy((void *)indexOutPtr, mesh.indices, indicesSize)
    });

    result.numIndices  = mesh.numIndices;
    result.numVertices = mesh.numVertices;
    result.numFaces    = mesh.numFaces;

    return result;
}

void BuildSceneBVHs(ScenePrimitives **scenes, int numScenes, int maxDepth)
{
    CommandList cmd;
    cmd = device->BeginCommandList(QueueType::Graphics);

    device->BeginEvent(cmd, "BLAS Build");

    for (int depth = maxDepth; depth >= 0; depth--)
    {
        for (int i = 0; i < numScenes; i++)
        {
            ScenePrimitives *scene = scenes[i];
            if (scene->depth.load(std::memory_order_acquire) == depth)
            {
                switch (scene->type)
                {
                    case GeometryType::TriangleMesh:
                    {
                        GPUMesh *meshes = (GPUMesh *)scene->primitives;
                        scene->nodePtr = device->CreateBLAS(cmd, meshes, scene->numPrimitives);
                    }
                    break;
                    default: Assert(0);
                }
            }
        }
    }
    device->EndEvent(cmd);
}
} // namespace rt
