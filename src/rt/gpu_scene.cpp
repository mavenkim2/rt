#include "vulkan.h"

namespace rt
{

GPUMesh CopyMesh(Arena *arena, Mesh &mesh)
{
    GPUMesh result;
    GPUBuffer &vertexBuffer = result.vertexBuffer;
    buffer.desc.size        = sizeof(mesh.p[0]) * mesh.numVertiices;

    device->CreateBuffer(&vertexBuffer, buffer.desc, mesh.p);

    GPUBuffer &indexBuffer = result.indexBuffer;
    indexBuffer.desc.size  = sizeof(mesh.indices[0]) * mesh.numIndices;

    device->CreateBuffer(&indexBuffer, indexBuffer.desc, mesh.indices);

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
