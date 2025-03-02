#ifndef USE_GPU
namespace rt
{

Mesh CopyMesh(Arena *arena, Mesh &mesh)
{
    Mesh newMesh        = {};
    newMesh.numVertices = mesh.numVertices;
    newMesh.numIndices  = mesh.numIndices;
    newMesh.numFaces    = mesh.numFaces;
    newMesh.p = PushArrayNoZeroTagged(arena, Vec3f, mesh.numVertices, MemoryType_Shape);
    MemoryCopy(newMesh.p, mesh.p, sizeof(Vec3f) * mesh.numVertices);
    if (mesh.n)
    {
        newMesh.n = PushArrayNoZeroTagged(arena, Vec3f, mesh.numVertices, MemoryType_Shape);
        MemoryCopy(newMesh.n, mesh.n, sizeof(Vec3f) * mesh.numVertices);
    }
    if (mesh.uv)
    {
        newMesh.uv = PushArrayNoZeroTagged(arena, Vec2f, mesh.numVertices, MemoryType_Shape);
        MemoryCopy(newMesh.uv, mesh.uv, sizeof(Vec2f) * mesh.numVertices);
    }
    if (mesh.indices)
    {
        newMesh.indices = PushArrayNoZero(arena, u32, mesh.numIndices);
        MemoryCopy(newMesh.indices, mesh.indices, sizeof(u32) * mesh.numIndices);
    }
    return newMesh;
}

void BuildSceneBVHs(ScenePrimitives **scenes, int numScenes, int maxDepth)
{
    for (int depth = maxDepth; depth >= 0; depth--)
    {
        ParallelFor(0, numScenes, 1, [&](int jobID, int start, int count) {
            for (int i = start; i < start + count; i++)
            {
                if (scenes[i]->depth.load(std::memory_order_acquire) == depth)
                    BuildSceneBVHs(arenas, scenes[i], NDCFromCamera, cameraFromRender,
                                   screenHeight);
            }
        });
    }
}

} // namespace rt
#endif
