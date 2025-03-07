#include "cpu_scene.h"
#include "scene.h"

namespace rt
{

SceneShapeParse StartSceneShapeParse() { return {}; }
void EndSceneShapeParse(SceneShapeParse *parse) {}

Mesh CopyMesh(SceneShapeParse *parse, Arena *arena, Mesh &mesh)
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

void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          MaterialHashMap *materialHashMap, Mesh &mesh,
                          ChunkedLinkedList<MeshType, MemoryType_Shape> &shapes,
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
        transform          = &scene->affineTransforms[transformIndex];
        Assert(mesh.n == 0);
        // Convert points to world space for BVH (since object space is
        // world space in this case)
        Assert(mesh.numVertices == 4);

        for (int i = 0; i < mesh.numVertices; i++)
        {
            Vec3f result = TransformP(worldFromRender * *transform, mesh.p[i]);
            mesh.p[i]    = result;
        }

        transform = &renderFromWorld;

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
