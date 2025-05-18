#include "cpu_scene.h"
#include "integrate.h"
#include "scene.h"

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

void BuildAllSceneBVHs(RenderParams2 *params, ScenePrimitives **scenes, int numScenes,
                       int maxDepth, Image *image)
{
    for (int depth = maxDepth; depth >= 0; depth--)
    {
        ParallelFor(0, numScenes, 1, [&](int jobID, int start, int count) {
            for (int i = start; i < start + count; i++)
            {
                if (scenes[i]->depth.load(std::memory_order_acquire) == depth)
                    BuildSceneBVHs(params->arenas, scenes[i], params->NDCFromCamera,
                                   params->cameraFromRender, params->height);
            }
        });
    }
}

} // namespace rt
