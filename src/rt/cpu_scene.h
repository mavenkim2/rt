#ifndef CPU_SCENE_H
#define CPU_SCENE_H

#include "containers.h"

namespace rt
{
enum class GeometryType;
struct AffineSpace;
struct Arena;
struct Light;
struct MaterialNode;
struct Mesh;
struct PrimitiveIndices;
struct RenderParams2;
struct ScenePrimitives;
struct string;

enum class TessellationStyle
{
    ClosestInstance,
    PerInViewInstancePerEdge,
};

void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          HashMap<MaterialNode> *materialHashMap, Mesh &mesh,
                          ChunkedLinkedList<Mesh, MemoryType_Shape> &shapes,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices,
                          ChunkedLinkedList<Light *, MemoryType_Light> &lights);
void ComputeEdgeRates(ScenePrimitives *scene, const AffineSpace &transform,
                      const Vec4f *planes, TessellationStyle style);
void BuildAllSceneBVHs(RenderParams2 *params, ScenePrimitives **scenes, int numScenes,
                       int maxDepth);
} // namespace rt
#endif
