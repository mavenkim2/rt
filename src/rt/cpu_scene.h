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
struct ScenePrimitives;
struct string;

struct SceneShapeParse
{
};
Mesh ProcessMesh(SceneShapeParse *parse, Arena *arena, Mesh &mesh);
SceneShapeParse StartSceneShapeParse();
void EndSceneShapeParse(ScenePrimitives *scene, SceneShapeParse *parse);
void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          HashMap<MaterialNode> *materialHashMap, Mesh &mesh,
                          ChunkedLinkedList<Mesh, MemoryType_Shape> &shapes,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices,
                          ChunkedLinkedList<Light *, MemoryType_Light> &lights);
} // namespace rt
#endif
