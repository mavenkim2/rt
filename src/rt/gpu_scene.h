#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "vulkan.h"

namespace rt
{
enum class GeometryType;
struct Light;
struct MaterialNode;
struct Mesh;
struct PrimitiveIndices;

struct SceneShapeParse
{
    TransferCommandBuffer buffer;
};

GPUMesh CopyMesh(SceneShapeParse *parse, Arena *arena, Mesh &mesh);
void AddMaterialAndLights(Arena *arena, Tokenizer &tokenizer,
                          HashMap<MaterialNode> *materialHashMap, GPUMesh &mesh,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices);
void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          HashMap<MaterialNode> *materialHashMap, GPUMesh &mesh,
                          ChunkedLinkedList<GPUMesh, MemoryType_Shape> &shapes,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices,
                          ChunkedLinkedList<Light *, MemoryType_Light> &lights);
SceneShapeParse StartSceneShapeParse();
void EndSceneShapeParse(SceneShapeParse *parse);
} // namespace rt
#endif
