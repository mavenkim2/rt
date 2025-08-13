#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "graphics/vulkan.h"

namespace rt
{
enum class GeometryType;
struct Light;
struct MaterialNode;
struct Mesh;
struct ScenePrimitives;
struct PrimitiveIndices;

void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          HashMap<MaterialNode> *materialHashMap, Mesh &mesh,
                          ChunkedLinkedList<Mesh, MemoryType_Shape> &shapes,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices,
                          ChunkedLinkedList<Light *, MemoryType_Light> &lights);

} // namespace rt

#endif
