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
    CommandBuffer *buffer;
    Semaphore semaphore;
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
void EndSceneShapeParse(ScenePrimitives *scene, SceneShapeParse *parse);

typedef Mat4 float4x4;
typedef Vec3f float3;
typedef AffineSpace float3x4;

} // namespace rt

#endif
