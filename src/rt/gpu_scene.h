#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "vulkan.h"

namespace rt
{
struct Mesh;

struct SceneShapeParse
{
    TransferCommandBuffer buffer;
};

GPUMesh CopyMesh(SceneShapeParse *parse, Arena *arena, Mesh &mesh);
SceneShapeParse StartSceneShapeParse();
void EndSceneShapeParse(SceneShapeParse *parse);
} // namespace rt
#endif
