#ifndef CPU_SCENE_H
#define CPU_SCENE_H
namespace rt
{
struct SceneShapeParse
{
};
Mesh CopyMesh(SceneShapeParse *parse, Arena *arena, Mesh &mesh);
SceneShapeParse StartSceneShapeParse();
void EndSceneShapeParse(SceneShapeParse *parse);
} // namespace rt
#endif
