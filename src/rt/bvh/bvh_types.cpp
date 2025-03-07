#include "bvh_types.h"
#include "../scene.h"
namespace rt
{

void TLASLeaf::GetData(const ScenePrimitives *scene, AffineSpace *&t,
                       ScenePrimitives *&childScene)
{
    t          = &scene->affineTransforms[transformIndex];
    childScene = scene->childScenes[GetSceneIndex()];
}

} // namespace rt
