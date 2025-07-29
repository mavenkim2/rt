#ifndef SCENE_SCENE_H_
#define SCENE_SCENE_H_

#include "../memory.h"
#include "../mesh.h"
#include "../handles.h"
#include "../surface_interaction.h"

namespace rt
{

struct Ray2;
template <int N>
struct StackEntry;
struct Texture;

struct ShapeSample
{
    Vec3f p;
    Vec3f n;
    Vec3f w;
    f32 pdf;
};

struct TessellationParams
{
    Bounds bounds;
    AffineSpace transform;
    f64 currentMinDistance;
    Mutex mutex;
};

struct PrimitiveIndices
{
    LightHandle lightID;
    // u32 volumeIndex;
    MaterialHandle materialID;
    Texture *alphaTexture;

    PrimitiveIndices() {}
    PrimitiveIndices(LightHandle lightID, MaterialHandle materialID);
    PrimitiveIndices(LightHandle lightID, MaterialHandle materialID, Texture *alpha);
};

struct Instance
{
    // TODO: materials
    u32 id;
    // GeometryID geomID;
    u32 transformIndex;
};

struct ScenePrimitives
{
    static const int maxSceneDepth = 4;
    typedef bool (*IntersectFunc)(ScenePrimitives *, StackEntry<DefaultN>, Ray2 &,
                                  SurfaceInteractions<1> &);
    typedef bool (*OccludedFunc)(ScenePrimitives *, StackEntry<DefaultN>, Ray2 &);

    string filename;

    GeometryType geometryType;

    Vec3f boundsMin;
    Vec3f boundsMax;
    BVHNodeN nodePtr;

    // NOTE: is one of PrimitiveType
    void *primitives;
    int bvhPrimSize;

    // NOTE: only set if not a leaf node in the scene hierarchy
    union
    {
        ScenePrimitives **childScenes;
        TessellationParams *tessellationParams;
    };
    u32 numChildScenes;
    AffineSpace *affineTransforms;
    PrimitiveIndices *primIndices;

    std::atomic<int> depth;
    u32 numTransforms;
    IntersectFunc intersectFunc;
    OccludedFunc occludedFunc;
    u32 numPrimitives, numFaces;

    int sceneIndex;
    int gpuInstanceID;

    ScenePrimitives() {}
    Bounds GetBounds() const { return Bounds(Lane4F32(boundsMin), Lane4F32(boundsMax)); }
    void SetBounds(const Bounds &inBounds)
    {
        boundsMin = ToVec3f(inBounds.minP);
        boundsMax = ToVec3f(inBounds.maxP);
    }

    template <GeometryType type>
    void BuildBVH(Arena **arenas);
    void BuildQuadBVH(Arena **arenas);
    void BuildTriangleBVH(Arena **arenas);
#ifndef USE_GPU
    void BuildCatClarkBVH(Arena **arenas);
#endif
    void BuildTLASBVH(Arena **arenas);
    void BuildSceneBVHs(Arena **arenas, const Mat4 &NDCFromCamera,
                        const Mat4 &cameraFromRender, int screenHeight);

    Bounds GetSceneBounds();
    Bounds GetTLASBounds(u32 start, u32 count);

    void GenerateBuildRefs(BRef *refs, u32 start, u32 count, RecordAOSSplits &record);
    BRef *GenerateBuildRefs(Arena *arena, RecordAOSSplits &record);

    bool Occluded(Ray2 &ray);
    bool Intersect(Ray2 &ray, SurfaceInteraction &si);

    ShapeSample SampleQuad(SurfaceInteraction &intr, Vec2f &u, AffineSpace *transform,
                           int geomID);
    ShapeSample Sample(SurfaceInteraction &intr, AffineSpace *space, Vec2f &u, int geomID);
};

extern ScenePrimitives **scenes_;
inline ScenePrimitives **GetScenes() { return scenes_; }
inline void SetScenes(ScenePrimitives **scenes) { scenes_ = scenes; }

inline Mesh *GetMesh(int sceneID, int geomID)
{
    Assert(scenes_);
    ScenePrimitives *s = scenes_[sceneID];
    Assert(s->geometryType != GeometryType::Instance);
    return (Mesh *)s->primitives + geomID;
}

} // namespace rt

#endif
