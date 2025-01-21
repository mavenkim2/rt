#define M_PI 3.1415926535897932
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/PatchTableFactory.h>
#include <opensubdiv/far/patchMap.h>

namespace rt
{

struct Vertex
{
    void Clear(void * = 0) { p = Vec3f(0.f); }
    void AddWithWeight(Vertex &src, f32 weight) { p += weight * src.p; }
    Vec3f p;
};

using namespace OpenSubdiv;
void Subdivide(Mesh *mesh)
{
    TempArena temp = ScratchStart(0, 0);
    // first do everything up front, and then do caching maybe, see embree's paper
    typedef Far::TopologyDescriptor Descriptor;
    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);

    i32 *numVertsPerFace = PushArrayNoZero(temp.arena, i32, mesh->numFaces);
    i32 *indices         = PushArrayNoZero(temp.arena, i32, mesh->numFaces * 4);
    for (u32 i = 0; i < mesh->numFaces; i++)
    {
        numVertsPerFace[i] = 4;
        indices[4 * i + 0] = 4 * i + 0;
        indices[4 * i + 1] = 4 * i + 1;
        indices[4 * i + 2] = 4 * i + 2;
        indices[4 * i + 3] = 4 * i + 3;
    }
    Descriptor desc;
    desc.numVertices        = mesh->numVertices;
    desc.numFaces           = mesh->numFaces;
    desc.numVertsPerFace    = numVertsPerFace;
    desc.vertIndicesPerFace = indices;

    Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Descriptor>::Create(
        desc, Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    int maxLevel = 2;
    // refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxLevel));

    // TODOs:
    // 1. how do you interpolate the normals durign subdivision?
    // 2. make this work with ptex
    // 3. adaptive subdivision based on screen space heuristics?
    // 4. multi resolution geometry cache?

    Far::PatchTableFactory::Options patchOptions;
    patchOptions.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
    auto *patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);

    Far::TopologyRefiner::AdaptiveOptions adaptiveOptions =
        patchOptions.GetRefineAdaptiveOptions();
    refiner->RefineAdaptive(adaptiveOptions);

    u32 size         = refiner->GetNumVerticesTotal();
    Vertex *vertices = PushArrayNoZero(temp.arena, Vertex, size);
    MemoryCopy(vertices, mesh->p, sizeof(Vec3f) * mesh->numVertices);

    Vertex *src        = vertices;
    i32 nRefinedLevels = refiner->GetNumLevels();
    Far::PrimvarRefiner primvarRefiner(*refiner);

    for (i32 i = 1; i <= maxLevel; i++)
    {
        Vertex *dst = src + refiner->GetLevel(i - 1).GetNumVertices();
        primvarRefiner.Interpolate(i, src, dst);
        src = dst;
    }

    OpenSubdiv::Far::PatchMap patchMap(*patchTable);
    Vec2f uv(0.5f);
    const Far::PatchTable::PatchHandle *handle = patchMap.FindPatch(0, uv[0], uv[1]);
    const auto &cvIndices                      = patchTable->GetPatchVertices(*handle);

    f32 pWeights[20];
    f32 duWeights[20];
    f32 dvWeights[20];
    patchTable->EvaluateBasis(*handle, uv[0], uv[1], duWeights, dvWeights);

    Vec3f pos  = {};
    Vec3f dpdu = {};
    Vec3f dpdv = {};
    for (i32 i = 0; i < cvIndices.size(); i++)
    {
        const Vec3f &p = vertices[cvIndices[i]].p;
        pos += pWeights[i] * p;
        dpdu += duWeights[i] * p;
        dpdv += dvWeights[i] * p;
    }

    delete refiner;
}

} // namespace rt
