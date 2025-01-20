#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>

namespace rt
{

struct Vertex
{
    void Clear(void * = 0)
    {
        p[0] = 0;
        p[1] = 0;
        p[2] = 0;
    }
    void AddWithWeights(Vertex &src, f32 weight)
    {
        p[0] += weight * src.p[0];
        p[1] += weight * src.p[1];
        p[2] += weight * src.p[2];
    }
    f32 p[3];
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

    Descriptor desc;
    desc.numVertices        = mesh->numVertices;
    desc.numFaces           = mesh->numFaces;
    desc.numVertsPerFace    = 4;
    desc.vertIndicesPerFace = ;

    Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Descriptor>::Create(
        desc, Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    int maxLevel = 2;
    // refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxLevel));
    refiner->RefineAdaptive(Far::TopologyRefiner::AdaptiveOptions();

    // TODOs:
    // 1. how do you interpolate the normals durign subdivision?
    // 2. make this work with ptex
    // 3. adaptive subdivision based on screen space heuristics?

    u32 size         = refiner->GetNumVerticesTotal();
    Vertex *vertices = PushArrayNoZero(temp.arena, Vertex, size);
    MemoryCopy(vertices, mesh->p, sizeof(Vec3f) * mesh->numVertices);

    Vertex *src = vertices;
    Far::PrimvarRefiner primvarRefiner(*refiner);

    for (i32 i = 1; i <= maxLevel; i++)
    {
        Vertex *dst = src + refiner->GetLevel(i - 1).GetNumVertices();
        primvarRefiner.Interpolate(i, src, dst);
        src = dst;
    }

    delete refiner;
}

} // namespace rt
