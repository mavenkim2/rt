#ifndef SCENE_H
#define SCENE_H

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/SampleFromVoxels.h>

// #include "lights.h"
namespace rt
{

class Sphere
{
public:
    Sphere() {}
    Sphere(Vec3f c, f32 r, Material *m) : center(c), radius(fmax(0.f, r)), material(m)
    {
        centerVec = Vec3f(0, 0, 0);
    }
    Sphere(Vec3f c1, Vec3f c2, f32 r, Material *m)
        : center(c1), radius(fmax(0.f, r)), material(m)
    {
        centerVec = c2 - c1;
    }
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const;
    Vec3f Center(f32 time) const;
    AABB GetAABB() const;
    static void GetUV(f32 &u, f32 &v, const Vec3f &p)
    {
        f32 zenith  = acos(-p.y);
        f32 azimuth = atan2(-p.z, p.x) + PI;

        u = azimuth / (2 * PI);
        v = zenith / PI;
    }
    f32 PdfValue(const Vec3f &origin, const Vec3f &direction) const;
    Vec3f Random(const Vec3f &origin, Vec2f u) const;

    Vec3f center;
    f32 radius;

private:
    Material *material;
    Vec3f centerVec;
};

typedef u32 PrimitiveType;
enum
{
    PrimitiveType_Sphere,
    PrimitiveType_Quad,
    PrimitiveType_Box,
    PrimitiveType_Triangle,
    PrimitiveType_Curve,
    PrimitiveType_Subdiv,

    // NOTE: triangle mesh instances only (for now)
    PrimitiveType_Instance     = 16,
    PrimitiveType_InstanceMask = PrimitiveType_Instance - 1,

    PrimitiveType_Count,
};

__forceinline b32 IsInstanced(u32 i) { return ~PrimitiveType_InstanceMask & i; }
__forceinline PrimitiveType GetBaseType(u32 i)
{
    return PrimitiveType(i & PrimitiveType_InstanceMask);
}

struct Disk
{
    f32 radius, height;
    const AffineSpace *objectFromRender;

    Disk(const AffineSpace *t, f32 radius = 1.f, f32 height = 0.f)
        : objectFromRender(t), radius(radius), height(height)
    {
    }
    bool Intersect(const Ray2 &r, SurfaceInteraction &intr, f32 tMax = pos_inf)
    {
        Vec3f oi = TransformP(*objectFromRender, Vec3f(r.o));
        Vec3f di = TransformV(*objectFromRender, Vec3f(r.d));

        // Compute plane intersection for disk
        // Reject disk intersections for rays parallel to the disk's plane
        if (f32(di.z) == 0) return false;

        f32 tShapeHit = (height - f32(oi.z)) / f32(di.z);
        if (tShapeHit <= 0 || tShapeHit >= tMax) return false;

        // See if hit point is inside disk radii and $\phimax$
        Vec3f pHit = Vec3f(oi) + (f32)tShapeHit * Vec3f(di);
        f32 dist2  = Sqr(pHit.x) + Sqr(pHit.y);
        if (dist2 > Sqr(radius)) return false;

        f32 phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0) phi += 2 * PI;

        const f32 phiMax = 2 * PI;
        // Find parametric representation of disk hit
        f32 u    = phi / phiMax;
        f32 rHit = Sqrt(dist2);
        f32 v    = (radius - rHit) / radius; //(radius - innerRadius);

        Vec3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
        Vec3f dpdv = Vec3f(pHit.x, pHit.y, 0) * (-radius) / rHit;
        // Vec3f dndu(0, 0, 0), dndv(0, 0, 0);

        // Refine disk intersection point
        pHit.z = height;

        // Return _SurfaceInteraction_ for quadric intersection
        intr = SurfaceInteraction(pHit, Normalize(Cross(dpdu, dpdv)), Vec2f(u, v));
        return true;

        // Return _QuadricIntersection_ for disk intersection
        // return QuadricIntersection{tShapeHit, pHit, phi};
    }
};

struct Quad
{
    Quad() {}
    Quad(const Vec3f &q, const Vec3f &u, const Vec3f &v, Material *mat)
        : q(q), u(u), v(v), material(mat)
    {
        Vec3f n = Cross(u, v);
        normal  = Normalize(n);
        d       = Dot(normal, q);
        w       = n / Dot(n, n);
        area    = Length(n);
    }

    inline AABB GetAABB() const
    {
        AABB bbox1  = AABB(q, q + u + v);
        AABB bbox2  = AABB(q + u, q + v);
        AABB result = AABB(bbox1, bbox2);
        return result;
    }

    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
    {
        f32 denom = Dot(normal, r.d);
        // if the ray is parallel to the plane
        if (fabs(denom) < 1e-8f) return false;

        f32 t = (d - Dot(normal, r.o)) / denom;

        if (t <= tMin || t >= tMax) return false;

        Vec3f intersection = r.at(t);

        Vec3f planarHitVector = intersection - q;
        f32 alpha             = Dot(w, Cross(planarHitVector, v));
        f32 beta              = Dot(w, Cross(u, planarHitVector));

        if (!(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1)) return false;

        record.u        = alpha;
        record.v        = beta;
        record.p        = intersection;
        record.t        = t;
        record.material = material;
        record.SetNormal(r, normal);
        return true;
    }

    f32 PdfValue(const Vec3f &origin, const Vec3f &direction) const
    {
        HitRecord rec;
        if (!this->Hit(Ray(origin, direction), 0.0001f, infinity, rec)) return 0;
        f32 distanceSquared = rec.t * rec.t * LengthSquared(direction);
        f32 cosine          = Abs(Dot(direction, rec.normal) / Length(direction));
        return distanceSquared / (cosine * area);
    };

    Vec3f Random(const Vec3f &origin, Vec2f random) const
    {
        Vec3f p = q + (random.x * u) + (random.y * v);
        return p - origin;
    }

    Vec3f q; // corner
    Vec3f u, v;
    Material *material;

    f32 area;
    // plane
    f32 d;
    Vec3f w;
    Vec3f normal;
};

struct Box
{
    Quad sides[6];
    Box() {}
    Box(const Vec3f &a, const Vec3f &b, Material *mat)
    {
        Vec3f min = Vec3f(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
        Vec3f max = Vec3f(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));

        Vec3f dx = Vec3f(max.x - min.x, 0, 0);
        Vec3f dy = Vec3f(0, max.y - min.y, 0);
        Vec3f dz = Vec3f(0, 0, max.z - min.z);

        sides[0] = Quad(Vec3f(min.x, min.y, max.z), dx, dy, mat);
        sides[1] = Quad(Vec3f(max.x, min.y, max.z), -dz, dy, mat);
        sides[2] = Quad(Vec3f(max.x, min.y, min.z), -dx, dy, mat);
        sides[3] = Quad(Vec3f(min.x, min.y, min.z), dz, dy, mat);
        sides[4] = Quad(Vec3f(min.x, max.y, max.z), dx, -dz, mat);
        sides[5] = Quad(Vec3f(min.x, min.y, min.z), dx, dz, mat);
    }

    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
    {
        bool hit    = false;
        f32 closest = tMax;
        HitRecord temp;
        for (u32 i = 0; i < ArrayLength(sides); i++)
        {
            const Quad &quad = sides[i];
            if (quad.Hit(r, tMin, tMax, temp))
            {
                if (temp.t < closest)
                {
                    closest = temp.t;
                    record  = temp;
                    hit     = true;
                }
            }
        }
        return hit;
    }
    AABB GetAABB() const
    {
        AABB result;
        for (u32 i = 0; i < ArrayLength(sides); i++)
        {
            result = AABB(result, sides[i].GetAABB());
        }
        return result;
    };
};

struct ConstantMedium
{
    f32 negInvDensity;
    Material *phaseFunction;

    ConstantMedium(f32 density, Material *material)
        : negInvDensity(-1 / density), phaseFunction(material)
    {
    }

    template <typename T>
    bool Hit(const T &primitive, const Ray &r, const f32 tMin, const f32 tMax,
             HitRecord &record) const
    {
        HitRecord rec1, rec2;
        if (!primitive.Hit(r, -infinity, infinity, rec1)) return false;
        // std::clog << "\n1. tMin=" << rec1.t;
        if (!primitive.Hit(r, rec1.t + 0.0001f, infinity, rec2))
        {
            // std::clog << "\ntMin=" << rec1.t << '\n'; //", tMax=" << rec2.t << '\n';
            return false;
        }

        if (rec1.t < tMin) rec1.t = tMin;
        if (rec2.t > tMax) rec2.t = tMax;

        if (rec1.t >= rec2.t)
        {
            // std::clog << "\ntMin=" << rec1.t << ", tMax=" << rec2.t;
            return false;
        }
        // std::clog << "\n2.";

        if (rec1.t < 0) rec1.t = 0;

        f32 rayLength              = Length(r.d);
        f32 distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;
        f32 hitDistance            = negInvDensity * log(RandomFloat());

        if (hitDistance > distanceInsideBoundary) return false;

        record.t           = rec1.t + hitDistance / rayLength;
        record.p           = r.at(record.t);
        record.normal      = Vec3f(1, 0, 0);
        record.isFrontFace = true;
        record.material    = phaseFunction;
        return true;
    }
};

// template <i32 N>
// struct Triangle
// {
//     LaneU32<N> geomIDs; // mesh
//     LaneU32<N> primIDs; // face
//
//     void Fill(Scene *scene, const PrimData *prim, u32 start, u32 end)
//     {
//         LaneU32<N> geomID;
//         LaneU32<N> geomID;
//
//         Assert(end - start < N);
//         u32 i = 0;
//         for (; i < N && start < end; i++, start++)
//         {
//             geomID[i] = prim[start].GeomID();
//             primID[i] = prim[start].PrimID();
//         }
//         for (; i < N - (end - start); i++, start++)
//         {
//             geomID[i] = geomID[0];
//             primID[i] = -1;
//         }
//     }
// };

struct TriangleMesh
{
    Vec3f *p;
    Vec3f *n;
    // Vec3f *t;
    Vec2f *uv;
    u32 *indices;
    u32 numVertices;
    u32 numIndices;

    static __forceinline AABB Bounds(TriangleMesh *mesh, u32 faceIndex)
    {
        Assert(faceIndex < mesh->numIndices / 3);
        AABB result;
        result.Extend(mesh->p[mesh->indices[faceIndex * 3 + 0]]);
        result.Extend(mesh->p[mesh->indices[faceIndex * 3 + 1]]);
        result.Extend(mesh->p[mesh->indices[faceIndex * 3 + 2]]);
        return result;
    }
};

struct QuadMesh
{
    Vec3f *p;
    Vec3f *n;
    u32 numVertices;

    // TEMP
    bool Intersect(const Ray2 &r, SurfaceInteraction &intr, f32 tMax = pos_inf)
    {
        bool hit = false;
        for (u32 quadIndex = 0; quadIndex < numVertices / 4.f; quadIndex++)
        {
            Vec3f p0     = p[quadIndex * 4 + 0];
            Vec3f p1     = p[quadIndex * 4 + 1];
            Vec3f p2     = p[quadIndex * 4 + 2];
            Vec3f p3     = p[quadIndex * 4 + 3];
            Vec3f u      = p1 - p0;
            Vec3f v      = p3 - p0;
            Vec3f normal = Cross(u, v);
            f32 denom    = 1.f / Dot(normal, r.d);
            if (Abs(denom) < 1e-8f) continue;
            // Find plane equation
            f32 d = Dot(normal, p0);
            f32 t = (d - Dot(normal, r.o)) * denom;
            if (t <= tMinEpsilon || t >= tMax) continue;
            Vec3f intersection    = r(t);
            Vec3f planarHitVector = intersection - p0;
            Vec3f w               = normal / LengthSquared(normal);
            f32 alpha             = Dot(w, Cross(planarHitVector, v));
            f32 beta              = Dot(w, Cross(u, planarHitVector));
            if (!(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1)) continue;
            intr.p    = intersection;
            intr.n    = normal;
            intr.uv   = Vec2f(alpha, beta);
            intr.tHit = t;
            hit       = true;
        }
        return hit;
    }
    // u32 numQuads;
};

//////////////////////////////

struct SceneHandle
{
    u32 offset;
    u32 count;
};

struct PrimitiveIndices
{
    i32 transformIndex     = -1;
    i32 constantMediaIndex = -1;
};

struct Light
{
    PrimitiveType type;
    void *primitive;
};

struct ScatterRecord
{
    Vec3f attenuation;
    Ray skipPDFRay;
    Vec3f sample;
};

struct HomogeneousTransform
{
    Vec3f translation;
    f32 rotateAngleY;
};

struct ScenePacket
{
    StringId *parameterNames;
    u8 **bytes;
    u32 *sizes;

    StringId type;

    // const string **parameterNames;
    // SceneByteType *types;
    u32 parameterCount;

    void Initialize(Arena *arena, u32 count)
    {
        // parameterCount = count;
        // parameterCount = 0;
        parameterNames = PushArray(arena, StringId, count);
        bytes          = PushArray(arena, u8 *, count);
        sizes          = PushArray(arena, u32, count);
        // types          = PushArray(arena, SceneByteType, count);
    }

    inline i32 GetInt(i32 i) const { return *(i32 *)(bytes[i]); }
    inline bool GetBool(i32 i) const { return *(bool *)(bytes[i]); }
    inline f32 GetFloat(i32 i) const { return *(f32 *)(bytes[i]); }
    // inline u8 *GetByParamName(const string &name) const
    // {
    //     for (u32 i = 0; i < parameterCount; i++)
    //     {
    //         if (*parameterNames[i] == name)
    //         {
    //             return bytes[i];
    //         }
    //     }
    //     return 0;
    // }
};

enum GeometryType
{
    GT_QuadMeshType = 0,
    GT_InstanceType = 1,
};

struct GeometryID
{
    static const u32 indexMask = 0x0fffffff;
    static const u32 typeShift = 28;

    // static const u32 sphereType       = 0;
    // static const u32 quadType         = 1;
    // static const u32 boxType          = 2;
    // static const u32 triangleMeshType = 3;
    // static const u32 curveType        = 4;
    // static const u32 subdivType       = 5;
    // static const u32 instanceType     = 16;
    static const u32 quadMeshType = GT_QuadMeshType;
    static const u32 instanceType = GT_InstanceType;

    u32 id;

    GeometryID(u32 id) : id(id) {}

    static GeometryID CreateInstanceID(u32 index)
    {
        Assert(index < (1 << 28));
        return (instanceType << typeShift) | (index & indexMask);
    }
    static GeometryID CreateQuadMeshID(u32 index)
    {
        Assert(index < (1 << 28));
        return (quadMeshType << typeShift) | (index & indexMask);
    }

    u32 GetIndex() const { return id & indexMask; }
    u32 GetType() const { return id >> typeShift; }
};

struct Instance
{
    // TODO: materials
    GeometryID geomID;
    u32 transformIndex;
};

// struct QuadMeshGroup
// {
//     QuadMesh *meshes;
//     BVHNode4 nodePtr;
//     u32 numMeshes;
// };

struct Volume
{
    u32 shapeIndex;
    f32 Extinction(const Vec3f &p, f32 time, f32 filterWidth) const;
    void QueryExtinction(const Bounds &bounds, f32 &cMin, f32 &cMaj) const;
    // PhaseFunction PhaseFunction() const;
};

struct NanoVDBBuffer
{
    TempArena arena;
    u64 allocSize;
    u8 *ptr;
    // NOTE: kind of messy, but the buffer owns the arena
    NanoVDBBuffer() = default;
    NanoVDBBuffer(u64 size, Arena *arena) : arena(TempBegin(arena)) { init(size); }
    u64 size() const { return allocSize; }
    const u8 *data() const { return ptr; }
    u8 *data() { return ptr; }

    void init(u64 size)
    {
        if (size == allocSize) return;
        if (allocSize > 0) clear();
        if (size == 0) return;
        allocSize = size;
        ptr       = PushArrayNoZero(arena.arena, u8, allocSize);
    }
    static NanoVDBBuffer create(u64 size, const NanoVDBBuffer *context = 0)
    {
        return NanoVDBBuffer(size, context ? context->arena.arena : ArenaAlloc());
    }
    void clear()
    {
        TempEnd(arena);
        allocSize = 0;
        ptr       = 0;
    }
};

f32 HenyeyGreenstein(f32 cosTheta, f32 g)
{
    g         = Clamp(g, -.99f, .99f);
    f32 denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4Pi * (1 - Sqr(g)) / (denom * SafeSqrt(denom));
}

f32 HenyeyGreenstein(Vec3f wo, Vec3f wi, f32 g) { return HenyeyGreenstein(Dot(wo, wi), g); }

Vec3f SampleHenyeyGreenstein(const Vec3f &wo, f32 g, Vec2f u, f32 *pdf = 0)
{
    f32 cosTheta;
    if (Abs(g) < 1e-3f) cosTheta = 1 - 2 * u[0];
    else cosTheta = -1 / (2 * g) * (1 + Sqr(g) - Sqr((1 - Sqr(g)) / (1 + g - 2 * g * u[0])));

    f32 sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    f32 phi      = TwoPi * u[1];

    // TODO: implement FromZ
    Vec3f wi;
    // Frame wFrame = Frame::FromZ(wo);
    // Vector3f wi  = wFrame.FromLocal(Vec3f(sinTheta * Cos(phi), sinTheta * Sin(phi),
    // cosTheta));

    if (pdf) *pdf = HenyeyGreenstein(cosTheta, g);
    return wi;
}

struct PhaseFunctionSample
{
    Vec3f wi;
    f32 p;
    f32 pdf = 0.f;
    PhaseFunctionSample() {}
    PhaseFunctionSample(const Vec3f &wi, f32 p, f32 pdf = 0.f) : wi(wi), p(p), pdf(pdf) {}
};

struct PhaseFunction
{
    f32 g;
    PhaseFunction() {}
    PhaseFunction(f32 g) : g(g) {}
    // NOTE: HG phase function is perfectly importance sampled, so the value of the
    // phasefunction = the pdf
    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, f32 *pdf) const
    {
        Assert(pdf);
        f32 p = HenyeyGreenstein(wo, wi, g);
        *pdf  = p;
        return SampledSpectrum(p);
    }
    PhaseFunctionSample GenerateSample(Vec3f wo, Vec2f u) const
    {
        f32 pdf;
        Vec3f wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{wi, pdf, pdf};
    }
    f32 PDF(Vec3f wo, Vec3f wi) const { return HenyeyGreenstein(wo, wi, g); }
};

struct NanoVDBVolume
{
    const AffineSpace *renderFromMedium;
    const AffineSpace mediumFromRender;
    static nanovdb::GridHandle<NanoVDBBuffer> ReadGrid(string str, string type)
    {
        nanovdb::GridHandle<NanoVDBBuffer> handle;
        try
        {
            handle = nanovdb::io::readGrid<NanoVDBBuffer>(
                std::string((const char *)str.str, str.size),
                std::string((const char *)type.str, type.size));
        } catch (std::exception)
        {
            Error(0, "Could not read file: %S\n", str);
        }
        return handle;
    }

    nanovdb::GridHandle<NanoVDBBuffer> densityGrid;
    nanovdb::GridHandle<NanoVDBBuffer> temperatureGrid;
    DenselySampledSpectrum cAbs;
    DenselySampledSpectrum cScatter;
    const nanovdb::FloatGrid *densityFloatGrid     = 0;
    const nanovdb::FloatGrid *temperatureFloatGrid = 0;
    // NOTE: world space bounds
    Bounds bounds;
    f32 LeScale, temperatureOffset, temperatureScale, cScale;
    PhaseFunction phaseFunction;

    NanoVDBVolume() {}
    NanoVDBVolume(string filename, const AffineSpace *renderFromMedium, Spectrum cAbs,
                  Spectrum cScatter, f32 g, f32 cScale, f32 LeScale = 1.f,
                  f32 temperatureOffset = 0.f, f32 temperatureScale = 1.f)
        : mediumFromRender(Inverse(*renderFromMedium)), cAbs(DenselySampledSpectrum(cAbs)),
          cScatter(DenselySampledSpectrum(cScatter)), phaseFunction(g), cScale(cScale),
          LeScale(LeScale), temperatureOffset(temperatureOffset),
          temperatureScale(temperatureScale)
    {
        densityGrid          = ReadGrid(filename, "density");
        temperatureGrid      = ReadGrid(filename, "temperature");
        densityFloatGrid     = densityGrid.grid<f32>();
        temperatureFloatGrid = temperatureGrid.grid<f32>();

        nanovdb::BBox<nanovdb::Vec3R> bbox = densityFloatGrid->worldBBox();
        bounds                             = Transform(
            *renderFromMedium,
            Bounds(Vec3f((f32)bbox.min()[0], (f32)bbox.min()[1], (f32)bbox.min()[2]),
                                               Vec3f((f32)bbox.max()[0], (f32)bbox.max()[1], (f32)bbox.max()[2])));

        nanovdb::BBox<nanovdb::Vec3R> bbox2 = temperatureFloatGrid->worldBBox();
        bounds.Extend(Transform(
            *renderFromMedium,
            Bounds(Vec3f((f32)bbox2.min()[0], (f32)bbox2.min()[1], (f32)bbox2.min()[2]),
                   Vec3f((f32)bbox2.max()[0], (f32)bbox2.max()[1], (f32)bbox2.max()[2]))));
    }

    SampledSpectrum Le(Vec3f p, const SampledWavelengths &lambda) const
    {
        // p = Transform(*mediumFromRender, p);
        if (!temperatureFloatGrid) return SampledSpectrum(0.f);
        nanovdb::Vec3<f32> pIndex =
            temperatureFloatGrid->worldToIndexF(nanovdb::Vec3<f32>(p.x, p.y, p.z));
        using TreeSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        f32 temp          = TreeSampler(temperatureFloatGrid->tree())(pIndex);
        temp              = (temp - temperatureOffset) * temperatureScale;
        if (temp <= 100.f) return SampledSpectrum(0.f);
        return LeScale * BlackbodySpectrum(temp).Sample(lambda);
    }
    void Extinction(Vec3f p, const SampledWavelengths &lambda, SampledSpectrum &outAbs,
                    SampledSpectrum &outScatter, SampledSpectrum &le) const //, f32, f32) const
    {
        // p = ApplyInverse(*renderFromMedium, p);
        p = TransformP(mediumFromRender, p);
        nanovdb::Vec3<f32> pIndex =
            densityFloatGrid->worldToIndexF(nanovdb::Vec3<f32>(p.x, p.y, p.z));
        using TreeSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        f32 density       = TreeSampler(densityFloatGrid->tree())(pIndex);

        outAbs     = cAbs.Sample(lambda) * density;
        outScatter = cScatter.Sample(lambda) * density;
        le         = Le(p, lambda);
    }
    const PhaseFunction &PhaseFunction() const { return phaseFunction; }
    void QueryExtinction(Bounds inBounds, f32 &cMin, f32 &cMaj) const
    {
        inBounds = Transform(mediumFromRender, inBounds);

        if (!Intersects(bounds, inBounds))
        {
            cMin = 0.f;
            cMaj = 0.f;
            return;
        }

        nanovdb::Vec3<f32> i0 = densityFloatGrid->worldToIndexF(
            nanovdb::Vec3<f32>(inBounds.minP[0], inBounds.minP[1], inBounds.minP[2]));
        nanovdb::Vec3<f32> i1 = densityFloatGrid->worldToIndexF(
            nanovdb::Vec3<f32>(inBounds.maxP[0], inBounds.maxP[1], inBounds.maxP[2]));

        struct MediumData
        {
            f32 cMin, cMaj;
        };

        Vec3i begin((i32)i0[0] - 1, (i32)i0[1] - 1, (i32)i0[2] - 1);
        Vec3i end((i32)i1[1] + 1, (i32)i1[1] + 1, (i32)i1[2] + 1);

        Vec3i width = end - begin;

        MediumData datum;
        ParallelReduce(
            &datum, 0, width.x * width.y * width.z, PARALLEL_THRESHOLD,
            [&](MediumData &data, u32 jobID, u32 start, u32 count) {
                auto accessor = densityFloatGrid->getAccessor();
                f32 cMin      = pos_inf;
                f32 cMax      = neg_inf;

                // TODO: see if loop carried dependency, or index computation, is significant
                // overhead vs access time
                for (u32 i = start; i < count; i++)
                {
                    i32 nx    = begin[0] + (i % width[0]);
                    i32 ny    = begin[1] + ((i / width[0]) % width[1]);
                    i32 nz    = begin[2] + (i / (width[0] * width[1]));
                    f32 value = accessor.getValue({nx, ny, nz});
                    cMin      = Min(cMin, value);
                    cMax      = Max(cMax, value);
                }
                datum.cMin = cMin;
                datum.cMaj = cMax;
            },
            [&](MediumData &left, const MediumData &right) {
                left.cMin = Min(left.cMin, right.cMin);
                left.cMaj = Max(left.cMaj, right.cMaj);
            });
        cMin = datum.cMin;
        cMaj = datum.cMaj;
    }
};

// template <typename Shape>
// struct Primitive
// {
//     ShapeHandle handle;
//     u32 lightIndex;
//     u32 volumeIndex;
//     u32 materialID;
// };

// NOTE: only leaf scenes can
struct Scene2
{
    // using ShapeTypes = TypePack<QuadMesh, Disk>;
    // using VolumeTypes        = TypePack<NanoVDBVolume>;
    // using MaterialTypes = TypePack<DielectricMaterialBase>;
    // using MaterialTypes = TypePack<CoatedDiffuseMaterial1>;

    // TODO: this really should adjacent in memory to the primitives
    struct PrimitiveIndices
    {
        // TODO: these are actaully ids (type + index)
        LightHandle lightID;
        // u32 volumeIndex;
        MaterialHandle materialID;
        PrimitiveIndices() {}
        PrimitiveIndices(LightHandle lightID, MaterialHandle materialID)
            : lightID(lightID), materialID(materialID)
        {
        }
    };

    union
    {
        struct
        {
            QuadMesh *meshes;
            u32 numMeshes;
            u32 numPrims;
            TriangleMesh *triangleMeshes;
            u32 numTriMeshes;
        };
        struct
        {
            Instance *instances;
            AffineSpace *affineTransforms;
        };
    };
    // Volumes
    // Volume *volumes;
    // ArrayTuple<ShapeTypes> primitives;
    // const PrimitiveIndices **primIndices;
    const PrimitiveIndices *primIndices;

    // ArrayTuple<VolumeTypes> volumes;
    // VolumeAggregate aggregate;

    // Lights
    ArrayTuple<LightTypes> lights;
    // u32 lightPDF[LightClass_Count];
    // u32 lightCount[LightClass_Count];
    // u32 numAreaLights;
    // u32 numInfiniteLights;
    u32 numLights; // total

    // Materials
    ArrayTuple<MaterialTypes> materials;

    // BVH
    BVHNodeN nodePtr;

    f32 minX, minY, minZ;
    f32 maxX, maxY, maxZ;
    u32 numInstances;
    // u32 numVolumes;
    // union
    // {
    //     struct
    //     {
    // TriangleMesh* triMeshes;
    // Curve* curves;
    //     };
    // };

    DiffuseAreaLight *GetAreaLights() { return lights.Get<DiffuseAreaLight>(); }
    const DiffuseAreaLight *GetAreaLights() const { return lights.Get<DiffuseAreaLight>(); }
    Bounds GetBounds() const
    {
        Bounds result;
        result.minP = Lane4F32::LoadU(&minX);
        result.maxP = Lane4F32::LoadU(&maxX);
        return result;
    }
    void SetBounds(const Bounds &bounds)
    {
        Lane4F32::StoreU(&minX, bounds.minP);
        Lane4F32::StoreU(&maxX, bounds.maxP);
    }
};

struct Scene2 *scene_;
Scene2 *GetScene() { return scene_; }

struct Scene
{
    static const u32 indexMask = 0x0fffffff;
    static const u32 typeShift = 28;

    Sampler sampler;

    Sphere *spheres;
    Quad *quads;
    Box *boxes;
    TriangleMesh *meshes;

    PrimitiveIndices *primitiveIndices;

    HomogeneousTransform *transforms;
    ConstantMedium *media;

    u32 sphereCount;
    u32 quadCount;
    u32 boxCount;
    u32 meshCount;
    u32 totalPrimitiveCount;

    // PrimitiveType_Count arrays
    u32 counts[PrimitiveType_Count];

    u32 transformCount;
    u32 mediaCount;

    void FinalizePrimitives();

    inline i32 GetIndex(PrimitiveType type, i32 primIndex) const;
    inline void GetTypeAndLocalindex(const u32 totalIndex, PrimitiveType *type,
                                     u32 *localIndex) const;

    void AddConstantMedium(PrimitiveType type, i32 primIndex, i32 constantMediumIndex);

    void AddTransform(PrimitiveType type, i32 primIndex, i32 transformIndex);

    void GetAABBs(AABB *aabbs);
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &temp, u32 index);

    template <typename T>
    __forceinline const T *Get(u32 i) const
    {
        StaticAssert(false, UnspecifiedPrimType);
    }

    template <>
    __forceinline const Sphere *Get<Sphere>(u32 i) const
    {
        Assert(i < sphereCount);
        return &spheres[i];
    }

    template <>
    __forceinline const Quad *Get<Quad>(u32 i) const
    {
        Assert(i < quadCount);
        return &quads[i];
    }

    template <>
    __forceinline const Box *Get<Box>(u32 i) const
    {
        Assert(i < boxCount);
        return &boxes[i];
    }

    template <>
    __forceinline const TriangleMesh *Get<TriangleMesh>(u32 i) const
    {
        Assert(i < meshCount);
        return &meshes[i];
    }
};
} // namespace rt
#endif
