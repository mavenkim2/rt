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
    Sphere(Vec3f c1, Vec3f c2, f32 r, Material *m) : center(c1), radius(fmax(0.f, r)), material(m)
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
__forceinline PrimitiveType GetBaseType(u32 i) { return PrimitiveType(i & PrimitiveType_InstanceMask); }

struct Quad
{
    Quad() {}
    Quad(const Vec3f &q, const Vec3f &u, const Vec3f &v, Material *mat) : q(q), u(u), v(v), material(mat)
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
        if (fabs(denom) < 1e-8f)
            return false;

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
        if (!this->Hit(Ray(origin, direction), 0.0001f, infinity, rec))
            return 0;
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

    ConstantMedium(f32 density, Material *material) : negInvDensity(-1 / density), phaseFunction(material) {}

    template <typename T>
    bool Hit(const T &primitive, const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
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

    inline i32 GetInt(i32 i) const
    {
        return *(i32 *)(bytes[i]);
    }
    inline bool GetBool(i32 i) const
    {
        return *(bool *)(bytes[i]);
    }
    inline f32 GetFloat(i32 i) const
    {
        return *(f32 *)(bytes[i]);
    }
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

template <i32 index, typename T, typename... Ts>
struct RemoveFirstN;

template <i32 index, typename T, typename... Ts>
struct RemoveFirstN<index, TypePack<T, Ts...>>
{
    using type = typename RemoveFirstN<index - 1, TypePack<Ts...>>::type;
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

    u32 GetIndex() const
    {
        return id & indexMask;
    }
    u32 GetType() const
    {
        return id >> typeShift;
    }
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

// NOTE: only leaf scenes can
struct Scene2
{
    union
    {
        struct
        {
            QuadMesh *meshes;
            u32 numMeshes;
            u32 numPrims;
        };
        struct
        {
            Instance *instances;
            AffineSpace *affineTransforms;
        };
    };
    struct DiffuseAreaLight *lights;
    struct InfiniteLight *infiniteLights;
    BVHNodeType nodePtr;

    f32 minX, minY, minZ;
    f32 maxX, maxY, maxZ;
    u32 numInstances;
    u32 numLights;
    u32 numInfiniteLights;
    // union
    // {
    //     struct
    //     {
    // TriangleMesh* triMeshes;
    // Curve* curves;
    //     };
    // };

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
    inline void GetTypeAndLocalindex(const u32 totalIndex, PrimitiveType *type, u32 *localIndex) const;

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
