#include "rt.h"

#define FINAL 1
#define EMISSIVE

using std::sqrt;

const f32 infinity = std::numeric_limits<f32>::infinity();
#define PI 3.1415926535897932385f

inline f32 DegreesToRadians(f32 degrees)
{
    return degrees * PI / 180.f;
}

//////////////////////////////
// Intervals
//
bool IsInInterval(f32 min, f32 max, f32 x)
{
    return x >= min && x <= max;
}

static vec3 BACKGROUND;

//////////////////////////////
// Random
//
inline f32 RandomFloat()
{
    return rand() / (RAND_MAX + 1.f);
}

inline f32 RandomFloat(f32 min, f32 max)
{
    return min + (max - min) * RandomFloat();
}

inline i32 RandomInt(i32 min, i32 max)
{
    return i32(RandomFloat(f32(min), f32(max)));
}

inline vec3 RandomVec3()
{
    return vec3(RandomFloat(), RandomFloat(), RandomFloat());
}

inline vec3 RandomVec3(f32 min, f32 max)
{
    return vec3(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max));
}

inline vec3 RandomUnitVector()
{
    while (true)
    {
        vec3 result = RandomVec3(-1, 1);
        if (result.lengthSquared() < 1)
        {
            return normalize(result);
        }
    }
}

inline vec3 RandomOnHemisphere(const vec3 &normal)
{
    // NOTE: why can't you just normalize a vector that has a length > 1?
    vec3 result = RandomUnitVector();
    result      = dot(normal, result) > 0 ? result : -result;
    return result;
}

inline vec3 RandomInUnitDisk()
{
    while (true)
    {
        vec3 p = vec3(RandomFloat(-1, 1), RandomFloat(-1, 1), 0);
        if (p.lengthSquared() < 1)
        {
            return p;
        }
    }
}

class Ray
{
public:
    Ray() {}
    Ray(const vec3 &origin, const vec3 &direction) : orig(origin), dir(direction), tm(0) {}
    Ray(const vec3 &origin, const vec3 &direction, const f32 time) : orig(origin), dir(direction), tm(time) {}

    const vec3 &origin() const { return orig; }
    const vec3 &direction() const { return dir; }
    const f32 &time() const { return tm; }

    vec3 at(f32 t) const
    {
        return orig + t * dir;
    }

private:
    vec3 orig;
    vec3 dir;
    f32 tm;
};

inline vec3 LinearToSRGB(const vec3 &v)
{
    return vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

f32 ExactLinearToSRGB(f32 l)
{
    if (l < 0.0f)
    {
        l = 0.0f;
    }

    if (l > 1.0f)
    {
        l = 1.0f;
    }

    f32 s = l * 12.92f;
    if (l > 0.0031308f)
    {
        s = 1.055f * pow(l, 1.0f / 2.4f) - 0.055f;
    }
    return s;
}

// void WriteColor(std::ostream &out, const vec3 &pixelColor)
// {
//     vec3 gammaCorrectedColor = LinearToSRGB(pixelColor);
//
//     int rByte = int(256 * Clamp(0.f, 0.999f, gammaCorrectedColor.r));
//     int gByte = int(256 * Clamp(0.f, 0.999f, gammaCorrectedColor.g));
//     int bByte = int(256 * Clamp(0.f, 0.999f, gammaCorrectedColor.b));
//
//     out << rByte << ' ' << gByte << ' ' << bByte << '\n';
// }

struct Material;

struct HitRecord
{
    vec3 normal;
    vec3 p;
    f32 t;
    f32 u, v;
    bool isFrontFace;
    Material *material;

    inline void SetNormal(const Ray &r, const vec3 &inNormal)
    {
        isFrontFace = dot(r.direction(), inNormal) < 0;
        normal      = isFrontFace ? inNormal : -inNormal;
    }
};

union AABB
{
    struct
    {
        vec3 minP;
        vec3 maxP;
    };
    struct
    {
        f32 minX;
        f32 minY;
        f32 minZ;
        f32 maxX;
        f32 maxY;
        f32 maxZ;
    };

    AABB()
    {
        minX = infinity;
        minY = infinity;
        minZ = infinity;
        maxX = -infinity;
        maxY = -infinity;
        maxZ = -infinity;
    }

    AABB(vec3 pt1, vec3 pt2)
    {
        minX = pt1.x <= pt2.x ? pt1.x : pt2.x;
        minY = pt1.y <= pt2.y ? pt1.y : pt2.y;
        minZ = pt1.z <= pt2.z ? pt1.z : pt2.z;

        maxX = pt1.x >= pt2.x ? pt1.x : pt2.x;
        maxY = pt1.y >= pt2.y ? pt1.y : pt2.y;
        maxZ = pt1.z >= pt2.z ? pt1.z : pt2.z;
        PadToMinimums();
    }
    AABB(AABB box1, AABB box2)
    {
        minX = box1.minX <= box2.minX ? box1.minX : box2.minX;
        minY = box1.minY <= box2.minY ? box1.minY : box2.minY;
        minZ = box1.minZ <= box2.minZ ? box1.minZ : box2.minZ;

        maxX = box1.maxX >= box2.maxX ? box1.maxX : box2.maxX;
        maxY = box1.maxY >= box2.maxY ? box1.maxY : box2.maxY;
        maxZ = box1.maxZ >= box2.maxZ ? box1.maxZ : box2.maxZ;
        PadToMinimums();
    }

    bool Hit(const Ray &r, f32 tMin, f32 tMax)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            f32 oneOverDir = 1.f / r.direction().e[axis];
            f32 t0         = (minP[axis] - r.origin()[axis]) * oneOverDir;
            f32 t1         = (maxP[axis] - r.origin()[axis]) * oneOverDir;
            if (t0 > t1)
            {
                f32 temp = t0;
                t0       = t1;
                t1       = temp;
            }
            tMin = t0 > tMin ? t0 : tMin;
            tMax = t1 < tMax ? t1 : tMax;
            if (tMax <= tMin)
                return false;
        }
        return true;
    }
    vec3 Center()
    {
        return (maxP + minP) * 0.5f;
    }
    vec3 GetHalfExtent()
    {
        return (maxP - minP) * 0.5f;
    }

    void Expand(f32 delta)
    {
        vec3 pad = vec3(delta / 2, delta / 2, delta / 2);
        minP -= pad;
        maxP += pad;
    }

    void PadToMinimums()
    {
        f32 delta        = 0.0001f;
        f32 deltaOverTwo = delta / 2;
        if (maxX - minX < delta)
        {
            minX -= deltaOverTwo;
            maxX += deltaOverTwo;
        }
        if (maxY - minY < delta)
        {
            minY -= deltaOverTwo;
            maxY += deltaOverTwo;
        }
        if (maxZ - minZ < delta)
        {
            minZ -= deltaOverTwo;
            maxZ += deltaOverTwo;
        }
    }
};

struct HomogeneousTransform
{
    vec3 translation;
    f32 rotateAngleY;
};

AABB Transform(const HomogeneousTransform &transform, const AABB &aabb)
{
    AABB result;
    vec3 vecs[] = {
        vec3(aabb.minX, aabb.minY, aabb.minZ),
        vec3(aabb.maxX, aabb.minY, aabb.minZ),
        vec3(aabb.maxX, aabb.maxY, aabb.minZ),
        vec3(aabb.minX, aabb.maxY, aabb.minZ),
        vec3(aabb.minX, aabb.minY, aabb.maxZ),
        vec3(aabb.maxX, aabb.minY, aabb.maxZ),
        vec3(aabb.maxX, aabb.maxY, aabb.maxZ),
        vec3(aabb.minX, aabb.maxY, aabb.maxZ),
    };
    f32 cosTheta = cos(transform.rotateAngleY);
    f32 sinTheta = sin(transform.rotateAngleY);
    for (u32 i = 0; i < ArrayLength(vecs); i++)
    {
        vec3 &vec = vecs[i];
        vec.x     = cosTheta * vec.x + sinTheta * vec.z;
        vec.z     = -sinTheta * vec.x + cosTheta * vec.z;
        vec += transform.translation;

        result.minX = result.minX < vec.x ? result.minX : vec.x;
        result.minY = result.minY < vec.y ? result.minY : vec.y;
        result.minZ = result.minZ < vec.z ? result.minZ : vec.z;

        result.maxX = result.maxX > vec.x ? result.maxX : vec.x;
        result.maxY = result.maxY > vec.y ? result.maxY : vec.y;
        result.maxZ = result.maxZ > vec.z ? result.maxZ : vec.z;
    }
    return result;
}

AABB Transform(const mat4 &mat, const AABB &aabb)
{
    AABB result;
    vec3 vecs[] = {
        mul(mat, vec3(aabb.minX, aabb.minY, aabb.minZ)),
        mul(mat, vec3(aabb.maxX, aabb.minY, aabb.minZ)),
        mul(mat, vec3(aabb.maxX, aabb.maxY, aabb.minZ)),
        mul(mat, vec3(aabb.minX, aabb.maxY, aabb.minZ)),
        mul(mat, vec3(aabb.minX, aabb.minY, aabb.maxZ)),
        mul(mat, vec3(aabb.maxX, aabb.minY, aabb.maxZ)),
        mul(mat, vec3(aabb.maxX, aabb.maxY, aabb.maxZ)),
        mul(mat, vec3(aabb.minX, aabb.maxY, aabb.maxZ)),
    };

    for (u32 i = 0; i < ArrayLength(vecs); i++)
    {
        vec3 &p     = vecs[i];
        result.minX = result.minX < p.x ? result.minX : p.x;
        result.minY = result.minY < p.y ? result.minY : p.y;
        result.minZ = result.minZ < p.z ? result.minZ : p.z;

        result.maxX = result.maxX > p.x ? result.maxX : p.x;
        result.maxY = result.maxY > p.y ? result.maxY : p.y;
        result.maxZ = result.maxZ > p.z ? result.maxZ : p.z;
    }
    return result;
}

class Sphere
{
public:
    Sphere() {}
    Sphere(vec3 c, f32 r, Material *m) : center(c), radius(fmax(0.f, r)), material(m)
    {
        centerVec = vec3(0, 0, 0);
    }
    Sphere(vec3 c1, vec3 c2, f32 r, Material *m) : center(c1), radius(fmax(0.f, r)), material(m)
    {
        centerVec = c2 - c1;
    }
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
    {
        // (C - P) dot (C - P) = r^2
        // (C - (O + Dt)) dot (C - (O + Dt)) - r^2 = 0
        // (-Dt + C - O) dot (-Dt + C - O) - r^2 = 0
        // t^2(D dot D) - 2t(D dot (C - O)) + (C - O dot C - O) - r^2 = 0
        vec3 oc = Center(r.time()) - r.origin();
        f32 a   = dot(r.direction(), r.direction());
        f32 h   = dot(r.direction(), oc);
        f32 c   = dot(oc, oc) - radius * radius;

        f32 discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        f32 result = (h - sqrt(discriminant)) / a;
        if (result <= tMin || result >= tMax)
        {
            result = (h + sqrt(discriminant)) / a;
            if (result <= tMin || result >= tMax)
                return false;
        }

        record.t    = result;
        record.p    = r.at(record.t);
        vec3 normal = (record.p - center) / radius;
        record.SetNormal(r, normal);
        record.material = material;
        Sphere::GetUV(record.u, record.v, normal);

        return true;
    }
    vec3 Center(f32 time) const
    {
        return center + centerVec * time;
    }
    AABB GetAABB() const
    {
        vec3 boxRadius = vec3(radius, radius, radius);
        vec3 center2   = center + centerVec;
        AABB box1      = AABB(center - boxRadius, center + boxRadius);
        AABB box2      = AABB(center2 - boxRadius, center2 + boxRadius);
        AABB aabb      = AABB(box1, box2);
        return aabb;
    }
    static void GetUV(f32 &u, f32 &v, const vec3 &p)
    {
        f32 zenith  = acos(-p.y);
        f32 azimuth = atan2(-p.z, p.x) + PI;

        u = azimuth / (2 * PI);
        v = zenith / PI;
    }

private:
    vec3 center;
    f32 radius;
    Material *material;
    vec3 centerVec;
};

struct Quad
{
    Quad() {}
    Quad(const vec3 &q, const vec3 &u, const vec3 &v, Material *mat) : q(q), u(u), v(v), material(mat)
    {
        vec3 n = cross(u, v);
        normal = normalize(n);
        d      = dot(normal, q);
        w      = n / dot(n, n);
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
        f32 denom = dot(normal, r.direction());
        // if the ray is parallel to the plane
        if (fabs(denom) < 1e-8f)
            return false;

        f32 t = (d - dot(normal, r.origin())) / denom;

        if (t < tMin || t > tMax) return false;

        vec3 intersection = r.at(t);

        vec3 planarHitVector = intersection - q;
        f32 alpha            = dot(w, cross(planarHitVector, v));
        f32 beta             = dot(w, cross(u, planarHitVector));

        if (!(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1)) return false;

        record.u        = alpha;
        record.v        = beta;
        record.p        = intersection;
        record.t        = t;
        record.material = material;
        record.SetNormal(r, normal);
        return true;
    }

    vec3 q; // corner
    vec3 u, v;
    Material *material;

    // plane
    f32 d;
    vec3 w;
    vec3 normal;
};

struct Box
{
    Quad sides[6];
    Box() {}
    Box(const vec3 &a, const vec3 &b, Material *mat)
    {
        vec3 min = vec3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
        vec3 max = vec3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));

        vec3 dx = vec3(max.x - min.x, 0, 0);
        vec3 dy = vec3(0, max.y - min.y, 0);
        vec3 dz = vec3(0, 0, max.z - min.z);

        sides[0] = Quad(vec3(min.x, min.y, max.z), dx, dy, mat);
        sides[1] = Quad(vec3(max.x, min.y, max.z), -dz, dy, mat);
        sides[2] = Quad(vec3(max.x, min.y, min.z), -dx, dy, mat);
        sides[3] = Quad(vec3(min.x, min.y, min.z), dz, dy, mat);
        sides[4] = Quad(vec3(min.x, max.y, max.z), dx, -dz, mat);
        sides[5] = Quad(vec3(min.x, min.y, min.z), dx, dz, mat);
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

        f32 rayLength              = r.direction().length();
        f32 distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;
        f32 hitDistance            = negInvDensity * log(RandomFloat());

        if (hitDistance > distanceInsideBoundary) return false;

        record.t           = rec1.t + hitDistance / rayLength;
        record.p           = r.at(record.t);
        record.normal      = vec3(1, 0, 0);
        record.isFrontFace = true;
        record.material    = phaseFunction;
        return true;
    }
};

typedef u32 PrimitiveFlags;
enum
{
    PrimitiveFlags_Transform      = 1 << 0,
    PrimitiveFlags_ConstantMedium = 1 << 1,
    PrimitiveFlags_Sphere         = 1 << 2,
    PrimitiveFlags_Quad           = 1 << 3,
};

bool EnumHasAnyFlags(u32 lhs, u32 rhs)
{
    return (lhs & rhs) != 0;
}

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

enum class PrimitiveType
{
    Sphere,
    Quad,
    Box,
};

struct Scene
{
    // std::vector<Sphere> spheres;
    // std::vector<Quad> quads;
    // std::vector<Box> boxes;
    //
    // std::vector<PrimitiveIndices> primitiveIndices;
    //
    // std::vector<HomogeneousTransform> transforms;
    // std::vector<ConstantMedium> media;

    Sphere *spheres;
    Quad *quads;
    Box *boxes;

    PrimitiveIndices *primitiveIndices;

    HomogeneousTransform *transforms;
    ConstantMedium *media;

    u32 sphereCount;
    u32 quadCount;
    u32 boxCount;
    u32 totalPrimitiveCount;

    u32 transformCount;
    u32 mediaCount;

    void FinalizePrimitives()
    {
        totalPrimitiveCount = sphereCount + quadCount + boxCount;
        primitiveIndices    = (PrimitiveIndices *)malloc(sizeof(PrimitiveIndices) * totalPrimitiveCount);
        for (u32 i = 0; i < totalPrimitiveCount; i++)
        {
            primitiveIndices[i].transformIndex     = -1;
            primitiveIndices[i].constantMediaIndex = -1;
        }
    }

    inline i32 GetIndex(PrimitiveType type, i32 primIndex) const
    {
        i32 index = -1;
        switch (type)
        {
            case PrimitiveType::Sphere:
            {
                index = primIndex;
            }
            break;
            case PrimitiveType::Quad:
            {
                index = primIndex + sphereCount;
            }
            break;
            case PrimitiveType::Box:
            {
                index = primIndex + sphereCount + quadCount;
            }
            break;
        }
        return index;
    }
    inline void GetTypeAndLocalindex(const u32 totalIndex, PrimitiveType *type, u32 *localIndex) const
    {
        if (totalIndex < sphereCount)
        {
            *type       = PrimitiveType::Sphere;
            *localIndex = totalIndex;
        }
        else if (totalIndex < quadCount + sphereCount)
        {
            *type       = PrimitiveType::Quad;
            *localIndex = totalIndex - sphereCount;
        }
        else if (totalIndex < quadCount + sphereCount + boxCount)
        {
            *type       = PrimitiveType::Box;
            *localIndex = totalIndex - sphereCount - quadCount;
        }
        else
        {
            assert(0);
        }
    }

    void AddConstantMedium(PrimitiveType type, i32 primIndex, i32 constantMediumIndex)
    {
        i32 index                                  = GetIndex(type, primIndex);
        primitiveIndices[index].constantMediaIndex = constantMediumIndex;
    }

    void AddTransform(PrimitiveType type, i32 primIndex, i32 transformIndex)
    {
        i32 index                              = GetIndex(type, primIndex);
        primitiveIndices[index].transformIndex = transformIndex;
    }

    // SceneHandle StartHandle(PrimitiveType type, u32 count)
    // {
    //     SceneHandle handle;
    //     switch (type)
    //     {
    //         case PrimitiveType::Sphere:
    //         {
    //             handle.offset = (u32)spheres.size();
    //         }
    //         break;
    //         case PrimitiveType::Quad:
    //         {
    //             handle.offset = (u32)spheres.size() + (u32)quads.size();
    //         }
    //         case PrimitiveType::Box:
    //         {
    //             handle.offset = (u32)spheres.size() + (u32)quads.size() + (u32)boxes.size();
    //         }
    //         break;
    //     }
    //     handle.count = count;
    //     return handle;
    // }

    // SceneHandle Add(Sphere &sphere)
    // {
    //     SceneHandle handle;
    //     handle.offset = (u32)spheres.size();
    //     handle.count  = 1;
    //     spheres.push_back(sphere);
    //     primitiveIndices.push_back({});
    //     return handle;
    // }

    // SceneHandle Add(Sphere &&sphere)
    // {
    //     SceneHandle handle;
    //     handle.offset = (u32)spheres.size();
    //     handle.count  = 1;
    //     spheres.push_back(std::move(sphere));
    //     primitiveIndices.push_back({});
    //     return handle;
    // }

    // SceneHandle Add(Quad &quad)
    // {
    //     SceneHandle handle;
    //     handle.offset = (u32)spheres.size() + (u32)quads.size();
    //     handle.count  = 1;
    //     quads.push_back(quad);
    //     primitiveIndices.push_back({});
    //     return handle;
    // }

    // SceneHandle Add(Quad &&quad)
    // {
    //     SceneHandle handle;
    //     handle.offset = (u32)spheres.size() + (u32)quads.size();
    //     handle.count  = 1;
    //     quads.push_back(std::move(quad));
    //     primitiveIndices.push_back({});
    //     return handle;
    // }

    // SceneHandle Add(Box &box)
    // {
    //     SceneHandle handle;
    //     handle.offset = (u32)spheres.size() + (u32)quads.size();
    //     handle.count  = ArrayLength(box.sides);
    //     for (u32 i = 0; i < ArrayLength(box.sides); i++)
    //     {
    //         Add(box.sides[i]);
    //         primitiveIndices.push_back({});
    //     }
    //     return handle;
    // }

    // SceneHandle Add(Box &&box)
    // {
    //     SceneHandle handle;
    //     handle.offset = (u32)spheres.size() + (u32)quads.size();
    //     handle.count  = ArrayLength(box.sides);
    //     for (u32 i = 0; i < ArrayLength(box.sides); i++)
    //     {
    //         Add(std::move(box.sides[i]));
    //         primitiveIndices.push_back({});
    //     }
    //     return handle;
    // }

    // void AddTransform(HomogeneousTransform transform, SceneHandle handle)
    // {
    //     u32 transformIndex = (u32)transforms.size();
    //     transforms.push_back(transform);
    //
    //     for (u32 i = handle.offset; i < handle.offset + handle.count; i++)
    //     {
    //         primitiveIndices[i].transformIndex = transformIndex;
    //     }
    // }

    // void AddConstantMedium(ConstantMedium medium, SceneHandle handle)
    // {
    //     u32 mediumIndex = (u32)media.size();
    //     media.push_back(std::move(medium));
    //     for (u32 i = handle.offset; i < handle.offset + handle.count; i++)
    //     {
    //         primitiveIndices[i].constantMediaIndex = mediumIndex;
    //     }
    // }

    void GetAABBs(AABB *aabbs)
    {
        for (u32 i = 0; i < sphereCount; i++)
        {
            Sphere &sphere = spheres[i];
            u32 index      = GetIndex(PrimitiveType::Sphere, i);
            aabbs[index]   = sphere.GetAABB();
        }
        for (u32 i = 0; i < quadCount; i++)
        {
            Quad &quad   = quads[i];
            u32 index    = GetIndex(PrimitiveType::Quad, i);
            aabbs[index] = quad.GetAABB();
        }
        for (u32 i = 0; i < boxCount; i++)
        {
            Box &box     = boxes[i];
            u32 index    = GetIndex(PrimitiveType::Box, i);
            aabbs[index] = box.GetAABB();
        }
        for (u32 i = 0; i < totalPrimitiveCount; i++)
        {
            if (primitiveIndices[i].transformIndex != -1)
            {
                aabbs[i] = Transform(transforms[primitiveIndices[i].transformIndex], aabbs[i]);
            }
        }
    }

    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &temp, u32 index)
    {
        bool result = false;

        Ray ray;
        HomogeneousTransform *transform = 0;
        f32 cosTheta;
        f32 sinTheta;

        if (primitiveIndices[index].transformIndex != -1)
        {
            transform             = &transforms[primitiveIndices[index].transformIndex];
            vec3 translatedOrigin = r.origin() - transform->translation;
            cosTheta              = cos(transform->rotateAngleY);
            sinTheta              = sin(transform->rotateAngleY);

            vec3 origin;
            origin.x = cosTheta * translatedOrigin.x - sinTheta * translatedOrigin.z;
            origin.y = translatedOrigin.y;
            origin.z = sinTheta * translatedOrigin.x + cosTheta * translatedOrigin.z;
            vec3 dir;
            dir.x = cosTheta * r.direction().x - sinTheta * r.direction().z;
            dir.y = r.direction().y;
            dir.z = sinTheta * r.direction().x + cosTheta * r.direction().z;
            // convert ray to object space
            ray = Ray(origin, dir, r.time());
        }
        else
        {
            ray = r;
        }

        PrimitiveType type;
        u32 localIndex;
        GetTypeAndLocalindex(index, &type, &localIndex);
        if (primitiveIndices[index].constantMediaIndex != -1)
        {
            ConstantMedium &medium = media[primitiveIndices[index].constantMediaIndex];
            switch (type)
            {
                case PrimitiveType::Sphere:
                {
                    Sphere &sphere = spheres[localIndex];
                    result         = medium.Hit(sphere, ray, tMin, tMax, temp);
                }
                break;
                case PrimitiveType::Quad:
                {
                    Quad &quad = quads[localIndex];
                    result     = medium.Hit(quad, ray, tMin, tMax, temp);
                }
                break;
                case PrimitiveType::Box:
                {
                    Box &box = boxes[localIndex];
                    result   = medium.Hit(box, ray, tMin, tMax, temp);
                }
                break;
            }
        }
        else
        {
            switch (type)
            {
                case PrimitiveType::Sphere:
                {
                    Sphere &sphere = spheres[localIndex];
                    result         = sphere.Hit(ray, tMin, tMax, temp);
                }
                break;
                case PrimitiveType::Quad:
                {
                    Quad &quad = quads[localIndex];
                    result     = quad.Hit(ray, tMin, tMax, temp);
                }
                break;
                case PrimitiveType::Box:
                {
                    Box &box = boxes[localIndex];
                    result   = box.Hit(ray, tMin, tMax, temp);
                }
                break;
            }
        }

        if (primitiveIndices[index].transformIndex != -1)
        {
            assert(transform);
            vec3 p;
            p.x = cosTheta * temp.p.x + sinTheta * temp.p.z;
            p.y = temp.p.y;
            p.z = -sinTheta * temp.p.x + cosTheta * temp.p.z;
            p += transform->translation;
            temp.p = p;

            vec3 normal;
            normal.x    = cosTheta * temp.normal.x + sinTheta * temp.normal.z;
            normal.y    = temp.normal.y;
            normal.z    = -sinTheta * temp.normal.x + cosTheta * temp.normal.z;
            temp.normal = normal;
        }
        return result;
    }
};

struct BVH
{
    Scene *scene;
    struct Node
    {
        AABB aabb;
        u32 left;
        u32 offset;
        u32 count;
        bool IsLeaf() { return count > 0; }
    };
    Node *nodes;
    u32 *leafIndices;
    u32 nodeCount;

    void Build(Scene *inScene)
    {
        scene                   = inScene;
        u32 totalPrimitiveCount = scene->totalPrimitiveCount;
        AABB *aabbs             = (AABB *)malloc(totalPrimitiveCount * sizeof(AABB));
        scene->GetAABBs(aabbs);
        Build(aabbs, totalPrimitiveCount);
    }

    void Build(AABB *aabbs, u32 count)
    {
        nodeCount = 0;
        assert(count != 0);
        const u32 nodeCapacity = count * 2 - 1;
        nodes                  = (Node *)malloc(sizeof(Node) * nodeCapacity);
        leafIndices            = (u32 *)malloc(sizeof(u32) * count);

        Node &node = nodes[nodeCount++];
        node       = {};
        node.count = count;
        for (u32 i = 0; i < count; i++)
        {
            node.aabb      = AABB(node.aabb, aabbs[i]);
            leafIndices[i] = i;
        }
        Subdivide(0, aabbs);
    }

    void Subdivide(u32 nodeIndex, AABB *aabbs)
    {
        Node &node = nodes[nodeIndex];
        if (node.count <= 2)
            return;
        vec3 extent = node.aabb.GetHalfExtent();
        vec3 min    = node.aabb.minP;
        int axis    = 0;
        if (extent.y > extent[axis]) axis = 1;
        if (extent.z > extent[axis]) axis = 2;
        f32 splitPos = min[axis] + extent[axis];

        int i = node.offset;
        int j = i + node.count - 1;
        while (i <= j)
        {
            vec3 center = aabbs[leafIndices[i]].Center();
            f32 value   = center[axis];
            if (value < splitPos)
            {
                i++;
            }
            else
            {
                u32 temp         = leafIndices[j];
                leafIndices[j--] = leafIndices[i];
                leafIndices[i]   = temp;
            }
        }
        u32 leftCount = i - node.offset;
        if (leftCount == 0 || leftCount == node.count)
            return;
        u32 leftChildIndex  = nodeCount++;
        u32 rightChildIndex = nodeCount++;
        node.left           = leftChildIndex;

        Node &leftChild  = nodes[leftChildIndex];
        leftChild        = {};
        leftChild.offset = node.offset;
        leftChild.count  = leftCount;

        Node &rightChild  = nodes[rightChildIndex];
        rightChild        = {};
        rightChild.offset = i;
        rightChild.count  = node.count - leftCount;

        node.count = 0;
        UpdateNodeBounds(leftChildIndex, aabbs);
        UpdateNodeBounds(rightChildIndex, aabbs);

        Subdivide(leftChildIndex, aabbs);
        Subdivide(rightChildIndex, aabbs);
    }

    inline void UpdateNodeBounds(u32 nodeIndex, AABB *aabbs)
    {
        Node &node = nodes[nodeIndex];
        for (u32 i = 0; i < node.count; i++)
        {
            node.aabb = AABB(node.aabb, aabbs[leafIndices[node.offset + i]]);
        }
    }

    inline bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
    {
        u32 stack[64];
        u32 stackPtr      = 0;
        stack[stackPtr++] = 0;
        f32 closest       = tMax;
        bool hit          = false;
        HitRecord temp;

        while (stackPtr > 0)
        {
            assert(stackPtr < 64);
            const u32 nodeIndex = stack[--stackPtr];
            Node &node          = nodes[nodeIndex];
            if (!node.aabb.Hit(r, 0, infinity))
                continue;
            if (node.IsLeaf())
            {
                for (u32 i = 0; i < node.count; i++)
                {
                    if (scene->Hit(r, tMin, tMax, temp, leafIndices[node.offset + i]))
                    {
                        if (temp.t < closest)
                        {
                            closest = temp.t;
                            record  = temp;
                            hit     = true;
                        }
                    }
                }
            }
            else
            {
                stack[stackPtr++] = node.left;
                stack[stackPtr++] = node.left + 1;
            }
        }
        return hit;
    }
};

struct Perlin
{
    // f32 *randFloat;
    vec3 *randVec;
    i32 *permX;
    i32 *permY;
    i32 *permZ;
    static const i32 pointCount = 256;

    void Init()
    {
        randVec = new vec3[pointCount];
        for (i32 i = 0; i < pointCount; i++)
        {
            randVec[i] = normalize(RandomVec3(-1, 1));
        }

        auto GeneratePerm = [&]() -> i32 * {
            i32 *perm = new i32[pointCount];
            for (i32 i = 0; i < pointCount; i++)
            {
                perm[i] = i;
            }

            for (i32 i = pointCount - 1; i > 0; i--)
            {
                i32 target   = RandomInt(0, i);
                i32 temp     = perm[i];
                perm[i]      = perm[target];
                perm[target] = temp;
            }
            return perm;
        };

        permX = GeneratePerm();
        permY = GeneratePerm();
        permZ = GeneratePerm();
    }

    f32 Noise(const vec3 &p) const
    {
        f32 u = p.x - floor(p.x);
        f32 v = p.y - floor(p.y);
        f32 w = p.z - floor(p.z);

        vec3 c[2][2][2];
        {
            i32 i = i32(floor(p.x));
            i32 j = i32(floor(p.y));
            i32 k = i32(floor(p.z));

            for (i32 di = 0; di < 2; di++)
            {
                for (i32 dj = 0; dj < 2; dj++)
                {
                    for (i32 dk = 0; dk < 2; dk++)
                    {
                        c[di][dj][dk] = randVec[permX[(i + di) & 255] ^ permY[(j + dj) & 255] ^ permZ[(k + dk) & 255]];
                    }
                }
            }
        }

        f32 accum = 0.0;
        {
            f32 uu = u * u * (3 - 2 * u);
            f32 vv = v * v * (3 - 2 * v);
            f32 ww = w * w * (3 - 2 * w);
            for (i32 i = 0; i < 2; i++)
            {
                for (i32 j = 0; j < 2; j++)
                {
                    for (i32 k = 0; k < 2; k++)
                    {
                        vec3 weightV(u - i, v - j, w - k);
                        accum += (i * uu + (1 - i) * (1 - uu)) *
                                 (j * vv + (1 - j) * (1 - vv)) *
                                 (k * ww + (1 - k) * (1 - ww)) *
                                 dot(c[i][j][k], weightV);
                    }
                }
            }
        }
        return accum;
    }

    f32 Turbulence(vec3 p, i32 depth) const
    {
        f32 accum  = 0.0;
        f32 weight = 1.0;
        for (i32 i = 0; i < depth; i++)
        {
            accum += weight * Noise(p);
            weight *= 0.5;
            p *= 2;
        }

        return fabs(accum);
    }
};

struct Texture
{
    enum class Type
    {
        Solid,
        Checkered,
        Image,
        Noise
    } type;

    static Texture CreateSolid(const vec3 &albedo)
    {
        Texture texture;
        texture.baseColor = albedo;
        texture.type      = Type::Solid;
        return texture;
    }
    static Texture CreateCheckered(f32 scale, const vec3 &even, const vec3 &odd)
    {
        Texture texture;
        texture.baseColor  = even;
        texture.baseColor2 = odd;
        texture.type       = Type::Checkered;
        texture.invScale   = 1.f / scale;
        return texture;
    }

    static Texture CreateImage(const char *filename)
    {
        Texture texture;
        texture.image = LoadFile(filename);
        texture.type  = Type::Image;
        return texture;
    }

    static Texture CreateNoise(f32 scale)
    {
        Texture texture;
        texture.perlin.Init();
        texture.type  = Type::Noise;
        texture.scale = scale;
        return texture;
    }

    vec3 Value(const f32 u, const f32 v, const vec3 &p) const
    {
        switch (type)
        {
            case Type::Solid:
            {
                return baseColor;
            }
            break;
            case Type::Checkered:
            {
                i32 x = i32(std::floor(p.x * invScale));
                i32 y = i32(std::floor(p.y * invScale));
                i32 z = i32(std::floor(p.z * invScale));
                return (x + y + z) % 2 == 0 ? baseColor : baseColor2;
            }
            break;
            case Type::Image:
            {
                assert(image.width);
                assert(image.height);
                i32 x = i32(u * image.width);
                i32 y = i32((1 - v) * image.height);

                u8 *data    = GetColor(&image, x, y);
                f32 divisor = 1 / 255.f;
                f32 r       = f32(data[0]) * divisor;
                f32 g       = f32(data[1]) * divisor;
                f32 b       = f32(data[2]) * divisor;
                return vec3(r, g, b);
            }
            break;
            case Type::Noise:
            {
                return vec3(.5f, .5f, .5f) * (1.f + sinf(scale * p.z + 10.f * perlin.Turbulence(p, 7)));
            }
            break;
            default: assert(0); return vec3(0, 0, 0);
        }
    }

    vec3 baseColor;

    // checkered
    vec3 baseColor2;
    f32 invScale;

    // image
    Image image;

    // perlin
    Perlin perlin;
    f32 scale;
};

enum class MaterialType
{
    Lambert,
    Metal,
    Dielectric,
    DiffuseLight,
    Isotropic,
};

struct Material
{
    MaterialType type;
    vec3 albedo;
    f32 fuzz;
    f32 refractiveIndex;

    Texture texture;

    static Material CreateLambert(vec3 inAlbedo)
    {
        Material result;
        result.type    = MaterialType::Lambert;
        result.texture = Texture::CreateSolid(inAlbedo);
        // result.albedo = inAlbedo;
        return result;
    }

    static Material CreateLambert(Texture *texture)
    {
        Material result;
        result.type    = MaterialType::Lambert;
        result.texture = *texture;
        return result;
    }

    static Material CreateMetal(vec3 inAlbedo, f32 inFuzz = 0.0)
    {
        Material result;
        result.type   = MaterialType::Metal;
        result.albedo = inAlbedo;
        result.fuzz   = inFuzz < 1 ? inFuzz : 1;
        return result;
    }

    static Material CreateDielectric(f32 inRefractiveIndex)
    {
        Material result;
        result.type            = MaterialType::Dielectric;
        result.refractiveIndex = inRefractiveIndex;
        return result;
    }

    static Material CreateDiffuseLight(Texture *texture)
    {
        Material result;
        result.type    = MaterialType::DiffuseLight;
        result.texture = *texture;
        return result;
    }

    static Material CreateDiffuseLight(vec3 inAlbedo)
    {
        Material result;
        result.type    = MaterialType::DiffuseLight;
        result.texture = Texture::CreateSolid(inAlbedo);
        return result;
    }

    static Material CreateIsotropic(const vec3 &albedo)
    {
        Material result;
        result.type    = MaterialType::Isotropic;
        result.texture = Texture::CreateSolid(albedo);
        return result;
    }

    static Material CreateIsotropic(Texture *texture)
    {
        Material result;
        result.type    = MaterialType::Isotropic;
        result.texture = *texture;
        return result;
    }

    bool LambertScatter(const Ray &r, const HitRecord &record, vec3 &attenuation, Ray &scatteredRay)
    {
        vec3 scatterDirection = record.normal + RandomUnitVector();
        scatterDirection      = NearZero(scatterDirection) ? record.normal : scatterDirection;
        scatteredRay          = Ray(record.p, scatterDirection, r.time());
        attenuation           = texture.Value(record.u, record.v, record.p);
        return true;
    }

    bool MetalScatter(const Ray &r, const HitRecord &record, vec3 &attenuation, Ray &scatteredRay)
    {
        vec3 reflectDir = Reflect(r.direction(), record.normal);
        reflectDir      = normalize(reflectDir) + fuzz * RandomUnitVector();
        scatteredRay    = Ray(record.p, reflectDir, r.time());
        attenuation     = albedo;
        return true;
    }

    bool DielectricScatter(const Ray &r, const HitRecord &record, vec3 &attenuation, Ray &scatteredRay)
    {
        attenuation = vec3(1, 1, 1);
        f32 ri      = record.isFrontFace ? 1.f / refractiveIndex : refractiveIndex;

        vec3 rayDir  = normalize(r.direction());
        f32 cosTheta = fmin(dot(-rayDir, record.normal), 1.f);
        f32 sinTheta = sqrt(1 - cosTheta * cosTheta);
        // total internal reflection
        bool cannotRefract = ri * sinTheta > 1.f;

        f32 f0          = (1 - ri) / (1 + ri);
        f0              = f0 * f0;
        f32 reflectance = f0 + (1 - f0) * powf(1 - cosTheta, 5.f);
        vec3 direction  = cannotRefract || reflectance > RandomFloat()
                              ? Reflect(rayDir, record.normal)
                              : Refract(rayDir, record.normal, ri);
        scatteredRay    = Ray(record.p, direction, r.time());

        return true;
    }

    bool IsotropicScatter(const Ray &r, const HitRecord &record, vec3 &attenuation, Ray &scatteredRay)
    {
        scatteredRay = Ray(record.p, RandomUnitVector(), r.time());
        attenuation  = texture.Value(record.u, record.v, record.p);
        return true;
    }

    inline bool Scatter(const Ray &r, const HitRecord &record, vec3 &attenuation, Ray &scatteredRay)
    {
        switch (type)
        {
            case MaterialType::Lambert: return LambertScatter(r, record, attenuation, scatteredRay);
            case MaterialType::Metal: return MetalScatter(r, record, attenuation, scatteredRay);
            case MaterialType::Dielectric: return DielectricScatter(r, record, attenuation, scatteredRay);
            case MaterialType::Isotropic: return IsotropicScatter(r, record, attenuation, scatteredRay);
            default: return false;
        }
    }

    inline vec3 Emitted(f32 u, f32 v, const vec3 &p) const
    {
        switch (type)
        {
            case MaterialType::DiffuseLight:
            {
                return texture.Value(u, v, p);
            }
            break;
            default: return vec3(0, 0, 0);
        }
    }
};

#ifndef EMISSIVE
vec3 RayColor(const Ray &r, const int depth, const BVH &bvh)
{
    if (depth <= 0)
        return vec3(0, 0, 0);

    vec3 sphereCenter = vec3(0, 0, -1);
    HitRecord record;

    if (bvh.Hit(r, 0.001f, infinity, record))
    {
        Ray scattered;
        vec3 attenuation;
        if (record.material->Scatter(r, record, attenuation, scattered))
        {
            return attenuation * RayColor(scattered, depth - 1, bvh);
        }
        return vec3(0, 0, 0);
    }

    const vec3 normalizedDirection = normalize(r.direction());
    f32 t                          = 0.5f * (normalizedDirection.y + 1.f);
    return (1 - t) * vec3(1, 1, 1) + t * vec3(0.5f, 0.7f, 1.f);
}
#else
vec3 RayColor(const Ray &r, const int depth, const BVH &bvh)
{
    if (depth <= 0)
        return vec3(0, 0, 0);

    vec3 sphereCenter = vec3(0, 0, -1);
    HitRecord record;

    if (!bvh.Hit(r, 0.001f, infinity, record))
        return BACKGROUND;
    Ray scattered;
    vec3 attenuation;
    vec3 emissiveColor = record.material->Emitted(record.u, record.v, record.p);
    if (!record.material->Scatter(r, record, attenuation, scattered))
        return emissiveColor;
    return emissiveColor + attenuation * RayColor(scattered, depth - 1, bvh);
}
#endif

bool RenderTile(WorkQueue *queue)
{
    u64 workItemIndex = InterlockedAdd(&queue->workItemIndex, 1);
    if (workItemIndex >= queue->workItemCount) return false;

    WorkItem *item = &queue->workItems[workItemIndex];

    i32 samplesPerPixel = queue->params->samplesPerPixel;
    vec3 cameraCenter   = queue->params->cameraCenter;

    for (u32 height = item->startY; height < item->onePastEndY; height++)
    {
        u32 *out = GetPixelPointer(queue->params->image, item->startX, height);
        for (u32 width = item->startX; width < item->onePastEndX; width++)
        {
            vec3 pixelColor(0, 0, 0);

            for (i32 i = 0; i < samplesPerPixel; i++)
            {
                const vec3 offset      = vec3(RandomFloat() - 0.5f, RandomFloat() - 0.5f, 0.f);
                const vec3 pixelSample = queue->params->pixel00 + ((width + offset.x) * queue->params->pixelDeltaU) +
                                         ((height + offset.y) * queue->params->pixelDeltaV);
                vec3 rayOrigin;
                if (queue->params->defocusAngle <= 0)
                {
                    rayOrigin = cameraCenter;
                }
                else
                {
                    vec3 sample = RandomInUnitDisk();
                    rayOrigin   = cameraCenter + sample[0] * queue->params->defocusDiskU +
                                sample[1] * queue->params->defocusDiskV;
                }
                const vec3 rayDirection = pixelSample - rayOrigin;
                const f32 rayTime       = RandomFloat();
                Ray r(rayOrigin, rayDirection, rayTime);

                pixelColor += RayColor(r, queue->params->maxDepth, *queue->params->bvh);
            }

            pixelColor /= (f32)samplesPerPixel;
            f32 r = 255.f * ExactLinearToSRGB(pixelColor.r);
            f32 g = 255.f * ExactLinearToSRGB(pixelColor.g);
            f32 b = 255.f * ExactLinearToSRGB(pixelColor.b);
            f32 a = 255.f;

            u32 color = (RoundFloatToU32(a) << 24) |
                        (RoundFloatToU32(r) << 16) |
                        (RoundFloatToU32(g) << 8) |
                        (RoundFloatToU32(b) << 0);
            *out++ = color;
        }
    }
    InterlockedAdd(&queue->tilesFinished, 1);
    return true;
}

int main(int argc, char *argv[])
{
#if SPHERES
    const f32 aspectRatio     = 16.f / 9.f;
    const vec3 lookFrom       = vec3(13, 2, 3);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0.6f;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = vec3(0.7f, 0.8f, 1.f);
#elif EARTH
    const f32 aspectRatio     = 16.f / 9.f;
    const vec3 lookFrom       = vec3(0, 0, 12);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = vec3(0.7f, 0.8f, 1.f);
#elif PERLIN
    const f32 aspectRatio     = 16.f / 9.f;
    const vec3 lookFrom       = vec3(13, 2, 3);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = vec3(0.7f, 0.8f, 1.f);
#elif QUADS
    const f32 aspectRatio     = 1.f;
    const vec3 lookFrom       = vec3(0, 0, 9);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 80;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = vec3(0.7f, 0.8f, 1.f);
#elif LIGHTS
    const f32 aspectRatio     = 16.f / 9.f;
    const vec3 lookFrom       = vec3(26, 3, 6);
    const vec3 lookAt         = vec3(0, 2, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int imageWidth      = 400;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    BACKGROUND                = vec3(0, 0, 0);
#elif CORNELL
    const f32 aspectRatio  = 1.f;
    const vec3 lookFrom    = vec3(278, 278, -800);
    const vec3 lookAt      = vec3(278, 278, 0);
    const vec3 worldUp     = vec3(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const int imageWidth      = 400;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    BACKGROUND                = vec3(0, 0, 0);
#elif CORNELL_SMOKE
    const f32 aspectRatio  = 1.0;
    const vec3 lookFrom    = vec3(278, 278, -800);
    const vec3 lookAt      = vec3(278, 278, 0);
    const vec3 worldUp     = vec3(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const u32 imageWidth      = 600;
    const u32 samplesPerPixel = 200;
    const u32 maxDepth        = 50;
    BACKGROUND                = vec3(0, 0, 0);
#elif FINAL
    const f32 aspectRatio  = 1.0;
    const vec3 lookFrom    = vec3(478, 278, -600);
    const vec3 lookAt      = vec3(278, 278, 0);
    const vec3 worldUp     = vec3(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const int imageWidth      = 400;
    const int samplesPerPixel = 250;
    const int maxDepth        = 4;
    BACKGROUND                = vec3(0, 0, 0);
#endif

    u32 imageHeight = u32(imageWidth / aspectRatio);
    imageHeight     = imageHeight < 1 ? 1 : imageHeight;
    f32 focalLength = (lookFrom - lookAt).length();
    f32 theta       = DegreesToRadians(verticalFov);
    f32 h           = tan(theta / 2);

    vec3 f = normalize(lookFrom - lookAt);
    vec3 s = cross(worldUp, f);
    vec3 u = cross(f, s);

    f32 viewportHeight = 2 * h * focusDist;
    f32 viewportWidth  = viewportHeight * (f32(imageWidth) / imageHeight);
    vec3 cameraCenter  = lookFrom;

    vec3 viewportU = viewportWidth * s;
    vec3 viewportV = viewportHeight * -u;

    vec3 pixelDeltaU = viewportU / imageWidth;
    vec3 pixelDeltaV = viewportV / (f32)imageHeight;

    vec3 viewportUpperLeft = cameraCenter - focusDist * f - viewportU / 2 - viewportV / 2;
    vec3 pixel00           = viewportUpperLeft + 0.5f * (pixelDeltaU + pixelDeltaV);

    f32 defocusRadius = focusDist * tan(DegreesToRadians(defocusAngle / 2));
    vec3 defocusDiskU = defocusRadius * s;
    vec3 defocusDiskV = defocusRadius * u;

    // std::cout << "P3\n"
    //           << imageWidth << ' ' << imageHeight << "\n255\n";

    Scene scene = {};

#if SPHERES
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            f32 chooseMat = RandomFloat();
            vec3 center(a + 0.9f * RandomFloat(), 0.2f, b + 0.9f * RandomFloat());

            if ((center - vec3(4, 0.2f, 0)).length() > 0.9f)
            {
                Material *material = (Material *)malloc(sizeof(Material));
                // Diffuse
                if (chooseMat < 0.8)
                {
                    vec3 albedo  = RandomVec3() * RandomVec3();
                    vec3 center2 = center + vec3(0, RandomFloat(0, .5f), 0);
                    *material    = Material::CreateLambert(albedo);
                    scene.Add(Sphere(center, center2, 0.2f, material));
                }
                // Metal
                else if (chooseMat < 0.95)
                {
                    vec3 albedo = RandomVec3(0.5f, 1);
                    f32 fuzz    = RandomFloat(0, 0.5f);
                    *material   = Material::CreateMetal(albedo, fuzz);
                    scene.Add(Sphere(center, 0.2f, material));
                }
                // Glass
                else
                {
                    *material = Material::CreateDielectric(1.5f);
                    scene.Add(Sphere(center, 0.2f, material));
                }
            }
        }
    }

    Texture checkered    = Texture::CreateCheckered(0.32f, vec3(.2f, .3f, .1f), vec3(.9f, .9f, .9f));
    Material materials[] = {
        Material::CreateDielectric(1.5f),
        Material::CreateLambert(vec3(0.4f, 0.2f, 0.1f)),
        Material::CreateMetal(vec3(0.7f, 0.6f, 0.5f), 0.f),
        Material::CreateLambert(&checkered),
    };

    scene.Add(Sphere(vec3(0, 1, 0), 1.f, &materials[0]));
    scene.Add(Sphere(vec3(-4, 1, 0), 1.f, &materials[1]));
    scene.Add(Sphere(vec3(4, 1, 0), 1.f, &materials[2]));

    // ground
    scene.Add(Sphere(vec3(0, -1000, 0), 1000, &materials[3]));

#elif EARTH
    Texture earth             = Texture::CreateImage("earthmap.jpg");
    Material surface          = Material::CreateLambert(&earth);

    scene.Add(Sphere(vec3(0, 0, 0), 2, &surface));
#elif PERLIN
    Texture noise             = Texture::CreateNoise(4.0);
    Material perlin           = Material::CreateLambert(&noise);

    scene.Add(Sphere(vec3(0, -1000, 0), 1000, &perlin));
    scene.Add(Sphere(vec3(0, 2, 0), 2, &perlin));
#elif QUADS
    Material materials[]      = {
        Material::CreateLambert(vec3(1.f, 0.2f, 0.2f)),
        Material::CreateLambert(vec3(0.2f, 1.f, 0.2f)),
        Material::CreateLambert(vec3(0.2f, 0.2f, 1.f)),
        Material::CreateLambert(vec3(1.f, 0.5f, 0.f)),
        Material::CreateLambert(vec3(0.2f, 0.8f, 0.8f)),
    };

    scene.Add(Quad(vec3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), &materials[0]));
    scene.Add(Quad(vec3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), &materials[1]));
    scene.Add(Quad(vec3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), &materials[2]));
    scene.Add(Quad(vec3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), &materials[3]));
    scene.Add(Quad(vec3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), &materials[4]));
#elif LIGHTS
    Texture texture           = Texture::CreateNoise(4);
    Material lambert          = Material::CreateLambert(&texture);
    scene.Add(Sphere(vec3(0, -1000, 0), 1000, &lambert));
    scene.Add(Sphere(vec3(0, 2, 0), 2, &lambert));

    Material diffuse = Material::CreateDiffuseLight(vec3(4, 4, 4));
    scene.Add(Sphere(vec3(0, 7, 0), 2, &diffuse));
    scene.Add(Quad(vec3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), &diffuse));
#elif CORNELL
    Material materials[]      = {
        Material::CreateLambert(vec3(.65f, .05f, .05f)),
        Material::CreateLambert(vec3(.73f, .73f, .73f)),
        Material::CreateLambert(vec3(.12f, .45f, .15f)),
        Material::CreateDiffuseLight(vec3(15, 15, 15)),
    };

    scene.Add(Quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), &materials[2]));
    scene.Add(Quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), &materials[0]));
    scene.Add(Quad(vec3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), &materials[3]));
    scene.Add(Quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), &materials[1]));
    scene.Add(Quad(vec3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), &materials[1]));
    scene.Add(Quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), &materials[1]));

    SceneHandle handle = scene.Add(Box(vec3(0, 0, 0), vec3(165, 330, 165), &materials[1]));
    f32 rotateAngle    = DegreesToRadians(15);
    vec3 translate     = vec3(265, 0, 295);
    HomogeneousTransform transform;
    transform.translation  = translate;
    transform.rotateAngleY = rotateAngle;
    scene.AddTransform(transform, handle);

    handle                 = scene.Add(Box(vec3(0, 0, 0), vec3(165, 165, 165), &materials[1]));
    rotateAngle            = DegreesToRadians(-18);
    translate              = vec3(130, 0, 65);
    transform.translation  = translate;
    transform.rotateAngleY = rotateAngle;
    scene.AddTransform(transform, handle);
#elif CORNELL_SMOKE
    Material materials[]      = {
        Material::CreateLambert(vec3(.65f, .05f, .05f)),
        Material::CreateLambert(vec3(.73f, .73f, .73f)),
        Material::CreateLambert(vec3(.12f, .45f, .15f)),
        Material::CreateDiffuseLight(vec3(7, 7, 7)),
        Material::CreateIsotropic(vec3(0, 0, 0)),
        Material::CreateIsotropic(vec3(1, 1, 1)),
    };

    Quad quads[] = {
        Quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), &materials[2]),
        Quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), &materials[0]),
        Quad(vec3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305), &materials[3]),
        Quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), &materials[1]),
        Quad(vec3(0, 555, 0), vec3(555, 0, 0), vec3(0, 0, 555), &materials[1]),
        Quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), &materials[1]),
    };

    Box boxes[] = {
        Box(vec3(0, 0, 0), vec3(165, 330, 165), &materials[1]),
        Box(vec3(0, 0, 0), vec3(165, 165, 165), &materials[1]),
    };

    HomogeneousTransform transforms[] = {
        {vec3(265, 0, 295), DegreesToRadians(15)},
        {vec3(130, 0, 65), DegreesToRadians(-18)},
    };

    ConstantMedium media[] = {
        {0.01f, &materials[4]},
        {0.01f, &materials[5]},
    };

    scene.quads          = quads;
    scene.quadCount      = ArrayLength(quads);
    scene.boxes          = boxes;
    scene.boxCount       = ArrayLength(boxes);
    scene.transforms     = transforms;
    scene.transformCount = ArrayLength(transforms);
    scene.media          = media;
    scene.mediaCount     = ArrayLength(media);

    scene.FinalizePrimitives();
    scene.AddConstantMedium(PrimitiveType::Box, 0, 0);
    scene.AddTransform(PrimitiveType::Box, 0, 0);

    scene.AddConstantMedium(PrimitiveType::Box, 1, 1);
    scene.AddTransform(PrimitiveType::Box, 1, 1);

#elif FINAL
    Texture texture           = Texture::CreateImage("earthmap.jpg");
    Texture noise             = Texture::CreateNoise(0.2f);
    Material materials[]{
        Material::CreateLambert(vec3(0.48f, 0.83f, 0.53f)),
        Material::CreateDiffuseLight(vec3(7, 7, 7)),
        Material::CreateLambert(vec3(0.7f, 0.3f, 0.1f)),
        Material::CreateDielectric(1.5f),
        Material::CreateMetal(vec3(0.8f, 0.8f, 0.9f), 1.f),
        Material::CreateIsotropic(vec3(0.2f, 0.4f, 0.9f)),
        Material::CreateIsotropic(vec3(1, 1, 1)),
        Material::CreateLambert(&texture),
        Material::CreateLambert(&noise),
        Material::CreateLambert(vec3(.73f, .73f, .73f)),
    };

    const i32 boxesPerSide = 20;
    Box boxes[boxesPerSide * boxesPerSide];
    for (i32 i = 0; i < boxesPerSide; i++)
    {
        for (i32 j = 0; j < boxesPerSide; j++)
        {
            f32 w  = 100.f;
            f32 x0 = -1000.f + i * w;
            f32 z0 = -1000.f + j * w;
            f32 y0 = 0.f;
            f32 x1 = x0 + w;
            f32 y1 = RandomFloat(1, 101);
            f32 z1 = z0 + w;

            boxes[i * boxesPerSide + j] = Box(vec3(x0, y0, z0), vec3(x1, y1, z1), &materials[0]);
        }
    }

    Quad quads[] = {
        Quad(vec3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), &materials[1]),
    };

    vec3 center1 = vec3(400, 400, 200);
    vec3 center2 = center1 + vec3(30, 0, 0);

    const i32 ns           = 1000;
    Sphere spheres[8 + ns] = {
        Sphere(center1, center2, 50, &materials[2]),
        Sphere(vec3(260, 150, 45), 50, &materials[3]),
        Sphere(vec3(0, 150, 45), 50, &materials[4]),
        Sphere(vec3(360, 150, 145), 70, &materials[3]),
        Sphere(vec3(360, 150, 145), 70, &materials[3]),
        Sphere(vec3(0, 0, 0), 5000, &materials[3]),
        Sphere(vec3(400, 200, 400), 100, &materials[7]),
        Sphere(vec3(220, 280, 300), 80, &materials[8]),
    };

    HomogeneousTransform transforms[] = {
        {vec3(-100, 270, 395), 15},
    };

    ConstantMedium media[] = {
        {0.2f, &materials[5]},
        {0.0001f, &materials[6]},
    };

    for (i32 j = 8; j < ArrayLength(spheres); j++)
    {
        spheres[j] = Sphere(RandomVec3(0, 165), 10, &materials[9]);
    }

    scene.spheres        = spheres;
    scene.sphereCount    = ArrayLength(spheres);
    scene.quads          = quads;
    scene.quadCount      = ArrayLength(quads);
    scene.boxes          = boxes;
    scene.boxCount       = ArrayLength(boxes);
    scene.transforms     = transforms;
    scene.transformCount = ArrayLength(transforms);
    scene.media          = media;
    scene.mediaCount     = ArrayLength(media);
    scene.FinalizePrimitives();

    scene.AddConstantMedium(PrimitiveType::Sphere, 4, 0);
    scene.AddConstantMedium(PrimitiveType::Sphere, 5, 1);

    for (i32 j = 8; j < ArrayLength(spheres); j++)
    {
        scene.AddTransform(PrimitiveType::Sphere, j, 0);
    }

#endif

    BVH bvh;
    bvh.Build(&scene);

    RenderParams params;
    params.pixel00         = pixel00;
    params.pixelDeltaU     = pixelDeltaU;
    params.pixelDeltaV     = pixelDeltaV;
    params.cameraCenter    = cameraCenter;
    params.defocusDiskU    = defocusDiskU;
    params.defocusDiskV    = defocusDiskV;
    params.defocusAngle    = defocusAngle;
    params.bvh             = &bvh;
    params.maxDepth        = maxDepth;
    params.samplesPerPixel = samplesPerPixel;

    Image image;
    image.width         = imageWidth;
    image.height        = imageHeight;
    image.bytesPerPixel = sizeof(u32);
    image.contents      = (u8 *)malloc(GetImageSize(&image));
    params.image        = &image;

    u32 tileWidth     = 64;
    u32 tileHeight    = 64;
    u32 tileCountX    = (imageWidth + tileWidth - 1) / tileWidth;
    u32 tileCountY    = (imageHeight + tileHeight - 1) / tileHeight;
    WorkQueue queue   = {};
    u32 workItemTotal = tileCountX * tileCountY;
    queue.workItems   = (WorkItem *)malloc(sizeof(WorkItem) * workItemTotal);
    queue.params      = &params;
    for (u32 tileY = 0; tileY < tileCountY; tileY++)
    {
        u32 startY      = tileY * tileHeight;
        u32 onePastEndY = startY + tileHeight;
        onePastEndY     = onePastEndY > imageHeight ? imageHeight : onePastEndY;
        for (u32 tileX = 0; tileX < tileCountX; tileX++)
        {
            u32 startX      = tileX * tileWidth;
            u32 onePastEndX = startX + tileWidth;
            onePastEndX     = onePastEndX > imageWidth ? imageWidth : onePastEndX;

            WorkItem *workItem    = &queue.workItems[queue.workItemCount++];
            workItem->startX      = startX;
            workItem->startY      = startY;
            workItem->onePastEndX = onePastEndX;
            workItem->onePastEndY = onePastEndY;
        }
    }

    assert(queue.workItemCount == workItemTotal);

    clock_t start = clock();
    for (u32 i = 0; i < GetCPUCoreCount(); i++)
    {
        CreateWorkThread(&queue);
    }

    while (queue.tilesFinished < workItemTotal)
    {
        fprintf(stderr, "\rRaycasting %d%%...    ", 100 * (u32)queue.tilesFinished / workItemTotal);
        fflush(stdout);
        RenderTile(&queue);
    }
    clock_t end = clock();

    fprintf(stderr, "\n");
    printf("Total time: %dms\n", end - start);
    WriteImage(&image, "image.bmp");
    fprintf(stderr, "Done.");
}
