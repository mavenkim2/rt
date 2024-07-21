#include <stdint.h>
#include <windows.h>
#include <iostream>
#include <vector>
#include <assert.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

typedef uint8_t u8;
typedef uint32_t u32;
typedef int32_t i32;
typedef float f32;
typedef double f64;

using std::sqrt;

const double infinity = std::numeric_limits<double>::infinity();
#define PI 3.1415926535897932385

inline double DegreesToRadians(double degrees)
{
    return degrees * PI / 180.0;
}

struct Image
{
    u8 *contents;
    i32 width;
    i32 height;
    i32 bytesPerPixel;
};

//////////////////////////////
// Intervals
//
bool IsInInterval(double min, double max, double x)
{
    return x >= min && x <= max;
}

template <typename T>
T Clamp(T min, T max, T x)
{
    return x < min ? min : (x > max ? max : x);
}

Image LoadFile(const char *file)
{
    Image image;
    i32 nComponents;
    image.contents      = stbi_load(file, &image.width, &image.height, &nComponents, 0);
    image.bytesPerPixel = nComponents;
    return image;
}

u8 *GetColor(const Image *image, i32 x, i32 y)
{
    x = Clamp(0, image->width - 1, x);
    y = Clamp(0, image->height - 1, y);

    return image->contents + x * image->bytesPerPixel + y * image->width * image->bytesPerPixel;
}

union vec3
{
    double e[3];
    struct
    {
        double x, y, z;
    };
    struct
    {
        double r, g, b;
    };

    vec3() : e{0, 0, 0} {}
    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double &operator[](int i) { return e[i]; }
    vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
    vec3 &operator*=(double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3 &operator/=(double t)
    {
        return *this *= 1 / t;
    }

    double length() const
    {
        return sqrt(lengthSquared());
    }

    double lengthSquared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

inline std::ostream &
operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(const vec3 &u, double d)
{
    return vec3(u.e[0] * d, u.e[1] * d, u.e[2] * d);
}

inline vec3 operator*(double d, const vec3 &v)
{
    return v * d;
}

inline vec3 operator/(const vec3 &v, double d)
{
    return (1 / d) * v;
}

inline double dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 normalize(const vec3 &v)
{
    return v / v.length();
}

inline bool NearZero(const vec3 &v)
{
    double s = 1e-8;
    return ((std::fabs(v.x) < s) && (std::fabs(v.y) < s) && (std::fabs(v.z) < s));
}

inline vec3 Reflect(const vec3 &v, const vec3 &norm)
{
    return v - 2 * dot(v, norm) * norm;
}

inline vec3 Refract(const vec3 &uv, const vec3 &n, double refractiveIndexRatio)
{
    double cosTheta = fmin(dot(-uv, n), 1.0);
    vec3 perp       = refractiveIndexRatio * (uv + cosTheta * n);
    vec3 parallel   = -sqrt(fabs(1 - perp.lengthSquared())) * n;
    return perp + parallel;
}

//////////////////////////////
// Random
//
inline double RandomDouble()
{
    return rand() / (RAND_MAX + 1.0);
}

inline double RandomDouble(double min, double max)
{
    return min + (max - min) * RandomDouble();
}

inline vec3 RandomVec3()
{
    return vec3(RandomDouble(), RandomDouble(), RandomDouble());
}

inline vec3 RandomVec3(double min, double max)
{
    return vec3(RandomDouble(min, max), RandomDouble(min, max), RandomDouble(min, max));
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
        vec3 p = vec3(RandomDouble(-1, 1), RandomDouble(-1, 1), 0);
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
    Ray(const vec3 &origin, const vec3 &direction, const double time) : orig(origin), dir(direction), tm(time) {}

    const vec3 &origin() const { return orig; }
    const vec3 &direction() const { return dir; }
    const double &time() const { return tm; }

    vec3 at(double t) const
    {
        return orig + t * dir;
    }

private:
    vec3 orig;
    vec3 dir;
    double tm;
};

inline vec3 LinearToSRGB(const vec3 &v)
{
    // const double exp = 1 / 2.2;
    // return vec3(pow(v.x, exp), pow(v.y, exp), pow(v.z, exp));
    return vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

void WriteColor(std::ostream &out, const vec3 &pixelColor)
{
    vec3 gammaCorrectedColor = LinearToSRGB(pixelColor);

    int rByte = int(256 * Clamp(0.0, 0.999, gammaCorrectedColor.r));
    int gByte = int(256 * Clamp(0.0, 0.999, gammaCorrectedColor.g));
    int bByte = int(256 * Clamp(0.0, 0.999, gammaCorrectedColor.b));

    out << rByte << ' ' << gByte << ' ' << bByte << '\n';
}

enum class MaterialType
{
    Lambert,
    Metal,
    Dielectric,
};

struct Material;

struct HitRecord
{
    vec3 normal;
    vec3 p;
    double t;
    double u, v;
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
        double minX;
        double minY;
        double minZ;
        double maxX;
        double maxY;
        double maxZ;
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

    AABB(vec3 min, vec3 max)
    {
        minP = min;
        maxP = max;
    }
    AABB(AABB box1, AABB box2)
    {
        minX = box1.minX <= box2.minX ? box1.minX : box2.minX;
        minY = box1.minY <= box2.minY ? box1.minY : box2.minY;
        minZ = box1.minZ <= box2.minZ ? box1.minZ : box2.minZ;

        maxX = box1.maxX >= box2.maxX ? box1.maxX : box2.maxX;
        maxY = box1.maxY >= box2.maxY ? box1.maxY : box2.maxY;
        maxZ = box1.maxZ >= box2.maxZ ? box1.maxZ : box2.maxZ;
    }

    bool Hit(const Ray &r, double tMin, double tMax)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            double oneOverDir = 1.f / r.direction().e[axis];
            double t0         = (minP.e[axis] - r.origin().e[axis]) * oneOverDir;
            double t1         = (maxP.e[axis] - r.origin().e[axis]) * oneOverDir;
            if (t0 > t1)
            {
                double temp = t0;
                t0          = t1;
                t1          = temp;
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
};

class Sphere
{
public:
    Sphere() {}
    Sphere(vec3 c, double r, Material *m) : center(c), radius(fmax(0, r)), material(m)
    {
        centerVec      = vec3(0, 0, 0);
        vec3 boxRadius = vec3(radius, radius, radius);
        aabb.minP      = c - boxRadius;
        aabb.maxP      = c + boxRadius;
    }
    Sphere(vec3 c1, vec3 c2, double r, Material *m) : center(c1), radius(fmax(0, r)), material(m)
    {
        vec3 boxRadius = vec3(radius, radius, radius);
        centerVec      = c2 - c1;
        AABB box1      = AABB(c1 - boxRadius, c1 + boxRadius);
        AABB box2      = AABB(c2 - boxRadius, c2 + boxRadius);
        aabb           = AABB(box1, box2);
    }
    bool Hit(const Ray &r, const double tMin, const double tMax, HitRecord &record) const
    {
        // (C - P) dot (C - P) = r^2
        // (C - (O + Dt)) dot (C - (O + Dt)) - r^2 = 0
        // (-Dt + C - O) dot (-Dt + C - O) - r^2 = 0
        // t^2(D dot D) - 2t(D dot (C - O)) + (C - O dot C - O) - r^2 = 0
        vec3 oc  = Center(r.time()) - r.origin();
        double a = dot(r.direction(), r.direction());
        double h = dot(r.direction(), oc);
        double c = dot(oc, oc) - radius * radius;

        double discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        double result = (h - sqrt(discriminant)) / a;
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
    vec3 Center(double time) const
    {
        return center + centerVec * time;
    }
    AABB &GetAABB()
    {
        return aabb;
    }
    static void GetUV(double &u, double &v, const vec3 &p)
    {
        double zenith  = acos(-p.y);
        double azimuth = atan2(-p.z, p.x) + PI;

        u = azimuth / (2 * PI);
        v = zenith / PI;
    }

private:
    vec3 center;
    double radius;
    Material *material;
    vec3 centerVec;
    AABB aabb;
};

struct Scene
{
    std::vector<Sphere> spheres;
    AABB sceneAABB;

    // bool Hit(const Ray &r, const double tMin, const double tMax, HitRecord &record) const
    // {
    //     HitRecord temp;
    //     double closest = tMax;
    //     bool hit       = false;
    //     for (const Sphere &sphere : spheres)
    //     {
    //         if (sphere.Hit(r, tMin, tMax, temp))
    //         {
    //             if (temp.t < closest)
    //             {
    //                 closest = temp.t;
    //                 record  = temp;
    //                 hit     = true;
    //             }
    //         }
    //     }
    //     return hit;
    // }

    void Clear()
    {
        spheres.clear();
    }

    void Add(Sphere &sphere)
    {
        spheres.push_back(sphere);
        sceneAABB = AABB(sceneAABB, sphere.GetAABB());
    }

    void Add(Sphere &&sphere)
    {
        spheres.push_back(std::move(sphere));
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
        scene       = inScene;
        AABB *aabbs = (AABB *)malloc(scene->spheres.size() * sizeof(AABB));
        for (u32 i = 0; i < scene->spheres.size(); i++)
        {
            Sphere &sphere = scene->spheres[i];
            aabbs[i]       = sphere.GetAABB();
        }
        Build(aabbs, (u32)scene->spheres.size());
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
        if (extent.y > extent.e[axis]) axis = 1;
        if (extent.z > extent.e[axis]) axis = 2;
        double splitPos = min.e[axis] + extent.e[axis];

        int i = node.offset;
        int j = i + node.count - 1;
        while (i <= j)
        {
            vec3 center  = aabbs[leafIndices[i]].Center();
            double value = center.e[axis];
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

    inline bool Hit(const Ray &r, const double tMin, const double tMax, HitRecord &record) const
    {
        u32 stack[64];
        u32 stackPtr      = 0;
        stack[stackPtr++] = 0;
        double closest    = tMax;
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
                    const Sphere &sphere = scene->spheres[leafIndices[node.offset + i]];
                    if (sphere.Hit(r, tMin, tMax, temp))
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

struct Texture
{
    enum class Type
    {
        Solid,
        Checkered,
        Image,
    } type;

    static Texture CreateSolid(const vec3 &albedo)
    {
        Texture texture;
        texture.baseColor = albedo;
        texture.type      = Type::Solid;
        return texture;
    }
    static Texture CreateCheckered(double scale, const vec3 &even, const vec3 &odd)
    {
        Texture texture;
        texture.baseColor  = even;
        texture.baseColor2 = odd;
        texture.type       = Type::Checkered;
        texture.invScale   = 1.0 / scale;
        return texture;
    }

    static Texture CreateImage(const char *filename)
    {
        Texture texture;
        texture.image = LoadFile(filename);
        texture.type  = Type::Image;
        return texture;
    }

    vec3 Value(const double u, const double v, const vec3 &p) const
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
                int x = int(std::floor(p.x * invScale));
                int y = int(std::floor(p.y * invScale));
                int z = int(std::floor(p.z * invScale));
                return (x + y + z) % 2 == 0 ? baseColor : baseColor2;
            }
            break;
            case Type::Image:
            {
                assert(image.width);
                assert(image.height);
                int x = int(u * image.width);
                int y = int((1 - v) * image.height);

                u8 *data    = GetColor(&image, x, y);
                f64 divisor = 1 / 255.0;
                f64 r       = f64(data[0]) * divisor;
                f64 g       = f64(data[1]) * divisor;
                f64 b       = f64(data[2]) * divisor;
                return vec3(r, g, b);
            }
            break;
            default: assert(0); return vec3(0, 0, 0);
        }
    }

    vec3 baseColor;

    // checkered
    vec3 baseColor2;
    double invScale;

    // image
    Image image;
};

struct Material
{
    MaterialType type;
    vec3 albedo;
    double fuzz;
    double refractiveIndex;

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

    static Material CreateMetal(vec3 inAlbedo, double inFuzz = 0.0)
    {
        Material result;
        result.type   = MaterialType::Metal;
        result.albedo = inAlbedo;
        result.fuzz   = inFuzz < 1 ? inFuzz : 1;
        return result;
    }

    static Material CreateDielectric(double inRefractiveIndex)
    {
        Material result;
        result.type            = MaterialType::Dielectric;
        result.refractiveIndex = inRefractiveIndex;
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
        double ri   = record.isFrontFace ? 1.0 / refractiveIndex : refractiveIndex;

        vec3 rayDir     = normalize(r.direction());
        double cosTheta = fmin(dot(-rayDir, record.normal), 1.0);
        double sinTheta = sqrt(1 - cosTheta * cosTheta);
        // total internal reflection
        bool cannotRefract = ri * sinTheta > 1.0;

        double f0          = (1 - ri) / (1 + ri);
        f0                 = f0 * f0;
        double reflectance = f0 + (1 - f0) * pow(1 - cosTheta, 5);
        vec3 direction     = cannotRefract || reflectance > RandomDouble()
                                 ? Reflect(rayDir, record.normal)
                                 : Refract(rayDir, record.normal, ri);
        scatteredRay       = Ray(record.p, direction, r.time());

        return true;
    }

    inline bool Scatter(const Ray &r, const HitRecord &record, vec3 &attenuation, Ray &scatteredRay)
    {
        switch (type)
        {
            case MaterialType::Lambert: return LambertScatter(r, record, attenuation, scatteredRay);
            case MaterialType::Metal: return MetalScatter(r, record, attenuation, scatteredRay);
            case MaterialType::Dielectric: return DielectricScatter(r, record, attenuation, scatteredRay);
            default: assert(0); return false;
        }
    }
};

vec3 RayColor(const Ray &r, const int depth, const BVH &bvh)
{
    if (depth <= 0)
        return vec3(0, 0, 0);

    vec3 sphereCenter = vec3(0, 0, -1);
    HitRecord record;

    if (bvh.Hit(r, 0.001, infinity, record))
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
    double t                       = 0.5 * (normalizedDirection.y + 1.0);
    return (1 - t) * vec3(1, 1, 1) + t * vec3(0.5, 0.7, 1.0);
}

#define EARTH 1
int main(int argc, char *argv[])
{

#if SPHERES
    const double aspectRatio  = 16.0 / 9.0;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    const vec3 lookFrom       = vec3(13, 2, 3);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const double verticalFov  = 20;
    const double defocusAngle = 0.6;
    const double focusDist    = 10;
#elif EARTH
    const double aspectRatio  = 16.0 / 9.0;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    const vec3 lookFrom       = vec3(0, 0, 12);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const double verticalFov  = 20;
    const double defocusAngle = 0;
    const double focusDist    = 10;
#endif

    int imageHeight    = int(imageWidth / aspectRatio);
    imageHeight        = imageHeight < 1 ? 1 : imageHeight;
    double focalLength = (lookFrom - lookAt).length();
    double theta       = DegreesToRadians(verticalFov);
    double h           = tan(theta / 2);

    vec3 f = normalize(lookFrom - lookAt);
    vec3 s = cross(worldUp, f);
    vec3 u = cross(f, s);

    double viewportHeight = 2 * h * focusDist;
    double viewportWidth  = viewportHeight * (double(imageWidth) / imageHeight);
    vec3 cameraCenter     = lookFrom;

    vec3 viewportU = viewportWidth * s;
    vec3 viewportV = viewportHeight * -u;

    vec3 pixelDeltaU = viewportU / imageWidth;
    vec3 pixelDeltaV = viewportV / imageHeight;

    vec3 viewportUpperLeft = cameraCenter - focusDist * f - viewportU / 2 - viewportV / 2;
    vec3 pixel00           = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

    double defocusRadius = focusDist * tan(DegreesToRadians(defocusAngle / 2));
    vec3 defocusDiskU    = defocusRadius * s;
    vec3 defocusDiskV    = defocusRadius * u;

    std::cout << "P3\n"
              << imageWidth << ' ' << imageHeight << "\n255\n";

    Scene scene;

#if SPHERES
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            double chooseMat = RandomDouble();
            vec3 center(a + 0.9 * RandomDouble(), 0.2, b + 0.9 * RandomDouble());

            if ((center - vec3(4, 0.2, 0)).length() > 0.9)
            {
                Material *material = (Material *)malloc(sizeof(Material));
                // Diffuse
                if (chooseMat < 0.8)
                {
                    vec3 albedo  = RandomVec3() * RandomVec3();
                    vec3 center2 = center + vec3(0, RandomDouble(0, .5), 0);
                    *material    = Material::CreateLambert(albedo);
                    scene.Add(Sphere(center, center2, 0.2, material));
                }
                // Metal
                else if (chooseMat < 0.95)
                {
                    vec3 albedo = RandomVec3(0.5, 1);
                    double fuzz = RandomDouble(0, 0.5);
                    *material   = Material::CreateMetal(albedo, fuzz);
                    scene.Add(Sphere(center, 0.2, material));
                }
                // Glass
                else
                {
                    *material = Material::CreateDielectric(1.5);
                    scene.Add(Sphere(center, 0.2, material));
                }
            }
        }
    }

    Texture checkered    = Texture::CreateCheckered(0.32, vec3(.2, .3, .1), vec3(.9, .9, .9));
    Material materials[] = {
        Material::CreateDielectric(1.5),
        Material::CreateLambert(vec3(0.4, 0.2, 0.1)),
        Material::CreateMetal(vec3(0.7, 0.6, 0.5), 0.0),
        Material::CreateLambert(&checkered),
    };

    scene.Add(Sphere(vec3(0, 1, 0), 1.0, &materials[0]));
    scene.Add(Sphere(vec3(-4, 1, 0), 1.0, &materials[1]));
    scene.Add(Sphere(vec3(4, 1, 0), 1.0, &materials[2]));

    // ground
    scene.Add(Sphere(vec3(0, -1000, 0), 1000, &materials[3]));

#elif EARTH
    Texture earth             = Texture::CreateImage("earthmap.jpg");
    Material surface          = Material::CreateLambert(&earth);

    scene.Add(Sphere(vec3(0, 0, 0), 2, &surface));
#endif

    BVH bvh;
    bvh.Build(&scene);

    for (int height = 0; height < imageHeight; height++)
    {
        std::clog << "\rScanlines remaining: " << (imageHeight - height) << ' ' << std::flush;
        for (int width = 0; width < imageWidth; width++)
        {
            vec3 pixelColor(0, 0, 0);

            for (int i = 0; i < samplesPerPixel; i++)
            {
                const vec3 offset      = vec3(RandomDouble() - 0.5, RandomDouble() - 0.5, 0);
                const vec3 pixelSample = pixel00 + ((width + offset.x) * pixelDeltaU) + ((height + offset.y) * pixelDeltaV);
                vec3 rayOrigin;
                if (defocusAngle <= 0)
                {
                    rayOrigin = cameraCenter;
                }
                else
                {
                    vec3 sample = RandomInUnitDisk();
                    rayOrigin   = cameraCenter + sample[0] * defocusDiskU + sample[1] * defocusDiskV;
                }
                const vec3 rayDirection = pixelSample - rayOrigin;
                const double rayTime    = RandomDouble();
                Ray r(rayOrigin, rayDirection, rayTime);

                pixelColor += RayColor(r, maxDepth, bvh);
            }

            pixelColor /= samplesPerPixel;
            WriteColor(std::cout, pixelColor);
        }
    }
    std::clog << "\rDone.                                   \n";
}
