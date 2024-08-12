struct Basis
{
    vec3 t;
    vec3 b;
    vec3 n;
};

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
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const;
    vec3 Center(f32 time) const;
    AABB GetAABB() const;
    static void GetUV(f32 &u, f32 &v, const vec3 &p)
    {
        f32 zenith  = acos(-p.y);
        f32 azimuth = atan2(-p.z, p.x) + PI;

        u = azimuth / (2 * PI);
        v = zenith / PI;
    }
    f32 PdfValue(const vec3 &origin, const vec3 &direction) const;
    vec3 Random(const vec3 &origin, vec2 u) const;

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
        vec3 n = Cross(u, v);
        normal = Normalize(n);
        d      = Dot(normal, q);
        w      = n / Dot(n, n);
        area   = n.length();
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
        f32 denom = Dot(normal, r.direction());
        // if the ray is parallel to the plane
        if (fabs(denom) < 1e-8f)
            return false;

        f32 t = (d - Dot(normal, r.origin())) / denom;

        if (t < tMin || t > tMax) return false;

        vec3 intersection = r.at(t);

        vec3 planarHitVector = intersection - q;
        f32 alpha            = Dot(w, Cross(planarHitVector, v));
        f32 beta             = Dot(w, Cross(u, planarHitVector));

        if (!(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1)) return false;

        record.u        = alpha;
        record.v        = beta;
        record.p        = intersection;
        record.t        = t;
        record.material = material;
        record.SetNormal(r, normal);
        return true;
    }

    f32 PdfValue(const vec3 &origin, const vec3 &direction) const
    {
        HitRecord rec;
        if (!this->Hit(Ray(origin, direction), 0.0001f, infinity, rec))
            return 0;
        f32 distanceSquared = rec.t * rec.t * direction.lengthSquared();
        f32 cosine          = fabs(Dot(direction, rec.normal) / direction.length());
        return distanceSquared / (cosine * area);
    };

    vec3 Random(const vec3 &origin, vec2 random) const
    {
        vec3 p = q + (random.x * u) + (random.y * v);
        return p - origin;
    }

    vec3 q; // corner
    vec3 u, v;
    Material *material;

    f32 area;
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

struct Light
{
    PrimitiveType type;
    void *primitive;
};
struct ScatterRecord
{
    vec3 attenuation;
    Ray skipPDFRay;
    vec3 sample;
};

struct HomogeneousTransform
{
    vec3 translation;
    f32 rotateAngleY;
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

    void FinalizePrimitives();

    inline i32 GetIndex(PrimitiveType type, i32 primIndex) const;
    inline void GetTypeAndLocalindex(const u32 totalIndex, PrimitiveType *type, u32 *localIndex) const;

    void AddConstantMedium(PrimitiveType type, i32 primIndex, i32 constantMediumIndex);

    void AddTransform(PrimitiveType type, i32 primIndex, i32 transformIndex);

    void GetAABBs(AABB *aabbs);
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &temp, u32 index);
};
