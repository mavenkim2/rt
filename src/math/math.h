#ifndef MATH_H
#define MATH_H

#include <cmath>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

namespace rt
{

//////////////////////////////
// Ray
//
struct Ray
{
    Ray() {}
    Ray(const Vec3f &origin, const Vec3f &direction) : o(origin), d(direction), t(0) {}
    Ray(const Vec3f &origin, const Vec3f &direction, const f32 time) : o(origin), d(direction), t(time) {}

    Vec3f at(f32 time) const
    {
        return o + time * d;
    }

    Vec3f o;
    Vec3f d;
    f32 t;
};

//////////////////////////////
// AABB
//
union AABB
{
    struct
    {
        Vec3f minP;
        Vec3f maxP;
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
    AABB(Vec3f pt1, Vec3f pt2)
    {
        minX = pt1.x <= pt2.x ? pt1.x : pt2.x;
        minY = pt1.y <= pt2.y ? pt1.y : pt2.y;
        minZ = pt1.z <= pt2.z ? pt1.z : pt2.z;

        maxX = pt1.x >= pt2.x ? pt1.x : pt2.x;
        maxY = pt1.y >= pt2.y ? pt1.y : pt2.y;
        maxZ = pt1.z >= pt2.z ? pt1.z : pt2.z;
        PadToMinimums();
    }
    AABB(const AABB &other)
    {
        minP = other.minP;
        maxP = other.maxP;
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

    __forceinline AABB &operator=(const AABB &other)
    {
        minP = other.minP;
        maxP = other.maxP;
        return *this;
    }
    bool Hit(const Ray &r, f32 tMin, f32 tMax)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            f32 oneOverDir = 1.f / r.d.e[axis];
            f32 t0         = (minP[axis] - r.o[axis]) * oneOverDir;
            f32 t1         = (maxP[axis] - r.o[axis]) * oneOverDir;
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

    bool Hit(const Ray &r, f32 tMin, f32 tMax, const int dirIsNeg[3]) const
    {
        for (int axis = 0; axis < 3; axis++)
        {
            f32 min = (*this)[dirIsNeg[axis]][axis];
            f32 max = (*this)[1 - dirIsNeg[axis]][axis];

            f32 oneOverDir = 1.f / r.d.e[axis];
            f32 t0         = (min - r.o[axis]) * oneOverDir;
            f32 t1         = (max - r.o[axis]) * oneOverDir;
            tMin           = t0 > tMin ? t0 : tMin;
            tMax           = t1 < tMax ? t1 : tMax;
            if (tMax <= tMin)
                return false;
        }
        return true;
    }

    inline Vec3f Center() const
    {
        return (maxP + minP) * 0.5f;
    }
    inline Vec3f Centroid() const
    {
        return Center();
    }
    inline Vec3f GetHalfExtent()
    {
        return (maxP - minP) * 0.5f;
    }

    Vec3f operator[](int i) const
    {
        return i == 0 ? minP : maxP;
    }

    inline Vec3f Offset(const Vec3f &p) const
    {
        Vec3f o = p - minP;
        if (maxX > minX)
            o.x /= (maxX - minX);
        if (maxY > minY)
            o.y /= (maxY - minY);
        if (maxZ > minZ)
            o.z /= (maxZ - minZ);
        return o;
    }

    inline void Expand(f32 delta)
    {
        Vec3f pad = Vec3f(delta / 2, delta / 2, delta / 2);
        minP -= pad;
        maxP += pad;
    }

    inline void PadToMinimums()
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
    Vec3f Diagonal() const
    {
        return maxP - minP;
    }
    f32 SurfaceArea() const
    {
        Vec3f d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    i32 MaxDimension() const
    {
        Vec3f d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        return 2;
    }

    __forceinline void Extend(const Vec3f &p)
    {
        minP = Min(minP, p);
        maxP = Max(maxP, p);
    }
};

inline AABB Union(const AABB &box1, const AABB &box2)
{
    AABB result;
    result.minP = Min(box1.minP, box2.minP);
    result.maxP = Max(box1.minP, box2.minP);
    return result;
}

inline AABB Union(const AABB &box1, const Vec3f &p)
{
    AABB result;
    result.minP = Min(box1.minP, p);
    result.maxP = Max(box1.maxP, p);
    return result;
}

inline Vec3f Min(const AABB &box1, const AABB &box2)
{
    Vec3f result = Min(box1.minP, box2.minP);
    return result;
}

inline Vec3f Max(const AABB &box1, const AABB &box2)
{
    Vec3f result = Max(box1.maxP, box2.maxP);
    return result;
}

//////////////////////////////
// Morton
//

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
inline u64 LeftShift2(u64 x)
{
    x &= 0xffffffff;
    x = (x ^ (x << 16)) & 0x0000ffff0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f0f0f0f0f;
    x = (x ^ (x << 2)) & 0x3333333333333333;
    x = (x ^ (x << 1)) & 0x5555555555555555;
    return x;
}

inline u64 EncodeMorton2(u32 x, u32 y)
{
    return (LeftShift2(y) << 1) | LeftShift2(x);
}

#if 0
inline LaneVec2i EncodeMorton2(LaneU32 x, LaneU32 y)
{
    LaneVec2i vecs[2];

    vecs[0].x = SignExtend(x);
    vecs[0].y = SignExtend(y);
    vecs[1].x = SignExtend(PermuteU32(x, 2, 3, 2, 3));
    vecs[1].y = SignExtend(PermuteU32(y, 2, 3, 2, 3));

    LaneU32 mask0 = LaneU32FromU64(0xffffffffull);
    LaneU32 mask1 = LaneU32FromU64(0x0000ffff0000ffffull);
    LaneU32 mask2 = LaneU32FromU64(0x00ff00ff00ff00ffull);
    LaneU32 mask3 = LaneU32FromU64(0x0f0f0f0f0f0f0f0full);
    LaneU32 mask4 = LaneU32FromU64(0x3333333333333333ull);
    LaneU32 mask5 = LaneU32FromU64(0x5555555555555555ull);

    for (u32 i = 0; i < ArrayLength(vecs); i++)
    {
        for (u32 j = 0; j < 2; j++)
        {
            vecs[i][j] = vecs[i][j] & mask0;
            vecs[i][j] = (vecs[i][j] ^ (vecs[i][j] << 16ull)) & mask1;
            vecs[i][j] = (vecs[i][j] ^ (vecs[i][j] << 8ull)) & mask2;
            vecs[i][j] = (vecs[i][j] ^ (vecs[i][j] << 4ull)) & mask3;
            vecs[i][j] = (vecs[i][j] ^ (vecs[i][j] << 2ull)) & mask4;
            vecs[i][j] = (vecs[i][j] ^ (vecs[i][j] << 1ull)) & mask5;
        }
    }

    LaneVec2i result;
    result.x = (vecs[0].y << 1ull) | vecs[0].x;
    result.y = (vecs[1].y << 1ull) | vecs[1].x;
    return result;
}
#endif

//////////////////////////////
// Complex numbers
//
struct complex
{
    complex(f32 real) : real(real), im(0) {}
    complex(f32 real, f32 im) : real(real), im(im) {}

    complex operator-() const { return {-real, -im}; }

    complex operator+(complex z) const { return {real + z.real, im + z.im}; }

    complex operator-(complex z) const { return {real - z.real, im - z.im}; }

    complex operator*(complex z) const
    {
        return {real * z.real - im * z.im, real * z.im + im * z.real};
    }

    complex operator/(complex z) const
    {
        f32 scale = 1 / (z.real * z.real + z.im * z.im);
        return {scale * (real * z.real + im * z.im), scale * (im * z.real - real * z.im)};
    }

    friend complex operator+(f32 value, complex z)
    {
        return complex(value) + z;
    }

    friend complex operator-(f32 value, complex z)
    {
        return complex(value) - z;
    }

    friend complex operator*(f32 value, complex z)
    {
        return complex(value) * z;
    }

    friend complex operator/(f32 value, complex z)
    {
        return complex(value) / z;
    }

    f32 Norm() const
    {
        return real * real + im * im;
    }

    f32 real, im;
};

f32 Abs(const complex &z)
{
    return Sqrt(z.Norm());
}

complex Sqrt(const complex &z)
{
    f32 n  = Abs(z);
    f32 t1 = sqrtf(.5f * (n + Abs(z.real)));
    f32 t2 = .5f * z.im / t1;

    if (n == 0)
        return 0;

    if (z.real >= 0)
        return {t1, t2};
    else
        return {Abs(t2), std::copysign(t1, z.im)};
}

struct Frame
{
    Vec3f x;
    Vec3f y;
    Vec3f z;

    Frame() : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}

    Frame(Vec3f x, Vec3f y, Vec3f z) : x(x), y(y), z(z) {}

    static Frame FromXZ(Vec3f x, Vec3f z)
    {
        return Frame(x, Cross(z, x), z);
    }
    static Frame FromXY(Vec3f x, Vec3f y)
    {
        return Frame(x, y, Cross(x, y));
    }
    Vec3f ToLocal(Vec3f a) const
    {
        return Vec3f(Dot(x, a), Dot(y, a), Dot(z, a));
    }

    Vec3f FromLocal(Vec3f a) const
    {
        return a.x * x + a.y * y + a.z * z;
    }
};

//////////////////////////////
// Octahedral encoding
//

struct OctahedralVector
{
    u16 x;
    u16 y;
};

inline u16 Encode(f32 f)
{
    return (u16)((f + 1) / 2 * 65535.f + 0.5f);
}

inline f32 Decode(u16 u)
{
    return (f32)(u / 65535.f * 2 - 1);
}

// TODO: simd
OctahedralVector EncodeOctahedral(Vec3f v)
{
    v /= Abs(v.x) + Abs(v.y) + Abs(v.z);
    OctahedralVector result;

    if (v.z >= 0)
    {
        result.x = Encode(v.x);
        result.y = Encode(v.y);
    }
    else
    {
        result.x = Encode(1 - Abs(v.y)) * (u16)std::copysign(1, v.x);
        result.y = Encode(1 - Abs(v.x)) * (u16)std::copysign(1, v.y);
    }
    return result;
}

Vec3f DecodeOctahedral(OctahedralVector in)
{
    Vec3f result;
    result.x = Decode(in.x);
    result.y = Decode(in.y);
    result.z = 1 - Abs(result.x) - Abs(result.y);
    if (result.z < 0)
    {
        f32 xo   = result.x;
        result.x = (1 - Abs(result.y)) * std::copysign(1.f, xo);
        result.y = (1 - Abs(xo)) * std::copysign(1.f, result.y);
    }
    return Normalize(result);
}
} // namespace rt

#endif
