#ifndef MATH_H
#define MATH_H

#include <cmath>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

namespace rt
{
__forceinline f32 AngleBetween(Vec3f v1, Vec3f v2)
{
    if (Dot(v1, v2) < 0) return PI - 2 * SafeASin(Length(v1 + v2) / 2.f);
    return 2 * SafeASin(Length(v2 - v1) / 2.f);
}

struct Basis
{
    Vec3f t;
    Vec3f b;
    Vec3f n;
};

//////////////////////////////
// Ray
//
struct Ray
{
    Ray() {}
    Ray(const Vec3f &origin, const Vec3f &direction) : o(origin), d(direction), t(0) {}
    Ray(const Vec3f &origin, const Vec3f &direction, const f32 time)
        : o(origin), d(direction), t(time)
    {
    }

    Vec3f at(f32 time) const { return o + time * d; }

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
            if (tMax <= tMin) return false;
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
            if (tMax <= tMin) return false;
        }
        return true;
    }

    inline Vec3f Center() const { return (maxP + minP) * 0.5f; }
    inline Vec3f Centroid() const { return Center(); }
    inline Vec3f GetHalfExtent() { return (maxP - minP) * 0.5f; }

    Vec3f operator[](int i) const { return i == 0 ? minP : maxP; }

    inline Vec3f Offset(const Vec3f &p) const
    {
        Vec3f o = p - minP;
        if (maxX > minX) o.x /= (maxX - minX);
        if (maxY > minY) o.y /= (maxY - minY);
        if (maxZ > minZ) o.z /= (maxZ - minZ);
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
    Vec3f Diagonal() const { return maxP - minP; }
    f32 SurfaceArea() const
    {
        Vec3f d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    i32 MaxDimension() const
    {
        Vec3f d = Diagonal();
        if (d.x > d.y && d.x > d.z) return 0;
        else if (d.y > d.z) return 1;
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

inline u64 EncodeMorton2(u32 x, u32 y) { return (LeftShift2(y) << 1) | LeftShift2(x); }

//////////////////////////////
// Complex numbers
//

template <typename T>
struct Complex
{
    T real, im;

    Complex() {}
    Complex(const T &real) : real(real), im(0) {}
    Complex(const T &real, const T &im) : real(real), im(im) {}

    Complex operator-() const { return {-real, -im}; }

    Complex operator+(const Complex<T> &z) const { return {real + z.real, im + z.im}; }

    Complex operator-(const Complex<T> &z) const { return {real - z.real, im - z.im}; }

    Complex operator*(const Complex<T> &z) const
    {
        return {real * z.real - im * z.im, real * z.im + im * z.real};
    }

    Complex operator/(const Complex<T> &z) const
    {
        f32 scale = 1 / (z.real * z.real + z.im * z.im);
        return {scale * (real * z.real + im * z.im), scale * (im * z.real - real * z.im)};
    }

    friend Complex operator+(const T &value, const Complex<T> &z)
    {
        return Complex(value) + z;
    }

    friend Complex operator-(const T &value, const Complex<T> &z)
    {
        return Complex(value) - z;
    }

    friend Complex operator*(const T &value, const Complex<T> &z)
    {
        return Complex(value) * z;
    }

    friend Complex operator/(const T &value, const Complex<T> &z)
    {
        return Complex(value) / z;
    }

    LaneNF32 Norm() const { return real * real + im * im; }
};

template <typename T>
T Abs(const Complex<T> &z)
{
    return Sqrt(z.Norm());
}

template <typename T>
Complex<T> Sqrt(const Complex<T> &z)
{
    T n  = Abs(z);
    T t1 = Sqrt(T(.5) * (n + Abs(z.real)));
    T t2 = T(.5) * z.im / t1;

    if (All(n == 0)) return Complex<T>(0);

    Complex<T> out;
    out.real = Select(z.real >= 0, t1, Abs(t2));
    out.im   = Select(z.real >= 0, t2, Copysign(t1, z.im));
    return out;
}

// struct Frame
// {
//     Vec3f x;
//     Vec3f y;
//     Vec3f z;
//
//     Frame() : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}
//
//     Frame(Vec3f x, Vec3f y, Vec3f z) : x(x), y(y), z(z) {}
//
//     static Frame FromXZ(Vec3f x, Vec3f z)
//     {
//         return Frame(x, Cross(z, x), z);
//     }
//     static Frame FromXY(Vec3f x, Vec3f y)
//     {
//         return Frame(x, y, Cross(x, y));
//     }
//     Vec3f ToLocal(Vec3f a) const
//     {
//         return Vec3f(Dot(x, a), Dot(y, a), Dot(z, a));
//     }
//
//     Vec3f FromLocal(Vec3f a) const
//     {
//         return a.x * x + a.y * y + a.z * z;
//     }
// };

//////////////////////////////
// Octahedral encoding
//

struct OctahedralVector
{
    u16 x;
    u16 y;
};

inline u16 Encode(f32 f) { return (u16)((f + 1) / 2 * 65535.f + 0.5f); }

inline f32 Decode(u16 u) { return (f32)(u / 65535.f * 2 - 1); }

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
