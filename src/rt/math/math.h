#ifndef MATH_H
#define MATH_H

#include <cmath>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

#include "vec3.h"

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

inline u32 Encode(f32 f) { return (u32)((f + 1) / 2 * 65535.f + 0.5f); }

inline f32 Decode(u16 u) { return (f32)(u / 65535.f * 2 - 1); }

inline u32 EncodeOctahedral(Vec3f v)
{
    Vec2f p = v.xy * (1.f / (Abs(v.x) + Abs(v.y) + Abs(v.z)));
    p       = v.z < 0 ? (1.f - Abs(p.yx()) * Vec2f(copysign(1, p.x), copysign(1, p.y))) : p;

    return (Encode(p[0]) << 16) | Encode(p[1]);
}

inline f32 PowerHeuristic(u32 numA, f32 pdfA, u32 numB, f32 pdfB)
{
    f32 a = Sqr(numA * pdfA);
    f32 b = Sqr(numB * pdfB);
    return a / (a + b);
}

} // namespace rt

#endif
