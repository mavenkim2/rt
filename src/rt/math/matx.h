#ifndef MATX_H
#define MATX_H

#include "bounds.h"
#include "vec4.h"

namespace rt
{
//////////////////////////////
// Mat3
//

union Mat3
{
    struct
    {
        f32 a1, a2, a3;
        f32 b1, b2, b3;
        f32 c1, c2, c3;
    };

    f32 elements[3][3];
    Vec3f columns[3];

    Mat3() {}
    Mat3(f32 a) : Mat3(a, a, a) {}

    Mat3(f32 a, f32 b, f32 c) : a1(a), a2(0), a3(0), b1(0), b2(b), b3(0), c1(0), c2(0), c3(c)
    {
    }

    Mat3(const Mat3 &other)
    {
        columns[0] = other.columns[0];
        columns[1] = other.columns[1];
        columns[2] = other.columns[2];
    }

    Mat3(Vec3f v) : Mat3(v.x, v.y, v.z) {}
    Mat3(const Vec3f &a, const Vec3f &b, const Vec3f &c) : columns{a, b, c} {}

    Vec3f &operator[](const i32 index) { return columns[index]; }

    Mat3 &operator/(const f32 f)
    {
        columns[0] /= f;
        columns[1] /= f;
        columns[2] /= f;
        return *this;
    }

    Mat3(f32 a1, f32 a2, f32 a3, f32 b1, f32 b2, f32 b3, f32 c1, f32 c2, f32 c3)
        : Mat3(Vec3f(a1, a2, a3), Vec3f(b1, b2, b3), Vec3f(c1, c2, c3))
    {
    }

    __forceinline Mat3 &operator=(const Mat3 &other)
    {
        columns[0] = other.columns[0];
        columns[1] = other.columns[1];
        columns[2] = other.columns[2];
        return *this;
    }

    Mat3 Inverse()
    {
        // a1, b1, c1,
        // a2, b2, c2,
        // a3, b3, c3
        f32 det1 = b2 * c3 - b3 * c2;
        f32 det2 = -(a2 * c3 - c2 * a3);
        f32 det3 = a2 * b3 - a3 * b2;

        f32 det4 = -(b1 * c3 - c1 * b3);
        f32 det5 = a1 * c3 - a3 * c1;
        f32 det6 = -(a1 * b3 - a3 * b1);

        f32 det7 = b1 * c2 - b2 * c1;
        f32 det8 = -(a1 * c2 - a2 * c1);
        f32 det9 = a1 * b2 - a2 * b1;

        Mat3 result(det1, det2, det3, det4, det5, det6, det7, det8, det9);

        f32 det = a1 * det1 + b1 * det2 + c1 * det3;

        Assert(det != 0.f);
        return result / det;
    }

    static Mat3 Diag(f32 a, f32 b, f32 c)
    {
        Mat3 result(a, b, c);
        return result;
    }
};

__forceinline Mat3 Transpose(const Mat3 &m)
{
    return Mat3(Vec3f(m.a1, m.b1, m.c1), Vec3f(m.a2, m.b2, m.c2), Vec3f(m.a3, m.b3, m.c3));
}

inline Mat3 operator*(Mat3 a, Mat3 b)
{
    Mat3 result;
    for (int j = 0; j < 3; j += 1)
    {
        for (int i = 0; i < 3; i += 1)
        {
            result.elements[i][j] =
                (a.elements[0][j] * b.elements[i][0] + a.elements[1][j] * b.elements[i][1] +
                 a.elements[2][j] * b.elements[i][2]);
        }
    }

    return result;
}

__forceinline f32 Determinant(const Mat3 &t)
{
    return Dot(t.columns[0], Cross(t.columns[1], t.columns[2]));
}

__forceinline Vec3f operator*(const Mat3 &m, const Vec3f &v)
{
    return m.columns[0] * v[0] + m.columns[1] * v[1] + m.columns[2] * v[2];
}

//////////////////////////////
// Mat4
//

struct AffineSpace;
struct Mat4
{
    union
    {
        f32 elements[4][4];
        Lane4F32 columns[4];
        Vec4f columns_[4];
        struct
        {
            f32 a1, a2, a3, a4;
            f32 b1, b2, b3, b4;
            f32 c1, c2, c3, c4;
            f32 d1, d2, d3, d4;
        };
    };

    __forceinline Mat4()
        : a1(0), a2(0), a3(0), a4(0), b1(0), b2(0), b3(0), b4(0), c1(0), c2(0), c3(0), c4(0),
          d1(0), d2(0), d3(0), d4(0)
    {
    }

    __forceinline Mat4(f32 a1, f32 a2, f32 a3, f32 a4, f32 b1, f32 b2, f32 b3, f32 b4, f32 c1,
                       f32 c2, f32 c3, f32 c4, f32 d1, f32 d2, f32 d3, f32 d4)
        : a1(a1), b1(a2), c1(a3), d1(a4), a2(b1), b2(b2), c2(b3), d2(b4), a3(c1), b3(c2),
          c3(c3), d3(c4), a4(d1), b4(d2), c4(d3), d4(d4)
    {
    }

    Mat4(f32 val)
        : a1(val), a2(0), a3(0), a4(0), b1(0), b2(val), b3(0), b4(0), c1(0), c2(0), c3(val),
          c4(0), d1(0), d2(0), d3(0), d4(val)
    {
    }

    explicit Mat4(const AffineSpace &s);

    // Copy constructor
    __forceinline Mat4(const Mat4 &other)
    {
        columns[0] = other.columns[0];
        columns[1] = other.columns[1];
        columns[2] = other.columns[2];
        columns[3] = other.columns[3];
    }

    // Assignment
    __forceinline Mat4 &operator=(const Mat4 &other)
    {
        columns[0] = other.columns[0];
        columns[1] = other.columns[1];
        columns[2] = other.columns[2];
        columns[3] = other.columns[3];
        return *this;
    }

    Vec4f GetRow(int i) const
    {
        Assert(i >= 0 && i < 4);
        return Vec4f(columns[0][i], columns[1][i], columns[2][i], columns[3][i]);
    }

    b32 operator==(const Mat4 &other)
    {
        return a1 == other.a1 && a2 == other.a2 && a3 == other.a3 && a4 == other.a4 &&
               b1 == other.b1 && b2 == other.b2 && b3 == other.b3 && b4 == other.b4 &&
               c1 == other.c1 && c2 == other.c2 && c3 == other.c3 && c4 == other.c4 &&
               d1 == other.d1 && d2 == other.d2 && d3 == other.d3 && d4 == other.d4;
    }

    b32 operator!=(const Mat4 &other)
    {
        return a1 != other.a1 || a2 != other.a2 || a3 != other.a3 || a4 != other.a4 ||
               b1 != other.b1 || b2 != other.b2 || b3 != other.b3 || b4 != other.b4 ||
               c1 != other.c1 || c2 != other.c2 || c3 != other.c3 || c4 != other.c4 ||
               d1 != other.d1 || d2 != other.d2 || d3 != other.d3 || d4 != other.d4;
    }

    Lane4F32 &operator[](const i32 index) { return columns[index]; }
    const Lane4F32 &operator[](const i32 index) const { return columns[index]; }

    static __forceinline Mat4 Identity()
    {
        Mat4 result(1.f);
        return result;
    }

    static Mat4 Rotate(Vec3f axis, f32 theta)
    {
        Mat4 result           = Mat4::Identity();
        axis                  = Normalize(axis);
        f32 sinTheta          = sin(theta);
        f32 cosTheta          = cos(theta);
        f32 cosValue          = 1.f - cosTheta;
        result.elements[0][0] = (axis.x * axis.x * cosValue) + cosTheta;
        result.elements[0][1] = (axis.x * axis.y * cosValue) + (axis.z * sinTheta);
        result.elements[0][2] = (axis.x * axis.z * cosValue) - (axis.y * sinTheta);
        result.elements[1][0] = (axis.y * axis.x * cosValue) - (axis.z * sinTheta);
        result.elements[1][1] = (axis.y * axis.y * cosValue) + cosTheta;
        result.elements[1][2] = (axis.y * axis.z * cosValue) + (axis.x * sinTheta);
        result.elements[2][0] = (axis.z * axis.x * cosValue) + (axis.y * sinTheta);
        result.elements[2][1] = (axis.z * axis.y * cosValue) - (axis.x * sinTheta);
        result.elements[2][2] = (axis.z * axis.z * cosValue) + cosTheta;
        return result;
    }

    static Mat4 Translate(Vec3f value)
    {
        Mat4 result          = Mat4::Identity();
        result.columns[3][0] = value.x;
        result.columns[3][1] = value.y;
        result.columns[3][2] = value.z;
        return result;
    }

    // NOTE: assumes viewing matrix is right hand, with -z into screen

    // NOTE: vertical fov
    __forceinline static Mat4 Perspective(f32 fov, f32 aspectRatio, f32 n = 1e-2f,
                                          f32 f = 1000.f)
    {
        f32 divTanHalf = 1.f / Tan(fov / 2.f);
        Mat4 result(divTanHalf / aspectRatio, 0.f, 0.f, 0.f, 0.f, divTanHalf, 0.f, 0.f, 0.f,
                    0.f, f / (n - f), n * f / (n - f), 0.f, 0.f, -1.f, 0.f);
        return result;
    }
    // NOTE: horizontal fov
    __forceinline static Mat4 Perspective2(f32 fov, f32 aspectRatio, f32 n = 1e-2f,
                                           f32 f = 1000.f)
    {
        f32 divTanHalf = 1.f / Tan(fov / 2.f);
        Mat4 result(divTanHalf, 0.f, 0.f, 0.f, 0.f, divTanHalf * aspectRatio, 0.f, 0.f, 0.f,
                    0.f, f / (n - f), n * f / (n - f), 0.f, 0.f, -1.f, 0.f);
        return result;
    }
};

inline Mat4 Scale(const Vec3f &value)
{
    Mat4 result           = Mat4::Identity();
    result.elements[0][0] = value.x;
    result.elements[1][1] = value.y;
    result.elements[2][2] = value.z;
    return result;
}

inline Mat4 Scale(f32 value)
{
    Mat4 result           = Mat4::Identity();
    result.elements[0][0] = value;
    result.elements[1][1] = value;
    result.elements[2][2] = value;
    return result;
}
inline Mat4 Translate(const Vec3f &value)
{
    Mat4 m(1.f, 0.f, 0.f, value.x, 0.f, 1.f, 0.f, value.y, 0.f, 0.f, 1.f, value.z, 0.f, 0.f,
           0.f, 1.f);
    return m;
}

inline Vec4f Transform(const Mat4 &a, const Vec4f &b)
{
    Vec4f result;
#ifdef __SSE2__
    Lane4F32 laneResult =
        a.columns[0] * b.x + a.columns[1] * b.y + a.columns[2] * b.z + a.columns[3] * b.w;
    Lane4F32::StoreU(&result, laneResult);
    return result;
#else
    result.x = a.columns[0][0] * b.x;
    result.y = a.columns[0][1] * b.x;
    result.z = a.columns[0][2] * b.x;
    result.w = a.columns[0][3] * b.x;

    result.x += a.columns[1][0] * b.y;
    result.y += a.columns[1][1] * b.y;
    result.z += a.columns[1][2] * b.y;
    result.w += a.columns[1][3] * b.y;

    result.x += a.columns[2][0] * b.z;
    result.y += a.columns[2][1] * b.z;
    result.z += a.columns[2][2] * b.z;
    result.w += a.columns[2][3] * b.z;

    result.x += a.columns[3][0] * b.w;
    result.y += a.columns[3][1] * b.w;
    result.z += a.columns[3][2] * b.w;
    result.w += a.columns[3][3] * b.w;
    return result;
#endif
}

inline Vec3f Mul(const Mat4 &a, const Vec3f &b)
{
    Vec4f vec(b.x, b.y, b.z, 1);
    Vec4f result = Transform(a, vec);
    return Vec3f(result.x, result.y, result.z) / result.w;
}
__forceinline Vec3f TransformP(const Mat4 &a, const Vec3f &b) { return Mul(a, b); }

// Ignores translation
inline Vec3f TransformV(const Mat4 &a, const Vec3f &b)
{
    Lane4F32 laneResult = a.columns[0] * b.x + a.columns[1] * b.y + a.columns[2] * b.z;
    return Vec3f(laneResult[0], laneResult[1], laneResult[2]);
}

inline Mat4 LookAt(Vec3f eye, Vec3f center, Vec3f up)
{
    Vec3f f = Normalize(eye - center);
    Vec3f s = Normalize(Cross(Normalize(up), f));
    Vec3f u = Cross(f, s);

    return Mat4(s.x, s.y, s.z, -Dot(s, eye), u.x, u.y, u.z, -Dot(u, eye), f.x, f.y, f.z,
                -Dot(f, eye), 0.f, 0.f, 0.f, 1.f);
}

inline Mat4 Mul(const Mat4 &a, const Mat4 &b)
{
    Mat4 result;
    for (int j = 0; j < 4; j += 1)
    {
        for (int i = 0; i < 4; i += 1)
        {
            result.elements[i][j] =
                (a.elements[0][j] * b.elements[i][0] + a.elements[1][j] * b.elements[i][1] +
                 a.elements[2][j] * b.elements[i][2] + a.elements[3][j] * b.elements[i][3]);
        }
    }

    return result;
}

inline Mat4 operator*(const Mat4 &a, const Mat4 &b)
{
    Mat4 result = Mul(a, b);
    return result;
}

inline Mat3 Mul(Mat3 a, Mat3 b)
{
    Mat3 result;
    for (int j = 0; j < 3; j += 1)
    {
        for (int i = 0; i < 3; i += 1)
        {
            result.elements[i][j] =
                (a.elements[0][j] * b.elements[i][0] + a.elements[1][j] * b.elements[i][1] +
                 a.elements[2][j] * b.elements[i][2]);
        }
    }

    return result;
}

inline Vec3f Mul(Mat3 a, Vec3f b)
{
    Vec3f result;
    result.x = a.a1 * b.x;
    result.y = a.a2 * b.x;
    result.z = a.a3 * b.x;

    result.x += a.b1 * b.y;
    result.y += a.b2 * b.y;
    result.z += a.b3 * b.y;

    result.x += a.c1 * b.z;
    result.y += a.c2 * b.z;
    result.z += a.c3 * b.z;

    return result;
}

inline f32 Determinant(const Mat4 &a)
{
    f32 result =
        a.a1 * a.b2 * a.c3 * a.d4 - a.a1 * a.b2 * a.c4 * a.d3 + a.a1 * a.b3 * a.c4 * a.d2 -
        a.a1 * a.b3 * a.c2 * a.d4 + a.a1 * a.b4 * a.c2 * a.d3 - a.a1 * a.b4 * a.c3 * a.d2 -
        a.a2 * a.b3 * a.c4 * a.d1 + a.a2 * a.b3 * a.c1 * a.d4 - a.a2 * a.b4 * a.c1 * a.d3 +
        a.a2 * a.b4 * a.c3 * a.d1 - a.a2 * a.b1 * a.c3 * a.d4 + a.a2 * a.b1 * a.c4 * a.d3 +
        a.a3 * a.b4 * a.c1 * a.d2 - a.a3 * a.b4 * a.c2 * a.d1 + a.a3 * a.b1 * a.c2 * a.d4 -
        a.a3 * a.b1 * a.c4 * a.d2 + a.a3 * a.b2 * a.c4 * a.d1 - a.a3 * a.b2 * a.c1 * a.d4 -
        a.a4 * a.b1 * a.c2 * a.d3 + a.a4 * a.b1 * a.c3 * a.d2 - a.a4 * a.b2 * a.c3 * a.d1 +
        a.a4 * a.b2 * a.c1 * a.d3 - a.a4 * a.b3 * a.c1 * a.d2 + a.a4 * a.b3 * a.c2 * a.d1;
    return result;
}

inline Mat4 Inverse(const Mat4 &a)
{
    Mat4 res   = {};
    f32 invdet = Determinant(a);
    if (invdet == 0)
    {
        return res;
    }
    invdet = 1.f / invdet;
    // NOTE: transpose
    res.a1 =
        invdet * (a.b2 * (a.c3 * a.d4 - a.d3 * a.c4) - a.c2 * (a.b3 * a.d4 - a.d3 * a.b4) +
                  a.d2 * (a.b3 * a.c4 - a.c3 * a.b4));
    res.a2 =
        -invdet * (a.a2 * (a.c3 * a.d4 - a.d3 * a.c4) - a.c2 * (a.a3 * a.d4 - a.d3 * a.a4) +
                   a.d2 * (a.a3 * a.c4 - a.c3 * a.a4));
    res.a3 =
        invdet * (a.a2 * (a.b3 * a.d4 - a.d3 * a.b4) - a.b2 * (a.a3 * a.d4 - a.d3 * a.a4) +
                  a.d2 * (a.a3 * a.b4 - a.b3 * a.a4));
    res.a4 =
        -invdet * (a.a2 * (a.b3 * a.c4 - a.b4 * a.c3) - a.b2 * (a.a3 * a.c4 - a.c3 * a.a4) +
                   a.c2 * (a.a3 * a.b4 - a.b3 * a.a4));
    res.b1 =
        -invdet * (a.b1 * (a.c3 * a.d4 - a.d3 * a.c4) - a.c1 * (a.b3 * a.d4 - a.d3 * a.b4) +
                   a.d1 * (a.b3 * a.c4 - a.c3 * a.b4));
    res.b2 =
        invdet * (a.a1 * (a.c3 * a.d4 - a.d3 * a.c4) - a.c1 * (a.a3 * a.d4 - a.d3 * a.a4) +
                  a.d1 * (a.a3 * a.c4 - a.c3 * a.a4));
    res.b3 =
        -invdet * (a.a1 * (a.b3 * a.d4 - a.d3 * a.b4) - a.b1 * (a.a3 * a.d4 - a.d3 * a.a4) +
                   a.d1 * (a.a3 * a.b4 - a.b3 * a.a4));
    res.b4 =
        invdet * (a.a1 * (a.b3 * a.c4 - a.b4 * a.c3) - a.b1 * (a.a3 * a.c4 - a.c3 * a.a4) +
                  a.c1 * (a.a3 * a.b4 - a.b3 * a.a4));
    res.c1 =
        invdet * (a.b1 * (a.c2 * a.d4 - a.d2 * a.c4) - a.c1 * (a.b2 * a.d4 - a.d2 * a.b4) +
                  a.d1 * (a.b2 * a.c4 - a.c2 * a.b4));
    res.c2 =
        -invdet * (a.a1 * (a.c2 * a.d4 - a.d2 * a.c4) - a.c1 * (a.a2 * a.d4 - a.d2 * a.a4) +
                   a.d1 * (a.a2 * a.c4 - a.c2 * a.a4));
    res.c3 =
        invdet * (a.a1 * (a.b2 * a.d4 - a.d2 * a.b4) - a.b1 * (a.a2 * a.d4 - a.d2 * a.a4) +
                  a.d1 * (a.a2 * a.b4 - a.b2 * a.a4));
    res.c4 =
        -invdet * (a.a1 * (a.b2 * a.c4 - a.c2 * a.b4) - a.b1 * (a.a2 * a.c4 - a.c2 * a.a4) +
                   a.c1 * (a.a2 * a.b4 - a.b2 * a.a4));
    res.d1 =
        -invdet * (a.b1 * (a.c2 * a.d3 - a.d2 * a.c3) - a.c1 * (a.b2 * a.d3 - a.d2 * a.b3) +
                   a.d1 * (a.b2 * a.c3 - a.c2 * a.b3));
    res.d2 =
        invdet * (a.a1 * (a.c2 * a.d3 - a.c3 * a.d2) - a.c1 * (a.a2 * a.d3 - a.d2 * a.a3) +
                  a.d1 * (a.a2 * a.c3 - a.c2 * a.a3));
    res.d3 =
        -invdet * (a.a1 * (a.b2 * a.d3 - a.d2 * a.b3) - a.b1 * (a.a2 * a.d3 - a.d2 * a.a3) +
                   a.d1 * (a.a2 * a.b3 - a.b2 * a.a3));
    res.d4 =
        invdet * (a.a1 * (a.b2 * a.c3 - a.c2 * a.b3) - a.b1 * (a.a2 * a.c3 - a.c2 * a.a3) +
                  a.c1 * (a.a2 * a.b3 - a.b2 * a.a3));

    return res;
}

inline bool Inverse(const Mat4 &a, Mat4 &res)
{
    f32 invdet = Determinant(a);
    if (invdet == 0) return false;
    invdet = 1.f / invdet;
    // NOTE: transpose
    res.a1 =
        invdet * (a.b2 * (a.c3 * a.d4 - a.d3 * a.c4) - a.c2 * (a.b3 * a.d4 - a.d3 * a.b4) +
                  a.d2 * (a.b3 * a.c4 - a.c3 * a.b4));
    res.a2 =
        -invdet * (a.a2 * (a.c3 * a.d4 - a.d3 * a.c4) - a.c2 * (a.a3 * a.d4 - a.d3 * a.a4) +
                   a.d2 * (a.a3 * a.c4 - a.c3 * a.a4));
    res.a3 =
        invdet * (a.a2 * (a.b3 * a.d4 - a.d3 * a.b4) - a.b2 * (a.a3 * a.d4 - a.d3 * a.a4) +
                  a.d2 * (a.a3 * a.b4 - a.b3 * a.a4));
    res.a4 =
        -invdet * (a.a2 * (a.b3 * a.c4 - a.b4 * a.c3) - a.b2 * (a.a3 * a.c4 - a.c3 * a.a4) +
                   a.c2 * (a.a3 * a.b4 - a.b3 * a.a4));
    res.b1 =
        -invdet * (a.b1 * (a.c3 * a.d4 - a.d3 * a.c4) - a.c1 * (a.b3 * a.d4 - a.d3 * a.b4) +
                   a.d1 * (a.b3 * a.c4 - a.c3 * a.b4));
    res.b2 =
        invdet * (a.a1 * (a.c3 * a.d4 - a.d3 * a.c4) - a.c1 * (a.a3 * a.d4 - a.d3 * a.a4) +
                  a.d1 * (a.a3 * a.c4 - a.c3 * a.a4));
    res.b3 =
        -invdet * (a.a1 * (a.b3 * a.d4 - a.d3 * a.b4) - a.b1 * (a.a3 * a.d4 - a.d3 * a.a4) +
                   a.d1 * (a.a3 * a.b4 - a.b3 * a.a4));
    res.b4 =
        invdet * (a.a1 * (a.b3 * a.c4 - a.b4 * a.c3) - a.b1 * (a.a3 * a.c4 - a.c3 * a.a4) +
                  a.c1 * (a.a3 * a.b4 - a.b3 * a.a4));
    res.c1 =
        invdet * (a.b1 * (a.c2 * a.d4 - a.d2 * a.c4) - a.c1 * (a.b2 * a.d4 - a.d2 * a.b4) +
                  a.d1 * (a.b2 * a.c4 - a.c2 * a.b4));
    res.c2 =
        -invdet * (a.a1 * (a.c2 * a.d4 - a.d2 * a.c4) - a.c1 * (a.a2 * a.d4 - a.d2 * a.a4) +
                   a.d1 * (a.a2 * a.c4 - a.c2 * a.a4));
    res.c3 =
        invdet * (a.a1 * (a.b2 * a.d4 - a.d2 * a.b4) - a.b1 * (a.a2 * a.d4 - a.d2 * a.a4) +
                  a.d1 * (a.a2 * a.b4 - a.b2 * a.a4));
    res.c4 =
        -invdet * (a.a1 * (a.b2 * a.c4 - a.c2 * a.b4) - a.b1 * (a.a2 * a.c4 - a.c2 * a.a4) +
                   a.c1 * (a.a2 * a.b4 - a.b2 * a.a4));
    res.d1 =
        -invdet * (a.b1 * (a.c2 * a.d3 - a.d2 * a.c3) - a.c1 * (a.b2 * a.d3 - a.d2 * a.b3) +
                   a.d1 * (a.b2 * a.c3 - a.c2 * a.b3));
    res.d2 =
        invdet * (a.a1 * (a.c2 * a.d3 - a.c3 * a.d2) - a.c1 * (a.a2 * a.d3 - a.d2 * a.a3) +
                  a.d1 * (a.a2 * a.c3 - a.c2 * a.a3));
    res.d3 =
        -invdet * (a.a1 * (a.b2 * a.d3 - a.d2 * a.b3) - a.b1 * (a.a2 * a.d3 - a.d2 * a.a3) +
                   a.d1 * (a.a2 * a.b3 - a.b2 * a.a3));
    res.d4 =
        invdet * (a.a1 * (a.b2 * a.c3 - a.c2 * a.b3) - a.b1 * (a.a2 * a.c3 - a.c2 * a.a3) +
                  a.c1 * (a.a2 * a.b3 - a.b2 * a.a3));

    return true;
}

template <typename T>
struct LinearSpace
{
    union
    {
        struct
        {
            Vec3<T> x, y, z;
        };
        Vec3<T> e[3];
    };
    LinearSpace() {}
    __forceinline LinearSpace(ZeroTy) : x(zero), y(zero), z(zero) {}
    __forceinline LinearSpace(const Vec3<T> &a, const Vec3<T> &b, const Vec3<T> &c)
        : x(a), y(b), z(c)
    {
    }
    __forceinline LinearSpace(const LinearSpace &space) : x(space.x), y(space.y), z(space.z) {}
    __forceinline LinearSpace &operator=(const LinearSpace &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    static LinearSpace<T> FromXZ(const Vec3<T> &x, const Vec3<T> &z)
    {
        return LinearSpace<T>(x, Cross(z, x), z);
    }
    static LinearSpace<T> FromXY(const Vec3<T> &x, const Vec3<T> &y)
    {
        return LinearSpace<T>(x, y, Cross(x, y));
    }

    Vec3<T> ToLocal(const Vec3<T> &a) const
    {
        return Vec3<T>(Dot(x, a), Dot(y, a), Dot(z, a));
    }

    Vec3<T> FromLocal(const Vec3<T> &a) const
    {
        return FMA(Vec3<T>(a.x), x, FMA(Vec3<T>(a.y), y, Vec3<T>(a.z) * z));
    }
};

template <typename T>
LinearSpace<T> operator/(const LinearSpace<T> &l, f32 d)
{
    return LinearSpace<T>(l.x / d, l.y / d, l.z / d);
}

template <typename T>
LinearSpace<T> &operator/=(LinearSpace<T> &l, f32 d)
{
    l = l / d;
    return l;
}

template <typename T>
LinearSpace<T> Transpose(const LinearSpace<T> &a)
{
    return LinearSpace<T>(Vec3<T>(a.x[0], a.y[0], a.z[0]), Vec3<T>(a.x[1], a.y[1], a.z[1]),
                          Vec3<T>(a.x[2], a.y[2], a.z[2]));
}

template <typename T>
__forceinline f32 Determinant(const LinearSpace<T> &t)
{
    return Dot(t.x, Cross(t.y, t.z));
}

template <typename T>
__forceinline Vec3f operator*(const LinearSpace<T> &m, const Vec3f &v)
{
    return FMA(Vec3<T>(v.x), m.x, FMA(Vec3<T>(v.y), m.y, Vec3<T>(v.z) * m.z));
}

typedef LinearSpace<f32> LinearSpace3f;
typedef LinearSpace<LaneNF32> LinearSpace3fn;

// NOTE: row major affine transformation matrix
struct AffineSpace
{
    union
    {
        struct
        {
            Vec3f c0;
            Vec3f c1;
            Vec3f c2;
            Vec3f c3;
        };
        Vec3f e[4];
    };

    AffineSpace() {}
    __forceinline AffineSpace(const AffineSpace &other)
        : c0(other.c0), c1(other.c1), c2(other.c2), c3(other.c3)
    {
    }
    __forceinline AffineSpace &operator=(const AffineSpace &other)
    {
        c0 = other.c0;
        c1 = other.c1;
        c2 = other.c2;
        c3 = other.c3;
        return *this;
    }
    __forceinline AffineSpace(const Mat4 &other)
        : c0(Vec3f(other.elements[0][0], other.elements[0][1], other.elements[0][2])),
          c1(Vec3f(other.elements[1][0], other.elements[1][1], other.elements[1][2])),
          c2(Vec3f(other.elements[2][0], other.elements[2][1], other.elements[2][2])),
          c3(Vec3f(other.elements[3][0], other.elements[3][1], other.elements[3][2]))
    {
    }

    AffineSpace(ZeroTy) : c0(0), c1(0), c2(0), c3(0) {}
    AffineSpace(const Vec3f &a, const Vec3f &b, const Vec3f &c) : c0(a), c1(b), c2(c), c3(0) {}
    AffineSpace(const Vec3f &a, const Vec3f &b, const Vec3f &c, const Vec3f &d)
        : c0(a), c1(b), c2(c), c3(d)
    {
    }
    AffineSpace(f32 a, f32 b, f32 c, f32 d, f32 e, f32 f, f32 g, f32 h, f32 i, f32 j, f32 k,
                f32 l)
        : c0(a, e, i), c1(b, f, j), c2(c, g, k), c3(d, h, l)
    {
    }
    AffineSpace(const LinearSpace3f &l, const Vec3f &t) : c0(l.x), c1(l.y), c2(l.z), c3(t) {}

    Vec3f &operator[](u32 i)
    {
        Assert(i < 4);
        return e[i];
    }
    const Vec3f &operator[](u32 i) const
    {
        Assert(i < 4);
        return e[i];
    }
    static AffineSpace Identity()
    {
        AffineSpace result = AffineSpace(zero);
        result[0][0]       = 1;
        result[1][1]       = 1;
        result[2][2]       = 1;
        return result;
    }
    static AffineSpace LookAt(Vec3f eye, Vec3f center, Vec3f up)
    {
        AffineSpace result;

        Vec3f f = Normalize(eye - center);
        Vec3f s = Normalize(Cross(Normalize(up), f));
        Vec3f u = Cross(f, s);

        result.e[0][0] = s.x;
        result.e[0][1] = u.x;
        result.e[0][2] = f.x;
        result.e[1][0] = s.y;
        result.e[1][1] = u.y;
        result.e[1][2] = f.y;
        result.e[2][0] = s.z;
        result.e[2][1] = u.z;
        result.e[2][2] = f.z;
        result.e[3][0] = -Dot(s, eye);
        result.e[3][1] = -Dot(u, eye);
        result.e[3][2] = -Dot(f, eye);

        return result;
    }
    static AffineSpace Scale(f32 a) { return AffineSpace(a, 0, 0, 0, 0, a, 0, 0, 0, 0, a, 0); }
    static AffineSpace Scale(const Vec3f &scale)
    {
        return AffineSpace(scale.x, 0, 0, 0, 0, scale.y, 0, 0, 0, 0, scale.z, 0);
    }
    static AffineSpace Scale(f32 a, f32 b, f32 c)
    {
        return AffineSpace(a, 0, 0, 0, 0, b, 0, 0, 0, 0, c, 0);
    }
    static AffineSpace Rotate(const Vec3f &axis, f32 theta)
    {
        const Vec3f a = Normalize(axis);
        f32 s         = Sin(theta);
        f32 c         = Cos(theta);
        f32 mc        = 1.f - c;

        return AffineSpace(
            a.x * a.x * mc + c, a.y * a.x * mc - a.z * s, a.z * a.x * mc + a.y * s, 0.f,
            a.x * a.y * mc + a.z * s, a.y * a.y * mc + c, a.z * a.y * mc - a.x * s, 0.f,
            a.x * a.z * mc - a.y * s, a.y * a.z * mc + a.x * s, a.z * a.z * mc + c, 0.f);
    }
    static AffineSpace Translate(const Vec3f &v)
    {
        return AffineSpace(1, 0, 0, v.x, 0, 1, 0, v.y, 0, 0, 1, v.z);
    }
    static AffineSpace Translate(f32 a, f32 b, f32 c)
    {
        return AffineSpace(1, 0, 0, a, 0, 1, 0, b, 0, 0, 1, c);
    }
};

__forceinline Mat4::Mat4(const AffineSpace &s)
{
    columns_[0] = Vec4f(s[0], 0.f);
    columns_[1] = Vec4f(s[1], 0.f);
    columns_[2] = Vec4f(s[2], 0.f);
    columns_[3] = Vec4f(s[3], 1.f);
}

__forceinline Vec3f operator*(const AffineSpace &t, const Vec3f &v)
{
    return FMA(t.c0, Vec3f(v.x), FMA(t.c1, Vec3f(v.y), FMA(t.c2, Vec3f(v.z), t.c3)));
}

// __forceinline Lane4F32 operator*(const AffineSpace &t, const Lane4F32 &v)
// {
//     return FMA(Lane4F32::LoadU(&t.c0), v[0],
//                FMA(Lane4F32::LoadU(&t.c1), v[1],
//                    FMA(Lane4F32::LoadU(&t.c2), v[2], Lane4F32::LoadU(&t.c3))));
// }

// __forceinline Lane4F32 Transform(const AffineSpace &t, const Lane4F32 &v) { return t * v; }

__forceinline bool operator==(const AffineSpace &a, const AffineSpace &b)
{
    return a.c0 == b.c0 && a.c1 == b.c1 && a.c2 == b.c2 && a.c3 == b.c3;
}
__forceinline bool operator!=(const AffineSpace &a, const AffineSpace &b)
{
    return a.c0 != b.c0 || a.c1 != b.c1 || a.c2 != b.c2 || a.c3 != b.c3;
}

__forceinline Vec3f TransformV(const AffineSpace &t, const Vec3f &v)
{
    return FMA(t.c0, Vec3f(v[0]), FMA(t.c1, Vec3f(v[1]), t.c2 * Vec3f(v[2])));
}
__forceinline Vec3f TransformP(const AffineSpace &a, const Vec3f &b) { return a * b; }

__forceinline Vec3f TransformP(const AffineSpace &t, const Vec3f &v, Vec3f &err)
{
    err = (gamma(3) + 1) * FMA(Abs(t.c0), Vec3f(err.x),
                               FMA(Abs(t.c1), Vec3f(err.y), Abs(t.c2) * Vec3f(err.z))) +
          gamma(3) * (Abs(t.c0 * v.x) + Abs(t.c1 * v.y) + Abs(t.c2 * v.z) + Abs(t.c3));
    return TransformP(t, v);
}

__forceinline Bounds Transform(const AffineSpace &t, const Bounds &b)
{
    Vec3f corners[8] = {
        Vec3f(b.minP[0], b.minP[1], b.minP[2]), Vec3f(b.maxP[0], b.minP[1], b.minP[2]),
        Vec3f(b.maxP[0], b.maxP[1], b.minP[2]), Vec3f(b.minP[0], b.maxP[1], b.minP[2]),
        Vec3f(b.minP[0], b.minP[1], b.maxP[2]), Vec3f(b.maxP[0], b.minP[1], b.maxP[2]),
        Vec3f(b.maxP[0], b.maxP[1], b.maxP[2]), Vec3f(b.minP[0], b.maxP[1], b.maxP[2]),
    };
    Bounds bounds;
    for (u32 i = 0; i < 8; i++)
    {
        bounds.Extend(Lane4F32(TransformP(t, corners[i])));
    }
    return bounds;
}

__forceinline AffineSpace operator*(const AffineSpace &a, const AffineSpace &b)
{
    return AffineSpace(TransformV(a, b.c0), TransformV(a, b.c1), TransformV(a, b.c2),
                       TransformP(a, b.c3));
}

__forceinline f32 Determinant(const AffineSpace &a) { return Dot(a.c0, Cross(a.c1, a.c2)); }

__forceinline Vec3f ApplyInverse(const AffineSpace &t, const Vec3f &v)
{
    Mat3 inv = Transpose(Mat3(Cross(t.c1, t.c2), Cross(t.c2, t.c0), Cross(t.c0, t.c1)));
    f32 det  = Determinant(t);
    Assert(det != 0.f);
    return inv / det * (v - t.c3);
}

__forceinline Vec3f ApplyInverseV(const AffineSpace &t, const Vec3f &v)
{
    Mat3 inv = Transpose(Mat3(Cross(t.c1, t.c2), Cross(t.c2, t.c0), Cross(t.c0, t.c1)));
    f32 det  = Determinant(t);
    Assert(det != 0.f);
    return inv / det * v;
}

// takes a normalized vector
// TODO: maybe take a look at this https://graphics.pixar.com/library/OrthonormalB/paper.pdf
__forceinline AffineSpace Frame(const Vec3f &n)
{
    Vec3f t0(0, n.z, -n.y);
    Vec3f b0(-n.z, 0, n.x);
    Vec3f t = Normalize(Select(Dot(t0, t0) > Dot(b0, b0), t0, b0));
    Vec3f b = Cross(n, t);
    return AffineSpace(t, b, n, Vec3f(0.f));
}

__forceinline AffineSpace Inverse(const AffineSpace &a)
{
    LinearSpace3f result =
        Transpose(LinearSpace3f(Cross(a.c1, a.c2), Cross(a.c2, a.c0), Cross(a.c0, a.c1)));
    f32 det = Determinant(a);
    Assert(det != 0.f);
    result /= det;
    Vec3f translation = result * a.c3;
    return AffineSpace(result, -translation);
}

inline void ExtractPlanes(Vec4f *planes, const Mat4 &NDCFromRender)
{
    Vec4f row0 = NDCFromRender.GetRow(0);
    Vec4f row1 = NDCFromRender.GetRow(1);
    Vec4f row2 = NDCFromRender.GetRow(2);
    Vec4f row3 = NDCFromRender.GetRow(3);

    planes[0] = row3 + row0;
    planes[1] = row3 - row0;
    planes[2] = row3 + row1;
    planes[3] = row3 - row1;
    planes[4] = row3 + row2;
    planes[5] = row3 - row2;

    for (u32 i = 0; i < 6; i++)
    {
        f32 oneOverLength = Length(planes[i].xyz);
        Assert(oneOverLength != 0);
        oneOverLength = 1 / (oneOverLength);
        planes[i] *= oneOverLength;
    }
}

template <i32 N>
int IntersectFrustumAABB(const Vec4f *planes, Bounds *bounds, int numPlanes = 6)
{
    LaneF32<N> minX, minY, minZ, maxX, maxY, maxZ;
    if constexpr (N == 4)
    {
        Transpose4x3(bounds[0].minP, bounds[1].minP, bounds[2].minP, bounds[3].minP, minX,
                     minY, minZ);
        Transpose4x3(bounds[0].maxP, bounds[1].maxP, bounds[2].maxP, bounds[3].maxP, maxX,
                     maxY, maxZ);
    }
    else if constexpr (N == 8)
    {
        Transpose8x3(bounds[0].minP, bounds[1].minP, bounds[2].minP, bounds[3].minP,
                     bounds[4].minP, bounds[5].minP, bounds[6].minP, bounds[7].minP, minX,
                     minY, minZ);
        Transpose8x3(bounds[0].maxP, bounds[1].maxP, bounds[2].maxP, bounds[3].maxP,
                     bounds[4].maxP, bounds[5].maxP, bounds[6].maxP, bounds[7].maxP, maxX,
                     maxY, maxZ);
    }
    else if constexpr (N == 1)
    {
        minX = bounds[0].minP[0];
        minY = bounds[0].minP[1];
        minZ = bounds[0].minP[2];

        maxX = bounds[0].maxP[0];
        maxY = bounds[0].maxP[1];
        maxZ = bounds[0].maxP[2];
    }
    else
    {
        Assert(0);
    }

    // NOTE: need to compare against - plane.w * 2
    LaneF32<N> centerY = maxY + minY;
    LaneF32<N> centerX = maxX + minX;
    LaneF32<N> centerZ = maxZ + minZ;

    LaneF32<N> extentX = maxX - minX;
    LaneF32<N> extentY = maxY - minY;
    LaneF32<N> extentZ = maxZ - minZ;

    LaneF32<N> signMask(-0.f);

    Mask<LaneF32<N>> results = Mask<LaneF32<N>>(true);
    Assert(All(results));

    for (int i = 0; i < numPlanes; i++)
    {
        LaneF32<N> planeX(planes[i].x);
        LaneF32<N> planeY(planes[i].y);
        LaneF32<N> planeZ(planes[i].z);
        LaneF32<N> planeW(planes[i].w * -2);

        // dot(center + extent, plane) > -plane.w
        // NOTE: to see if the box is partially inside, need to maximize the dot product.
        // dot(center + extent, plane) = dot(center, plane) + dot(extent, plane) <-- this needs
        // to be maximized. this can be done by xor the extent with the sign bit of the plane,
        // so the dot product is always +

        LaneF32<N> test = planeX + signMask;
        LaneF32<N> dot, t;

        if constexpr (N == 1)
        {
            dot = planeX * (centerX + Copysign(extentX, planeX));
            dot += planeY * (centerY + Copysign(extentY, planeY));
            dot += planeZ * (centerZ + Copysign(extentZ, planeZ));
            results = results && (dot > planeW);
        }
        else
        {
            t       = centerX + (extentX ^ (planeX & signMask));
            dot     = t * planeX;
            t       = centerY + (extentY ^ (planeY & signMask));
            dot     = FMA(t, planeY, dot);
            t       = centerZ + (extentZ ^ (planeZ & signMask));
            dot     = FMA(t, planeZ, dot);
            results = results & (dot > planeW);
        }

        if (None(results)) return 0;
    }
    return Movemask(results);
}

} // namespace rt
#endif
