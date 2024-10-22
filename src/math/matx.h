#ifndef MATX_H
#define MATX_H
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

    Mat3(f32 a, f32 b, f32 c)
        : a1(a), a2(0), a3(0), b1(0), b2(b), b3(0), c1(0), c2(0), c3(c) {}

    Mat3(const Mat3 &other)
    {
        columns[0] = other.columns[0];
        columns[1] = other.columns[1];
        columns[2] = other.columns[2];
    }

    Mat3(Vec3f v) : Mat3(v.x, v.y, v.z) {}

    Vec3f &operator[](const i32 index)
    {
        return columns[index];
    }

    Mat3 &operator/(const f32 f)
    {
        columns[0] /= f;
        columns[1] /= f;
        columns[2] /= f;
        return *this;
    }

    Mat3(Vec3f c1, Vec3f c2, Vec3f c3)
    {
        columns[0] = c1;
        columns[1] = c2;
        columns[2] = c3;
    }
    Mat3(f32 a1, f32 a2, f32 a3, f32 b1, f32 b2, f32 b3, f32 c1, f32 c2, f32 c3)
        : Mat3(Vec3f(a1, a2, a3), Vec3f(b1, b2, b3), Vec3f(c1, c2, c3)) {}

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

inline Mat3 operator*(Mat3 a, Mat3 b)
{
    Mat3 result;
    for (int j = 0; j < 3; j += 1)
    {
        for (int i = 0; i < 3; i += 1)
        {
            result.elements[i][j] = (a.elements[0][j] * b.elements[i][0] + a.elements[1][j] * b.elements[i][1] +
                                     a.elements[2][j] * b.elements[i][2]);
        }
    }

    return result;
}

//////////////////////////////
// Mat4
//
struct Mat4
{
    union
    {
        f32 elements[4][4];
        Lane4F32 columns[4];
        struct
        {
            f32 a1, a2, a3, a4;
            f32 b1, b2, b3, b4;
            f32 c1, c2, c3, c4;
            f32 d1, d2, d3, d4;
        };
    };

    __forceinline Mat4() : a1(0), a2(0), a3(0), a4(0), b1(0), b2(0), b3(0), b4(0),
                           c1(0), c2(0), c3(0), c4(0), d1(0), d2(0), d3(0), d4(0)
    {
    }

    __forceinline Mat4(f32 a1, f32 a2, f32 a3, f32 a4,
                       f32 b1, f32 b2, f32 b3, f32 b4,
                       f32 c1, f32 c2, f32 c3, f32 c4,
                       f32 d1, f32 d2, f32 d3, f32 d4)
        : a1(a1), a2(a2), a3(a3), a4(a4), b1(b1), b2(b2), b3(b3), b4(b4),
          c1(c1), c2(c2), c3(c3), c4(c4), d1(d1), d2(d2), d3(d3), d4(d4)
    {
    }

    Mat4(f32 val) : a1(val), a2(0), a3(0), a4(0),
                    b1(0), b2(val), b3(0), b4(0),
                    c1(0), c2(0), c3(val), c4(0),
                    d1(0), d2(0), d3(0), d4(val)
    {
    }

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

    Lane4F32 &operator[](const i32 index)
    {
        return columns[index];
    }

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
};

Vec4f Mul(Mat4 a, Vec4f b)
{
    Vec4f result;
#ifdef __SSE2__
    Lane4F32 laneResult = a.columns[0] * b.x + a.columns[1] * b.y + a.columns[2] * b.z + a.columns[3] * b.w;
    Lane4F32::Store(&result, laneResult);
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
    result.w += a.columns[3][2] * b.w;
    return result;
#endif
}

Vec3f Mul(Mat4 a, Vec3f b)
{
    Vec4f vec(b.x, b.y, b.z, 1);
    return Mul(a, vec).xyz;
}

// Ignores translation
inline Vec3f NormalTransform(const Mat4 &a, const Vec3f &b)
{
    Lane4F32 laneResult = a.columns[0] * b.x + a.columns[1] * b.y + a.columns[2] * b.z;
    return Vec3f(laneResult[0], laneResult[1], laneResult[2]);
}

// inline Mat4 LookAt(Vec3f eye, Vec3f center, Vec3f up)
// {
//     Mat4 result;
//
//     Vec3f f = Normalize(eye - center);
//     Vec3f s = Normalize(Cross(up, f));
//     Vec3f u = Cross(f, s);
//
//     result.elements[0][0] = s.x;
//     result.elements[0][1] = u.x;
//     result.elements[0][2] = f.x;
//     result.elements[0][3] = 0.0f;
//     result.elements[1][0] = s.y;
//     result.elements[1][1] = u.y;
//     result.elements[1][2] = f.y;
//     result.elements[1][3] = 0.0f;
//     result.elements[2][0] = s.z;
//     result.elements[2][1] = u.z;
//     result.elements[2][2] = f.z;
//     result.elements[2][3] = 0.0f;
//     result.elements[3][0] = -Dot(s, eye);
//     result.elements[3][1] = -Dot(u, eye);
//     result.elements[3][2] = -Dot(f, eye);
//     result.elements[3][3] = 1.0f;
//
//     return result;
// }

inline Mat4 Mul(Mat4 a, Mat4 b)
{
    Mat4 result;
    for (int j = 0; j < 4; j += 1)
    {
        for (int i = 0; i < 4; i += 1)
        {
            result.elements[i][j] = (a.elements[0][j] * b.elements[i][0] + a.elements[1][j] * b.elements[i][1] +
                                     a.elements[2][j] * b.elements[i][2] + a.elements[3][j] * b.elements[i][3]);
        }
    }

    return result;
}

inline Mat3 Mul(Mat3 a, Mat3 b)
{
    Mat3 result;
    for (int j = 0; j < 3; j += 1)
    {
        for (int i = 0; i < 3; i += 1)
        {
            result.elements[i][j] = (a.elements[0][j] * b.elements[i][0] + a.elements[1][j] * b.elements[i][1] +
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
    __forceinline AffineSpace(const AffineSpace &other) : c0(other.c0), c1(other.c1), c2(other.c2), c3(other.c3) {}
    __forceinline AffineSpace &operator=(const AffineSpace &other)
    {
        c0 = other.c0;
        c1 = other.c1;
        c2 = other.c2;
        c3 = other.c3;
        return *this;
    }

    AffineSpace(ZeroTy) : c0(0), c1(0), c2(0), c3(0) {}
    AffineSpace(const Vec3f &a, const Vec3f &b, const Vec3f &c, const Vec3f &d)
        : c0(a), c1(b), c2(c), c3(d) {}
    AffineSpace(f32 a, f32 b, f32 c, f32 d,
                f32 e, f32 f, f32 g, f32 h,
                f32 i, f32 j, f32 k, f32 l)
        : c0(a, e, i), c1(b, f, j), c2(c, g, k), c3(d, h, l) {}

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
        Vec3f s = Normalize(Cross(up, f));
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
    static AffineSpace Scale(const Vec3f &scale)
    {
        return AffineSpace(scale.x, 0, 0, 0,
                           0, scale.y, 0, 0,
                           0, 0, scale.z, 0);
    }
    static AffineSpace Rotate(const Vec3f &axis, f32 theta)
    {
        const Vec3f a = Normalize(axis);
        f32 s         = Sin(theta);
        f32 c         = Cos(theta);
        f32 mc        = 1.f - c;

        return AffineSpace(a.x * a.x * mc + c, a.y * a.x * mc - a.z * s, a.z * a.x * mc + a.y * s, 0.f,
                           a.x * a.y * mc + a.z * s, a.y * a.y * mc + c, a.z * a.y * mc - a.x * s, 0.f,
                           a.x * a.z * mc - a.y * s, a.y * a.z * mc + a.x * s, a.z * a.z * mc + c, 0.f);
    }
    static AffineSpace Translate(const Vec3f &v)
    {
        return AffineSpace(0, 0, 0, v.x,
                           0, 0, 0, v.y,
                           0, 0, 0, v.z);
    }
};

__forceinline Vec3f operator*(const AffineSpace &t, const Vec3f &v)
{
    return FMA(t.c0, Vec3f(v.x), FMA(t.c1, Vec3f(v.y), t.c2 * Vec3f(v.z) + t.c3));
}

// NOTE: can only be used when there is at least 4 bytes of padding after AffineSpace
__forceinline Lane4F32 operator*(const AffineSpace &t, const Lane4F32 &v)
{
    return FMA(Lane4F32::LoadU(&t.c0), v[0],
               FMA(Lane4F32::LoadU(&t.c1), v[1],
                   Lane4F32::LoadU(&t.c2) * v[2] +
                       Lane4F32::LoadU(&t.c3)));
}

__forceinline AffineSpace operator*(const AffineSpace &a, const AffineSpace &b)
{
    return AffineSpace(a * b.c0, a * b.c1, a * b.c2, a * b.c3);
}

__forceinline bool operator==(const AffineSpace &a, const AffineSpace &b)
{
    return a.c0 == b.c0 && a.c1 == b.c1 && a.c2 == b.c2 && a.c3 == b.c3;
}

} // namespace rt
#endif
