#ifndef MATH_H
#define MATH_H

#include <cmath>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

using std::sqrt;
const f32 infinity = std::numeric_limits<f32>::infinity();

#define PI      3.14159265358979323846f
#define InvPi   0.31830988618379067154f
#define Inv2Pi  0.15915494309189533577f
#define Inv4Pi  0.07957747154594766788f
#define PiOver2 1.57079632679489661923f
#define PiOver4 0.78539816339744830961f
#define Sqrt2   1.41421356237309504880f
#define U32Max  0xffffffff

static const __m128 SIMDInfinity = _mm_set1_ps(infinity);

f32 IsInf(f32 x)
{
    return std::isinf(x);
}

f32 IsNaN(f32 x)
{
    return std::isnan(x);
}

f64 IsNaN(f64 x)
{
    return std::isnan(x);
}

f32 Sqr(f32 x)
{
    return x * x;
}

f64 Sqrt(f64 x)
{
    return x * x;
}

template <typename T>
T Lerp(f32 t, T a, T b)
{
    return (1 - t) * a + t * b;
}

template <typename T>
T Clamp(T min, T max, T x)
{
    return x < min ? min : (x > max ? max : x);
}

template <typename T>
T Min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
T Max(T a, T b)
{
    return a > b ? a : b;
}

inline int Log2Int(u64 v)
{
#if _WIN32
    unsigned long lz = 0;
    BitScanReverse64(&lz, v);
    return lz;
#else
#error
#endif
}

f32 SafeSqrt(f32 x)
{
    return std::sqrt(Max(0.f, x));
}

template <int n>
constexpr f32 Pow(f32 v)
{
    if constexpr (n < 0) return 1 / Pow<-n>(v);
    f32 n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

template <>
constexpr f32 Pow<0>(f32 v) { return 1; }

template <>
constexpr f32 Pow<1>(f32 v) { return v; }

f32 FMA(f32 a, f32 b, f32 c)
{
    return std::fma(a, b, c);
}

template <typename f32, typename C>
constexpr f32 EvaluatePolynomial(f32 t, C c) { return c; }

template <typename f32, typename C, typename... Args>
constexpr f32 EvaluatePolynomial(f32 t, C c, Args... cRemaining)
{
    return FMA(t, EvaluatePolynomial(t, cRemaining...), c);
}

inline f32 BitsToFloat(u32 src)
{
    f32 dst;
    std::memcpy(&dst, &src, sizeof(dst));
    return dst;
}

inline u32 FloatToBits(f32 src)
{
    u32 dst;
    std::memcpy(&dst, &src, sizeof(dst));
    return dst;
}

inline i32 Exponent(f32 v) { return (FloatToBits(v) >> 23) - 127; }

f32 FastExp(f32 x)
{
    f32 xp  = x * 1.442695041f;
    f32 fxp = std::floor(xp), f = xp - fxp;
    i32 i = (i32)fxp;

    f32 twoToF = EvaluatePolynomial(f, 1.f, 0.695556856f,
                                    0.226173572f, 0.0781455737f);

    i32 exponent = Exponent(twoToF) + i;
    if (exponent < -126) return 0;
    if (exponent > 127) return infinity;
    u32 bits = FloatToBits(twoToF);
    bits &= 0b10000000011111111111111111111111u;
    bits |= (exponent + 127) << 23;
    return BitsToFloat(bits);
}

inline constexpr i32 NextPowerOfTwo(i32 v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

inline u16 SafeTruncateU32(u32 val)
{
    u16 result = (u16)val;
    Assert(val == result);
    return result;
}

inline u32 ReverseBits32(u32 n)
{
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

union vec2
{
    f32 e[2];
    struct
    {
        f32 x, y;
    };
    struct
    {
        f32 r, g;
    };
    vec2() : e{0, 0} {}
    vec2(f32 e0, f32 e1) : e{e0, e1} {}
    vec2 operator-() const { return vec2(-e[0], -e[1]); }
    f32 operator[](int i) const { return e[i]; }
    f32 &operator[](int i) { return e[i]; }

    vec2 &operator+=(const vec2 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        return *this;
    }
    vec2 &operator-=(const vec2 &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        return *this;
    }
    vec2 &operator*=(f32 t)
    {
        e[0] *= t;
        e[1] *= t;
        return *this;
    }

    vec2 &operator/=(f32 t)
    {
        return *this *= 1 / t;
    }

    f32 length() const
    {
        return sqrt(lengthSquared());
    }

    f32 lengthSquared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

union vec2i
{
    i32 e[2];
    struct
    {
        i32 x, y;
    };
    struct
    {
        i32 r, g;
    };
    vec2i() : e{0, 0} {}
    vec2i(i32 e0, i32 e1) : e{e0, e1} {}
    vec2i operator-() const { return vec2i(-e[0], -e[1]); }
    i32 operator[](int i) const { return e[i]; }
    i32 &operator[](int i) { return e[i]; }

    vec2i &operator+=(const vec2i &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        return *this;
    }
    vec2i &operator-=(const vec2i &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        return *this;
    }
    vec2i &operator*=(i32 t)
    {
        e[0] *= t;
        e[1] *= t;
        return *this;
    }
    vec2i &operator/=(i32 t)
    {
        return *this *= 1 / t;
    }
};

union vec3
{
    f32 e[3];
    struct
    {
        f32 x, y, z;
    };
    struct
    {
        f32 r, g, b;
    };
    struct
    {
        vec2 xy;
        f32 z;
    };
    struct
    {
        f32 x;
        vec2 yz;
    };

    vec3() : e{0, 0, 0} {}
    vec3(f32 e0, f32 e1, f32 e2) : e{e0, e1, e2} {}
    vec3(vec2 a, f32 b) : xy(a), z(b) {}

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    f32 operator[](int i) const { return e[i]; }
    f32 &operator[](int i) { return e[i]; }
    vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
    vec3 &operator-=(const vec3 &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }
    vec3 &operator*=(f32 t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }
    vec3 &operator*=(vec3 t)
    {
        e[0] *= t.x;
        e[1] *= t.y;
        e[2] *= t.z;
        return *this;
    }

    vec3 &operator/=(f32 t)
    {
        return *this *= 1 / t;
    }

    f32 length() const
    {
        return sqrt(lengthSquared());
    }

    f32 lengthSquared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

union vec4
{
    f32 e[4];
    struct
    {
        f32 x, y, z, w;
    };
    struct
    {
        f32 r, g, b, a;
    };

    struct
    {
        vec3 xyz;
        f32 w;
    };

    vec4() : e{0, 0, 0, 0} {}
    vec4(f32 e0, f32 e1, f32 e2, f32 e3) : e{e0, e1, e2, e3} {}

    vec4 operator-() const { return vec4(-e[0], -e[1], -e[2], -e[3]); }
    f32 operator[](int i) const { return e[i]; }
    f32 &operator[](int i) { return e[i]; }
    vec4 &operator+=(const vec4 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        e[3] += v.e[3];
        return *this;
    }
    vec4 &operator-=(const vec4 &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        e[3] -= v.e[3];
        return *this;
    }
    vec4 &operator*=(f32 t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        e[3] *= t;
        return *this;
    }

    vec4 &operator/=(f32 t)
    {
        return *this *= 1 / t;
    }

    f32 length() const
    {
        return sqrt(lengthSquared());
    }

    f32 lengthSquared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3];
    }
};

//////////////////////////////
// Vec2
//
inline vec2 operator+(const vec2 &u, const vec2 &v)
{
    return vec2(u[0] + v[0], u[1] + v[1]);
}

inline vec2 operator-(const vec2 &u, const vec2 &v)
{
    return vec2(u[0] - v[0], u[1] - v[1]);
}

inline vec2 operator*(const vec2 &u, const vec2 &v)
{
    return vec2(u[0] * v[0], u[1] * v[1]);
}

inline vec2 operator*(const vec2 &u, f32 d)
{
    return vec2(u[0] * d, u[1] * d);
}

inline vec2 operator*(f32 d, const vec2 &v)
{
    return v * d;
}

inline vec2 operator/(const vec2 &v, f32 d)
{
    return (1 / d) * v;
}

inline f32 Dot(const vec2 &u, const vec2 &v)
{
    return u[0] * v[0] + u[1] * v[1];
}

inline vec2 Normalize(const vec2 &v)
{
    return v / v.length();
}

inline bool NearZero(const vec2 &v)
{
    f32 s = 1e-8f;
    return ((std::fabs(v.x) < s) && (std::fabs(v.y) < s));
}

inline vec2 Min(const vec2 &a, const vec2 &b)
{
    vec2 result;
    result.x = a.x < b.x ? a.x : b.x;
    result.y = a.y < b.y ? a.y : b.y;
    return result;
}

inline vec2 Max(const vec2 &a, const vec2 &b)
{
    vec2 result;
    result.x = a.x > b.x ? a.x : b.x;
    result.y = a.y > b.y ? a.y : b.y;
    return result;
}

//////////////////////////////
// Vec2i
//
inline vec2i operator+(const vec2i &u, const vec2i &v)
{
    return vec2i(u[0] + v[0], u[1] + v[1]);
}

inline vec2i operator-(const vec2i &u, const vec2i &v)
{
    return vec2i(u[0] - v[0], u[1] - v[1]);
}

inline vec2i operator*(const vec2i &u, const vec2i &v)
{
    return vec2i(u[0] * v[0], u[1] * v[1]);
}

inline vec2i operator*(const vec2i &u, i32 d)
{
    return vec2i(u[0] * d, u[1] * d);
}

inline vec2i operator*(i32 d, const vec2i &v)
{
    return v * d;
}

inline vec2i operator/(const vec2i &v, i32 d)
{
    return (1 / d) * v;
}

inline vec2 operator*(const vec2i &u, f32 d)
{
    return vec2(u[0] * d, u[1] * d);
}

inline vec2 operator*(f32 d, const vec2i &v)
{
    return v * d;
}

inline vec2 operator/(const vec2i &v, f32 d)
{
    return (1 / d) * v;
}

inline i32 Dot(const vec2i &u, const vec2i &v)
{
    return u[0] * v[0] + u[1] * v[1];
}

inline bool NearZero(const vec2i &v)
{
    f32 s = 1e-8f;
    return ((std::fabs(v.x) < s) && (std::fabs(v.y) < s));
}

inline vec2i Min(const vec2i &a, const vec2i &b)
{
    vec2i result;
    result.x = a.x < b.x ? a.x : b.x;
    result.y = a.y < b.y ? a.y : b.y;
    return result;
}

inline vec2i Max(const vec2i &a, const vec2i &b)
{
    vec2i result;
    result.x = a.x > b.x ? a.x : b.x;
    result.y = a.y > b.y ? a.y : b.y;
    return result;
}
//////////////////////////////
// Vec3
//
inline std::ostream &
operator<<(std::ostream &out, const vec3 &v)
{
    return out << v[0] << ' ' << v[1] << ' ' << v[2];
}

inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

inline vec3 operator*(const vec3 &u, f32 d)
{
    return vec3(u[0] * d, u[1] * d, u[2] * d);
}

inline vec3 operator*(f32 d, const vec3 &v)
{
    return v * d;
}

inline vec3 operator/(const vec3 &v, f32 d)
{
    return (1 / d) * v;
}

inline f32 Dot(const vec3 &u, const vec3 &v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

inline f32 AbsDot(const vec3 &u, const vec3 &v)
{
    return fabsf(Dot(u, v));
}

inline vec3 Cross(const vec3 &u, const vec3 &v)
{
    return vec3(u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]);
}

inline vec3 Normalize(const vec3 &v)
{
    return v / v.length();
}

inline bool NearZero(const vec3 &v)
{
    f32 s = 1e-8f;
    return ((std::fabs(v.x) < s) && (std::fabs(v.y) < s) && (std::fabs(v.z) < s));
}

inline vec3 Reflect(const vec3 &v, const vec3 &norm)
{
    return v - 2 * Dot(v, norm) * norm;
}

inline vec3 Refract(const vec3 &uv, const vec3 &n, f32 refractiveIndexRatio)
{
    f32 cosTheta  = fmin(Dot(-uv, n), 1.f);
    vec3 perp     = refractiveIndexRatio * (uv + cosTheta * n);
    vec3 parallel = -sqrt(fabs(1 - perp.lengthSquared())) * n;
    return perp + parallel;
}

inline vec3 Min(const vec3 &a, const vec3 &b)
{
    vec3 result;
    result.x = a.x < b.x ? a.x : b.x;
    result.y = a.y < b.y ? a.y : b.y;
    result.z = a.z < b.z ? a.z : b.z;
    return result;
}

inline vec3 Max(const vec3 &a, const vec3 &b)
{
    vec3 result;
    result.x = a.x > b.x ? a.x : b.x;
    result.y = a.y > b.y ? a.y : b.y;
    result.z = a.z > b.z ? a.z : b.z;
    return result;
}

inline vec3 ClampZero(const vec3 &v)
{
    return vec3(Max(0.f, v.x), Max(0.f, v.y), Max(0.f, v.z));
}

//////////////////////////////
// Vec4
//
inline std::ostream &
operator<<(std::ostream &out, const vec4 &v)
{
    return out << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3];
}

inline vec4 operator+(const vec4 &u, const vec4 &v)
{
    return vec4(u[0] + v[0], u[1] + v[1], u[2] + v[2], u[3] + v[3]);
}

inline vec4 operator-(const vec4 &u, const vec4 &v)
{
    return vec4(u[0] - v[0], u[1] - v[1], u[2] - v[2], u[3] - v[3]);
}

inline vec4 operator*(const vec4 &u, const vec4 &v)
{
    return vec4(u[0] * v[0], u[1] * v[1], u[2] * v[2], u[3] * v[3]);
}

inline vec4 operator*(const vec4 &u, f32 d)
{
    return vec4(u[0] * d, u[1] * d, u[2] * d, u[3] * d);
}

inline vec4 operator*(f32 d, const vec4 &v)
{
    return v * d;
}

inline vec4 operator/(const vec4 &v, f32 d)
{
    return (1 / d) * v;
}

inline f32 Dot(const vec4 &u, const vec4 &v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3];
}

inline vec4 Normalize(const vec4 &v)
{
    return v / v.length();
}

inline bool NearZero(const vec4 &v)
{
    f32 s = 1e-8f;
    return ((std::fabs(v.x) < s) && (std::fabs(v.y) < s) && (std::fabs(v.z) < s) && (std::fabs(v.w) < s));
}

//////////////////////////////
// mat3
//
union mat3
{
    struct
    {
        f32 a1, a2, a3;
        f32 b1, b2, b3;
        f32 c1, c2, c3;
    };

    f32 elements[3][3];
    vec3 columns[3];

    mat3() {}
    mat3(f32 a) : mat3(a, a, a) {}

    mat3(f32 a, f32 b, f32 c)
        : a1(a), a2(0), a3(0), b1(0), b2(b), b3(0), c1(0), c2(0), c3(c) {}

    mat3(vec3 v) : mat3(v.x, v.y, v.z) {}

    vec3 &operator[](const i32 index)
    {
        return columns[index];
    }

    mat3 &operator/(const f32 f)
    {
        columns[0] /= f;
        columns[1] /= f;
        columns[2] /= f;
        return *this;
    }

    mat3(vec3 c1, vec3 c2, vec3 c3)
    {
        columns[0] = c1;
        columns[1] = c2;
        columns[2] = c3;
    }
    mat3(f32 a1, f32 a2, f32 a3, f32 b1, f32 b2, f32 b3, f32 c1, f32 c2, f32 c3)
        : mat3(vec3(a1, a2, a3), vec3(b1, b2, b3), vec3(c1, c2, c3)) {}

    mat3 Inverse()
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

        mat3 result(det1, det2, det3, det4, det5, det6, det7, det8, det9);

        f32 det = a1 * det1 + b1 * det2 + c1 * det3;

        Assert(det != 0.f);
        return result / det;
    }

    static mat3 Diag(f32 a, f32 b, f32 c)
    {
        mat3 result(a, b, c);
        return result;
    }
};

inline mat3 operator*(mat3 a, mat3 b)
{
    mat3 result;
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
// mat4
//
union mat4
{
    f32 elements[4][4];
    vec4 columns[4];
    struct
    {
        f32 a1, a2, a3, a4;
        f32 b1, b2, b3, b4;
        f32 c1, c2, c3, c4;
        f32 d1, d2, d3, d4;
    };

    mat4() : a1(0), a2(0), a3(0), a4(0), b1(0), b2(0), b3(0), b4(0),
             c1(0), c2(0), c3(0), c4(0), d1(0), d2(0), d3(0), d4(0)
    {
    }

    mat4(f32 val) : a1(val), b2(val), c3(val), d4(val)
    {
    }

    vec4 &operator[](const i32 index)
    {
        return columns[index];
    }

    static inline mat4 Identity()
    {
        mat4 result(1.f);
        return result;
    }

    static mat4 Rotate(vec3 axis, f32 theta)
    {
        mat4 result           = mat4::Identity();
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

    static mat4 Translate(vec3 value)
    {
        mat4 result         = mat4::Identity();
        result.columns[3].x = value.x;
        result.columns[3].y = value.y;
        result.columns[3].z = value.z;
        return result;
    }
};
vec4 mul(mat4 a, vec4 b)
{
    vec4 result;
#ifdef SSE42
    __m128 c0 = _mm_load_ps((f32 *)(&a.a1));
    __m128 c1 = _mm_load_ps((f32 *)(&a.b1));
    __m128 c2 = _mm_load_ps((f32 *)(&a.c1));
    __m128 c3 = _mm_load_ps((f32 *)(&a.d1));

    __m128 vx = _mm_set1_ps(b.x);
    __m128 vy = _mm_set1_ps(b.y);
    __m128 vz = _mm_set1_ps(b.z);
    __m128 vw = _mm_set1_ps(b.w);

    __m128 vec = _mm_mul_ps(c0, vx);
    vec        = _mm_add_ps(vec, _mm_mul_ps(c1, vy));
    vec        = _mm_add_ps(vec, _mm_mul_ps(c2, vz));
    vec        = _mm_add_ps(vec, _mm_mul_ps(c3, vw));

    _mm_store_ps((f32 *)&result.e[0], vec);

#else
    result.x = a.columns[0].x * b.x;
    result.y = a.columns[0].y * b.x;
    result.z = a.columns[0].z * b.x;
    result.w = a.columns[0].w * b.x;

    result.x += a.columns[1].x * b.y;
    result.y += a.columns[1].y * b.y;
    result.z += a.columns[1].z * b.y;
    result.w += a.columns[1].w * b.y;

    result.x += a.columns[2].x * b.z;
    result.y += a.columns[2].y * b.z;
    result.z += a.columns[2].z * b.z;
    result.w += a.columns[2].w * b.z;

    result.x += a.columns[3].x * b.w;
    result.y += a.columns[3].y * b.w;
    result.z += a.columns[3].z * b.w;
    result.w += a.columns[3].w * b.w;
#endif
    return result;
}

vec3 mul(mat4 a, vec3 b)
{
    vec4 vec(b.x, b.y, b.z, 1);
    return mul(a, vec).xyz;
}

// Ignores translation
inline vec3 NormalTransform(const mat4 &a, const vec3 &b)
{
    vec3 result;
    result.x = a.columns[0].x * b.x;
    result.y = a.columns[0].y * b.x;
    result.z = a.columns[0].z * b.x;

    result.x += a.columns[1].x * b.y;
    result.y += a.columns[1].y * b.y;
    result.z += a.columns[1].z * b.y;

    result.x += a.columns[2].x * b.z;
    result.y += a.columns[2].y * b.z;
    result.z += a.columns[2].z * b.z;

    return result;
}

inline mat4 mul(mat4 a, mat4 b)
{
    mat4 result;
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

inline mat3 mul(mat3 a, mat3 b)
{
    mat3 result;
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

inline vec3 mul(mat3 a, vec3 b)
{
    vec3 result;
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

// only scale and rotation
inline vec3 mul2(mat4 a, vec3 b)
{
    vec3 result;

    result.x = a.columns[0].x * b.x;
    result.y = a.columns[0].y * b.x;
    result.z = a.columns[0].z * b.x;

    result.x += a.columns[1].x * b.y;
    result.y += a.columns[1].y * b.y;
    result.z += a.columns[1].z * b.y;

    result.x += a.columns[2].x * b.z;
    result.y += a.columns[2].y * b.z;
    result.z += a.columns[2].z * b.z;

    return result;
}

struct LaneU32
{
    __m128i v;
    LaneU32() {}
    LaneU32(__m128i v) : v(v) {}
};

struct LaneF32
{
    __m128 v;
    LaneF32() {}
    LaneF32(__m128 v) : v(v) {}
};

struct LaneVec2
{
    LaneF32 x;
    LaneF32 y;
};

struct LaneVec2i
{
    LaneU32 x;
    LaneU32 y;

    LaneU32 operator[](i32 i) const
    {
        return i == 0 ? x : y;
    }

    LaneU32 &operator[](i32 i)
    {
        return i == 0 ? x : y;
    }
};

struct LaneVec3
{
    LaneF32 x;
    LaneF32 y;
    LaneF32 z;
};

inline LaneF32 LaneF32Zero()
{
    LaneF32 result;
    result.v = _mm_setzero_ps();
    return result;
}

inline LaneU32 LaneU32Zero()
{
    LaneU32 result;
    result.v = _mm_setzero_si128();
    return result;
}

inline LaneF32 Load(f32 *val)
{
    LaneF32 result;
    // result.v = _mm_load_ps(val);
    result.v = _mm_setr_ps(val[0], val[1], val[2], val[3]);
    return result;
}

inline LaneF32 Load(f32 a, f32 b, f32 c, f32 d) //, f32, f32, f32, f32)
{
    LaneF32 result;
    result.v = _mm_setr_ps(a, b, c, d);
    return result;
}

inline LaneU32 Load(u32 *val)
{
    LaneU32 result;
    // result.v = _mm_load_si128((__m128i *)val);
    result.v = _mm_setr_epi32(val[0], val[1], val[2], val[3]);
    return result;
}

inline LaneU32 Load(u32 a, u32 b, u32 c, u32 d)
{
    LaneU32 result;
    // result.v = _mm_load_si128((__m128i *)val);
    result.v = _mm_setr_epi32(a, b, c, d);
    return result;
}

inline LaneVec2i Load(vec2i *val)
{
    LaneVec2i result;
    result.x.v = _mm_setr_epi32(val[0].x, val[1].x, val[2].x, val[3].x);
    result.y.v = _mm_setr_epi32(val[0].y, val[1].y, val[2].y, val[3].y);
    return result;
}

inline LaneF32 LaneF32FromF32(f32 repl)
{
    LaneF32 result;
    result.v = _mm_set1_ps(repl);
    return result;
}

inline LaneU32 LaneU32FromU32(u32 repl)
{
    LaneU32 result;
    result.v = _mm_set1_epi32(repl);
    return result;
}

// NOTE: in this case LaneU32 contains two 64-bit integers
inline LaneU32 LaneU32FromU64(u64 val)
{
    LaneU32 result;
    result.v = _mm_set1_epi64x(val);
    return result;
}

inline LaneU32 CastLaneU32FromLaneF32(LaneF32 lane)
{
    LaneU32 result;
    result.v = _mm_castps_si128(lane.v);
    return result;
}

inline LaneF32 CastLaneF32FromLaneU32(const LaneU32 &lane)
{
    LaneF32 result;
    result.v = _mm_castsi128_ps(lane.v);
    return result;
}

inline LaneU32 ConvertLaneF32ToLaneU32(const LaneF32 &lane)
{
    LaneU32 result;
    result.v = _mm_cvtps_epi32(lane.v);
    return result;
}

inline LaneF32 ConvertLaneU32ToLaneF32(const LaneU32 &lane)
{
    LaneF32 result;
    result.v = _mm_cvtepi32_ps(lane.v);
    return result;
}

inline LaneU32 TruncateU32ToU8(const LaneU32 &lane)
{
    LaneU32 result;
#ifdef __AVX512F__
    result.v = _mm_cvtepi32_epi8(lane.v);
#else if __SSE3__
    static const __m128i mask = _mm_setr_epi8((u8)0, (u8)4, (u8)8, (u8)12,
                                              (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80,
                                              (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80,
                                              (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80);
    result.v                  = _mm_shuffle_epi8(lane.v, mask);
#endif
    return result;
}

inline LaneU32 UnpackLowU32(const LaneU32 &a, const LaneU32 &b)
{
    LaneU32 result;
    result.v = _mm_unpacklo_epi32(a.v, b.v);
    return result;
}

inline LaneU32 UnpackHiU32(const LaneU32 &a, const LaneU32 &b)
{
    LaneU32 result;
    result.v = _mm_unpackhi_epi32(a.v, b.v);
    return result;
}

#define ExtractU32(lane, index) _mm_extract_epi32(lane.v, index)

inline u32 RoundFloatToU32(f32 a)
{
    u32 result = (u32)(a + 0.5f);
    return result;
}

//////////////////////////////

inline LaneVec2 LaneV2FromV2(const vec2 &v2)
{
    LaneVec2 result;
    result.x = LaneF32FromF32(v2.x);
    result.y = LaneF32FromF32(v2.y);
    return result;
}

inline LaneVec2i LaneV2IFromV2I(const vec2i &v)
{
    LaneVec2i result;
    result.x = LaneU32FromU32(v.x);
    result.y = LaneU32FromU32(v.y);
    return result;
}

inline LaneVec3 LaneV3FromV3(const vec3 &v3)
{
    LaneVec3 result;
    result.x = LaneF32FromF32(v3.x);
    result.y = LaneF32FromF32(v3.y);
    result.z = LaneF32FromF32(v3.z);
    return result;
}

inline LaneVec3 LaneV3FromV3(f32 a, f32 b, f32 c)
{
    LaneVec3 result;
    result.x = LaneF32FromF32(a);
    result.y = LaneF32FromF32(b);
    result.z = LaneF32FromF32(c);
    return result;
}

inline LaneF32 operator+(LaneF32 a, LaneF32 b)
{
    LaneF32 result;
    result.v = _mm_add_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator-(LaneF32 a, LaneF32 b)
{
    LaneF32 result;
    result.v = _mm_sub_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator*(LaneF32 a, LaneF32 b)
{
    LaneF32 result;
    result.v = _mm_mul_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator/(LaneF32 a, LaneF32 b)
{
    LaneF32 result;
    result.v = _mm_div_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator*(f32 a, LaneF32 b)
{
    LaneF32 aVec = LaneF32FromF32(a);
    return aVec * b;
}

inline LaneF32 FMA(LaneF32 a, LaneF32 b, LaneF32 c)
{
    LaneF32 result;
    result.v = _mm_fmadd_ps(a.v, b.v, c.v);
    return result;
}

inline LaneF32 FMA(f32 a, LaneF32 b, LaneF32 c)
{
    LaneF32 aVec = LaneF32FromF32(a);
    return FMA(aVec, b, c);
}

inline LaneF32 FMA(LaneF32 a, LaneF32 b, f32 c)
{
    LaneF32 cVec = LaneF32FromF32(c);
    return FMA(a, b, cVec);
}

inline LaneF32 FMA(f32 a, LaneF32 b, f32 c)
{
    LaneF32 aVec = LaneF32FromF32(c);
    LaneF32 cVec = LaneF32FromF32(c);
    return FMA(aVec, b, cVec);
}

inline LaneF32 rsqrt(LaneF32 a)
{
    LaneF32 result;
    result.v = _mm_rsqrt_ps(a.v);
    return result;
}

inline LaneF32 Min(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_min_ps(a.v, b.v);
    return result;
}

inline LaneF32 Max(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_max_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator<(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_cmplt_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator<=(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_cmple_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator>(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_cmpgt_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator==(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_cmpeq_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator&(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_and_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator|(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_or_ps(a.v, b.v);
    return result;
}

inline LaneF32 AndNot(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_andnot_ps(a.v, b.v);
    return result;
}

inline LaneVec3 operator*(const LaneVec3 &a, const LaneVec3 &b)
{
    LaneVec3 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    return result;
}

inline LaneVec3 operator-(const LaneVec3 &a, const LaneVec3 &b)
{
    LaneVec3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

inline LaneVec3 FMA(const LaneVec3 &a, const LaneVec3 &b, const LaneVec3 &c)
{
    LaneVec3 result;
    result.x = FMA(a.x, b.x, c.x);
    result.y = FMA(a.y, b.y, c.y);
    result.z = FMA(a.z, b.z, c.z);
    return result;
}

inline void ConditionalAssign(LaneF32 &dest, const LaneF32 &src, const LaneF32 &mask)
{
    dest.v = _mm_or_ps(_mm_andnot_ps(mask.v, dest.v), _mm_and_ps(mask.v, src.v));
}

// For each packed float, if the corresponding region of the mask is set, choose b
inline LaneF32 Blend(const LaneF32 &a, const LaneF32 &b, const LaneF32 &mask)
{
    LaneF32 result;
    result.v = _mm_blendv_ps(a.v, b.v, mask.v);
    return result;
}

inline bool MaskIsZeroed(const LaneF32 &a)
{
    i32 result = _mm_movemask_ps(a.v);
    return result == 0;
}

inline i32 FlattenMask(const LaneF32 &a)
{
    i32 result = _mm_movemask_ps(a.v);
    return result;
}

//////////////////////////////

inline LaneU32 operator^(const LaneU32 &a, const LaneU32 &b)
{
    LaneU32 result;
    result.v = _mm_xor_si128(a.v, b.v);
    return result;
}

inline LaneU32 operator^=(const LaneU32 &a, const LaneU32 &b)
{
    return a ^ b;
}

inline LaneU32 operator*(const LaneU32 &a, const u64 b)
{
    LaneU32 result;
    result.v = _mm_mullo_epi32(a.v, _mm_set1_epi64x(b));
    return result;
}

inline LaneU32 operator*=(const LaneU32 &a, const u64 b)
{
    return a * b;
}

inline LaneU32 operator&(const LaneU32 &a, const LaneU32 &b)
{
    LaneU32 result;
    result.v = _mm_and_si128(a.v, b.v);
    return result;
}

inline LaneU32 operator&(const LaneU32 &a, u32 inMask)
{
    LaneU32 mask = LaneU32FromU32(inMask);
    LaneU32 result;
    result.v = _mm_and_si128(a.v, mask.v);
    return result;
}

inline LaneU32 operator&=(LaneU32 lane, u32 val)
{
    return lane & LaneU32FromU32(val);
}

inline LaneU32 operator&(LaneU32 lane, u64 val)
{
    return lane & LaneU32FromU64(val);
}

inline LaneU32 operator|(const LaneU32 &a, const LaneU32 &b)
{
    LaneU32 result;
    result.v = _mm_or_si128(a.v, b.v);
    return result;
}

inline LaneU32 operator<<(const LaneU32 &a, u32 inShift)
{
    LaneU32 result;
    result.v = _mm_slli_epi32(a.v, inShift);
    return result;
}

inline LaneU32 operator>>(const LaneU32 &a, u32 inShift)
{
    LaneU32 result;
    result.v = _mm_srli_epi32(a.v, inShift);
    return result;
}

inline LaneU32 operator<<(const LaneU32 &a, u64 inShift)
{
    LaneU32 result;
    result.v = _mm_slli_epi64(a.v, (i32)inShift);
    return result;
}

inline LaneU32 operator>>(const LaneU32 &a, u64 inShift)
{
    LaneU32 result;
    result.v = _mm_srli_epi64(a.v, (i32)inShift);
    return result;
}

inline LaneU32 SignExtend(LaneU32 l)
{
    LaneU32 result;
    result.v = _mm_cvtepi32_epi64(l.v);
    return result;
}

inline LaneU32 SignExtendU8ToU32(LaneU32 l)
{
    LaneU32 result;
    // result.v = _mm_cvtepi8_epi32(l.v);
    result.v = _mm_cvtepu8_epi32(l.v);
    return result;
}

inline u32 PopCount(u32 val)
{
    return _mm_popcnt_u32(val);
}

#define PermuteU32(l, a0, a1, a2, a3) LaneU32(_mm_shuffle_epi32(l.v, _MM_SHUFFLE(a3, a2, a1, a0)))
#define PermuteF32(l, a0, a1, a2, a3) LaneF32(_mm_permute_ps(l.v, _MM_SHUFFLE(a3, a2, a1, a0)))

// NOTE: a3 is stored last
#define PermuteReverseF32(l, a3, a2, a1, a0) LaneF32(_mm_permute_ps(l.v, _MM_SHUFFLE(a3, a2, a1, a0)))

//////////////////////////////
// Ray
//
#define LANE_WIDTH 4
struct Ray
{
    Ray() {}
    Ray(const vec3 &origin, const vec3 &direction) : o(origin), d(direction), t(0) {}
    Ray(const vec3 &origin, const vec3 &direction, const f32 time) : o(origin), d(direction), t(time) {}

    const vec3 &origin() const { return o; }
    const vec3 &direction() const { return d; }
    const f32 &time() const { return t; }

    vec3 at(f32 time) const
    {
        return o + time * d;
    }

    vec3 o;
    vec3 d;
    f32 t;
};

struct SOARay
{
    LaneVec3 *o;
    LaneVec3 *d;
    LaneF32 *t;
};

//////////////////////////////
// AABB
//
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

    AABB();
    AABB(vec3 pt1, vec3 pt2);
    AABB(AABB box1, AABB box2);

    bool Hit(const Ray &r, f32 tMin, f32 tMax);
    bool Hit(const Ray &r, f32 tMin, f32 tMax, const int dirIsNeg[3]) const;
    inline vec3 Center() const
    {
        return (maxP + minP) * 0.5f;
    }
    inline vec3 Centroid() const
    {
        return Center();
    }
    inline vec3 GetHalfExtent()
    {
        return (maxP - minP) * 0.5f;
    }

    vec3 operator[](int i) const
    {
        return i == 0 ? minP : maxP;
    }

    inline vec3 Offset(const vec3 &p) const
    {
        vec3 o = p - minP;
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
        vec3 pad = vec3(delta / 2, delta / 2, delta / 2);
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
    vec3 Diagonal() const
    {
        return maxP - minP;
    }
    f32 SurfaceArea() const
    {
        vec3 d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    i32 MaxDimension() const
    {
        vec3 d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        return 2;
    }
};

inline AABB Union(const AABB &box1, const AABB &box2)
{
    AABB result;
    result.minP = Min(box1.minP, box2.minP);
    result.maxP = Max(box1.minP, box2.minP);
    return result;
}

inline AABB Union(const AABB &box1, const vec3 &p)
{
    AABB result;
    result.minP = Min(box1.minP, p);
    result.maxP = Max(box1.maxP, p);
    return result;
}

inline vec3 Min(const AABB &box1, const AABB &box2)
{
    vec3 result = Min(box1.minP, box2.minP);
    return result;
}

inline vec3 Max(const AABB &box1, const AABB &box2)
{
    vec3 result = Max(box1.maxP, box2.maxP);
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

    f32 norm() const
    {
        return real * real + im * im;
    }

    f32 real, im;
};

f32 Abs(const complex &z)
{
    return sqrtf(z.norm());
}

complex sqrt(const complex &z)
{
    f32 n  = Abs(z);
    f32 t1 = sqrtf(.5f * (n + fabsf(z.real)));
    f32 t2 = .5f * z.im / t1;

    if (n == 0)
        return 0;

    if (z.real >= 0)
        return {t1, t2};
    else
        return {fabsf(t2), std::copysign(t1, z.im)};
}

struct Frame
{
    vec3 x;
    vec3 y;
    vec3 z;

    Frame() : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}

    Frame(vec3 x, vec3 y, vec3 z) : x(x), y(y), z(z) {}

    static Frame FromXZ(vec3 x, vec3 z)
    {
        return Frame(x, Cross(z, x), z);
    }
    static Frame FromXY(vec3 x, vec3 y)
    {
        return Frame(x, y, Cross(x, y));
    }
    vec3 ToLocal(vec3 a) const
    {
        return vec3(Dot(x, a), Dot(y, a), Dot(z, a));
    }

    vec3 FromLocal(vec3 a) const
    {
        return a.x * x + a.y * y + a.z * z;
    }
};

#endif
