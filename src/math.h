#ifndef MATH_H
#define MATH_H

#include <xmmintrin.h>

inline u16 SafeTruncateU32(u32 val)
{
    u16 result = (u16)val;
    assert(val == result);
    return result;
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

inline f32 dot(const vec2 &u, const vec2 &v)
{
    return u[0] * v[0] + u[1] * v[1];
}

inline vec2 normalize(const vec2 &v)
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

inline i32 dot(const vec2i &u, const vec2i &v)
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

inline f32 dot(const vec3 &u, const vec3 &v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]);
}

inline vec3 normalize(const vec3 &v)
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
    return v - 2 * dot(v, norm) * norm;
}

inline vec3 Refract(const vec3 &uv, const vec3 &n, f32 refractiveIndexRatio)
{
    f32 cosTheta  = fmin(dot(-uv, n), 1.f);
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

inline f32 dot(const vec4 &u, const vec4 &v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3];
}

inline vec4 normalize(const vec4 &v)
{
    return v / v.length();
}

inline bool NearZero(const vec4 &v)
{
    f32 s = 1e-8f;
    return ((std::fabs(v.x) < s) && (std::fabs(v.y) < s) && (std::fabs(v.z) < s) && (std::fabs(v.w) < s));
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
        axis                  = normalize(axis);
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
};

struct LaneF32
{
    __m128 v;
};

struct LaneVec3
{
    LaneF32 x;
    LaneF32 y;
    LaneF32 z;
};

inline LaneF32 Load(f32 *val)
{
    LaneF32 result;
    result.v = _mm_load_ps(val);
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

inline u32 RoundFloatToU32(f32 a)
{
    u32 result = (u32)(a + 0.5f);
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

inline LaneF32 operator<(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_cmplt_ps(a.v, b.v);
    return result;
}

inline LaneF32 operator>(const LaneF32 &a, const LaneF32 &b)
{
    LaneF32 result;
    result.v = _mm_cmpgt_ps(a.v, b.v);
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

inline void ConditionalAssign(LaneF32 &dest, const LaneF32 &src, const LaneF32 &mask)
{
    dest.v = _mm_or_ps(_mm_andnot_ps(mask.v, dest.v), _mm_and_ps(mask.v, src.v));
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
// Ray
//
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

#endif
