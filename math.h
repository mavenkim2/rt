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

inline LaneF32 operator+(LaneF32 a, LaneF32 b)
{
    LaneF32 result;
    result.v = _mm_add_ps(a.v, b.v);
    return result;
}

inline LaneF32 LaneF32FromF32(f32 repl)
{
    LaneF32 result;
    result.v = _mm_set1_ps(repl);
    return result;
}

inline LaneU32 CastLaneU32FromLaneF32(LaneF32 lane)
{
    LaneU32 result;
    result.v = _mm_castps_si128(lane.v);
    return result;
}

inline u32 RoundFloatToU32(f32 a)
{
    u32 result = (u32)(a + 0.5f);
    return result;
}
