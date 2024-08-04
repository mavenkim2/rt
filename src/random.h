inline f32 RandomFloat()
{
    return rand() / (RAND_MAX + 1.f);
}

inline f32 RandomFloat(f32 min, f32 max)
{
    return min + (max - min) * RandomFloat();
}

inline i32 RandomInt(i32 min, i32 max)
{
    return i32(RandomFloat(f32(min), f32(max)));
}

inline vec3 RandomVec3()
{
    return vec3(RandomFloat(), RandomFloat(), RandomFloat());
}

inline vec3 RandomVec3(f32 min, f32 max)
{
    return vec3(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max));
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
        vec3 p = vec3(RandomFloat(-1, 1), RandomFloat(-1, 1), 0);
        if (p.lengthSquared() < 1)
        {
            return p;
        }
    }
}

inline vec3 RandomCosineDirection()
{
    f32 r1 = RandomFloat();
    f32 r2 = RandomFloat();

    f32 phi = 2 * PI * r1;
    f32 x   = cos(phi) * sqrt(r2);
    f32 y   = sin(phi) * sqrt(r2);
    f32 z   = sqrt(1 - r2);
    return vec3(x, y, z);
}
