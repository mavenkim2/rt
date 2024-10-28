struct BSpline
{
    // NOTE: the w component contains the radius
    Vec4f v0, v1, v2, v3;
    static Vec4f Eval(f32 u)
    {
        const f32 t = u;
        const f32 s = 1.f - u;

        const f32 t0 = t * t * t;
        const f32 t1 = (4.0f * (s * s * s) + (t * t * t)) + (12.0f * ((s * t) * s) + 6.0f * ((t * s) * t));
        const f32 t2 = (4.0f * (t * t * t) + (s * s * s)) + (12.0f * ((t * s) * t) + 6.0f * ((s * t) * s));
        const f32 t3 = s * s * s;
        return (1.0f / 6.f) * Vec4f(t0, t1, t2, t3);
    }
    Vec3f Eval(f32 u)
    {
        Vec4f t      = Eval(u);
        Vec3f result = FMA(t.x, v0, FMA(t.y, v1, FMA(t.z, v2, t.w * v3)));
        return result;
    }
}
