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
};

// https://research.nvidia.com/sites/default/files/pubs/2018-08_Phantom-Ray-Hair-Intersector//Phantom-HPG%202018.pdf
void PhantomCurveIntersector(BSpline &curve, AffineSpace &raySpace)
{
    // Intersect against cylinder first

    // Transform curve into ray space (i.e, o = (0, 0, 0), d = (0, 0, 1))
    BSpline curve2D = Transform(raySpace, curve);
    Vec3f c0        = curve2D.Eval(0);
    Vec3f cPrime    = curve2D.Dervative(0);

    f32 dt;

    // Recursively intersect until abs(dt) < 1e-5
    for (;;)
    {
        // a = c(t) + c'(t) * dt
        // ray intersects plane with normal c'(t) and point a when p1 = (0, 0, s):
        // |p1 - a| = r(t) + dr(t) * dt
        // also, (p1 - a) dot c'(t) = 0 (since p1 and a are co planar,
        // cancel dt, then solve for s. then calculate dt

        // s * c'(t).z - c(t) dot c'(t) - c'(t) dot c'(t) * dt = 0
        // dt = (s * c'(t).z - c(t) dot c'(t)) / (c'(t) dot c'(t))

        // plug dt back into original equation
        // (p1 - a) dot (p1 - a) = r^2 + 2rdrdt + dr^2 * dt^2
        // s^2 - 2(a dot p1) + a dot a = r^2 + 2rdr(s * c'z - c0cP) / (cPcP) + dr^2((c'z * s - c0cP) / (cPcP))^2

        // multiply both sides by cPcP
        // s^2 * cPcP - 2 * cPcP(c0.z * s + c'z * s * (s * c'z - c0cP) / cPcP) + c0c0*cPcP +
        // 2c0cP*(c'z * s - c0cP) + cPcP((c'z * s)^2 - 2 * (c'z * s * c0cP) + c0cP^2)/cPcP
        // = r^2 * cPcP + 2rdr(s * c'z - c0cP) + dr^2((c'z*s)^2 - 2 * (c'z * s * c0cp) + (c0cP) ^2) / cPcP

        // s^2 terms: cPcP - 2 * (c'z)^2 + (c'z)^2 - dr^2(c'z)^2
        // (c'z)^2 all cancel
        f32 cPcP = cP.x * cP.x + cP.y * cP.y;
        f32 cPz2 = cP.z * cP.z;

        f32 qs = dr * dr / Dot(cP, cP);
        f32 a  = cPcP - qs * cPz2;

        f32 cPcP = Dot(cPrime, cPrime);
        f32 c0cP = Dot(c0, cPrime);

        // s terms: -2 * c0.z * cPcP + (-2 * c'z * c0cP + 2 * c0cP * c'z) - 2 * c'z * s * c0cP - 2rdr * c'z
        // middle two terms cancel
        // = - 2 * c0.z * cPcP - 2 * c'z * c0cP - 2rdr * c'z + 2dr^2(c'z * s * c0cp) / cPcP
        // divide by -2
        f32 rdr = r * dr;
        f32 b   = c0.z * cPcP + cPrime.z * (rdr - c0cP - qs * c0cP);

        Vec3f c0xcP = Cross(c0, cP);
        // constant terms:
        // c0c0*cPcP - 2(c0cP)^2  + c0cP^2 - 2rdr * c0cP - r^2 * cPcP
        // c0c0 * cPcP - (c0cP)^2 - 2rdr * c0cP - r^2 * cPcP

        // c0c0 * cPcP - (c0cP)^2 = |c0 x cP|^2
        f32 c = Dot(c0xcP, c0xcP) - 2 * rdr * c0cP - r * r * cPcP;
    }
}
