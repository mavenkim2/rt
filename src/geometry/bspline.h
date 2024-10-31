struct BSpline
{
    // NOTE: the w component contains the radius
    Vec4f v0, v1, v2, v3;
    BSpline() {}
    BSpline(const Vec4f &v0, const Vec4f &v1, const Vec4f &v2, const Vec4f &v3) : v0(v0), v1(v1), v2(v2), v3(v3) {}
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
    static Vec4f Derivative(f32 u)
    {
        const f32 t  = u;
        const f32 s  = 1.0f - u;
        const f32 n0 = -s * s;
        const f32 n1 = -t * t - 4.0f * (t * s);
        const f32 n2 = s * s + 4.0f * (s * t);
        const f32 n3 = t * t;
        return 0.5f * Vec4f(n0, n1, n2, n3);
    }
    Vec3f Eval(f32 u)
    {
        Vec4f t      = Eval(u);
        Vec3f result = FMA(t.x, v0, FMA(t.y, v1, FMA(t.z, v2, t.w * v3)));
        return result;
    }
    Vec3f Derivative(f32 u)
    {
        Vec4f t      = Derivative(u);
        Vec3f result = FMA(t.x, v0, FMA(t.y, v1, FMA(t.z, v2, t.w * v3)));
        return result;
    }
};

struct CurvePrecalculations
{
    f32 invLength;
    AffineSpace space;
    CurvePrecalculations(const Ray &ray)
    {
        invLength = Rsqrt(Dot(ray.d, ray.d));
        space     = Frame(ray.d * invLength);
        // NOTE: this makes it so frame * ray.d = (0, 0, 1)
        space.c2 *= invLength;
        space    = AffineSpace::Transpose3x3(space);
        space.c3 = TransformV(space, -ray.o);
    }
};

BSpline Transform(const AffineSpace &space, const BSpline &curve)
{
    return BSpline(Vec4f(TransformP(space, curve.v0.xyz), curve.v0.w),
                   Vec4f(TransformP(space, curve.v1.xyz), curve.v1.w),
                   Vec4f(TransformP(space, curve.v2.xyz), curve.v2.w),
                   Vec4f(TransformP(space, curve.v3.xyz), curve.v3.w));
}

struct Curve
{
    BSpline *curve;
};

// https://www.embree.org/papers/2014-HPG-hair.pdf
void IntersectCurve(const CurvePrecalculations &pre, const Ray &ray, const Curve &inCurve)
{
    BSpline curve = Transform(space, *inCurve.curve);
    /* directly evaluate 8 line segments (P0, P1) */
    Vec3lf8 p0, p1;
    curve.Eval(p0, p1);
    /* project ray origin (0, 0) onto 8 line segments */
    Vec3lf8 a   = -p0;
    Vec3lf8 b   = p1 - p0;
    Lane8F32 d0 = FMA(a.x, b.x, a.y * b.y);
    Lane8F32 d1 = FMA(b.x, b.x, b.y * b.y);
    /* calculate closest points P on line segments */
    Lane8F32 u = Clamp(d0 * Rcp(d1), 0.0f, 1.0f);
    Vec3lf8 p  = FMA(u, b, p0);
    /* the z-component holds hit distance */
    Lane8F32 t = p.z * pre.invLength;
    /* the w-component interpolates the curve radius */
    Lane8F32 r = p.w;
    /* if distance to nearest point P <= curve radius ... */
    Lane8F32 r2   = r * r;
    Lane8F32 d2   = p.x * p.x + p.y * p.y;
    Lane8F32 mask = d2 <= r2 & ray.tNear < t & t < ray.tFar;
    /* find closest hit along ray by horizontal reduction */
    if (Any(mask))
    {
        f32 tOut = ReduceMin(t);
        // TODO: using u, linearly interpolate between the the uMin and uMax of the closest segment
        f32 uOut   = ? ? ;
        Vec3f dpdu = inCurve.Derivative(uOut);
        // if ribbon

        size_t i = select_horizontal_min(mask, t);
    }
}

// https://research.nvidia.com/sites/default/files/pubs/2018-08_Phantom-Ray-Hair-Intersector//Phantom-HPG%202018.pdf
void IntersectCurve(const Ray &ray, BSpline &curve)
{
    // Intersect against cylinder first

    // Transform curve into ray space (i.e, o = (0, 0, 0), d = (0, 0, 1))
    f32 t    = 0;
    Vec4f c0 = curve.Eval(t);
    Vec4f cd = curve.Derivative(t);

    f32 r  = c0.w;
    f32 dr = ? ;

    f32 dt = 0;
    f32 prevT;
    f32 prevDt;

    f32 dc;
    f32 sp;
    f32 dp;

    // Recursively intersect until abs(dt) < 1e-5
    // https://www.highperformancegraphics.org/wp-content/uploads/2018/Papers-Session4/HPG2018_PhantomRayHairIntersection.pdf
    u32 iteration = 0;
    for (;;)
    {
        // a = c0(t) + c'(t)dt
        // p1 = o + sd (o is (0, 0, 0)), p1 is intersection of ray with cone base plane
        // |p1 - a| = r(t) + r'(t)dt, intersection with cone
        // (p1 - a) dot c'(t) = 0 (since p1 and a are coplanar)
        Vec3f cr         = c0 - ray.o;
        f32 cdcd         = Dot(cd, cd);
        f32 crcd         = Dot(cr, cd);
        f32 crcr         = Dot(cr, cr);
        f32 crrd         = Dot(cr, ray.d);
        f32 cdrd         = Dot(cd, ray.d);
        f32 oneover_cdcd = 1.f / cdcd;
        f32 cdrdn        = cdrd * oneover_cdcd;
        f32 crcdn        = crcd * oneover_cdcd;
        r                = r - dr * crcdn;
        dr               = dr * cdrdn;
        f32 q00          = crcr - crcd * crcdn;
        f32 q01          = cdrd * crcdn - crrd;
        f32 q11          = 1 - cdrd * cdrdn;
        // NOTE: coefficients of quadratic equation, cx^2 - 2bx + a = 0
        f32 a = q00 - r * r;
        f32 b = q01 - r * dr;
        f32 c = dr * dr - q11;

        // Solve the quadratic equation for s
        f32 det = b * b - a * c;
        s       = (b + (det < 0 ? 0 : Sqrt(det))) / c; // ray.s for ray ^ cone

        prevDt = dt;
        prevT  = t;
        dt     = cdrdn * s - crcdn; // dt to the (ray ^ cone) from t

        // TODO: what are these values supposed to be used for???
        // also ray direction needs to be normalized
        // dc     = crcr - 2 * crrd * s + s * s;    // |((ray ^ cone) - c0)|^2
        sp = crcd / cdrd; // ray.s for ray ^ plane
        // dp     = crcr - 2 * crrd * sp + sp * sp; // |((ray ^ plane) - c0)|^2

        dt = Clamp(dt, -0.5f, 0.5f);

        const f32 threshold = 5e-5;
        if (det > 0.f && Abs(dt) < thresold) break;

        // regula falsi
        if (dt * prevDt < 0)
        {
            // bisection every 4th iteration
            if ((iteration & 3) == 0)
            {
                t = 0.5f * (prevT + t);
            }
            else
            {
                t = (dt * prevT - prevDt * t) / (dt - prevDt);
            }
        }
        // normal case
        else
        {
            t += dt;
        }
        c0 = curve.Eval(t);
        cd = curve.Derivative(t);

        iteration++;
    }

    // final regula falsi
    t = (dt * prevT - prevDt * t) / (dt - prevDt);
    // Intersection point
    Vec3f p = ray.at(sp);
    // Normal (for cylinder)
    // TODO: orient towards ray for "flat" curves, handle ribbons
    Vec3f ng = Normalize(p - c0);
}
