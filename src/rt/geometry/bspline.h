#ifndef BSPLINE_H
#define BSPLINE_H
namespace rt
{
namespace curve
{

static f32 splitTable[8 + 1];

void InitTables()
{
    u32 length = 8;
    for (u32 i = 0; i <= length; i++)
    {
        splitTable[i] = (f32)i / length;
    }
}

void GetSegments(Lane8F32 &a, Lane8F32 &b)
{
    a = Lane8F32::LoadU(splitTable);
    b = Lane8F32::LoadU(splitTable + 1);
}

struct BSpline
{
    // NOTE: the w component contains the radius
    Vec4f v0, v1, v2, v3;
    BSpline() {}
    BSpline(const Vec4f &v0, const Vec4f &v1, const Vec4f &v2, const Vec4f &v3)
        : v0(v0), v1(v1), v2(v2), v3(v3)
    {
    }

    template <typename T>
    static Vec4<T> Eval(const T &u)
    {
        const T t = u;
        const T s = 1.f - u;

        const T t0 = t * t * t;
        const T t1 = (4.0f * (s * s * s) + (t * t * t)) +
                     (12.0f * ((s * t) * s) + 6.0f * ((t * s) * t));
        const T t2 = (4.0f * (t * t * t) + (s * s * s)) +
                     (12.0f * ((t * s) * t) + 6.0f * ((s * t) * s));
        const T t3 = s * s * s;
        return (1.0f / 6.f) * Vec4<T>(t0, t1, t2, t3);
    }

    template <typename T>
    static Vec4<T> Derivative(const T &u)
    {
        const T t  = u;
        const T s  = 1.0f - u;
        const T n0 = -s * s;
        const T n1 = -t * t - 4.0f * (t * s);
        const T n2 = s * s + 4.0f * (s * t);
        const T n3 = t * t;
        return 0.5f * Vec4<T>(n0, n1, n2, n3);
    }
    template <typename T>
    Vec3<T> Eval(const T &u)
    {
        Vec4<T> t      = Eval(u);
        Vec3<T> result = FMA(t.x, v0, FMA(t.y, v1, FMA(t.z, v2, t.w * v3)));
        return result;
    }

    template <typename T>
    Vec3<T> Derivative(const T &u)
    {
        Vec4<T> t      = Derivative(u);
        Vec3<T> result = FMA(t.x, v0, FMA(t.y, v1, FMA(t.z, v2, t.w * v3)));
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
    enum CurveType
    {
        Ribbon,
        Cylinder,
        Flat,
    };
    BSpline *curve;
    CurveType type;
};

struct SurfaceInteraction
{
    Vec3f p;
    Vec3f n;
};

// https://www.embree.org/papers/2014-HPG-hair.pdf
void IntersectCurve(const CurvePrecalculations &pre, const Ray &ray, const Curve &inCurve)
{
    BSpline curve = Transform(space, *inCurve.curve);
    /* directly evaluate 8 line segments (P0, P1) */

    Lane8F32 t0, t1;
    GetSegments(t0, t1);
    Vec3lf8 p0 = curve.Eval(t0);
    Vec3lf8 p1 = curve.Eval(t1);
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
    /* the w-component interpolates the curve width */
    Lane8F32 w = p.w;
    /* if distance to nearest point P <= curve radius ... */
    Lane8F32 r2   = w * w * 0.25f;
    Lane8F32 d2   = p.x * p.x + p.y * p.y;
    Lane8F32 mask = d2 <= r2 & ray.tNear < t & t < ray.tFar;
    /* find closest hit along ray by horizontal reduction */
    if (Any(mask))
    {
        f32 tOut  = ReduceMin(t);
        u32 index = 0;
        for (u32 i = 0; i < 8; i++)
        {
            if (t[i] == tOut)
            {
                index = i;
                break;
            }
        }
        f32 uOut     = (index + u[index]) * 1.f / 8.f; // Lerp(u[index], t0[index], t1[index]);
        Vec3f dpdu   = inCurve.Derivative(uOut);
        f32 edgeFunc = dpdu.x * -p.y[index] + dpdu.y * p.x[index];
        f32 v        = 0.5f - Copysign(Sqrt(d2[index]), edgeFunc) / (r[index]);
        if (inCurve.type == Curve::Ribbon)
        {
            // not implemented
            Assert(0);
        }
        else
        {
            Vec3f dpduPlane = Transform(space, dpdu);
            Vec3f dpdvPlane = Normalize(Vec3f(-dpduPlane.y, dpduPlane.x, 0)) * w[index];
            if (Curve::Cylinder)
            {
                f32 theta = Lerp(v, PI / 2, -PI / 2);
                dpdvPlane = Transform(Rotate(dpduPlane, theta), dpdvPlane);
            }
            dpdv = Transform(Inverse(space), dpdvPlane);
        }
    }
}

struct Cylinder
{
    Vec3f o;
    Vec3f dir;
    f32 dist;
};

Cylinder ComputeCurveCylinder(BSpline &curve)
{
    Vec3f c0   = curve.Eval(0);
    Vec3f c1   = curve.Eval(1);
    Vec3f cMid = curve.Eval(0.5f);

    Vec3f dir = Normalize(c1 - c0);
    Vec3f oe  = ((c0 + c1) / 2.f + cMid) / 2.f;

    // TODO: subdivide for tighter bounds?
    f32 distSq = 0.f;
    distSq     = Max(distSq, LengthSquared(Cross(curve.v0 - oe, dir)));
    distSq     = Max(distSq, LengthSquared(Cross(curve.v1 - oe, dir)));
    distSq     = Max(distSq, LengthSquared(Cross(curve.v2 - oe, dir)));
    distSq     = Max(distSq, LengthSquared(Cross(curve.v3 - oe, dir)));

    return Cylinder{oe, dir, distSq};
}

bool IntersectCurve(const Ray &ray, BSpline &curve, SurfaceInteraction &si)
{
    f32 t      = 0.f;
    f32 prevDt = 0.f;

    // First intersect against enclosing cylinder
    Cylinder cylinder;

    Vec3f n    = Cross(ray.d, cylinder.dir);
    f32 distSq = Sqr(Dot(ray.o - cylinder.o, n)) / Dot(n, n);
    if (distSq > cylinder.distSq) return false;

    // Choose end to start
    Vec3f w0 = curve.Eval(0);
    Vec3f w1 = curve.Eval(1.f);

    f32 sign = 1.f;
    // Start at other end
    if (Dot(w1 - w0, ray.d) <= 0.f)
    {
        t    = 1.f;
        sign = -1.f;
    }

    // NOTE: this assumes that c' * (w3 - w0) > 0 for all t (0, 1)
    // Then, begin cone intersection
    // TODO: I'm not sure if this is the best
    bool intersect                 = false;
    f32 minT                       = 0.f;
    f32 hitWidth                   = 0.f;
    static const int maxIterations = 50;
    for (int i = 0; i < maxIterations; i++)
    {
        Vec3f c0 = curve.Eval(t);
        Vec3f cd = curve.Derivative(t); // * sign;

        f32 r = curve.Radius(t);
        f32 dr;

        // Solve quadratic equation to find s
        Vec3f cr = c0 - ray.o;
        f32 cdcd = Dot(cd, cd);
        f32 crcd = Dot(cr, cd);
        f32 crcr = Dot(cr, cr);
        f32 crrd = Dot(cr, ray.d);
        f32 cdrd = Dot(cd, ray.d);

        f32 cdrdn = cdrd / cdcd;
        f32 crcdn = crcd / cdcd;
        r         = r - dr * crcdn;
        dr        = dr * cdrdn;
        f32 q00   = crcr - crcd * crcdn;
        f32 q01   = cdrd * crcdn - crrd;
        f32 q11   = 1 - cdrd * cdrdn;
        f32 a     = q00 - r * r;
        f32 b     = q01 - r * dr;
        f32 c     = dr * dr - q11;
        f32 det   = b * b + a * c;

        f32 s  = (b + (det < 0 ? 0 : Sqrt(det))) / c;
        f32 dt = cdrdn * s - crcdn;

        if (t == 0.f && dt < 0.f || t == 1.f && dt > 1.f) break;

        // Compute |cr - s rd|, i.e. length of c0 - ray(s)
        f32 dc = crcr - 2 * crrd * s + s * s;
        f32 sp = crcd / cdrd;
        f32 dp = crcr - 2 * crrd * sp + sp * sp;

        // Buttend check
        if (dt > ?)
        {
        }
        dt = Clamp(dt, -0.5f, 0.5f);

        const f32 threshold = 5e-5;
        if (det > 0.f && Abs(dt) < threshold)
        {
            intersect = true;
            hitWidth  = r;
            break;
        }

        // Regula falsi
        if (dt * prevDt < 0)
        {
            f32 newT;
            // Bisection every 4th iteration
            if ((iteration & 3) == 0)
            {
                newT = 0.5f * (prevT + t);
            }
            else
            {
                newT = (dt * prevT - prevDt * t) / (dt - prevDt);
            }
            prevT  = t;
            prevDt = dt;
            t      = newT;
        }
        else
        {
            f32 newT = t + dt;
            prevT    = t;
            prevDt   = dt;
            t        = newT;
        }
    }
    if (intersect)
    {
        if (curveType == CurveType::Cylinder)
        {
            Vec3f n =
        }

        f32 u              = minT;
        f32 curvePoint     = curve.Eval(u);
        f32 intersectPoint = ray(u);

        f32 hitWidth = curve.Radius(u) / 2.f;

        // TODO: this needs to be a projected distance
        f32 v = .5f + Length(intersectPoint - curvePoint) / hitWidth;
    }
}

} // namespace curve
} // namespace rt
#endif
