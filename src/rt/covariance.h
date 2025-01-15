#ifndef COVARIANCE_H
#define COVARIANCE_H

namespace rt
{
#define COV_MIN_FLOAT 1e-5
struct CovarianceMatrix
{
    FixedArray<f32, 10> matrix;
    LinearSpace<f32> frame;

    // * C =  ( 0  1  3  6)
    // *      ( *  2  4  7)
    // *      ( *  *  5  8)
    // *      ( *  *  *  9)

    // T = { 1 0 -d  0
    //     { 0 1  0 -d
    //     { 0 0  1  0
    //     { 0 0  0  1
    //
    // Tt * cov * T
    void Travel(f32 d)
    {
        matrix[5] += (matrix[0] * d - 2 * matrix[3]) * d;
        matrix[3] -= matrix[0] * d;
        matrix[8] += matrix[1] * d * d - (matrix[4] * d + matrix[6] * d);
        matrix[4] -= matrix[1] * d;
        matrix[6] -= matrix[1] * d;
        matrix[9] += (matrix[2] * d - 2 * matrix[7]) * d;
        matrix[7] -= matrix[2] * d;
    }

    void Curvature(f32 cx, f32 cy)
    {
        matrix[0] += (matrix[5] * cx - 2 * matrix[3]) * cx;
        matrix[1] += matrix[8] * cx * cy - (matrix[4] * cy + matrix[6] * cx);
        matrix[2] += (matrix[9] * cy - 2 * matrix[7]) * cy;
        matrix[3] -= matrix[5] * cx;
        matrix[4] -= matrix[8] * cy;
        matrix[6] -= matrix[8] * cx;
        matrix[7] -= matrix[9] * cy;
    }

    void Projection(const Vec3f &n)
    {
        const f32 cx = Dot(x, n);
        const f32 cy = Dot(y, n);

        // Rotate the Frame to be aligned with plane.
        const f32 alpha = (cx != 0.0) ? Atan2(cx, cy) : 0.0;
        const f32 c     = Cos(alpha);
        const f32 s     = -Sin(alpha);
        Rotate(c, s);

        // Scale the componnent that project by the cosine of the ray direction
        // and the normal.
        const f32 cosine = Dot(z, n);
        ScaleY(Abs(cosine));

        // Update direction vectors.
        x = c * x + s * y;
        z = (cosine < 0.0f) ? -n : n;
        y = (cosine < 0.0f) ? Cross(x, z) : Cross(z, x);
    }

    inline void InverseProjection(const Vec3f &d)
    {
        const f32 cx = Dot(x, d);
        const f32 cy = Dot(y, d);

        // Rotate the Frame to be aligned with plane.
        const f32 alpha = (cx != 0.0) ? Atan2(cx, cy) : 0.0;
        const f32 c = Cos(alpha), s = -Sin(alpha);
        Rotate(c, s); // Rotate of -alpha

        // Scale the componnent that project by the inverse cosine of the ray
        // direction and the normal.
        const f32 cosine = Dot(z, d);
        if (cosine < 0.0f)
        {
            ScaleV(-1.0f);
            ScaleU(-1.0f);
        }
        ScaleY(1.0 / Max(Abs(cosine), COV_MIN_FLOAT));

        // Update direction vectors.
        x = c * x + s * y;
        z = d;
        y = Cross(z, x);
    }

    inline void ScaleY(f32 alpha)
    {
        matrix[1] *= alpha;
        matrix[2] *= alpha * alpha;
        matrix[4] *= alpha;
        matrix[7] *= alpha;
    }
    inline void ScaleU(f32 alpha)
    {
        matrix[3] *= alpha;
        matrix[4] *= alpha;
        matrix[5] *= alpha * alpha;
        matrix[8] *= alpha;
    }

    inline void ScaleV(f32 alpha)
    {
        matrix[6] *= alpha;
        matrix[7] *= alpha;
        matrix[8] *= alpha;
        matrix[9] *= alpha * alpha;
    }

    inline void Rotate(f32 c, f32 s)
    {
        const f32 cs = c * s;
        const f32 c2 = c * c;
        const f32 s2 = s * s;

        const f32 cov_xx = matrix[0];
        const f32 cov_xy = matrix[1];
        const f32 cov_yy = matrix[2];
        const f32 cov_xu = matrix[3];
        const f32 cov_yu = matrix[4];
        const f32 cov_uu = matrix[5];
        const f32 cov_xv = matrix[6];
        const f32 cov_yv = matrix[7];
        const f32 cov_uv = matrix[8];
        const f32 cov_vv = matrix[9];

        // Rotation of the space
        matrix[0] = c2 * cov_xx + 2 * cs * cov_xy + s2 * cov_yy;
        matrix[1] = (c2 - s2) * cov_xy + cs * (cov_yy - cov_xx);
        matrix[2] = c2 * cov_yy - 2 * cs * cov_xy + s2 * cov_xx;

        // Rotation of the angle
        matrix[5] = c2 * cov_uu + 2 * cs * cov_uv + s2 * cov_vv;
        matrix[8] = (c2 - s2) * cov_uv + cs * (cov_vv - cov_uu);
        matrix[9] = c2 * cov_vv - 2 * cs * cov_uv + s2 * cov_uu;

        // Covariances
        matrix[3] = c2 * cov_xu + cs * (cov_xv + cov_yu) + s2 * cov_yv;
        matrix[4] = c2 * cov_yu + cs * (cov_yv - cov_xu) - s2 * cov_xv;
        matrix[6] = c2 * cov_xv + cs * (cov_yv - cov_xu) - s2 * cov_yu;
        matrix[7] = c2 * cov_yv - cs * (cov_xv + cov_yu) + s2 * cov_xu;
    }

    // problems:
    // 1. how do I handle bsdfs? how do I handle transmission?
    // - it seems that specular reflection at least has an infinite hessian lmao
    // 2. do I have to have a change of bases or something?
    // - i think this is just the projection, see the example code
    // 3. how do I handle lights? do i have to do a FFT or something?
    // - i have no idea
    // 4. curvature

    __forceinline Mat4 ConvertToMatrix() const
    {
        return Mat4(matrix[0], matrix[1], matrix[3], matrix[6],  //
                    matrix[1], matrix[2], matrix[4], matrix[7],  //
                    matrix[3], matrix[4], matrix[5], matrix[8],  //
                    matrix[6], matrix[7], matrix[8], matrix[9]); //
    }
    _forceinline void ConvertFromMatrix(const Mat4 &mat)
    {
        matrix[0] = mat[0];
        matrix[1] = mat[1];
        matrix[2] = mat[5];
        matrix[3] = mat[8];
        matrix[4] = mat[9];
        matrix[5] = mat[10];
        matrix[6] = mat[12];
        matrix[7] = mat[13];
        matrix[8] = mat[14];
        matrix[9] = mat[15];
    }

    bool Inverse(Mat4 *result) { return Inverse(ConvertToMatrix(), *result); }

    void Reflection(f32 h11, f32 h22)
    {
        if (h11 == pos_inf && h22 == pos_inf) return;
        // TODO: ensure this handles 0 case properly (i.e brdf doesn't change)

        Mat4 invCov;
        bool result = Inverse(&invCov);
        invCov[10] += 1.f / Max(h11, COV_MIN_FLOAT);
        invCov[15] += 1.f / Max(h22, COV_MIN_FLOAT);

        Mat4 cov;
        Inverse(invCov, &cov);
        ConvertFromMatrix(cov);
    }

    // curvature, symmetry, and alignment must be performed
    // how do you do alignment?
    void Refraction()
    {
        // for specular transmission:
        // St * (cov  + W) * S
        // see pg. 81 of belcour's thesis

        // "rough refraction is done like the brdf operator"
        //
    }

    void ConvertMatrixToDifferentials(Vec3f &dpdx, Vec3f &dpdy, f32 x00, f32 x01,
                                      f32 x11) const
    {
        f32 b = x00 + x11;
        f32 c = x00 * x11 - Sqr(x01);

        // Compute eigenvalues using quadratic formula
        // f32 lambda = (-b + Sqrt(Sqr(b) - 4.f * c)) / 2.f;
        f32 d = .25f * Sqr(b) - c;
        if (d < 0.f) return;
        d           = Sqrt(d);
        f32 lambda1 = d - .5f * b;
        f32 lambda2 = d + .5f * b;

        // Compute eigenvectors by solving system of equations (A - Diag(lambda))v = 0
        // NOTE: there are infinitely many solutions
        // (x00 - lambda) * x.x + x01 * x.y = 0
        // (x11 - lambda) * x.y + x01 * x.x = 0
        //
        // x.x = (lambda - x11) * x.y / x01

        if (Abs(x01) > COV_MIN_FLOAT)
        {
            dpdx.x = lambda1 - x11;
            dpdx.y = x01;
            dpdx.z = 0.f;

            dpdy.x = lambda2 - x11;
            dpdy.y = x01;
            dpdy.z = 0.f;

            dpdx = Normalize(dpdx);
            dpdy = Normalize(dpdy);

            // Extract ray differential from covariance matrix
            dpdx = Inv2Pi * Sqrt(lambda1) * dpdx;
            dpdy = Inv2Pi * Sqrt(lambda2) * dpdy;
        }
        else
        {
            dpdx.x = Inv2Pi * Sqrt(x00);
            dpdx.y = 0.f;
            dpdx.z = 0.f;

            dpdy.x = 0.f;
            dpdy.y = Inv2Pi * Sqrt(x11);
            dpdy.z = 0.f;
        }

        // Convert from local light field to world space
        dpdx = frame.FromLocal(dpdx);
        dpdy = frame.FromLocal(dpdy);
    }

    void ComputeRayDifferentials(Vec3f &dpdx, Vec3f &dpdy, Vec3f &dddx, Vec3f &dddy) const
    {
        Mat4 invCov;
        bool invertible = Inverse(invCov);

        if (!invertible)
        {
            return;
        }

        ConvertMatrixToDifferentials(dpdx, dpdy, invCov[0], invCov[1], invCov[5]);
        ConvertMatrixToDifferentials(dddx, dddy, invCov[10], invCov[14], invCov[15]);
    }
};

} // namespace rt
#endif
