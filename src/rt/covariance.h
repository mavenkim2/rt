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

    // 2 problems:
    // 1. how do I handle bsdfs? how do I handle transmission?
    // - it seems that specular reflection at least has an infinite hessian lmao
    // 2. do I have to have a change of bases or something?
    // 3. how do I handle lights? do i have to do a FFT or something?
    // 4. curvature

    void InverseProjection(f32 d) {}

    bool Inverse(Mat4 *result)
    {
        Mat4 inv(matrix[0], matrix[1], matrix[3], matrix[6],  //
                 matrix[1], matrix[2], matrix[4], matrix[7],  //
                 matrix[3], matrix[4], matrix[5], matrix[8],  //
                 matrix[6], matrix[7], matrix[8], matrix[9]); //
        return Inverse(inv, *result);
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
