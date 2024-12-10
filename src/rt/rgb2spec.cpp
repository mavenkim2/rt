// Similar to pbrt, using Gauss-Newton to solve non-linear least squaRES for overspecified
// problem (rgb to spectrum) conversion.

#include "base.h"
#include <assert.h>
using namespace rt;

#define CIE_SAMPLES      95
#define CIE_FINE_SAMPLES ((CIE_SAMPLES - 1) * 3 + 1)
#define CIE_LAMBDA_MIN   360.0
#define CIE_LAMBDA_MAX   830.0
#define RGB2SPEC_EPSILON 1e-4

const double CIE_X[CIE_SAMPLES] = {
    0.000129900000, 0.000232100000, 0.000414900000, 0.000741600000, 0.001368000000,
    0.002236000000, 0.004243000000, 0.007650000000, 0.014310000000, 0.023190000000,
    0.043510000000, 0.077630000000, 0.134380000000, 0.214770000000, 0.283900000000,
    0.328500000000, 0.348280000000, 0.348060000000, 0.336200000000, 0.318700000000,
    0.290800000000, 0.251100000000, 0.195360000000, 0.142100000000, 0.095640000000,
    0.057950010000, 0.032010000000, 0.014700000000, 0.004900000000, 0.002400000000,
    0.009300000000, 0.029100000000, 0.063270000000, 0.109600000000, 0.165500000000,
    0.225749900000, 0.290400000000, 0.359700000000, 0.433449900000, 0.512050100000,
    0.594500000000, 0.678400000000, 0.762100000000, 0.842500000000, 0.916300000000,
    0.978600000000, 1.026300000000, 1.056700000000, 1.062200000000, 1.045600000000,
    1.002600000000, 0.938400000000, 0.854449900000, 0.751400000000, 0.642400000000,
    0.541900000000, 0.447900000000, 0.360800000000, 0.283500000000, 0.218700000000,
    0.164900000000, 0.121200000000, 0.087400000000, 0.063600000000, 0.046770000000,
    0.032900000000, 0.022700000000, 0.015840000000, 0.011359160000, 0.008110916000,
    0.005790346000, 0.004109457000, 0.002899327000, 0.002049190000, 0.001439971000,
    0.000999949300, 0.000690078600, 0.000476021300, 0.000332301100, 0.000234826100,
    0.000166150500, 0.000117413000, 0.000083075270, 0.000058706520, 0.000041509940,
    0.000029353260, 0.000020673830, 0.000014559770, 0.000010253980, 0.000007221456,
    0.000005085868, 0.000003581652, 0.000002522525, 0.000001776509, 0.000001251141};

const double CIE_Y[CIE_SAMPLES] = {
    0.000003917000, 0.000006965000, 0.000012390000, 0.000022020000, 0.000039000000,
    0.000064000000, 0.000120000000, 0.000217000000, 0.000396000000, 0.000640000000,
    0.001210000000, 0.002180000000, 0.004000000000, 0.007300000000, 0.011600000000,
    0.016840000000, 0.023000000000, 0.029800000000, 0.038000000000, 0.048000000000,
    0.060000000000, 0.073900000000, 0.090980000000, 0.112600000000, 0.139020000000,
    0.169300000000, 0.208020000000, 0.258600000000, 0.323000000000, 0.407300000000,
    0.503000000000, 0.608200000000, 0.710000000000, 0.793200000000, 0.862000000000,
    0.914850100000, 0.954000000000, 0.980300000000, 0.994950100000, 1.000000000000,
    0.995000000000, 0.978600000000, 0.952000000000, 0.915400000000, 0.870000000000,
    0.816300000000, 0.757000000000, 0.694900000000, 0.631000000000, 0.566800000000,
    0.503000000000, 0.441200000000, 0.381000000000, 0.321000000000, 0.265000000000,
    0.217000000000, 0.175000000000, 0.138200000000, 0.107000000000, 0.081600000000,
    0.061000000000, 0.044580000000, 0.032000000000, 0.023200000000, 0.017000000000,
    0.011920000000, 0.008210000000, 0.005723000000, 0.004102000000, 0.002929000000,
    0.002091000000, 0.001484000000, 0.001047000000, 0.000740000000, 0.000520000000,
    0.000361100000, 0.000249200000, 0.000171900000, 0.000120000000, 0.000084800000,
    0.000060000000, 0.000042400000, 0.000030000000, 0.000021200000, 0.000014990000,
    0.000010600000, 0.000007465700, 0.000005257800, 0.000003702900, 0.000002607800,
    0.000001836600, 0.000001293400, 0.000000910930, 0.000000641530, 0.000000451810};

const double CIE_Z[CIE_SAMPLES] = {
    0.000606100000, 0.001086000000, 0.001946000000, 0.003486000000, 0.006450001000,
    0.010549990000, 0.020050010000, 0.036210000000, 0.067850010000, 0.110200000000,
    0.207400000000, 0.371300000000, 0.645600000000, 1.039050100000, 1.385600000000,
    1.622960000000, 1.747060000000, 1.782600000000, 1.772110000000, 1.744100000000,
    1.669200000000, 1.528100000000, 1.287640000000, 1.041900000000, 0.812950100000,
    0.616200000000, 0.465180000000, 0.353300000000, 0.272000000000, 0.212300000000,
    0.158200000000, 0.111700000000, 0.078249990000, 0.057250010000, 0.042160000000,
    0.029840000000, 0.020300000000, 0.013400000000, 0.008749999000, 0.005749999000,
    0.003900000000, 0.002749999000, 0.002100000000, 0.001800000000, 0.001650001000,
    0.001400000000, 0.001100000000, 0.001000000000, 0.000800000000, 0.000600000000,
    0.000340000000, 0.000240000000, 0.000190000000, 0.000100000000, 0.000049999990,
    0.000030000000, 0.000020000000, 0.000010000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000};

#define N(x) (x / 10566.864005283874576)

const double CIE_D65[CIE_SAMPLES] = {
    N(46.6383), N(49.3637), N(52.0891), N(51.0323), N(49.9755), N(52.3118), N(54.6482),
    N(68.7015), N(82.7549), N(87.1204), N(91.486),  N(92.4589), N(93.4318), N(90.057),
    N(86.6823), N(95.7736), N(104.865), N(110.936), N(117.008), N(117.41),  N(117.812),
    N(116.336), N(114.861), N(115.392), N(115.923), N(112.367), N(108.811), N(109.082),
    N(109.354), N(108.578), N(107.802), N(106.296), N(104.79),  N(106.239), N(107.689),
    N(106.047), N(104.405), N(104.225), N(104.046), N(102.023), N(100.0),   N(98.1671),
    N(96.3342), N(96.0611), N(95.788),  N(92.2368), N(88.6856), N(89.3459), N(90.0062),
    N(89.8026), N(89.5991), N(88.6489), N(87.6987), N(85.4936), N(83.2886), N(83.4939),
    N(83.6992), N(81.863),  N(80.0268), N(80.1207), N(80.2146), N(81.2462), N(82.2778),
    N(80.281),  N(78.2842), N(74.0027), N(69.7213), N(70.6652), N(71.6091), N(72.979),
    N(74.349),  N(67.9765), N(61.604),  N(65.7448), N(69.8856), N(72.4863), N(75.087),
    N(69.3398), N(63.5927), N(55.0054), N(46.4182), N(56.6118), N(66.8054), N(65.0941),
    N(63.3828), N(63.8434), N(64.304),  N(61.8779), N(59.4519), N(55.7054), N(51.959),
    N(54.6998), N(57.4406), N(58.8765), N(60.3125)};

// NOTE: SRGB
const f64 XYZToRGB[3][3] = {{3.240479, -1.537150, -0.498535},
                            {-0.969256, 1.875991, 0.041556},
                            {0.055648, -0.204043, 1.057311}};

const f64 RGBToXYZ[3][3] = {{0.412453, 0.357580, 0.180423},
                            {0.212671, 0.715160, 0.072169},
                            {0.019334, 0.119193, 0.950227}};

double RGBTable[3][CIE_FINE_SAMPLES];
double LambdaTable[CIE_FINE_SAMPLES];
double XYZWhitepoint[3];

double CIE_Interp(const double *data, double x)
{
    x -= CIE_LAMBDA_MIN;
    x *= (CIE_SAMPLES - 1) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
    int offset = (int)x;
    offset     = offset < 0 ? 0 : (offset > CIE_SAMPLES - 2 ? CIE_SAMPLES - 2 : offset);
    double t   = x - offset;

    return (1.0 - t) * data[offset] + t * data[offset + 1];
}

// Normalization of the rgb value minus the roundtrip rgb value is done in CIE Lab space

void CIE_Lab(double *p)
{
    double X = 0.0;
    double Y = 0.0;
    double Z = 0.0;

    double Xw = XYZWhitepoint[0];
    double Yw = XYZWhitepoint[1];
    double Zw = XYZWhitepoint[2];

    for (int j = 0; j < 3; j++)
    {
        X += p[j] * RGBToXYZ[0][j];
        Y += p[j] * RGBToXYZ[1][j];
        Z += p[j] * RGBToXYZ[2][j];
    }
    auto f = [](double t) -> double {
        double delta = 6.0 / 29.0;
        if (t > delta * delta * delta)
        {
            return cbrt(t);
        }
        return t / (delta * delta * 3.0) + (4.0 / 29.0);
    };

    p[0] = 116.0 * f(Y / Yw) - 16.0;
    p[1] = 500.0 * (f(X / Xw) - f(Y / Yw));
    p[2] = 200.0 * (f(Y / Yw) - f(Z / Zw));
}

f64 Smoothstep(f64 x) { return x * x * (3.0 - 2.0 * x); }

void EvalResidual(const double *coeffs, const double *rgb, double *residual)
{
    double out[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < CIE_FINE_SAMPLES; i++)
    {
        double lambda = (LambdaTable[i] - CIE_LAMBDA_MIN) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
        // Polynomial
        double x = 0.0;
        for (int p = 0; p < 3; p++)
        {
            x = x * lambda + coeffs[p];
        }
        // Sigmoid
        double s = 0.5 * x / sqrt(1.0 + x * x) + 0.5;
        for (int p = 0; p < 3; p++)
        {
            out[p] += RGBTable[p][i] * s;
        }
    }
    CIE_Lab(out);
    MemoryCopy(residual, rgb, sizeof(double) * 3);
    CIE_Lab(residual);
    for (int j = 0; j < 3; j++)
    {
        residual[j] -= out[j];
    }
}

// Find the jacobian by manually computing partial derivatives for each coefficient
void EvalJacobian(const double *coeffs, const double *rgb, double **jac)
{
    double r0[3];
    double r1[3];
    double temp[3];
    for (int i = 0; i < 3; i++)
    {
        MemoryCopy(temp, coeffs, sizeof(double) * 3);
        temp[i] -= RGB2SPEC_EPSILON;
        EvalResidual(temp, rgb, r0);

        MemoryCopy(temp, coeffs, sizeof(double) * 3);
        temp[i] += RGB2SPEC_EPSILON;
        EvalResidual(temp, rgb, r1);

        for (int j = 0; j < 3; j++)
        {
            jac[j][i] = (r1[j] - r0[j]) / (2 * RGB2SPEC_EPSILON);
        }
    }
}

// https://en.wikipedia.org/wiki/LU_decomposition
int LUPDecompose(double **A, int N, double Tol, int *P)
{

    int i, j, k, imax;
    double maxA, *ptr, absA;

    for (i = 0; i <= N; i++) P[i] = i; // Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < N; i++)
    {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA)
            {
                maxA = absA;
                imax = k;
            }

        if (maxA < Tol) return 0; // failure, matrix is degenerate

        if (imax != i)
        {
            // pivoting P
            j       = P[i];
            P[i]    = P[imax];
            P[imax] = j;

            // pivoting rows of A
            ptr     = A[i];
            A[i]    = A[imax];
            A[imax] = ptr;

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++)
        {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++) A[j][k] -= A[j][i] * A[i][k];
        }
    }

    return 1; // decomposition done
}

/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
void LUPSolve(double **A, int *P, double *b, int N, double *x)
{

    for (int i = 0; i < N; i++)
    {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++) x[i] -= A[i][k] * x[k];
    }

    for (int i = N - 1; i >= 0; i--)
    {
        for (int k = i + 1; k < N; k++) x[i] -= A[i][k] * x[k];

        x[i] /= A[i][i];
    }
}

// https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
void GaussNewton(const double rgb[3], double coeffs[3], int it = 15)
{
    double r = 0;
    for (int i = 0; i < it; i++)
    {
        double J0[3], J1[3], J2[3], *J[3] = {J0, J1, J2};
        double residual[3];
        EvalResidual(coeffs, rgb, residual);
        EvalJacobian(coeffs, rgb, J);
        int P[4];
        int rv = LUPDecompose(J, 3, 1e-15, P);
        if (rv != 1)
        {
            printf("%f %f %f\n%f %f %f\n", rgb[0], rgb[1], rgb[2], coeffs[0], coeffs[1],
                   coeffs[2]);
            assert(0);
        }

        double x[3];
        LUPSolve(J, P, residual, 3, x);
        r = 0.0;
        for (int j = 0; j < 3; j++)
        {
            coeffs[j] -= x[j];
            r += residual[j] * residual[j];
        }
        double max = std::max<double>(std::max<double>(coeffs[0], coeffs[1]), coeffs[2]);
        if (max > 200)
        {
            for (int j = 0; j < 3; j++)
            {
                coeffs[j] *= 200 / max;
            }
        }

        if (r < 1e-6) break;
    }
}

#define RES 64
int main(int argc, char **argv)
{
    // Initialize spec to rgb tables, with discretization 1.6nm
    double h = (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / (CIE_FINE_SAMPLES - 1);
    MemorySet(XYZWhitepoint, 0, sizeof(XYZWhitepoint));
    MemorySet(RGBTable, 0, sizeof(XYZWhitepoint));
    for (int i = 0; i < CIE_FINE_SAMPLES; i++)
    {
        double lambda = CIE_LAMBDA_MIN + i * h;
        double xyz[3] = {CIE_Interp(CIE_X, lambda), CIE_Interp(CIE_Y, lambda),
                         CIE_Interp(CIE_Z, lambda)};

        double I = CIE_Interp(CIE_D65, lambda);

        double weight = 3.0 / 8.0 * h;
        // Simpson's 3/8 rule
        if (i == 0 || i == CIE_FINE_SAMPLES - 1)
            ;
        else if ((i - 1) % 3 == 2)
        {
            weight *= 2.f;
        }
        else
        {
            weight *= 3.f;
        }

        LambdaTable[i] = lambda;
        for (int k = 0; k < 3; k++)
        {
            for (int j = 0; j < 3; j++)
            {
                RGBTable[k][i] += XYZToRGB[k][j] * xyz[j] * I * weight;
            }
        }
        for (int k = 0; k < 3; k++)
        {
            XYZWhitepoint[k] += xyz[k] * I * weight;
        }
    }

    float *scale = new float[RES];
    for (int k = 0; k < RES; k++)
    {
        scale[k] = (float)Smoothstep(Smoothstep(k / double(RES - 1)));
    }
    size_t bufsize = 3 * 3 * RES * RES * RES;
    float *out     = new float[bufsize];
    for (int l = 0; l < 3; l++)
    {
        for (int j = 0; j < RES; j++)
        {
            const double y = j / double(RES - 1);
            for (int i = 0; i < RES; i++)
            {
                const double x   = i / double(RES - 1);
                double coeffs[3] = {};
                double rgb[3]    = {};

                int start = RES / 5;
                for (int k = start; k < RES; k++)
                {
                    double b = (double)scale[k];

                    rgb[l]           = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    GaussNewton(rgb, coeffs);
                    double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
                    double A = coeffs[0], B = coeffs[1], C = coeffs[2];
                    int idx      = (((l * RES + k) * RES + j) * RES + i) * 3;
                    out[idx + 0] = float(A * c1 * c1);
                    out[idx + 1] = float(B * c1 - 2 * A * c0 * c1 * c1);
                    out[idx + 2] = float(C - B * c0 * c1 + A * (c0 * c1) * (c0 * c1));
                }
                MemorySet(coeffs, 0, sizeof(double) * 3);
                for (int k = start; k >= 0; k--)
                {
                    double b = (double)scale[k];

                    rgb[l]           = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    GaussNewton(rgb, coeffs);
                    double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
                    double A = coeffs[0], B = coeffs[1], C = coeffs[2];
                    int idx      = (((l * RES + k) * RES + j) * RES + i) * 3;
                    out[idx + 0] = float(A * c1 * c1);
                    out[idx + 1] = float(B * c1 - 2 * A * c0 * c1 * c1);
                    out[idx + 2] = float(C - B * c0 * c1 + A * (c0 * c1) * (c0 * c1));
                }
            }
        }
    }

    FILE *f;
    fopen_s(&f, argv[1], "w");
    fprintf(f, "namespace rt {\n");
    fprintf(f, "extern const int sRGBToSpectrumTable_Res = %d;\n", RES);
    fprintf(f, "extern const float sRGBToSpectrumTable_Scale[%d] = {\n", RES);
    for (int i = 0; i < RES; i++)
    {
        fprintf(f, "%.9g, ", scale[i]);
    }
    fprintf(f, "};\n");
    fprintf(f, "extern const float sRGBToSpectrumTable_Data[3][%d][%d][%d][3] = {\n", RES, RES,
            RES);
    const float *ptr = out;
    for (int maxc = 0; maxc < 3; ++maxc)
    {
        fprintf(f, "{ ");
        for (int z = 0; z < RES; ++z)
        {
            fprintf(f, "{ ");
            for (int y = 0; y < RES; ++y)
            {
                fprintf(f, "{ ");
                for (int x = 0; x < RES; ++x)
                {
                    fprintf(f, "{ ");
                    for (int c = 0; c < 3; ++c) fprintf(f, "%.9g, ", *ptr++);
                    fprintf(f, "}, ");
                }
                fprintf(f, "},\n    ");
            }
            fprintf(f, "}, ");
        }
        fprintf(f, "}, ");
    }
    fprintf(f, "};\n");
    fprintf(f, "} // namespace rt\n");
    fclose(f);
    return 0;
}
