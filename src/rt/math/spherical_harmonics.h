#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

#include "../base.h"
#include "basemath.h"
#include "vec3.h"
#include "simd_base.h"

namespace rt
{

struct L2
{
    f32 coefficients[9];

    f32 operator[](i32 index) const
    {
        Assert(index < 9);
        return coefficients[index];
    }
    f32 &operator[](i32 index)
    {
        Assert(index < 9);
        return coefficients[index];
    }
};

static const f32 SqrtPi = Sqrt(PI);

static const f32 CosineA0 = PI;
static const f32 CosineA1 = (2.0f * PI) / 3.0f;
static const f32 CosineA2 = (0.25f * PI);

static const f32 BasisL0     = 1 / (2 * SqrtPi);
static const f32 BasisL1     = Sqrt(3) / (2 * SqrtPi);
static const f32 BasisL2_MN2 = Sqrt(15) / (2 * SqrtPi);
static const f32 BasisL2_MN1 = Sqrt(15) / (2 * SqrtPi);
static const f32 BasisL2_M0  = Sqrt(5) / (4 * SqrtPi);
static const f32 BasisL2_M1  = Sqrt(15) / (2 * SqrtPi);
static const f32 BasisL2_M2  = Sqrt(15) / (4 * SqrtPi);

inline L2 EvaluateL2Basis(const Vec3f &direction)
{
    L2 sh;

    // L0
    sh.coefficients[0] = BasisL0;

    // coefficients1
    sh.coefficients[1] = BasisL1 * direction.y;
    sh.coefficients[2] = BasisL1 * direction.z;
    sh.coefficients[3] = BasisL1 * direction.x;

    // coefficients2
    sh.coefficients[4] = BasisL2_MN2 * direction.x * direction.y;
    sh.coefficients[5] = BasisL2_MN1 * direction.y * direction.z;
    sh.coefficients[6] = BasisL2_M0 * (3 * direction.z * direction.z - 1);
    sh.coefficients[7] = BasisL2_M1 * direction.x * direction.z;
    sh.coefficients[8] = BasisL2_M2 * (direction.x * direction.x - direction.y * direction.y);

    return sh;
}

inline f32 EvaluateIrradiance(const L2 &c, const Vec3f &n)
{
    L2 l2 = EvaluateL2Basis(n);

    f32 result = 0.f;
    result += l2[0] * CosineA0 * c[0];
    for (int i = 1; i <= 3; i++)
    {
        result += l2[i] * CosineA1 * c[i];
    }
    for (int i = 4; i < 9; i++)
    {
        result += l2[i] * CosineA2 * c[i];
    }
    return result;
}

} // namespace rt
#endif
