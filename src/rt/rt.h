#ifndef RT_H
#define RT_H

#include "base.h"
#include "thread_statistics.h"
#include "macros.h"
#include "string.h"
#include "template.h"
#include "algo.h"
#include "math/math_include.h"

#define CORNELL 1
#define EMISSIVE

namespace rt
{

struct Options
{
    string filename;
    i32 pixelX = -1;
    i32 pixelY = -1;
    bool useValidation;
};

const Vec3f INVALID_VEC = Vec3f((f32)U32Max, (f32)U32Max, (f32)U32Max);

struct HitRecord
{
    Vec3f normal;
    Vec3f p;
    f32 t;
    f32 u, v;
    bool isFrontFace;
    struct Material *material;

    inline void SetNormal(const Ray &r, const Vec3f &inNormal)
    {
        isFrontFace = Dot(r.d, inNormal) < 0;
        normal      = isFrontFace ? inNormal : -inNormal;
    }
};

struct RayQueueItem
{
    Ray ray;
    i32 radianceIndex;
};

} // namespace rt

#endif
