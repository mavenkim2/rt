#ifndef MESH_SIMPLIFICATION_H_
#define MESH_SIMPLIFICATION_H_

#include "../base.h"

namespace rt
{

struct Quadric
{
    f32 a2;
    f32 ab;
    f32 ac;

    f32 b2;
    f32 bc;

    f32 c2;

    Vec3f nd;

    f32 d2;

    f32 Evaluate(const Vec3f &p);
};

template <u32 numAttributes>
struct QuadricAttr : Quadric
{
    QuadricAttr();
    Vec3f gradients[numAttributes];
    f32 d[numAttributes];

    f32 Evaluate(const Vec3f &p);
};

} // namespace rt

#endif
