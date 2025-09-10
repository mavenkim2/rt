#ifndef SPHERE_H_
#define SPHERE_H_

#include "math_include.h"
namespace rt
{

inline Vec4f ConstructSphereFromPoints(Vec3f *points, u32 numPoints)
{
    u32 min[3] = {};
    u32 max[3] = {};
    for (u32 i = 0; i < numPoints; i++)
    {
        for (u32 axis = 0; axis < 3; axis++)
        {
            min[axis] = points[i][axis] < points[min[axis]][axis] ? i : min[axis];
            max[axis] = points[i][axis] > points[max[axis]][axis] ? i : max[axis];
        }
    }

    f32 largestDistSqr = 0.f;
    u32 chosenAxis     = 0;
    for (u32 axis = 0; axis < 3; axis++)
    {
        f32 distSqr = LengthSquared(points[min[axis]] - points[max[axis]]);
        if (distSqr > largestDistSqr)
        {
            largestDistSqr = distSqr;
            chosenAxis     = axis;
        }
    }

    Vec3f center  = 0.5f * (points[min[chosenAxis]] + points[max[chosenAxis]]);
    f32 radius    = Length(center - points[min[chosenAxis]]);
    f32 radiusSqr = Sqr(radius);

    for (u32 i = 0; i < numPoints; i++)
    {
        f32 distSqr = LengthSquared(center - points[i]);
        if (distSqr > radiusSqr)
        {
            f32 dist = Sqrt(distSqr);
            f32 t    = 0.5f + 0.5f * (radius / dist);
            center   = Lerp(t, points[i], center);
            radius   = 0.5f * (radius + dist);
        }
    }

    return Vec4f(center, radius);
}

inline Vec4f ConstructSphereFromSpheres(Vec4f *spheres, u32 numSpheres)
{
    u32 min[3] = {};
    u32 max[3] = {};
    for (u32 i = 0; i < numSpheres; i++)
    {
        for (u32 axis = 0; axis < 3; axis++)
        {
            min[axis] = spheres[i][axis] < spheres[min[axis]][axis] ? i : min[axis];
            max[axis] = spheres[i][axis] > spheres[max[axis]][axis] ? i : max[axis];
        }
    }

    f32 largestDistSqr = 0.f;
    u32 chosenAxis     = 0;
    for (u32 axis = 0; axis < 3; axis++)
    {
        f32 distSqr = LengthSquared(spheres[min[axis]].xyz - spheres[max[axis]].xyz);
        if (distSqr > largestDistSqr)
        {
            largestDistSqr = distSqr;
            chosenAxis     = axis;
        }
    }

    // Start adding spheres
    auto AddSpheres = [&](const Vec4f &sphere0, const Vec4f &sphere1) {
        Vec3f toOther = sphere1.xyz - sphere0.xyz;
        f32 distSqr   = LengthSquared(toOther);
        if (Sqr(sphere0.w - sphere1.w) >= distSqr)
        {
            return sphere0.w < sphere1.w ? sphere1 : sphere0;
        }
        f32 dist        = Sqrt(distSqr);
        f32 newRadius   = (dist + sphere0.w + sphere1.w) * 0.5f;
        Vec3f newCenter = sphere0.xyz;
        if (dist > 1e-8f) newCenter += toOther * ((newRadius - sphere0.w) / dist);
        f32 tolerance = 1e-4f;

        return Vec4f(newCenter, newRadius);
    };

    Vec4f newSphere = spheres[min[chosenAxis]];
    newSphere       = AddSpheres(newSphere, spheres[max[chosenAxis]]);

    for (u32 i = 0; i < numSpheres; i++)
    {
        newSphere = AddSpheres(newSphere, spheres[i]);
    }

    return newSphere;
}
} // namespace rt
#endif
