#ifndef CAMERA_H_
#define CAMERA_H_

#include "base.h"
#include "math/math_include.h"

namespace rt
{

struct ViewCamera
{
    f32 pitch;
    f32 yaw;
    Vec3f position;
    Vec3f forward;
    Vec3f right;

    void RotateCamera(Vec2f dMouse, f32 rotationSpeed)
    {
        pitch -= rotationSpeed * dMouse.y;
        f32 epsilon = 0.1f;
        pitch       = Clamp(pitch, -PI / 2.f + epsilon, PI / 2.f - epsilon);
        yaw -= rotationSpeed * dMouse.x;

        while (yaw > 2 * PI) yaw -= 2 * PI;
        while (yaw < -2 * PI) yaw += 2 * PI;
    }
};

} // namespace rt
#endif
