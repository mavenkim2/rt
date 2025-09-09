#include "image.h"
#include "color.h"
#include "memory.h"
#include "parallel.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"

namespace rt
{
f32 *Image::GetSamplingDistribution(Arena *arena)
{
    u8 *ptr = contents;

    f32 *result = PushArrayNoZero(arena, f32, height * width);
    u32 count   = 0;

    ParallelFor2D(Vec2i(0), Vec2i(width, height), Vec2i(32),
                  [&](int jobID, Vec2i start, Vec2i end) {
                      for (i32 h = start[1]; h < end[1]; h++)
                      {
                          for (i32 w = start[0]; w < end[0]; w++)
                          {
                              Vec3f values = SRGBToLinear(GetColor(this, w, h));
                              f32 val      = (values[0] + values[1] + values[2]) / 3.f;
                              Assert(val == val);
                              result[h * width + w] = val;
                              ptr += bytesPerPixel;
                          }
                      }
                  });
    return result;
}

Image LoadFile(const char *file, int numComponents)
{
    Image image;
    i32 nComponents;
    image.contents =
        (u8 *)stbi_load(file, &image.width, &image.height, &nComponents, numComponents);
    image.bytesPerPixel = nComponents;
    return image;
}

Image LoadHDR(const char *file)
{
    Image image;
    i32 nComponents;
    image.contents      = (u8 *)stbi_loadf(file, &image.width, &image.height, &nComponents, 0);
    image.bytesPerPixel = nComponents * sizeof(f32);
    return image;
}

} // namespace rt
