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

struct Image
{
    u8 *contents;
    i32 width;
    i32 height;
    i32 bytesPerPixel;

    f32 *GetSamplingDistribution(struct Arena *arena);
    Vec2i GetPixel(Vec2f uv) const
    {
        return Vec2i(Clamp(i32(uv[0] * width), 0, width - 1),
                     Clamp(i32(uv[1] * height), 0, height - 1));
    }
};

#pragma pack(push, 1)
struct BitmapHeader
{
    u16 fileType;
    u32 fileSize;
    u16 reserved1;
    u16 reserved2;
    u32 bitmapOffset;
    u32 size;
    i32 width;
    i32 height;
    u16 planes;
    u16 bitsPerPixel;
    u32 compression;
    u32 sizeOfBitmap;
    i32 horzResolution;
    i32 vertResolution;
    u32 colorsUsed;
    u32 colorsImportant;
};
#pragma pack(pop)

struct RayQueueItem
{
    Ray ray;
    i32 radianceIndex;
};

inline u32 *GetPixelPointer(Image *image, u32 x, u32 y)
{
    Assert(x < (u32)image->width && y < (u32)image->height);
    u32 *ptr = (u32 *)(image->contents + x * image->bytesPerPixel +
                       (image->height - y - 1) * image->width * image->bytesPerPixel);
    return ptr;
}

inline u32 GetImageSize(Image *image)
{
    u32 size = image->width * image->height * image->bytesPerPixel;
    return size;
}

Image LoadFile(const char *file, int numComponents = 0);
Image LoadHDR(const char *file);

inline u8 *GetColor(const Image *image, i32 x, i32 y)
{
    x = Clamp(x, 0, image->width - 1);
    y = Clamp(y, 0, image->height - 1);

    return image->contents + x * image->bytesPerPixel +
           y * image->width * image->bytesPerPixel;
}

inline Vec3f GetRGB(const Image *image, i32 x, i32 y)
{
    Assert(x < image->width && x >= 0 && y < image->height && y >= 0);
    return *(Vec3f *)(image->contents + image->bytesPerPixel * (x + y * image->width));
}

inline void WriteImage(Image *image, const char *filename)
{
    u32 imageSize = GetImageSize(image);
    BitmapHeader header;
    header.fileType        = 0x4D42; // 'BM' little endian
    header.fileSize        = sizeof(header) + imageSize;
    header.bitmapOffset    = sizeof(header);
    header.size            = sizeof(header) - 14; // 40
    header.width           = image->width;
    header.height          = image->height;
    header.planes          = 1;
    header.bitsPerPixel    = 32;
    header.compression     = 0;
    header.sizeOfBitmap    = imageSize;
    header.horzResolution  = 0;
    header.vertResolution  = 0;
    header.colorsUsed      = 0;
    header.colorsImportant = 0;

    FILE *outFile;
    fopen_s(&outFile, filename, "wb");
    if (outFile)
    {
        fwrite(&header, sizeof(header), 1, outFile);
        fwrite(image->contents, imageSize, 1, outFile);
        fclose(outFile);
    }
    else
    {
        fprintf(stderr, "[ERROR] Unable to write file %s.\n", filename);
    }
}
f32 ExactLinearToSRGB(f32 l);

} // namespace rt

#endif
