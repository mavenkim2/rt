#ifndef RT_H
#define RT_H

#include "base.h"
#include "template.h"
#include "math.h"
#include "algo.h"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define CORNELL 1
#define EMISSIVE

const vec3 INVALID_VEC = vec3((f32)U32Max, (f32)U32Max, (f32)U32Max);

// NOTE: all member have to be u64
struct ThreadStatistics
{
    // u64 rayPrimitiveTests;
    // u64 rayAABBTests;
    u64 bvhIntersectionTime;
    u64 primitiveIntersectionTime;
    u64 integrationTime;
    u64 samplingTime;

    u64 dumb;
};

struct ThreadMemoryStatistics
{
    u64 totalFileMemory;
    u64 totalShapeMemory;
    u64 totalMaterialMemory;
    u64 totalTextureMemory;
    u64 totalLightMemory;
    u64 totalInstanceMemory;
    u64 totalTransformMemory;
    u64 totalStringMemory;
    u64 totalOtherMemory;
};

static ThreadStatistics *threadLocalStatistics;
static ThreadMemoryStatistics *threadMemoryStatistics;

struct HitRecord
{
    vec3 normal;
    vec3 p;
    f32 t;
    f32 u, v;
    bool isFrontFace;
    struct Material *material;

    inline void SetNormal(const Ray &r, const vec3 &inNormal)
    {
        isFrontFace = Dot(r.direction(), inNormal) < 0;
        normal      = isFrontFace ? inNormal : -inNormal;
    }
};

#define OffsetOf(type, member) (u64) & (((type *)0)->member)

struct Image
{
    u8 *contents;
    i32 width;
    i32 height;
    i32 bytesPerPixel;
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
    u32 *ptr = (u32 *)(image->contents + x * image->bytesPerPixel + (image->height - y - 1) * image->width * image->bytesPerPixel);
    return ptr;
}

inline u32 GetImageSize(Image *image)
{
    u32 size = image->width * image->height * image->bytesPerPixel;
    return size;
}

Image LoadFile(const char *file)
{
    Image image;
    i32 nComponents;
    image.contents      = stbi_load(file, &image.width, &image.height, &nComponents, 0);
    image.bytesPerPixel = nComponents;
    return image;
}

u8 *GetColor(const Image *image, i32 x, i32 y)
{
    x = Clamp(0, image->width - 1, x);
    y = Clamp(0, image->height - 1, y);

    return image->contents + x * image->bytesPerPixel + y * image->width * image->bytesPerPixel;
}

void WriteImage(Image *image, char *filename)
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

#endif
