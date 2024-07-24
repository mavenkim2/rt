#include "base.h"
#include "math.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

template <typename T>
T Clamp(T min, T max, T x)
{
    return x < min ? min : (x > max ? max : x);
}

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

struct BVH;
struct Light;
struct WorkItem
{
    u32 startX;
    u32 startY;
    u32 onePastEndX;
    u32 onePastEndY;
};

struct RenderParams
{
    BVH *bvh;
    Image *image;
    Light *lights;
    vec3 cameraCenter;
    vec3 pixel00;
    vec3 pixelDeltaU;
    vec3 pixelDeltaV;
    vec3 defocusDiskU;
    vec3 defocusDiskV;
    f32 defocusAngle;
    u32 maxDepth;
    u32 samplesPerPixel;
    u32 squareRootSamplesPerPixel;
    u32 numLights;
};

struct WorkQueue
{
    WorkItem *workItems;
    RenderParams *params;
    u64 volatile workItemIndex;
    u64 volatile tilesFinished;
    u32 workItemCount;
};

bool RenderTile(WorkQueue *queue);
// platform specific
#if _WIN32
DWORD WorkerThread(void *parameter)
{
    WorkQueue *queue = (WorkQueue *)parameter;
    while (RenderTile(queue)) continue;
    return 0;
}

void CreateWorkThread(void *parameter)
{
    DWORD threadID;
    HANDLE threadHandle = CreateThread(0, 0, WorkerThread, parameter, 0, &threadID);
    CloseHandle(threadHandle);
}

inline u64 InterlockedAdd(u64 volatile *addend, u64 value)
{
    return InterlockedExchangeAdd64((volatile LONG64 *)addend, value);
}

inline u32 GetCPUCoreCount()
{
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    u32 result = info.dwNumberOfProcessors;
    return result;
}
#endif
