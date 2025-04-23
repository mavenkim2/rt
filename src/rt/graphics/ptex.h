#ifndef PTEX_H_
#define PTEX_H_

#include <Ptexture.h>
#include "../rt.h"
#include "../platform.h"
#include "vulkan.h"
#include "../shader_interop/virtual_textures_shaderinterop.h"

namespace Utils
{
using namespace rt;
void Copy(const void *src, int sstride, void *dst, int dstride, int vres, int rowlen);
void *GetContentsAbsoluteIndex(void *contents, const Vec2u &p, u32 width, u32 height,
                               u32 stride, u32 borderSize, u32 bytesPerPixel);
void Copy(void *src, const Vec2u &srcIndex, u32 srcWidth, u32 srcHeight, void *dst,
          const Vec2u &dstIndex, u32 dstWidth, u32 dstHeight, u32 vRes, u32 rowLen,
          u32 bytesPerBlock);
} // namespace Utils

namespace rt
{

extern Ptex::PtexCache *cache;
struct PtexErrHandler : public PtexErrorHandler
{
    void reportError(const char *error) override { ErrorExit(0, "%s", error); }
};
extern PtexErrHandler errorHandler;

struct PtexHandle
{
    i64 offset;
    OS_Handle osHandle;

    i64 bufferOffset;
    i64 fileSize;
    static const i64 bufferSize = 8192;
    void *buffer;
};

struct PtexInpHandler : public PtexInputHandler
{

public:
    /** Open a file in read mode.
        Returns null if there was an error.
        If an error occurs, the error string is available via lastError().
    */
    virtual Handle open(const char *path) override
    {
        // TODO: free list pool this
        PtexHandle *handle = (PtexHandle *)malloc(sizeof(PtexHandle) + PtexHandle::bufferSize);
        handle->offset     = 0;
        handle->osHandle   = OS_CreateFile(Str8C(path));
        handle->buffer     = (void *)(handle + 1);
        handle->bufferOffset = -1;
        handle->fileSize     = OS_GetFileSize2(Str8C(path));
        return handle;
    }

    /** Seek to an absolute byte position in the input stream. */
    // virtual void seek(Handle handle, int64_t pos) override
    // {
    //     u8 **ptr = (u8 **)handle;
    //     // Assert(pos < (int64_t)str.size);
    //     *ptr = str.str + pos;
    // }
    virtual void seek(Handle handle, int64_t pos) override
    {
        PtexHandle *offset = (PtexHandle *)handle;
        offset->offset     = pos;
    }

    /** Read a number of bytes from the file.
        Returns the number of bytes successfully read.
        If less than the requested number of bytes is read, the error string
        is available via lastError().
    */
    virtual size_t read(void *buffer, size_t size, Handle handle) override
    {
        // u8 **ptr = (u8 **)handle;
        // Assert(size_t(*ptr - str.str) + size < (size_t)str.size);
        PtexHandle *ptexHandle = (PtexHandle *)handle;
        u64 offset             = ptexHandle->offset;
        Assert(ptexHandle->osHandle.handle);

        size_t result = 0;
        if (ptexHandle->buffer && ptexHandle->bufferOffset != -1 &&
            ptexHandle->offset >= ptexHandle->bufferOffset &&
            ptexHandle->offset + size <=
                Min(ptexHandle->bufferOffset + ptexHandle->bufferSize, ptexHandle->fileSize))
        {
            MemoryCopy(buffer,
                       (u8 *)ptexHandle->buffer +
                           (ptexHandle->offset - ptexHandle->bufferOffset),
                       size);
            result = size;
            ptexHandle->offset += size;
        }
        else
        {
            if (size <= ptexHandle->bufferSize)
            {
                i64 readSize =
                    Min(ptexHandle->bufferSize, ptexHandle->fileSize - ptexHandle->offset);
                result =
                    OS_ReadFile(ptexHandle->osHandle, ptexHandle->buffer, readSize, offset)
                        ? size
                        : 0;
                MemoryCopy(buffer, ptexHandle->buffer, size);
            }
            else
            {
                result = OS_ReadFile(ptexHandle->osHandle, buffer, size, offset) ? size : 0;
            }
            ptexHandle->bufferOffset = offset;
            ptexHandle->offset += result;
        }
        Assert(result);
        return result;
    }

    /** Close a file.  Returns false if an error occurs, and the error
        string is available via lastError().  */
    virtual bool close(Handle handle) override
    {
        PtexHandle *h = (PtexHandle *)handle;
        bool result   = OS_CloseFile(h->osHandle);
        free((PtexHandle *)handle);
        return result;
    }
    virtual const char *lastError() override { return 0; }

    /** Return the last error message encountered. */
};
extern PtexInpHandler ptexInputHandler;

struct PtexTexture;
class Ptex::PtexTexture;

enum class TileType
{
    Corner,
    Edge,
    Center,
};

enum EdgeId
{
    e_bottom, ///< Bottom edge, from UV (0,0) to (1,0)
    e_right,  ///< Right edge, from UV (1,0) to (1,1)
    e_top,    ///< Top edge, from UV (1,1) to (0,1)
    e_left,   ///< Left edge, from UV (0,1) to (0,0)
    e_max,
};

struct Tile
{
    u8 *contents;
    TileType type;
    EdgeId edgeId;
    u32 parentFace;
};

struct TileMetadata
{
    u32 offset;
    int startLevel;
    int endLevel;

    int log2Width;
    int log2Height;
};

struct TileRequest
{
    int faceIndex;
    int startLevelIndex;
    int numLevels;
};

struct TileFileHeader
{
    int numFaces;
    int tileSizes[MAX_LEVEL];
    int tileCounts[MAX_LEVEL];
};

struct PaddedImage : Image
{
    // Does not include border
    int log2Width;
    int log2Height;
    int strideNoBorder;

    // Includes border
    int borderSize;
    int strideWithBorder;

    Vec2u ConvertRelativeToAbsolute(const Vec2u &p);
    u8 *GetContentsAbsoluteIndex(const Vec2u &p);
    u8 *GetContentsRelativeIndex(const Vec2u &p);

    u32 GetPaddedWidth() const;
    u32 GetPaddedHeight() const;
    void WriteRotatedBorder(PaddedImage &other, Vec2u srcStart, Vec2u dstStart, int edgeIndex,
                            int rotate, int srcVLen, int srcRowLen, int dstVLen, int dstRowLen,
                            Vec2u scale);
};

enum class AllocationStatus
{
    Free,
    Allocated,
    PartiallyAllocated,
};

struct BlockRange
{
    static const u32 InvalidRange = ~0u;
    AllocationStatus status;

    u32 start;
    u32 onePastEnd;

    u32 prevRange;
    u32 nextRange;

    u32 prevFree;
    u32 nextFree;

    BlockRange() {}

    BlockRange(AllocationStatus status, u32 start, u32 onePastEnd, u32 prevRange,
               u32 nextRange, u32 prevFree, u32 nextFree)
        : status(status), start(start), onePastEnd(onePastEnd), prevRange(prevRange),
          nextRange(nextRange), prevFree(prevFree), nextFree(nextFree)
    {
    }

    u32 GetNum() const;
    static u32 FindBestFree(const StaticArray<BlockRange> &ranges, u32 freeIndex, u32 num,
                            u32 leftover = ~0u);
    static void Split(StaticArray<BlockRange> &ranges, u32 index, u32 &freeIndex, u32 num);
};

// NOTE: quad tree node
struct AllocationNode
{
    AllocationStatus status;
    Vec2i start;
    Vec2i end;

    int firstSibling;
    int nextSibling;
    int firstChild;
    int parent;

    int prevFree;
    int nextFree;

    AllocationNode(AllocationStatus status, Vec2i start, Vec2i end, int firstSibling,
                   int nextSibling, int firstChild, int parent, int prevFree, int nextFree)
        : status(status), start(start), end(end), firstSibling(firstSibling),
          nextSibling(nextSibling), firstChild(firstChild), parent(parent), prevFree(prevFree),
          nextFree(nextFree)
    {
    }
};

struct PhysicalPagePool
{
    StaticArray<BlockRange> ranges;
    StaticArray<AllocationNode> nodes;

    u32 freeRange;
    u32 freePages;

    u32 freeNode;

    u32 prevFree;
    u32 nextFree;
};

struct PhysicalPageAllocation
{
    u32 virtualPage;
    u32 poolIndex;
    u32 pageIndex;
};

struct VirtualTextureManager
{
    static const u32 InvalidPool = ~0u;

    VkFormat format;

    u32 maxNumLayers;
    StaticArray<BlockRange> pageRanges;
    u32 freeRange;

    StaticArray<PhysicalPagePool> pools;
    u32 partiallyFreePool;
    u32 completelyFreePool;

    u32 pageWidthPerPool;

    struct LevelInfo
    {
        GPUImage gpuPhysicalPool;
        u32 texelWidthPerPage;

        // int pageTableSubresourceIndex;
    };
    StaticArray<LevelInfo> levelInfo;

    Shader shader;
    DescriptorSetLayout descriptorSetLayout;
    VkPipeline pipeline;
    GPUImage pageTable;
    PushConstant push;

    VirtualTextureManager(Arena *arena, u32 numVirtualPages, u32 numPhysicalPages,
                          int numLevels, u32 inPageWidthPerPool, u32 inTexelWidthPerPage,
                          u32 borderSize, VkFormat format);
    u32 AllocateVirtualPages(u32 numPages);
    void AllocatePhysicalPages(CommandBuffer *cmd, TileMetadata *metadata, u32 numFaces,
                               u8 *contents, u32 allocIndex);
};

void InitializePtex();
PaddedImage GenerateMips(Arena *arena, PaddedImage &input, u32 width, u32 height, Vec2u scale,
                         u32 borderSize);
string Convert(Arena *arena, PtexTexture *texture, int filterWidth = 4);
void Convert(string filename);
TileType GetTileType(int tileX, int tileY, int numTilesX, int numTilesY);
} // namespace rt
#endif
