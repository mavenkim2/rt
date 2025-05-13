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

struct FaceMetadata
{
    u32 bufferOffset;
    u32 totalSize_rotate;
    int log2Width;
    int log2Height;
};

struct FaceMetadata2
{
    u32 offsetX;
    u32 offsetY;
};

struct TileRequest
{
    int faceIndex;
    int startLevel;
    int numLevels;
};

struct TileRequest2
{
    Vec2u virtualPage;
    u32 virtualAddressIndex;
    u32 mipLevel;
};

struct TextureMetadata
{
    u32 numFaces;
    u32 sqrtNumPages;
    u32 mipPageOffsets[MAX_LEVEL];
};

struct TileFileHeader
{
    int numFaces;
    u32 numTilesX;
    u32 numTilesY;
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
    void WriteRotated(PaddedImage &other, Vec2u srcStart, Vec2u dstStart, int rotate,
                      int srcVLen, int srcRowLen, int dstVLen, int dstRowLen, Vec2u scale);
};

enum class AllocationStatus
{
    Free,
    Allocated,
    PartiallyAllocated,
};

struct PhysicalPagePool
{
    StaticArray<Vec2u> pagePool;
    u32 freePage;
};

struct PhysicalPageAllocation
{
    u32 virtualPage;
    u32 poolIndex;
    u32 pageIndex;
};

template <typename T>
struct RingBuffer
{
    Mutex mutex;
    StaticArray<T> entries;
    u64 readOffset;
    u64 writeOffset;

    u32 max;

    RingBuffer(Arena *arena, u32 max);
    bool Write(T *vals, u32 num);
    void WriteWithOverwrite(T *vals, u32 num);
    void SynchronizedWrite(Mutex *mutex, T *vals, u32 num);
    T *Read(Arena *arena, u32 &num);
    T *SynchronizedRead(Mutex *mutex, Arena *arena, u32 &num);
};

struct AllocationColumn
{
    u32 numPagesWide;
    u32 numPagesX;
    u32 currentPageHeight;
    u32 maxPageHeight;
};

struct VirtualTextureManager
{
    static const u32 InvalidPool = ~0u;

    struct RequestHandle
    {
        u8 sortKey;
        int requestIndex;
    };

    VkFormat format;

    StaticArray<AllocationColumn> allocationColumns;
    std::vector<Vec2u> baseVirtualPage;

    StaticArray<PhysicalPagePool> pools;
    u32 freePool;

    u32 pageWidthPerPool;
    GPUImage gpuPhysicalPool;
    GPUImage pageTable;

    Shader shader;
    DescriptorSetLayout descriptorSetLayout;
    VkPipeline pipeline;
    PushConstant push;

    // Streaming
    static const u32 numPendingSubmissions = 2;
    static const u32 maxUploadSize         = megabytes(512);
    static const u32 maxCopies             = 1u << 20u;
    static const u32 maxFeedback           = 32 * 1024 * 1024;

    // Clock replacement
    u32 clockPoolIndex;
    u32 clockPageIndex;

    // Double buffered, update once per virtual texture thread tick
    FixedArray<StaticArray<u8>, numPendingSubmissions> uploadBuffers;
    FixedArray<StaticArray<BufferImageCopy>, numPendingSubmissions> uploadCopyCommands;
    FixedArray<Semaphore, numPendingSubmissions> uploadSemaphores;
    std::atomic<u64> writeSubmission;

    // Double buffered, updated once per frame
    FixedArray<GPUBuffer, numPendingSubmissions> uploadDeviceBuffers;
    FixedArray<TransferBuffer, numPendingSubmissions> pageTableRequestBuffers;
    std::atomic<u64> readSubmission;

    Mutex updateRequestMutex;
    RingBuffer<PageTableUpdateRequest> updateRequestRingBuffer;

    Mutex feedbackMutex;
    FixedArray<TransferBuffer, 2> feedbackBuffers;
    RingBuffer<u32> feedbackRingBuffer;

    VirtualTextureManager(Arena *arena, u32 virtualTextureWidth, u32 virtualTextureHeight,
                          u32 physicalTextureWidth, u32 physicalTextureHeight, u32 numPools,
                          VkFormat format);
    Vec2u AllocateVirtualPages(const Vec2u &virtualSize, u32 &index);
    void AllocatePhysicalPages(CommandBuffer *cmd, u32 allocIndex, FaceMetadata *metadata,
                               u32 numFaces, u8 *contents);
    void AllocatePhysicalPages(CommandBuffer *cmd, u32 allocIndex, FaceMetadata *metadata,
                               u32 numFaces, u8 *contents, TileRequest *requests,
                               u32 numRequests, RequestHandle *handles);

    // Streaming
    // u32 Evict(StaticArray<PageTableUpdateRequest> &evictRequests,
    //         StaticArray<PageTableUpdateRequest> &mapRequests, u32 *feedbackRequests,
    //         u32 numRequests, u8 *uploadBuffer, u32 uploadOffset);
    // void WriteFeedback(TransferBuffer *transfer);
    // void Update(CommandBuffer *computeCmd, CommandBuffer *transferCmd);
    // void Callback();

    static PageTableUpdateRequest CreatePageTableUpdateRequest(int faceIndex, u32 x, u32 y,
                                                               u32 layer, int log2Width,
                                                               int log2Height, int startLevel,
                                                               bool rotate);
};

void InitializePtex();
PaddedImage GenerateMips(Arena *arena, PaddedImage &input, u32 width, u32 height, Vec2u scale,
                         u32 borderSize);
string Convert(Arena *arena, PtexTexture *texture, int filterWidth = 4);
void Convert(string filename);
TileType GetTileType(int tileX, int tileY, int numTilesX, int numTilesY);
} // namespace rt
#endif
