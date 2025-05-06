#include "../base.h"
#include "../memory.h"
#include "../bit_packing.h"
#include "../string.h"
#include "../radix_sort.h"
#include "../scene.h"
#include "../integrate.h"
#include "vulkan.h"
#include "ptex.h"
#include <Ptexture.h>
#include <PtexReader.h>
#include <cstring>
#include <thread>

namespace Utils
{
void Copy(const void *src, int sstride, void *dst, int dstride, int vres, int rowlen)
{
    // regular non-tiled case
    if (sstride == rowlen && dstride == rowlen)
    {
        // packed case - copy in single block
        MemoryCopy(dst, src, vres * rowlen);
    }
    else
    {
        // copy a row at a time
        const char *sptr = (const char *)src;
        char *dptr       = (char *)dst;
        for (const char *end = sptr + vres * sstride; sptr != end;)
        {
            MemoryCopy(dptr, sptr, rowlen);
            dptr += dstride;
            sptr += sstride;
        }
    }
}

// NOTE: absolute index
void *GetContentsAbsoluteIndex(void *contents, const Vec2u &p, u32 width, u32 height,
                               u32 stride, u32 bytesPerPixel)
{
    Assert(p.x < width && p.y < height);
    return (u8 *)contents + stride * p.y + p.x * bytesPerPixel;
}

void Copy(void *src, const Vec2u &srcIndex, u32 srcWidth, u32 srcHeight, void *dst,
          const Vec2u &dstIndex, u32 dstWidth, u32 dstHeight, u32 vRes, u32 rowLen,
          u32 bytesPerBlock)
{
    u32 srcStride = srcWidth * bytesPerBlock;
    u32 dstStride = dstWidth * bytesPerBlock;
    src = Utils::GetContentsAbsoluteIndex(src, srcIndex, srcWidth, srcHeight, srcStride,
                                          bytesPerBlock);
    dst = Utils::GetContentsAbsoluteIndex(dst, dstIndex, dstWidth, dstHeight, dstStride,
                                          bytesPerBlock);

    vRes   = Min(srcHeight, vRes);
    rowLen = Min(rowLen, srcStride);
    Copy(src, srcStride, dst, dstStride, vRes, rowLen);
}

// Zeroes out unfilled texels in a NxN region; used for block compression
void CopyAndPad(const void *src, int sstride, void *dst, int dstride, int vres, int rowlen,
                int minVres, int minRowlen)
{
    // regular non-tiled case
    if (sstride == rowlen && dstride == rowlen)
    {
        Assert(0);
        // packed case - copy in single block
        MemoryCopy(dst, src, vres * rowlen);
    }
    else
    {
        // copy a row at a time
        const char *sptr = (const char *)src;
        char *dptr       = (char *)dst;
        for (const char *end = sptr + vres * sstride; sptr != end;)
        {
            MemoryCopy(dptr, sptr, rowlen);
            if (minRowlen > rowlen)
            {
                MemoryZero(dptr + rowlen, minRowlen - rowlen);
            }
            dptr += dstride;
            sptr += sstride;
        }

        rowlen = Max(rowlen, minRowlen);
        for (int i = 0; i < minVres - vres; i++)
        {
            MemoryZero(dptr, rowlen);
            dptr += dstride;
        }
    }
}

} // namespace Utils

namespace rt
{

Ptex::PtexCache *cache;
PtexErrHandler errorHandler;
PtexInpHandler ptexInputHandler;

void InitializePtex()
{
    u32 maxFiles  = 400;
    size_t maxMem = gigabytes(8);
    cache = Ptex::PtexCache::create(maxFiles, maxMem, true, &ptexInputHandler, &errorHandler);
}

TileType GetTileType(int tileX, int tileY, int numTilesX, int numTilesY)
{

    if ((tileX == 0 && tileY == 0) || (tileX == numTilesX - 1 && tileY == 0) ||
        (tileX == 0 && tileY == numTilesY - 1) ||
        (tileX == numTilesX - 1 && tileY == numTilesY - 1))
    {
        return TileType::Corner;
    }
    else if (tileX == 0 || tileX == numTilesX - 1 || tileY == 0 || tileY == numTilesY - 1)
    {
        return TileType::Edge;
    }
    return TileType::Center;
}

Vec2u PaddedImage::ConvertRelativeToAbsolute(const Vec2u &p)
{
    return Vec2u(p.x + borderSize, p.y + borderSize);
}

// NOTE: absolute index
u8 *PaddedImage::GetContentsAbsoluteIndex(const Vec2u &p)
{
    return (u8 *)Utils::GetContentsAbsoluteIndex(
        contents, p, GetPaddedWidth(), GetPaddedHeight(), strideWithBorder, bytesPerPixel);
}

// NOTE: ignores border
u8 *PaddedImage::GetContentsRelativeIndex(const Vec2u &p)
{
    Assert(p.x < width && p.y < height);
    return GetContentsAbsoluteIndex(ConvertRelativeToAbsolute(p));
}

u32 PaddedImage::GetPaddedWidth() const { return width + 2 * borderSize; }

u32 PaddedImage::GetPaddedHeight() const { return height + 2 * borderSize; }

// NOTE: start is an absolute index
void PaddedImage::WriteRotated(PaddedImage &other, Vec2u srcStart, Vec2u dstStart, int rotate,
                               int srcVLen, int srcRowLen, int dstVLen, int dstRowLen,
                               Vec2u scale)
{
    int uStart = srcStart.x;
    int vStart = srcStart.y;

    Assert(bytesPerPixel == other.bytesPerPixel);
    Assert(borderSize == other.borderSize);

    u8 *src       = other.GetContentsAbsoluteIndex(srcStart);
    u32 srcStride = other.strideWithBorder;

    ScratchArena scratch;

    int pixelsToWriteX = 1 << scale.x;
    int pixelsToWriteY = 1 << scale.y;

    u32 size       = dstVLen * dstRowLen * bytesPerPixel;
    u8 *tempBuffer = PushArray(scratch.temp.arena, u8, size);
    for (int v = 0; v < srcVLen; v++)
    {
        for (int u = 0; u < srcRowLen; u++)
        {
            Vec2i dstAddress;
            switch (rotate)
            {
                case 0:
                {
                    dstAddress.x = u;
                    dstAddress.y = v;
                }
                break;
                case 1:
                {
                    dstAddress.x = srcVLen - 1 - v;
                    dstAddress.y = u;
                }
                break;
                case 2:
                {
                    dstAddress.x = srcRowLen - 1 - u;
                    dstAddress.y = srcVLen - 1 - v;
                }
                break;
                case 3:
                {
                    dstAddress.y = srcRowLen - 1 - u;
                    dstAddress.x = v;
                }
                break;
                default: Assert(0); break;
            }

            Vec2i dstPos(dstAddress);
            dstPos.x <<= scale.x;
            dstPos.y <<= scale.y;

            Vec2u srcPos(u, v);
            srcPos += srcStart;

            u32 offset = (dstPos.y * dstRowLen + dstPos.x) * bytesPerPixel;

            for (int y = 0; y < pixelsToWriteY; y++)
            {
                u32 dstOffset = offset + y * dstRowLen * bytesPerPixel;
                for (int x = 0; x < pixelsToWriteX; x++)
                {
                    Assert(dstOffset < size);
                    MemoryCopy(tempBuffer + dstOffset, other.GetContentsAbsoluteIndex(srcPos),
                               bytesPerPixel);
                    dstOffset += bytesPerPixel;
                }
            }
        }
    }

    Utils::Copy(tempBuffer, dstRowLen * bytesPerPixel, GetContentsAbsoluteIndex(dstStart),
                strideWithBorder, dstVLen, dstRowLen * bytesPerPixel);
}

PaddedImage PtexToImg(Arena *arena, Ptex::PtexTexture *ptx, int faceID, int borderSize)
{
    TempArena temp = ScratchStart(&arena, 1);
    Assert(faceID >= 0 && faceID < ptx->numFaces());

    u32 numChannels = ptx->numChannels();
    int aChan       = ptx->alphaChannel();

    Ptex::FaceInfo fi = ptx->getFaceInfo(faceID);

    int u = fi.res.u();
    int v = fi.res.v();

    Assert(IsPow2(u) && IsPow2(v));

    u32 bytesPerChannel = Ptex::DataSize(ptx->dataType());
    u32 bytesPerPixel   = numChannels * bytesPerChannel;
    int stride          = (u + 2 * borderSize) * bytesPerPixel;
    int size            = stride * (v + 2 * borderSize);
    u8 *data   = aChan == -1 ? PushArray(temp.arena, u8, size) : PushArray(arena, u8, size);
    int rowlen = (u * bytesPerPixel);
    // if (flip)
    // {
    //     data += rowlen * (img.h - 1);
    //     stride = -rowlen;
    // }

    PaddedImage result;
    result.contents         = data;
    result.width            = u;
    result.height           = v;
    result.log2Width        = fi.res.ulog2;
    result.log2Height       = fi.res.vlog2;
    result.bytesPerPixel    = bytesPerPixel;
    result.strideNoBorder   = rowlen;
    result.borderSize       = borderSize;
    result.strideWithBorder = stride;

    ptx->getData(faceID, (char *)result.GetContentsRelativeIndex({0, 0}), stride);

    // Convert image from 3 to 4 channels
    if (aChan == -1)
    {
        result.bytesPerPixel = (numChannels + 1) * Ptex::DataSize(ptx->dataType());
        stride               = (u + 2 * borderSize) * result.bytesPerPixel;
        size                 = stride * (v + 2 * borderSize);
        rowlen               = (u * result.bytesPerPixel);

        result.contents         = PushArray(arena, u8, size);
        result.strideNoBorder   = rowlen;
        result.strideWithBorder = stride;

        u32 dstOffset = 0;
        u32 srcOffset = 0;
        u8 alpha      = 255;
        for (int i = 0; i < result.GetPaddedWidth() * result.GetPaddedHeight(); i++)
        {
            MemoryCopy(result.contents + dstOffset, data + srcOffset, bytesPerPixel);
            MemorySet(result.contents + dstOffset + bytesPerPixel, alpha, bytesPerChannel);
            dstOffset += result.bytesPerPixel;
            srcOffset += bytesPerPixel;
        }
    }

    return result;
}

Vec4f GammaToLinear(const u8 rgb[4])
{
    return Vec4f(Pow(rgb[0] / 255.f, 2.2f), Pow(rgb[1] / 255.f, 2.2f),
                 Pow(rgb[2] / 255.f, 2.2f), rgb[3] / 255.f);
}

void LinearToGamma(const Vec4f &rgb, u8 *out)
{
    out[0] = u8(Clamp(Pow(rgb.x, 1 / 2.2f) * 255.f, 0.f, 255.f));
    out[1] = u8(Clamp(Pow(rgb.y, 1 / 2.2f) * 255.f, 0.f, 255.f));
    out[2] = u8(Clamp(Pow(rgb.z, 1 / 2.2f) * 255.f, 0.f, 255.f));
    out[3] = u8(Clamp(rgb.w * 255.f, 0.f, 255.f));
}

// steps:
// 1. write borders, including corners
//      - how do I handle extraordinary vertices?
// 2. block compress on gpu
// 3. group into tiles, write to disk
//      - how do I load the right tiles from disk at runtime?

// 4. load the right tiles from disk at runtime, keeping highest mip level always resident
// 5. create a page table texture
// 6. upload the tiles to the gpu, and update the page table texture
// 7. on gpu, look at page table texture to find right tiles. do the texture lookup + filter
// 8. if tile isn't found, populate feedback buffer
// 9. asynchronously read feedback buffer on cpu. go to step 4

// NOTE: scale = (2, 2) reduces in both dimension. scale = (2, 1) reduces only in u
PaddedImage GenerateMips(Arena *arena, PaddedImage &input, u32 width, u32 height, Vec2u scale,
                         u32 borderSize)
{
    PaddedImage output;
    output.width            = width;
    output.height           = height;
    output.log2Width        = Log2Int(width);
    output.log2Height       = Log2Int(height);
    output.bytesPerPixel    = input.bytesPerPixel;
    output.strideNoBorder   = width * input.bytesPerPixel;
    output.borderSize       = borderSize;
    output.strideWithBorder = (width + 2 * borderSize) * input.bytesPerPixel;
    output.contents =
        PushArray(arena, u8, output.strideWithBorder * (height + 2 * borderSize));

    for (u32 v = 0; v < height; v++)
    {
        for (u32 u = 0; u < width; u++)
        {
            Vec2u xy = scale * Vec2u(u, v);
            Vec2u zw = xy + scale - 1u;

            Vec4f topleft     = GammaToLinear(input.GetContentsRelativeIndex(xy));
            Vec4f topright    = GammaToLinear(input.GetContentsRelativeIndex({zw.x, xy.y}));
            Vec4f bottomleft  = GammaToLinear(input.GetContentsRelativeIndex({xy.x, zw.y}));
            Vec4f bottomright = GammaToLinear(input.GetContentsRelativeIndex({zw.x, zw.y}));

            Vec4f avg = (topleft + topright + bottomleft + bottomright) * .25f;

            LinearToGamma(avg, output.GetContentsRelativeIndex({u, v}));
        }
    }

    return output;
}

// TODO: explore a borderless solution if the borders cost too much
void Convert(string filename)
{
    ScratchArena scratch;
    string outFilename =
        PushStr8F(scratch.temp.arena, "%S.tiles", RemoveFileExtension(filename));

    // if (OS_FileExists(outFilename)) return;

    if (!Contains(outFilename, "mountainb0001_geo") || Contains(outFilename, "displacement"))
        return;

    const Vec2u uvTable[] = {
        Vec2u(0, 0),
        Vec2u(1, 0),
        Vec2u(1, 1),
        Vec2u(0, 1),
    };

    Ptex::String error;
    Ptex::PtexTexture *t     = cache->get((char *)filename.str, error);
    Ptex::PtexReader *reader = static_cast<Ptex::PtexReader *>(t);
    int numFaces             = reader->numFaces();

    const u32 bytesPerPixel    = 3;
    const u32 gpuBytesPerPixel = 4;
    const u32 borderSize       = 4;

    const u32 texelWidthPerPage      = 128;
    const u32 log2TexelWidth         = Log2Int(texelWidthPerPage);
    const u32 totalTexelWidthPerPage = 128 + 2 * borderSize;

    const VkFormat baseFormat    = VK_FORMAT_R8G8B8A8_UNORM;
    const VkFormat blockFormat   = VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    const u32 bytesPerTexel      = GetFormatSize(baseFormat);
    const u32 bytesPerBlock      = GetFormatSize(blockFormat);
    const u32 blockSize          = GetBlockSize(blockFormat);
    const u32 log2BlockSize      = Log2Int(blockSize);
    const u32 blocksPerPage      = texelWidthPerPage >> log2BlockSize;
    const u32 totalBlocksPerPage = totalTexelWidthPerPage >> log2BlockSize;

    const u32 blockBorderSize = borderSize >> log2BlockSize;

    static DescriptorSetLayout bcLayout = {};
    static bool initialized             = false;
    static int inputBinding             = 0;
    static int outputBinding            = 0;
    static VkPipeline pipeline;
    static Shader shader;
    if (!initialized)
    {
        string shaderName = "../src/shaders/block_compress.spv";
        string data       = OS_ReadFile(scratch.temp.arena, shaderName);
        shader            = device->CreateShader(ShaderStage::Compute, "block_compress", data);

        inputBinding =
            bcLayout.AddBinding(0, DescriptorType::SampledImage, VK_SHADER_STAGE_COMPUTE_BIT);
        outputBinding =
            bcLayout.AddBinding(1, DescriptorType::StorageImage, VK_SHADER_STAGE_COMPUTE_BIT);
        bcLayout.AddImmutableSamplers();
        pipeline    = device->CreateComputePipeline(&shader, &bcLayout);
        initialized = true;
    }

    // Create all face images and mip maps
    StaticArray<FixedArray<PaddedImage, MAX_LEVEL>> images(scratch.temp.arena, numFaces,
                                                           numFaces);

    struct FaceUploadInfo
    {
        int faceIndex;
        Vec2u srcDim;
        Vec2i offset;

        bool base;
    };

    FixedArray<StaticArray<FaceUploadInfo>, 2> faceUploads(2);
    faceUploads[0] = StaticArray<FaceUploadInfo>(scratch.temp.arena, numFaces * MAX_LEVEL);
    faceUploads[1] = StaticArray<FaceUploadInfo>(scratch.temp.arena, numFaces * MAX_LEVEL);

    StaticArray<TileMetadata> metaData(scratch.temp.arena, numFaces, numFaces);

    Arena **arenas = GetArenaArray(scratch.temp.arena);
    // Generate mips for faces
    ParallelFor(0, numFaces, 1024, [&](int jobID, int start, int count) {
        u32 threadIndex = GetThreadIndex();
        Arena *arena    = arenas[threadIndex];
        for (int i = start; i < start + count; i++)
        {
            PaddedImage img = PtexToImg(arena, t, i, borderSize);
            int levels      = Max(img.log2Width, img.log2Height) + 1;
            Assert(levels < MAX_LEVEL);

            images[i]    = FixedArray<PaddedImage, MAX_LEVEL>(levels);
            images[i][0] = img;

            // Generate mip maps
            Assert(IsPow2(img.width) && IsPow2(img.height));

            int width  = Max(img.width >> 1, 1);
            int height = Max(img.height >> 1, 1);

            int log2Width  = Max(img.log2Width - 1, 0);
            int log2Height = Max(img.log2Height - 1, 0);

            PaddedImage inPaddedImage = img;
            int depth                 = 1;

            Vec2u scale = 2u;

            while (depth < levels)
            {
                u32 mipBorderSize = GetBorderSize(log2Width, log2Height);
                PaddedImage outPaddedImage =
                    GenerateMips(arena, inPaddedImage, width, height, scale, mipBorderSize);

                images[i][depth] = outPaddedImage;
                inPaddedImage    = outPaddedImage;

                u32 prevWidth = width;
                width         = Max(width >> 1, 1);
                scale[0]      = prevWidth / width;

                u32 prevHeight = height;
                height         = Max(height >> 1, 1);
                scale[1]       = prevHeight / height;

                log2Width  = Max(log2Width - 1, 0);
                log2Height = Max(log2Height - 1, 0);

                depth++;
            }
        }
    });

    auto GetNeighborFaceImage = [&](PaddedImage &currentFaceImg, u32 neighborFace, u32 rot,
                                    Vec2u &scale) {
        Vec2i dstBaseSize(currentFaceImg.log2Width, currentFaceImg.log2Height);

        int neighborMaxLevel                   = images[neighborFace].Length();
        const PaddedImage &baseNeighborFaceImg = images[neighborFace][0];
        const Vec2i srcBaseSize(baseNeighborFaceImg.log2Width, baseNeighborFaceImg.log2Height);

        if (rot & 1) Swap(dstBaseSize[0], dstBaseSize[1]);
        Vec2i srcBaseDepth = srcBaseSize - dstBaseSize;
        int depthIndex = Clamp(Min(srcBaseDepth.x, srcBaseDepth.y), 0, neighborMaxLevel - 1);

        PaddedImage neighborFaceImg = images[neighborFace][depthIndex];
        u32 log2Width               = neighborFaceImg.log2Width;
        u32 log2Height              = neighborFaceImg.log2Height;

        u32 mipBorderSize = GetBorderSize(dstBaseSize.x, dstBaseSize.y);
        // reduce u
        if (log2Width > dstBaseSize.x)
        {
            for (int i = 0; i < log2Width - dstBaseSize.x; i++)
            {
                u32 width = neighborFaceImg.width;
                width >>= 1;
                neighborFaceImg = GenerateMips(scratch.temp.arena, neighborFaceImg, width,
                                               neighborFaceImg.height, {2, 1}, mipBorderSize);
            }
        }
        // reduce v
        else if (log2Height > dstBaseSize.y)
        {
            for (int i = 0; i < log2Height - dstBaseSize.y; i++)
            {
                u32 height = neighborFaceImg.height;
                height >>= 1;
                neighborFaceImg =
                    GenerateMips(scratch.temp.arena, neighborFaceImg, neighborFaceImg.width,
                                 height, {1, 2}, mipBorderSize);
            }
        }
        srcBaseDepth =
            Vec2i(neighborFaceImg.log2Width, neighborFaceImg.log2Height) - dstBaseSize;
        scale = Vec2u(Max(Vec2i(0), -srcBaseDepth));
        return neighborFaceImg;
    };

    const u32 flushSize = 256;
    u32 runningCount    = 0;

    // NOTE: pack multiple face textures/mips into one texture atlas that gets submitted to the
    // GPU
    const u32 gpuSubmissionWidth  = 4096;
    const u32 gpuSubmissionHeight = 4096;
    const u32 gpuOutputWidth      = gpuSubmissionWidth >> log2BlockSize;
    const u32 gpuOutputHeight     = gpuSubmissionHeight >> log2BlockSize;

    GPUImage gpuSrcImages[2];
    GPUImage uavImages[2];
    GPUBuffer submissionBuffers[2];
    void *mappedPtrs[2];
    GPUBuffer readbackBuffers[2];
    DescriptorSet descriptorSets[2];
    Semaphore semaphores[2];
    CommandBuffer *cmds[2];
    u32 numSubmissions  = 0;
    u32 submissionIndex = 0;

    // Allocate GPU resources
    {
        ImageDesc blockCompressedImageDesc(
            ImageType::Type2D, gpuSubmissionWidth, gpuSubmissionHeight, 1, 1, 1, baseFormat,
            MemoryUsage::GPU_ONLY,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_TILING_OPTIMAL);
        gpuSrcImages[0] = device->CreateImage(blockCompressedImageDesc);
        gpuSrcImages[1] = device->CreateImage(blockCompressedImageDesc);

        ImageDesc uavDesc(ImageType::Type2D, gpuOutputWidth, gpuOutputHeight, 1, 1, 1,
                          VK_FORMAT_R32G32_UINT, MemoryUsage::GPU_ONLY,
                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          VK_IMAGE_TILING_OPTIMAL);

        uavImages[0] = device->CreateImage(uavDesc);
        uavImages[1] = device->CreateImage(uavDesc);

        submissionBuffers[0] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    gpuSubmissionWidth * gpuSubmissionHeight *
                                                        GetFormatSize(baseFormat),
                                                    MemoryUsage::CPU_TO_GPU);
        submissionBuffers[1] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    gpuSubmissionWidth * gpuSubmissionHeight *
                                                        GetFormatSize(baseFormat),
                                                    MemoryUsage::CPU_TO_GPU);

        Assert(submissionBuffers[0].mappedPtr && submissionBuffers[1].mappedPtr);
        mappedPtrs[0] = submissionBuffers[0].mappedPtr;
        mappedPtrs[1] = submissionBuffers[1].mappedPtr;

        readbackBuffers[0] = device->CreateBuffer(
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpuOutputWidth * gpuOutputHeight * bytesPerBlock,
            MemoryUsage::GPU_TO_CPU);
        readbackBuffers[1] = device->CreateBuffer(
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpuOutputWidth * gpuOutputHeight * bytesPerBlock,
            MemoryUsage::GPU_TO_CPU);

        descriptorSets[0] = bcLayout.CreateNewDescriptorSet();
        descriptorSets[1] = bcLayout.CreateNewDescriptorSet();

        semaphores[0] = device->CreateSemaphore();
        semaphores[1] = device->CreateSemaphore();

        cmds[0] = device->BeginCommandBuffer(QueueType_Compute);
        cmds[0]->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        cmds[1] = device->BeginCommandBuffer(QueueType_Compute);
        cmds[1]->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    }

    struct FaceHandle
    {
        u8 sortKey;
        int faceIndex;
    };

    // Generate radix sort keys
    FaceHandle *handles = PushArrayNoZero(scratch.temp.arena, FaceHandle, numFaces);
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        PaddedImage &img = images[faceIndex][0];

        u32 key        = 0;
        u32 offset     = 0;
        u32 log2Width  = img.log2Width;
        u32 log2Height = img.log2Height;
        if (log2Height > log2Width)
        {
            Swap(log2Width, log2Height);
        }
        key                          = BitFieldPackU32(key, log2Width, offset, 4);
        key                          = BitFieldPackU32(key, log2Height, offset, 4);
        handles[faceIndex].sortKey   = (u8)key;
        handles[faceIndex].faceIndex = faceIndex;
    }

    // Not worth sorting if there aren't enough faces
    if (numFaces > 1000) SortHandles<FaceHandle, false>(handles, numFaces);

    int currentHorizontalOffset = 0;
    u32 currentShelfHeight      = 0;
    u32 totalHeight             = 0;

    StringBuilderMapped builder(outFilename);
    TileFileHeader header = {};
    header.numFaces       = numFaces;
    PutData(&builder, &header, sizeof(header));
    FaceMetadata *outFaceMetadata =
        (FaceMetadata *)AllocateSpace(&builder, sizeof(FaceMetadata) * numFaces);

    StaticArray<FaceMetadata> faceMetadata(scratch.temp.arena, numFaces, numFaces);

    auto CopyBlockCompressedResultsToDisk = [&]() {
        // Write to face images to disk
        if (numSubmissions > 0)
        {
            int lastSubmissionIndex   = (submissionIndex - 1) & 1;
            GPUBuffer *readbackBuffer = &readbackBuffers[lastSubmissionIndex];
            device->Wait(semaphores[lastSubmissionIndex]);

            descriptorSets[lastSubmissionIndex].Reset();

            // Loop through requests and copy out to disk
            for (auto &upload : faceUploads[lastSubmissionIndex])
            {
                Assert(!upload.base ||
                       builder.totalSize == faceMetadata[upload.faceIndex].bufferOffset);

                u32 faceStride = upload.srcDim.x * bytesPerBlock;
                u32 srcStride  = gpuOutputWidth * bytesPerBlock;

                void *src = Utils::GetContentsAbsoluteIndex(
                    readbackBuffer->mappedPtr, upload.offset, gpuOutputWidth, gpuOutputHeight,
                    srcStride, bytesPerBlock);

                void *dst = AllocateSpace(&builder, faceStride * upload.srcDim.y);
                Utils::Copy(src, srcStride, dst, faceStride, upload.srcDim.y, faceStride);
            }
            faceUploads[lastSubmissionIndex].size_ = 0;
        }
    };

    auto SubmitBlockCompressionCommandsToGPU = [&]() {
        // Submission currently packed image to GPU
        GPUImage *srcImage        = &gpuSrcImages[submissionIndex];
        GPUImage *uavImage        = &uavImages[submissionIndex];
        GPUBuffer *transferBuffer = &submissionBuffers[submissionIndex];
        GPUBuffer *readbackBuffer = &readbackBuffers[submissionIndex];
        DescriptorSet *set        = &descriptorSets[submissionIndex];
        CommandBuffer *cmd        = cmds[submissionIndex];

        BufferImageCopy copy = {};
        copy.layerCount      = 1;
        copy.extent          = Vec3u(gpuSubmissionWidth, gpuSubmissionHeight, 1);

        // Copy from buffer to src image
        {
            cmd->Barrier(srcImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            cmd->FlushBarriers();
            cmd->CopyImage(transferBuffer, srcImage, &copy, 1);
        }

        // Block compress
        {
            cmd->Barrier(srcImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            cmd->Barrier(uavImage, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
            cmd->FlushBarriers();

            set->Bind(inputBinding, srcImage);
            set->Bind(outputBinding, uavImage);
            cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, set,
                                    bcLayout.pipelineLayout);
            cmd->Dispatch((gpuSubmissionWidth + 7) >> 3, (gpuSubmissionHeight + 7) >> 3, 1);
        }

        // Copy from uav to buffer
        {
            copy.extent = Vec3u(gpuOutputWidth, gpuOutputHeight, 1);
            cmd->Barrier(uavImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            cmd->FlushBarriers();

            cmd->CopyImageToBuffer(readbackBuffer, uavImage, copy);
        }

        // Submit command buffer
        semaphores[submissionIndex].signalValue++;
        cmd->Signal(semaphores[submissionIndex]);
        device->SubmitCommandBuffer(cmds[submissionIndex]);
        cmds[submissionIndex] = device->BeginCommandBuffer(QueueType_Compute);
        cmds[submissionIndex]->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    };

    u32 totalSize = builder.totalSize;

    // Add borders to all images
    for (int handleIndex = 0; handleIndex < numFaces; handleIndex++)
    {
        FaceHandle &handle = handles[handleIndex];
        int faceIndex      = handle.faceIndex;

        const Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
        Assert(!f.isSubface());

        for (int levelIndex = 0; levelIndex < images[faceIndex].Length(); levelIndex++)
        {
            PaddedImage &currentFaceImg = images[faceIndex][levelIndex];

            if (currentHorizontalOffset + currentFaceImg.GetPaddedWidth() > gpuSubmissionWidth)
            {
                currentHorizontalOffset = 0;
                totalHeight += currentShelfHeight;
                currentShelfHeight = 0;
            }

            currentShelfHeight =
                Max(currentShelfHeight, AlignPow2(currentFaceImg.GetPaddedHeight(), 4u));

            if (currentShelfHeight + totalHeight > gpuSubmissionHeight)
            {
                CopyBlockCompressedResultsToDisk();
                SubmitBlockCompressionCommandsToGPU();

                numSubmissions++;
                submissionIndex         = numSubmissions & 1;
                currentHorizontalOffset = 0;
                totalHeight             = 0;
                currentShelfHeight      = AlignPow2(currentFaceImg.GetPaddedHeight(), 4u);
            }

            for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
            {
                // Add edge borders
                int aeid         = f.adjedge(edgeIndex);
                int neighborFace = f.adjface(edgeIndex);
                int rot          = (edgeIndex - aeid + 2) & 3;

                Vec2u scale;
                PaddedImage neighborFaceImg =
                    GetNeighborFaceImage(currentFaceImg, neighborFace, rot, scale);

                Vec2u start;
                int vRes;
                int rowLen;

                Assert(neighborFaceImg.borderSize == currentFaceImg.borderSize);
                u32 mipBorderSize = neighborFaceImg.borderSize;

                start.x = (edgeIndex == e_left ? 0 : mipBorderSize) +
                          (edgeIndex == e_right ? currentFaceImg.width : 0);
                start.y = (edgeIndex == e_bottom ? 0 : mipBorderSize) +
                          (edgeIndex == e_top ? currentFaceImg.height : 0);

                Vec2u srcStart;
                srcStart.x =
                    (aeid == e_right ? neighborFaceImg.width - (mipBorderSize >> scale.x)
                                     : 0) +
                    mipBorderSize;
                srcStart.y =
                    (aeid == e_top ? neighborFaceImg.height - (mipBorderSize >> scale.y) : 0) +
                    mipBorderSize;

                Assert(!(mipBorderSize == 1 && scale.y != 0 && scale.x != 0));
                int srcVRes = (aeid & 1) ? neighborFaceImg.height : (mipBorderSize >> scale.y);
                int srcRowLen =
                    (aeid & 1) ? (mipBorderSize >> scale.x) : neighborFaceImg.width;
                int dstVRes   = (edgeIndex & 1) ? currentFaceImg.height : mipBorderSize;
                int dstRowLen = (edgeIndex & 1) ? mipBorderSize : currentFaceImg.width;

                currentFaceImg.WriteRotated(neighborFaceImg, srcStart, start, rot, srcVRes,
                                            srcRowLen, dstVRes, dstRowLen,
                                            (rot & 1) ? scale.yx() : scale);

                // Add corner borders
                int afid                 = faceIndex;
                int cornerAeid           = edgeIndex;
                const Ptex::FaceInfo *af = &f;

                const int MaxValence = 10;
                int cfaceId[MaxValence];
                int cedgeId[MaxValence];
                const Ptex::FaceInfo *cface[MaxValence];

                int numCorners = 0;
                for (int i = 0; i < MaxValence; i++)
                {
                    int prevFace = afid;
                    afid         = af->adjface(cornerAeid);
                    cornerAeid   = (af->adjedge(cornerAeid) + 1) & 3;

                    if (afid < 0 || (afid == faceIndex && cornerAeid == edgeIndex))
                    {
                        numCorners = i - 2;
                        break;
                    }

                    af         = &reader->getFaceInfo(afid);
                    cfaceId[i] = afid;
                    cedgeId[i] = cornerAeid;
                    cface[i]   = af;

                    Assert(!af->isSubface());
                }

                if (numCorners == 1)
                {
                    int cornerRotate = (edgeIndex - cedgeId[1] + 2) & 3;

                    Vec2u cornerScale;
                    PaddedImage cornerImg = GetNeighborFaceImage(currentFaceImg, cfaceId[1],
                                                                 cornerRotate, cornerScale);

                    u32 cornerMipBorderSize = cornerImg.borderSize;
                    Assert(cornerMipBorderSize == currentFaceImg.borderSize);
                    Vec2u srcStart =
                        uvTable[cedgeId[1]] * Vec2u(cornerImg.width, cornerImg.height);
                    Vec2u dstStart = uvTable[edgeIndex] *
                                     Vec2u(currentFaceImg.width + cornerMipBorderSize,
                                           currentFaceImg.height + cornerMipBorderSize);

                    currentFaceImg.WriteRotated(cornerImg, srcStart, dstStart, cornerRotate,
                                                cornerMipBorderSize >> cornerScale.y,
                                                cornerMipBorderSize >> cornerScale.x,
                                                cornerMipBorderSize, cornerMipBorderSize,
                                                (cornerRotate & 1) ? cornerScale.yx()
                                                                   : cornerScale);
                }
                else if (numCorners > 1)
                {
                    // TODO: what do I do here?
                }
                else
                {
                }
            }

            u32 dstStride = gpuSubmissionWidth * GetFormatSize(baseFormat);
            u32 offset =
                dstStride * totalHeight + currentHorizontalOffset * GetFormatSize(baseFormat);
            Assert(offset <
                   gpuSubmissionWidth * gpuSubmissionHeight * GetFormatSize(baseFormat));

            // Rotate so that all textures are taller than wide
            PaddedImage img = currentFaceImg;
            bool rotate     = false;
            if (currentFaceImg.height > currentFaceImg.width)
            {
                u32 size = img.GetPaddedWidth() * img.GetPaddedHeight() * img.bytesPerPixel;
                img.contents = PushArray(scratch.temp.arena, u8, size);
                Swap(img.width, img.height);
                Swap(img.log2Width, img.log2Height);
                img.strideNoBorder   = img.width * img.bytesPerPixel;
                img.strideWithBorder = img.GetPaddedWidth() * img.bytesPerPixel;

                img.WriteRotated(currentFaceImg, Vec2u(0, 0), Vec2u(0, 0), 1,
                                 currentFaceImg.GetPaddedHeight(),
                                 currentFaceImg.GetPaddedWidth(),
                                 currentFaceImg.GetPaddedWidth(),
                                 currentFaceImg.GetPaddedHeight(), Vec2u(0, 0));
                rotate = true;
            }

            u32 alignedHeight = AlignPow2(img.GetPaddedHeight(), 4);
            u32 alignedWidth  = AlignPow2(img.GetPaddedWidth(), 4);
            u32 alignedRowLen = alignedWidth * GetFormatSize(baseFormat);
            Utils::CopyAndPad(img.contents, img.strideWithBorder,
                              (u8 *)mappedPtrs[submissionIndex] + offset, dstStride,
                              img.GetPaddedHeight(), img.strideWithBorder, alignedHeight,
                              alignedRowLen);

            Assert((currentHorizontalOffset & 3) == 0);
            Assert((totalHeight & 3) == 0);
            FaceUploadInfo upload;
            upload.faceIndex = faceIndex;
            upload.srcDim.x  = Max(alignedWidth >> log2BlockSize, 1u);
            upload.srcDim.y  = Max(alignedHeight >> log2BlockSize, 1u);
            upload.offset =
                Vec2i(currentHorizontalOffset >> log2BlockSize, totalHeight >> log2BlockSize);
            upload.base = levelIndex == 0;
            faceUploads[submissionIndex].Push(upload);

            u32 numBytes = upload.srcDim.x * upload.srcDim.y * bytesPerBlock;
            Assert((faceMetadata[faceIndex].totalSize_rotate & 0x7fffffff) + numBytes <
                   0x80000000);
            if (levelIndex == 0)
            {
                faceMetadata[faceIndex].bufferOffset = totalSize;
                faceMetadata[faceIndex].log2Width    = img.log2Width;
                faceMetadata[faceIndex].log2Height   = img.log2Height;
            }
            faceMetadata[faceIndex].totalSize_rotate |= (rotate << 31u);
            faceMetadata[faceIndex].totalSize_rotate += numBytes;

            totalSize += numBytes;
            currentHorizontalOffset += alignedWidth;
        }
    }

    CopyBlockCompressedResultsToDisk();
    SubmitBlockCompressionCommandsToGPU();

    numSubmissions++;
    submissionIndex = numSubmissions & 1;

    CopyBlockCompressedResultsToDisk();

    MemoryCopy(outFaceMetadata, faceMetadata.data, numFaces * sizeof(FaceMetadata));

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        device->DestroyImage(&gpuSrcImages[i]);
        device->DestroyImage(&uavImages[i]);
        device->DestroyBuffer(&submissionBuffers[i]);
        device->DestroyBuffer(&readbackBuffers[i]);
        device->DestroyPool(descriptorSets[i].pool);
    }

    // Output to disk
    OS_UnmapFile(builder.ptr);
    OS_ResizeFile(builder.filename, builder.totalSize);

    t->release();
}

VirtualTextureManager::VirtualTextureManager(Arena *arena, u32 numVirtualFaces,
                                             u32 physicalTextureWidth,
                                             u32 physicalTextureHeight, u32 numPools,
                                             VkFormat format)
    : format(format), updateRequestRingBuffer(arena, maxCopies)
{
    string shaderName = "../src/shaders/update_page_tables.spv";
    string data       = OS_ReadFile(arena, shaderName);
    shader            = device->CreateShader(ShaderStage::Compute, "update page tables", data);

    descriptorSetLayout = {};
    descriptorSetLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    descriptorSetLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    push     = PushConstant(ShaderStage::Compute, 0, sizeof(PageTableUpdatePushConstant));
    pipeline = device->CreateComputePipeline(&shader, &descriptorSetLayout, &push);

    ImageLimits limits = device->GetImageLimits();

    // Allocate page table
    pageTable = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     sizeof(Vec2u) * numVirtualFaces);

    {
        numPools = Min(numPools, limits.maxNumLayers);
        Assert(physicalTextureWidth <= limits.max2DImageDim);
        Assert(physicalTextureHeight <= limits.max2DImageDim);
        physicalTextureWidth  = Min(physicalTextureWidth, limits.max2DImageDim);
        physicalTextureHeight = Min(physicalTextureHeight, limits.max2DImageDim);

        // Allocate physical page pool
        pools = StaticArray<PhysicalPagePool>(arena, numPools);
        for (int i = 0; i < numPools; i++)
        {
            PhysicalPagePool pool;
            pool.maxWidth    = physicalTextureWidth;
            pool.maxHeight   = physicalTextureHeight;
            pool.totalHeight = 0;
            pool.layerIndex  = i;
            pool.shelfStarts = FixedArray<int, MAX_LEVEL>(MAX_LEVEL);
            for (int j = 0; j < MAX_LEVEL; j++)
            {
                pool.shelfStarts[j] = InvalidShelf;
            }
            pools.Push(pool);
        }

        // Allocate block ranges
        const u32 invalidRange = BlockRange::InvalidRange;
        pageRanges             = StaticArray<BlockRange>(arena, numVirtualFaces);
        pageRanges.Push(BlockRange(AllocationStatus::Free, 0, numVirtualFaces, invalidRange,
                                   invalidRange, invalidRange, invalidRange));

        // Allocate texture arrays
        ImageDesc poolDesc(ImageType::Array2D, physicalTextureWidth, physicalTextureHeight, 1,
                           1, numPools, format, MemoryUsage::GPU_ONLY,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        gpuPhysicalPool = device->CreateImage(poolDesc);
    }

    // Instantiate streaming system
    {
        writeSubmission.store(0);
        readSubmission.store(0);
        uploadBuffers =
            FixedArray<StaticArray<u8>, numPendingSubmissions>(numPendingSubmissions);
        uploadBuffers[0] = StaticArray<u8>(threadScratch.temp.arena, maxUploadSize);
        uploadBuffers[1] = StaticArray<u8>(threadScratch.temp.arena, maxUploadSize);

        uploadCopyCommands = FixedArray<StaticArray<BufferImageCopy>, numPendingSubmissions>(
            numPendingSubmissions);
        uploadCopyCommands[0] =
            StaticArray<BufferImageCopy>(threadScratch.temp.arena, maxCopies);
        uploadCopyCommands[1] =
            StaticArray<BufferImageCopy>(threadScratch.temp.arena, maxCopies);

        uploadDeviceBuffers =
            FixedArray<GPUBuffer, numPendingSubmissions>(numPendingSubmissions);
        uploadDeviceBuffers[0] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                      maxUploadSize, MemoryUsage::CPU_TO_GPU);
        uploadDeviceBuffers[1] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                      maxUploadSize, MemoryUsage::CPU_TO_GPU);

        pageTableRequestBuffers =
            FixedArray<TransferBuffer, numPendingSubmissions>(numPendingSubmissions);
        pageTableRequestBuffers[0] = device->GetStagingBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sizeof(PageTableUpdateRequest) * updateRequestRingBuffer.max);
        pageTableRequestBuffers[1] = device->GetStagingBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sizeof(PageTableUpdateRequest) * updateRequestRingBuffer.max);

        uploadSemaphores = FixedArray<Semaphore, numPendingSubmissions>(numPendingSubmissions);
        uploadSemaphores[0] = device->CreateSemaphore();
        uploadSemaphores[1] = device->CreateSemaphore();

        updateRequestMutex = {};
    }

    // callback
    Scheduler::Counter counter = {};
    scheduler.Schedule(&counter, [&](int jobID) { Callback(); });

    freeRange = 0;
}

template <typename Array>
u32 BlockRange::FindBestFree(const Array &ranges, u32 freeIndex, u32 num, u32 leftover)
{
    u32 allocIndex = ~0u;
    while (freeIndex != InvalidRange)
    {
        const BlockRange &range = ranges[freeIndex];
        u32 numFree             = range.GetNum();
        if (numFree >= num && numFree - num < leftover)
        {
            allocIndex = freeIndex;
            leftover   = numFree - num;
        }
        freeIndex = range.nextFree;
    }
    return allocIndex;
}

template <typename Array>
void BlockRange::Split(Array &ranges, u32 index, u32 &freeIndex, u32 num)
{
    BlockRange &range = ranges[index];

    u32 oldOnePastEnd = range.onePastEnd;
    u32 oldNextRange  = range.nextRange;
    u32 oldPrevFree   = range.prevFree;
    u32 oldNextFree   = range.nextFree;
    u32 newRangeIndex = ranges.size();

    Assert(range.status == AllocationStatus::Free);
    range.status     = AllocationStatus::Allocated;
    range.onePastEnd = range.start + num;

    if (range.onePastEnd != oldOnePastEnd)
    {
        range.nextRange = newRangeIndex;

        UnlinkFreeList(ranges, index, freeIndex, InvalidRange);

        BlockRange newRange(AllocationStatus::Free, range.start + num, oldOnePastEnd, index,
                            oldNextRange, oldPrevFree, oldNextFree);

        ranges.push_back(newRange);
        LinkFreeList(ranges, newRangeIndex, freeIndex, InvalidRange);
    }
    else
    {
        UnlinkFreeList(ranges, index, freeIndex, InvalidRange);
    }
}

u32 BlockRange::GetNum() const { return onePastEnd - start; }
u32 BlockRange::GetStartLevel() const
{
    return startLevel_requested & ~TopLevelMipRequestedBit;
}

u32 BlockRange::GetTopLevelWidth() const { return 0; }
// NOTE: Follows a clock replacement policy
bool BlockRange::CheckTopLevelMipRequested() const
{
    bool result = (startLevel_requested & TopLevelMipRequestedBit) != 0;
    return result;
}

void BlockRange::SetRangeAsRequested() { startLevel_requested |= TopLevelMipRequestedBit; }

PageTableUpdateRequest
VirtualTextureManager::CreatePageTableUpdateRequest(int faceIndex, u32 x, u32 y, u32 layer,
                                                    int log2Width, int log2Height,
                                                    int startLevel, bool rotate)
{
    u32 packed       = 0;
    u32 packedOffset = 0;
    packed           = BitFieldPackU32(packed, x, packedOffset, 15);
    packed           = BitFieldPackU32(packed, y, packedOffset, 15);
    packed           = BitFieldPackU32(packed, layer, packedOffset, 2);

    u32 packed2  = 0;
    packedOffset = 0;
    packed2      = BitFieldPackU32(packed2, log2Width, packedOffset, 4);
    packed2      = BitFieldPackU32(packed2, log2Height, packedOffset, 4);
    packed2      = BitFieldPackU32(packed2, startLevel, packedOffset, 4);
    packed2      = BitFieldPackU32(packed2, rotate, packedOffset, 1);

    PageTableUpdateRequest request;
    request.faceIndex                     = faceIndex;
    request.packed_x_y_layer              = packed;
    request.packed_width_height_baseLayer = packed2;
    return request;
}

u32 VirtualTextureManager::AllocateVirtualPages(u32 numPages)
{
    Assert(numPages);
    u32 freeIndex = freeRange;

    u32 leftover   = ~0u;
    u32 allocIndex = ~0u;

    allocIndex = BlockRange::FindBestFree(pageRanges, freeRange, numPages);
    Assert(allocIndex != ~0u);

    BlockRange::Split(pageRanges, allocIndex, freeRange, numPages);

    return allocIndex;
}

void VirtualTextureManager::AllocatePhysicalPages(CommandBuffer *cmd, u32 allocIndex,
                                                  FaceMetadata *metadata, u32 numFaces,
                                                  u8 *contents)
{
    ScratchArena scratch;

    TileRequest *requests = PushArrayNoZero(scratch.temp.arena, TileRequest, numFaces);
    RequestHandle *requestHandles =
        PushArrayNoZero(scratch.temp.arena, RequestHandle, numFaces);

    for (int i = 0; i < numFaces; i++)
    {
        requests[i].faceIndex  = i;
        requests[i].startLevel = 0;
        requests[i].numLevels  = Max(metadata[i].log2Width, metadata[i].log2Height) + 1;

        u32 key    = 0;
        u32 offset = 0;
        key        = BitFieldPackU32(key, metadata[i].log2Width, offset, 4);
        key        = BitFieldPackU32(key, metadata[i].log2Height, offset, 4);

        requestHandles[i].sortKey      = (u8)key;
        requestHandles[i].requestIndex = i;
    }
    SortHandles<RequestHandle, false>(requestHandles, numFaces);
    AllocatePhysicalPages(cmd, allocIndex, metadata, numFaces, contents, requests, numFaces,
                          requestHandles);
}

ShelfRequest VirtualTextureManager::AllocateShelf(Vec2i allocationSize, int currentLog2Height)
{
    u32 freePoolIndex      = 0;
    PhysicalPagePool *pool = &pools[freePoolIndex];
    u32 shelfStart         = pool->shelfStarts[currentLog2Height];

    ShelfRequest shelfRequest = {};

    u32 blockRangeIndex = BlockRange::InvalidRange;
    int shelfIndex      = -1;

    for (;;)
    {
        while (shelfStart == InvalidShelf)
        {
            if (pool->totalHeight + allocationSize.y > pool->maxHeight)
            {
                // Go to next free pool if there is no more space in current pool
                freePoolIndex++;

                // Must evict
                if (freePoolIndex >= pools.Length())
                {
                    shelfRequest.shelfIndex = -1;
                    return shelfRequest;
                }
                Assert(freePoolIndex < pools.Length());
                pool = &pools[freePoolIndex];
            }
            else
            {
                // Allocate a new row in the current pool
                shelves.emplace_back();
                Shelf &newRow    = shelves.back();
                newRow.startY    = pool->totalHeight;
                newRow.height    = allocationSize.y;
                newRow.freeRange = ranges.size();
                newRow.prevFree  = InvalidShelf;
                newRow.nextFree  = InvalidShelf;

                ranges.push_back(BlockRange(AllocationStatus::Free, 0, pool->maxWidth));
                LinkFreeList(shelves, shelves.size() - 1, pool->shelfStarts[currentLog2Height],
                             InvalidShelf);

                pool->totalHeight += allocationSize.y;
            }
            shelfStart = pool->shelfStarts[currentLog2Height];
        }

        // Find suitable span in row
        Shelf *shelf   = &shelves[shelfStart];
        u32 rangeIndex = BlockRange::FindBestFree(ranges, shelf->freeRange, allocationSize.x);

        if (rangeIndex == ~0u)
        {
            u32 leftover = ~0u;

            // First try to fit into rows with larger height
            for (int testDimIndex = faceMetadata.log2Height;
                 testDimIndex < pool->shelfStarts.Length(); testDimIndex++)
            {
                int freeShelf = pool->shelfStarts[testDimIndex];
                while (freeShelf != InvalidShelf)
                {
                    Shelf *testShelf   = &shelves[freeShelf];
                    u32 testRangeIndex = BlockRange::FindBestFree(ranges, testShelf->freeRange,
                                                                  allocationSize.x, leftover);
                    if (testRangeIndex != BlockRange::InvalidRange)
                    {
                        shelfIndex      = freeShelf;
                        blockRangeIndex = testRangeIndex;
                        leftover        = ranges[testRangeIndex].GetNum() - allocationSize.x;
                    }
                    freeShelf = testShelf->nextFree;
                }
            }

            // If a row was found that can fit the current entry
            if (shelfIndex != -1)
            {
                shelf = &shelves[shelfIndex];
                BlockRange::Split(ranges, blockRangeIndex, shelf->freeRange, allocationSize.x);
            }
            // Repeat the loop
            else
            {
                shelfStart = InvalidShelf;
                continue;
            }
        }
        else
        {
            BlockRange::Split(ranges, rangeIndex, shelf->freeRange, allocationSize.x);
            shelfIndex      = shelfStart;
            blockRangeIndex = rangeIndex;
        }
        break;
    }

    shelfRequest.shelfIndex      = shelfIndex;
    shelfRequest.blockRangeIndex = (int)blockRangeIndex;
    shelfRequest.layerIndex      = pool->layerIndex;

    return shelfRequest;
}

// Pack textures into shelves
void VirtualTextureManager::AllocatePhysicalPages(CommandBuffer *cmd, u32 allocIndex,
                                                  FaceMetadata *metadata, u32 numFaces,
                                                  u8 *contents, TileRequest *requests,
                                                  u32 numRequests, RequestHandle *handles)
{
    ScratchArena scratch;

    u32 baseFaceIndex = pageRanges[allocIndex].start;

    const u32 blockShift  = GetBlockShift(format);
    Shelf *currentRow     = 0;
    u32 currentLog2Height = ~0u;

    u32 bufferOffset = 0;

    StaticArray<PageTableUpdateRequest> updateRequests(scratch.temp.arena, numRequests,
                                                       numRequests);
    StaticArray<BufferImageCopy> copies(scratch.temp.arena, numRequests * MAX_LEVEL);

    for (int handleIndex = 0; handleIndex < numRequests; handleIndex++)
    {
        int requestIndex = handles[handleIndex].requestIndex;

        const TileRequest &request = requests[requestIndex];
        int faceIndex              = request.faceIndex;

        const FaceMetadata &faceMetadata = metadata[faceIndex];

        Vec2i allocationSize =
            CalculateFaceSize(faceMetadata.log2Width, faceMetadata.log2Height);

        // Allocation space for subsequent mip levels to the right
        allocationSize.x +=
            (1u << Max(faceMetadata.log2Width - 1, 0)) +
            2 * GetBorderSize(faceMetadata.log2Width - 1, faceMetadata.log2Height - 1);

        // First find the pool to use
        u32 freePoolIndex      = 0;
        PhysicalPagePool *pool = &pools[freePoolIndex];
        currentLog2Height      = faceMetadata.log2Height;

        // Allocate on a suitable shelf
        ShelfRequest shelfRequest = AllocateShelf(allocationSize, currentLog2Height);

        // Create the buffer image copy commands and page table update requests
        const Shelf &shelf           = shelves[shelfRequest.shelfIndex];
        const BlockRange &blockRange = ranges[shelfRequest.blockRangeIndex];

        updateRequests[handleIndex] = CreatePageTableUpdateRequest(
            baseFaceIndex + faceIndex, blockRange.start, shelf.startY, shelfRequest.layerIndex,
            faceMetadata.log2Width, faceMetadata.log2Height, request.startLevel,
            faceMetadata.totalSize_rotate >> 31);

        Vec3i start    = Vec3i(blockRange.start, shelf.startY, 0);
        Vec2i begin    = start.xy;
        int log2Width  = Max(faceMetadata.log2Width - request.startLevel, 0);
        int log2Height = Max(faceMetadata.log2Height - request.startLevel, 0);
        for (int levelIndex = 0; levelIndex < request.numLevels; levelIndex++)
        {
            Assert(start.xy - begin < allocationSize);
            BufferImageCopy copy = {};
            copy.bufferOffset    = bufferOffset;
            copy.baseLayer       = shelfRequest.layerIndex;
            copy.layerCount      = 1;
            copy.offset          = start;

            Vec2i extent = CalculateFaceSize(log2Width, log2Height);
            extent.x     = AlignPow2(extent.x, 4);
            extent.y     = AlignPow2(extent.y, 4);
            copy.extent  = Vec3u(extent.x, extent.y, 1);

            u32 index = Min(log2Width, log2Height) == 0 ? 0 : (levelIndex & 1);
            start[index] += extent[index];

            u32 texSize = Max(extent.y >> blockShift, 1) * Max(extent.x >> blockShift, 1) *
                          GetFormatSize(format);
            bufferOffset += texSize;
            log2Width  = Max(log2Width - 1, 0);
            log2Height = Max(log2Height - 1, 0);

            copies.Push(copy);
        }
    }

    GPUBuffer transferBuffer = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    bufferOffset, MemoryUsage::CPU_TO_GPU);
    Assert(transferBuffer.mappedPtr);
    u8 *mappedPtr = (u8 *)transferBuffer.mappedPtr;
    u32 dstOffset = 0;

    // Allocate staging buffer
    for (int handleIndex = 0; handleIndex < numRequests; handleIndex++)
    {
        int requestIndex = handles[handleIndex].requestIndex;

        const TileRequest &request = requests[requestIndex];
        int faceIndex              = request.faceIndex;

        const FaceMetadata &faceMetadata = metadata[faceIndex];

        Vec2u offsetAndSize = faceMetadata.CalculateOffsetAndSize(
            request.startLevel, blockShift, GetFormatSize(format));
        u32 offset = offsetAndSize.x;
        u32 size   = offsetAndSize.y;

        Assert(dstOffset + size <= bufferOffset);
        MemoryCopy(mappedPtr + dstOffset, contents + offset, size);
        dstOffset += size;
    }

    PageTableUpdatePushConstant pc;
    pc.numRequests = numRequests;

    cmd->CopyImage(&transferBuffer, &gpuPhysicalPool, copies.data, copies.Length());
    TransferBuffer pageTableUpdateRequestBuffer =
        cmd->SubmitBuffer(updateRequests.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          sizeof(PageTableUpdateRequest) * pc.numRequests);

    DescriptorSet ds = descriptorSetLayout.CreateDescriptorSet();
    ds.Bind(0, &pageTableUpdateRequestBuffer.buffer).Bind(1, &pageTable);
    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmd->PushConstants(&push, &pc, descriptorSetLayout.pipelineLayout);
    cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                            descriptorSetLayout.pipelineLayout);
    cmd->Dispatch((numRequests + 63) >> 6, 1, 1);
}

u32 VirtualTextureManager::Evict(StaticArray<PageTableUpdateRequest> &evictRequests,
                                 StaticArray<PageTableUpdateRequest> &mapRequests,
                                 StaticArray<BufferImageCopy> &copies, u32 *feedbackRequests,
                                 u32 numRequests, u8 *uploadBuffer, u32 uploadOffset)
{
    ScratchArena scratch;
    // Clock replacement policy; ranges that will have their request bit reset, losing their
    // "first chance"
    // StaticArray<u32> requestedRangeIndices(scratch.temp.arena, numRequests)

    for (int i = 0; i < numRequests; i++)
    {
        u32 feedback         = feedbackRequests[i];
        u32 virtualFaceIndex = feedbackRequest & 0x0fffffff;
        u32 mipLevel         = feedbackRequest >> 28;

        int shelfIndex = -1;
        while (shelfIndex != InvalidShelf)
        {
            Shelf &shelf   = shelves[shelfIndex];
            u32 rangeIndex = shelf.rangeStart;
            while (rangeIndex != BlockRange::InvalidRange)
            {
                BlockRange &range = ranges[rangeIndex];
                u32 topLevelWidth = = range.GetTopLevelWidth();
                bool fits           = topLevelWidth >= requestedWidth;
                if (fits && !range.CheckTopLevelMipRequested())
                {
                    // Split the range we're evicting
                    BlockRange newRange(AllocationStatus::Allocated, range.start,
                                        range.start + topLevelWidth, range.prevRange,
                                        rangeIndex, BlockRange::InvalidRange,
                                        BlockRange::InvalidRange);

                    ranges.push_back(newRange);
                    range.start += topLevelWidth;
                    range.prevRange = ranges.size() - 1;

                    int faceIndex = requests[i].faceIndex;

                    // Update page table information for entries we're evicting
                    range.startLevel_requested++;
                    int startLevel                      = range.GetStartLevel();
                    PageTableUpdateRequest evictRequest = CreatePageTableUpdateRequest(
                        virtualFaceIndex, range.start, shelf.startY, pool->layerIndex,
                        range.log2Width, range.log2Height, startLevel,
                        range.totalSize_rotate >> 31);
                    evictRequests.push_back(evictRequest);

                    // Issue copy commands to new location
                    Vec3i start(newRange.start, shelf.startY, 0);
                    CreateBufferImageCopies(copies, start, int numLevels, int layerIndex,
                                            int log2Width, int log2Height, u32 texelSize,
                                            u32 &bufferOffset);
                    int oldRangeIndex = virtualFaceIndexToRangeIndex[virtualFaceIndex];
                    // Remap the range previously associated with this data if present
                    if (oldRangeIndex != -1)
                    {
                        ranges[oldRangeIndex].allocationStatus = AllocationStatus::Free;
                        LinkFreeList(ranges, oldRangeIndex, shelf.freeRange,
                                     BlockRange::InvalidRange);
                        virtualFaceIndexToRangeIndex[virtualFaceIndex] = ranges.size() - 1;
                    }
                    virtualFaceInfo[virtualFaceIndex].startLevel_requested =
                        mipLevel | BlockRange::TopLevelMipRequestedBit;
                    // Update page table information for entries we're mapping
                    PageTableUpdateRequest mapRequest = CreatePageTableUpdateRequest(
                        virtualFaceIndex, newRange.start, shelf.startY, pool->layerIndex,
                        int log2Width, int log2Height, requests[i].startLevel, bool rotate);
                    mapRequests.push_back(mapRequest);
                    break;
                }
                rangeIndex = range.nextRange;
            }
        }
    }

    // Update page table information to prepare for eviction

    // Issue copy commands to new location
    cmd->CopyImage(&transferBuffer, &gpuPhysicalPool, copies.data, copies.Length());

    // Update page table information for entries we're mapping
    pc.numRequests = mapRequests.Length();
    return uploadOffset;
}

///////////////////////////////////////
// Streaming/Feedback
//

template <typename T>
RingBuffer<T>::RingBuffer(Arena *arena, u32 max) : max(max)
{
    entries = StaticArray<T>(arena, max);
    // mutex       = {};
    readOffset  = 0;
    writeOffset = 0;
}

template <typename T>
bool RingBuffer<T>::Write(T *vals, u32 num)
{
    bool result  = true;
    u32 capacity = entries.capacity;
    if (writeOffset + num >= readOffset + capacity)
    {
        result = false;
    }
    else
    {
        u32 writeIndex = entries.size() & (capacity - 1);
        u32 numToEnd   = capacity - writeIndex;
        MemoryCopy(entries.data + writeIndex, vals, sizeof(T) * Min(num, numToEnd));
        if (num > numToEnd)
        {
            MemoryCopy(entries.data, vals + numToEnd, sizeof(T) * (num - numToEnd));
        }
        writeOffset += num;
    }

    return result;
}

template <typename T>
void RingBuffer<T>::WriteWithOverwrite(T *vals, u32 num)
{
    u32 capacity = entries.capacity;
    if (writeOffset + num >= readOffset + capacity)
    {
        readOffset = writeOffset + num - capacity;
    }
    u32 writeIndex = entries.size() & (capacity - 1);
    u32 numToEnd   = capacity - writeIndex;
    MemoryCopy(entries.data + writeIndex, vals, sizeof(T) * Min(num, numToEnd));
    if (num > numToEnd)
    {
        MemoryCopy(entries.data, vals + numToEnd, sizeof(T) * (num - numToEnd));
    }
    writeOffset += num;

    return result;
}

template <typename T>
void RingBuffer<T>::SynchronizedWrite(Mutex *mutex, T *vals, u32 num)
{
    for (;;)
    {
        BeginMutex(mutex);
        bool result = Write(vals, num);

        if (result)
        {
            EndMutex(mutex);
            break;
        }

        EndMutex(mutex);

        std::this_thread::yield();
    }
}

template <typename T>
T *RingBuffer<T>::Read(Arena *arena, u32 &num)
{
    // BeginMutex(&mutex);
    Assert(readOffset <= writeOffset);
    u32 numToRead = writeOffset - readOffset;
    u32 capacity  = entries.capacity;
    T *vals       = 0;
    if (numToRead)
    {
        vals          = (T *)PushArrayNoZero(arena, u8, sizeof(T) * numToRead);
        u32 readIndex = readOffset & (capacity - 1);
        u32 numToEnd  = capacity - readIndex;
        MemoryCopy(vals, entries.data + readIndex, sizeof(T) * Min(numToRead, numToEnd));
        if (numToRead > numToEnd)
        {
            MemoryCopy(vals + numToEnd, entries.data, sizeof(T) * numToRead - numToEnd);
        }
        readOffset = writeOffset;
    }
    num = numToRead;
    // EndMutex(&mutex);
    return vals;
}

template <typename T>
T *RingBuffer<T>::SynchronizedRead(Mutex *mutex, Arena *arena, u32 &num)
{
    BeginMutex(mutex);
    T *result = Read(arena, num);
    EndMutex(mutex);
    return result;
}

// Executes on main thread
void VirtualTextureManager::Update(CommandBuffer *computeCmd, CommandBuffer *transferCmd)
{
    // Update page table
    ScratchArena scratch;

    u32 currentBuffer = device->GetCurrentBuffer();

    // Write last frame's feedback buffer
    if (device->frameCount >= 2)
    {
        u32 numFeedbackRequests = 
            ((u32 *)feedbackBuffers[currentBuffer].mappedPtr)[0]);
        u32 *feedbackRequests = PushArrayNoZero(scratch.temp.arena, u32, numFeedbackRequests);

        BeginMutex(&feedbackMutex);
        feedbackRingBuffer.WriteWithOverwrite(
            (u32 *)feedbackBuffers[currentBuffer].mappedPtr + 1, numFeedbackRequests);
        EndMutex(&feedbackMutex);
    }

    u32 numRequests = 0;
    PageTableUpdateRequest *updateRequests =
        virtualTextureManager.updateRequestRingBuffer.SynchronizedRead(
            virtualTextureManager.updateRequestMutex, scratch.temp.arena, numRequests);

    if (numRequests)
    {
        Assert(updateRequests);

        TransferBuffer *pageTableRequestBuffer = &pageTableRequestBuffers[currentBuffer];
        MemoryCopy(pageTableRequestBuffer->mappedPtr, updateRequests,
                   sizeof(PageTableUpdateRequest) * numRequests);
        computeCmd->SubmitTransfer(pageTableRequestBuffer);

        // WAR for page table (only execution dependency), RAW for transfer buffer
        // (memory + execution dependency)
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        computeCmd->FlushBarriers();

        PageTableUpdatePushConstant pc;
        pc.numRequests = numRequests;

        DescriptorSet updatePageTableDescriptorSet = descriptorSetLayout.CreateDescriptorSet();
        updatePageTableDescriptorSet.Bind(0, &pageTableRequestBuffer->buffer)
            .Bind(1, &pageTable);
        computeCmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        cmd->PushConstants(&push, &pc, descriptorSetLayout.pipelineLayout);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &updatePageTableDescriptorSet,
                                descriptorSetLayout.pipelineLayout);
        computeCmd->Dispatch(numRequests >> 6, 1, 1);

        // RAW for page table
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->FlushBarriers();
    }

    u64 readIndex = readSubmission.load(std::memory_order_relaxed);

    for (;;)
    {
        u64 writeIndex = writeSubmission.load(std::memory_order_acquire);
        if (readIndex < writeIndex)
        {
            break;
        }
    }

    // Update physical texture with new entries
    u32 readDoubleBufferIndex = readIndex & 1;

    StaticArray<BufferImageCopy> &writtenCopyCommands =
        uploadCopyCommands[readDoubleBufferIndex];
    u32 numCopies = writtenCopyCommands.size();

    if (numCopies)
    {
        StaticArray<u8> &uploadBuffer = uploadBuffers[readDoubleBufferIndex];
        // Transition physical texture to allow copies
        transferCmd->Barrier(&gpuPhysicalPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        transferCmd->FlushBarriers();

        // Copy data to staging buffer
        GPUBuffer *uploadDeviceBuffer = &uploadDeviceBuffers[currentBuffer];
        MemoryCopy(uploadDeviceBuffer->mappedPtr, uploadBuffer.data, uploadBuffer.size());
        transferCmd->CopyImage(uploadDeviceBuffer, &gpuPhysicalPool, writtenCopyCommands.data,
                               numCopies);

        // Transition physical texture to allow reads
        transferCmd->Barrier(&gpuPhysicalPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_READ_BIT);
        transferCmd->FlushBarriers();

        // Synchronize
        uploadSemaphores[currentBuffer].signalValue++;
        computeCmd->Signal(uploadSemaphores[currentBuffer]);
        transferCmd->Wait(uploadSemaphores[currentBuffer]);
    }

    // Make sure all memory writes are visible
    readSubmission.store(readIndex + 1, std::memory_order_release);
}

// Executes on virtual texture thread
void VirtualTextureManager::Callback()
{
    struct SortKey
    {
        u32 sortKey;
        u32 requestIndex;
    };

    const u32 blockShift    = GetBlockShift(format);
    const u32 bytesPerBlock = GetFormatSize(format);

    ScratchArena threadScratch;

    for (;;)
    {
        ScratchArena scratch;
        // Get feedback requests

        u32 numFeedbackRequests = 0;
        u32 *feedbackRequests   = feedbackRingBuffer.SynchronizedRead(
            &feedbackMutex, scratch.temp.arena, numFeedbackRequests);

        SortKey *keys = PushArrayNoZero(scratch.temp.arena, SortKey, numFeedbackRequests);
        u32 numKeys   = 0;

        // TODO: should data already be compacted or not?

        // Sort requests based on diff between requested mip level and resident mip level
        for (int requestIndex = 0; requestIndex < numFeedbackRequests; requestIndex++)
        {
            u32 virtualFaceIndex = feedbackRequests[requestIndex] & 0x0fffffff;
            u32 mipLevel         = feedbackRequests[requestIndex] >> 28;

            // The larger the difference between top resident mip and requested mip, the higher
            // the priority
            int rangeIndex    = virtualFaceIndexToRangeIndex[virtualFaceIndex];
            BlockRange &range = ranges[rangeIndex];
            int startLevel    = range.GetStartLevel();
            range.SetRangeAsRequested();

            int diff = startLevel - mipLevel;

            int materialIndex = ? ;
            int faceIndex     = ? ;

            if (diff > 0)
            {
                FaceMetadata &f = faceMetadata[materialIndex][faceIndex];
                u32 key         = 0;
                u32 offset      = 0;
                key             = BitFieldPackU32(key, f.log2Width, offset, 4);
                key             = BitFieldPackU32(key, f.log2Height, offset, 4);
                key             = BitFieldPackU32(key, materialIndex, offset, 16);
                key             = BitFieldPackU32(key, diff, offset, 4);

                keys[numKeys].sortKey      = key;
                keys[numKeys].requestIndex = requestIndex;
                numKeys++;
            }
        }

        SortHandles<SortKey, false>(keys, numKeys);

        StaticArray<BufferImageCopy> copies(scratch.temp.arena,
                                            numFeedbackRequests * MAX_LEVEL);

        StaticArray<u32> feedbackRequiringEvict(scratch.temp.arena, numFeedbackRequests);
        StaticArray<PageTableUpdateRequest> evictRequests(scratch.temp.arena,
                                                          numFeedbackRequests);

        StaticArray<PageTableUpdateRequest> mapRequests(scratch.temp.arena,
                                                        numFeedbackRequests);

        // Wait until the transfer buffer is read on the GPU before writing new data
        u64 writeIndex = writeSubmission.load(std::memory_order_relaxed);
        for (;;)
        {
            u64 readIndex = readSubmission.load(std::memory_order_acquire);
            if (readIndex > writeIndex - numPendingSubmissions)
            {
                break;
            }
            std::this_thread::yield();
        }

        u32 writeDoubleBufferIndex = writeIndex & 1;

        // Load face data from disk/cache
        u8 *uploadBuffer  = uploadBuffers[writeDoubleBufferIndex].data;
        u32 uploadOffset  = 0;
        bool bufferIsFull = false;
        for (int keyIndex = 0; keyIndex < numKeys; keyIndex++)
        {
            int feedbackRequestIndex = keys[keyIndex].feedbackRequestIndex;
            u32 feedbackRequest      = feedbackRequests[feedbackRequestIndex];
            u32 virtualFaceIndex     = feedbackRequest & 0x0fffffff;
            u32 mipLevel             = feedbackRequest >> 28;

            int materialIndex = ? ;
            string filename   = ? ;

            FaceMetadata metadata;

            Vec2u offsetAndSize =
                metadata.CalculateOffsetAndSize(mipLevel, blockShift, bytesPerBlock);

            Vec2i allocationSize = CalculateFaceSize(metadata.log2Width, metadata.log2Height);

            // Allocation space for subsequent mip levels to the right
            allocationSize.x +=
                (1u << Max(metadata.log2Width - 1, 0)) +
                2 * GetBorderSize(metadata.log2Width - 1, metadata.log2Height - 1);

            int log2Width  = Max(metadata.log2Width - mipLevel, 0);
            int log2Height = Max(metadata.log2Height - mipLevel, 0);
            int numLevels  = Max(log2Width, log2Height) + 1;

            if (offsetAndSize.y + uploadOffset > maxUploadSize ||
                copies.size() + numLevels > maxCopies)
            {
                bufferIsFull = true;
                break;
            }

            ShelfRequest shelfRequest = AllocateShelf(allocationSize, log2Height);

            // Save for eviction later
            if (shelfRequest.shelfIndex == -1)
            {
                feedbackRequiringEvict.Push(feedbackRequest);
                continue;
            }

            // Create buffer to image copy commands
            const Shelf &shelf           = shelves[shelfRequest.shelfIndex];
            const BlockRange &blockRange = ranges[shelfRequest.blockRangeIndex];

            Vec3i start = Vec3i(blockRange.start, shelf.startY, 0);
            CreateBufferImageCopies(copies, start, numLevels, pool->layerIndex, log2Width,
                                    log2Height, GetFormatSize(format), bufferOffset);

            // TODO: cache
            // Load from disk and write to buffer
            OS_Handle handle = OS_CreateFile(filename);
            bool result      = OS_ReadFile(handle, uploadBuffer + uploadOffset + uploadOffset,
                                           offsetAndSize.y, offsetAndSize.x);
            Assert(result);

            OS_CloseFile(handle);
            uploadOffset += offsetAndSize.y;

            // Write map request
        }

        // Evict
        if (evictRequests.size() && !bufferIsFull)
        {
            uploadOffset =
                Evict(evictRequests, mapRequests, copies, feedbackRequiringEvict.data,
                      feedbackRequiringEvict.size(), uploadBuffer, uploadOffset);
        }

        // Write evict requests if present
        if (evictRequests.size())
        {
            updateRequestRingBuffer.SynchronizedWrite(&updateRequestMutex, evictRequests.data,
                                                      evictRequests.size());
        }

        // Write copy commands
        uploadBuffers[writeDoubleBufferIndex].size_ = uploadOffset;
        MemoryCopy(uploadCopyCommands[writeDoubleBufferIndex], copies.data,
                   sizeof(BufferImageCopy) * copies.size());
        uploadCopyCommands[writeDoubleBufferIndex].size_ = copies.size();

        writeSubmission.store(writeIndex + 1, std::memory_order_release);

        // Write map requests
        if (mapRequests.size())
        {
            updateRequestRingBuffer.SynchronizedWrite(&updateRequestMutex, mapRequests.data,
                                                      mapRequests.size());
        }
    }
}

Vec2u FaceMetadata::CalculateOffsetAndSize(u32 mipLevel, u32 blockShift, u32 bytesPerBlock)
{
    int currentLog2Width  = Max(log2Width - (int)mipLevel, 0);
    int currentLog2Height = Max(log2Height - (int)mipLevel, 0);

    Vec2u offsetAndSize(bufferOffset, totalSize_rotate & 0x7fffffff);
    for (int levelIndex = 0; levelIndex < mipLevel; levelIndex++)
    {
        Vec2i extent = CalculateFaceSize(currentLog2Width, currentLog2Height);
        extent.x     = AlignPow2(extent.x, 4);
        extent.y     = AlignPow2(extent.y, 4);
        u32 texSize =
            Max(extent.y >> blockShift, 1) * Max(extent.x >> blockShift, 1) * bytesPerBlock;
        offsetAndSize.x += texSize;
        offsetAndSize.y -= texSize;
        currentLog2Width  = Max(currentLog2Width - 1, 0);
        currentLog2Height = Max(currentLog2Height - 1, 0);
    }
}

void VirtualTextureManager::CreateBufferImageCopies(StaticArray<BufferImageCopy> &copies,
                                                    Vec3i start, int numLevels, int layerIndex,
                                                    int log2Width, int log2Height,
                                                    u32 texelSize, u32 &bufferOffset)
{
    Vec2i begin = start.xy;
    for (int levelIndex = 0; levelIndex < numLevels; levelIndex++)
    {
        Assert(start.xy - begin < allocationSize);
        BufferImageCopy copy = {};
        copy.bufferOffset    = bufferOffset;
        copy.baseLayer       = layerIndex;
        copy.layerCount      = 1;
        copy.offset          = start;

        Vec2i extent = CalculateFaceSize(log2Width, log2Height);
        extent.x     = AlignPow2(extent.x, 4);
        extent.y     = AlignPow2(extent.y, 4);
        copy.extent  = Vec3u(extent.x, extent.y, 1);

        u32 index = Min(log2Width, log2Height) == 0 ? 0 : (levelIndex & 1);
        start[index] += extent[index];

        u32 texSize =
            Max(extent.y >> blockShift, 1) * Max(extent.x >> blockShift, 1) * texelSize;
        bufferOffset += texSize;
        log2Width  = Max(log2Width - 1, 0);
        log2Height = Max(log2Height - 1, 0);

        copies.Push(copy);
    }
}

} // namespace rt
