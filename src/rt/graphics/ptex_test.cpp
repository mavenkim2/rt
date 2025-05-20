#include "../base.h"
#include "../containers.h"
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

// NOTE: all values in terms of blocks/texels, not bytes
void Copy(void *src, const Vec2u &srcIndex, u32 srcWidth, u32 srcHeight, void *dst,
          const Vec2u &dstIndex, u32 dstWidth, u32 dstHeight, u32 vRes, u32 rowLen,
          u32 bytesPerBlock)
{
    u32 srcStride = srcWidth * bytesPerBlock;
    u32 dstStride = dstWidth * bytesPerBlock;
    rowLen *= bytesPerBlock;
    src = Utils::GetContentsAbsoluteIndex(src, srcIndex, srcWidth, srcHeight, srcStride,
                                          bytesPerBlock);
    dst = Utils::GetContentsAbsoluteIndex(dst, dstIndex, dstWidth, dstHeight, dstStride,
                                          bytesPerBlock);

    vRes   = Min(srcHeight, vRes);
    rowLen = Min(rowLen, srcStride);
    Copy(src, srcStride, dst, dstStride, vRes, rowLen);
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

    Arena **arenas = GetArenaArray(scratch.temp.arena);
    // Generate mips for faces
    ParallelFor(0, numFaces, 1024, [&](int jobID, int start, int count) {
        u32 threadIndex = GetThreadIndex();
        Arena *arena    = arenas[threadIndex];
        for (int i = start; i < start + count; i++)
        {
            PaddedImage img = PtexToImg(arena, t, i, 0);
            Assert(img.log2Width >= 2 && img.log2Height >= 2);
            int levels = Max(img.log2Width, img.log2Height) + 1;
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
                PaddedImage outPaddedImage =
                    GenerateMips(arena, inPaddedImage, width, height, scale, 0);

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
    u8 *debugSrc[2];
    GPUBuffer readbackBuffers[2];
    DescriptorSet descriptorSets[2];
    Semaphore semaphores[2];
    CommandBuffer *cmds[2];
    u32 numSubmissions  = 0;
    u32 submissionIndex = 0;

    const u32 submissionSize =
        gpuSubmissionWidth * gpuSubmissionHeight * GetFormatSize(baseFormat);
    const u32 outputSize = gpuOutputWidth * gpuOutputHeight * GetFormatSize(blockFormat);
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
                                                    submissionSize, MemoryUsage::CPU_TO_GPU);
        submissionBuffers[1] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    submissionSize, MemoryUsage::CPU_TO_GPU);

        Assert(submissionBuffers[0].mappedPtr && submissionBuffers[1].mappedPtr);
        mappedPtrs[0] = submissionBuffers[0].mappedPtr;
        mappedPtrs[1] = submissionBuffers[1].mappedPtr;

        debugSrc[0] = PushArray(scratch.temp.arena, u8, submissionSize);
        debugSrc[1] = PushArray(scratch.temp.arena, u8, submissionSize);

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
        if (log2Width > log2Height)
        {
            Swap(log2Width, log2Height);
        }
        key                          = BitFieldPackU32(key, log2Width, offset, 4);
        key                          = BitFieldPackU32(key, log2Height, offset, 4);
        handles[faceIndex].sortKey   = (u8)key;
        handles[faceIndex].faceIndex = faceIndex;
    }

    SortHandles<FaceHandle, false>(handles, numFaces);

    StringBuilderMapped builder(outFilename);

    TextureMetadata *textureMetadata =
        (TextureMetadata *)AllocateSpace(&builder, sizeof(TextureMetadata));
    textureMetadata->numFaces = numFaces;
    FaceMetadata2 *outFaceMetadata =
        (FaceMetadata2 *)AllocateSpace(&builder, sizeof(FaceMetadata2) * numFaces);
    StaticArray<FaceMetadata2> faceMetadatas(scratch.temp.arena, numFaces, numFaces);

    const int pageTexelWidth = PAGE_WIDTH;
    const int pageBlockWidth = pageTexelWidth >> log2BlockSize;
    const int pageByteSize   = Sqr(pageBlockWidth) * bytesPerBlock;
    const int numpagesX      = gpuOutputWidth / pageBlockWidth;
    const int numpagesY      = gpuOutputHeight / pageBlockWidth;

    const int gpuSrcStride = gpuOutputWidth * bytesPerBlock;
    const int pageStride   = pageBlockWidth * bytesPerBlock;

    auto CopyBlockCompressedResultsToDisk = [&](std::vector<u8 *> &blockCompressedResults) {
        // Write to face images to disk
        if (numSubmissions > 0)
        {
            int lastSubmissionIndex   = (submissionIndex - 1) & 1;
            GPUBuffer *readbackBuffer = &readbackBuffers[lastSubmissionIndex];
            device->Wait(semaphores[lastSubmissionIndex]);

            descriptorSets[lastSubmissionIndex].Reset();

            MemoryZero(debugSrc[lastSubmissionIndex], submissionSize);

            u8 *blockCompressedResult = PushArrayNoZero(scratch.temp.arena, u8, outputSize);
            MemoryCopy(blockCompressedResult, readbackBuffer->mappedPtr, outputSize);
            blockCompressedResults.push_back(blockCompressedResult);
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

        // Debug copy
        {
            MemoryCopy((u8 *)mappedPtrs[submissionIndex], debugSrc[submissionIndex],
                       submissionSize);
        }

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

    // Find max level
    u32 maxLevel = 0;
    for (int handleIndex = 0; handleIndex < numFaces; handleIndex++)
    {
        FaceHandle &handle = handles[handleIndex];
        int faceIndex      = handle.faceIndex;
        maxLevel           = Max(maxLevel, images[faceIndex].Length());
    }

    textureMetadata->numLevels = maxLevel;

    u32 currentPageOffset = 0;

    struct FaceInfo
    {
        Vec2u srcOffset;
        Vec2u size;
        u32 submissionIndex;
        Vec2u dstOffset;

        int faceIndex;
    };

    StaticArray<FaceInfo> faceInfos(scratch.temp.arena, numFaces);

    // How this is packed:
    // 1. calculate the total texture area to get a naive lower bound estimate on the
    // number of pages
    // 2. pack face data onto shelves in the estimated square virtual texture space to
    // calculate the minimum necessary # of pages. shelves are packed such that all shelves
    // have height >= shelves below . When the width changes, a new shelf is also created. This
    // way, high mip levels of 2x1 and 4x1 aspect ratio faces are handled by clamping the u
    // coordinate to the entry index on the shelf.
    // 3. repeat #2 in the actual virtual texture space.
    //
    // e.g.
    //
    // Mip 0:
    // 256x256 256x256 ...
    // 128x256 128x256 ...
    // 16x256 16x256 16x256
    // 128x128 128x128
    //
    // Mip 5:
    // 8x8 8x8 ...
    // 4x8 4x8 ...
    // 1x8 1x8 <-- note: 16 >> 5 = 0, so max with the entry index on this shelf to get u.
    // 4x4 4x4
    //
    // Mip 8:
    // 1x1 1x1 ...
    // 1x1 1x1 ...
    // 1x1 1x1 1x1 ...
    // <-- 128x128 faces and below vanish. this doesn't impact the addressing of faces actually
    // present at this mip level.
    //
    // Data on disk is stored using these shelves minimize wasted space.

    u32 totalTextureArea = 0;
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        PaddedImage &currentFaceImg = images[faceIndex][0];
        totalTextureArea += currentFaceImg.width * currentFaceImg.height;
    }

    const u32 estimateSqrtNumPages      = std::sqrt(totalTextureArea >> (2 * PAGE_SHIFT)) + 1;
    const u32 estimateVirtualTexelWidth = estimateSqrtNumPages * pageTexelWidth;
    Assert(Sqr(estimateVirtualTexelWidth) >= totalTextureArea);

    PaddedImage &firstFaceImg   = images[handles[0]][0];
    int currentHorizontalOffset = 0;
    int currentShelfHeight      = Max(firstFaceImg.width, firstFaceImg.height);
    int currentShelfEntryWidth  = Min(firstFaceImg.width, firstFaceImg.height);
    int currentTotalHeight      = 0;
    for (int handleIndex = 0; handleIndex < numFaces; handleIndex++)
    {
        FaceHandle &handle          = handles[handleIndex];
        int faceIndex               = handle.faceIndex;
        PaddedImage &currentFaceImg = images[faceIndex][0];
        int cmpWidth                = Max(currentFaceImg.width, currentFaceImg.height);
        int cmpHeight               = Max(currentFaceImg.width, currentFaceImg.height);

        if (cmpWidth != currentShelfEntryWidth || cmpHeight != currentShelfHeight ||
            currentHorizontalOffset + cmpWidth > estimateVirtualTexelWidth)
        {
            currentHorizontalOffset = 0;
            currentTotalHeight += currentShelfHeight;
            currentShelfEntryWidth = cmpWidth;
            currentShelfHeight     = cmpHeight;
        }
        currentHorizontalOffset += cmpWidth;
    }
    currentTotalHeight += currentShelfHeight;

    struct Shelf
    {
        int width;
        int height;
        int totalWidth;
        int offsetY;
    };

    std::vector<Shelf> shelves;
    const u32 virtualTextureWidth = Min(currentTotalHeight, estimateVirtualTexelWidth);
    u8 *temp = PushArray(scratch.temp.arena, u8, Sqr(virtualTextureWidth) * bytesPerBlock);

    int currentHorizontalOffset = 0;
    int currentShelfHeight      = Max(firstFaceImg.width, firstFaceImg.height);
    int currentShelfEntryWidth  = Min(firstFaceImg.width, firstFaceImg.height);
    int currentTotalHeight      = 0;
    int currentEntryIndex       = 0;
    for (int handleIndex = 0; handleIndex < numFaces; handleIndex++)
    {
        FaceHandle &handle          = handles[handleIndex];
        int faceIndex               = handle.faceIndex;
        PaddedImage &currentFaceImg = images[faceIndex][0];
        int cmpWidth                = Max(currentFaceImg.width, currentFaceImg.height);
        int cmpHeight               = Max(currentFaceImg.width, currentFaceImg.height);

        if (cmpWidth != currentShelfEntryWidth || cmpHeight != currentShelfHeight ||
            currentHorizontalOffset + cmpWidth > virtualTextureWidth)
        {
            Shelf newShelf;
            newShelf.width      = currentShelfEntryWidth;
            newShelf.height     = currentShelfHeight;
            newShelf.totalWidth = currentHorizontalOffset;
            newShelf.offsetY    = currentTotalHeight;
            shelves.push_back(newShelf);

            currentHorizontalOffset = 0;
            currentTotalHeight += currentShelfHeight;
            currentShelfEntryWidth = cmpWidth;
            currentShelfHeight     = cmpHeight;
        }
        Assert(currentTotalHeight + currentShelfHeight <= virtualTextureWidth);

        // Rotate face
        PaddedImage img = currentFaceImg;
        bool rotate     = false;
        if (currentFaceImg.width > currentFaceImg.height)
        {
            u32 size     = img.GetPaddedWidth() * img.GetPaddedHeight() * img.bytesPerPixel;
            img.contents = PushArray(scratch.temp.arena, u8, size);
            Swap(img.width, img.height);
            Swap(img.log2Width, img.log2Height);
            img.strideNoBorder   = img.width * img.bytesPerPixel;
            img.strideWithBorder = img.GetPaddedWidth() * img.bytesPerPixel;

            img.WriteRotated(currentFaceImg, Vec2u(0, 0), Vec2u(0, 0), 1,
                             currentFaceImg.GetPaddedHeight(), currentFaceImg.GetPaddedWidth(),
                             currentFaceImg.GetPaddedWidth(), currentFaceImg.GetPaddedHeight(),
                             Vec2u(0, 0));
            rotate = true;
            faceMetadatas[faceIndex].rotate |= 0x80000000;
        }

        faceMetadatas[faceIndex].log2Width  = img.log2Width;
        faceMetadatas[faceIndex].log2Height = img.log2Height;
        faceMetadatas[faceIndex].offsetX    = currentHorizontalOffset;
        faceMetadatas[faceIndex].offsetY    = currentTotalHeight;

        for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
        {
            // Add edge borders
            int aeid         = f.adjedge(edgeIndex);
            int neighborFace = f.adjface(edgeIndex);
            int rot          = (edgeIndex - aeid + 2) & 3;

            faceMetadatas[faceIndex].neighborFaces[edgeIndex] = neighborFace;
            faceMetadatas[faceIndex].rotate |= rot << (2 * edgeIndex);
        }

        Vec2u dstLoc(currentHorizontalOffset, currentTotalHeight);
        Utils::Copy(img.contents, Vec2u(0, 0), img.width, img.height, temp, dstLoc,
                    virtualTextureWidth, virtualTextureWidth, img.height, img.width,
                    img.bytesPerPixel);

        currentHorizontalOffset += cmpWidth;
    }
    Shelf newShelf;
    newShelf.width      = currentShelfEntryWidth;
    newShelf.height     = currentShelfHeight;
    newShelf.totalWidth = currentHorizontalOffset;
    newShelf.offsetY    = currentTotalHeight;
    shelves.push_back(newShelf);

    // TODO IMPORTANT: if I had to support the 2x1 1x1 mips, I'd need to:
    // 1. create a separate page table that has layers instead of a pyramid
    // 2. clamp the virtual offset in x to be at minimum virtualOffsetX / faceData.width (or
    // instead of storing virtualX, store the entry index on the shelf, and virtualX is
    // Max(entryIndex, entryIndex * faceData.width >> mipLevel);

    // Submit to GPU to calculate mip and block compress, then write to disk
    const u32 totalBlockCompressedSize =
        Sqr(virtualTextureWidth >> log2BlockSize) * bytesPerBlock;
    u8 *dst = (u8 *)AllocateSpace(&builder, totalBlockCompressedSize);
    for (int levelIndex = 0; levelIndex < maxLevel; levelIndex++)
    {
        int submissionTotalHeight = 0;
        int submissionTotalWidth  = 0;
        int shelfIndex            = 0;
        for (; shelfIndex < shelves.size(); shelfIndex++)
        {
            Shelf &shelf         = shelves[shelfIndex];
            submissionTotalWidth = Max(submissionTotalWidth, shelf.submissionTotalWidth);
            if (submissionTotalHeight + shelf.height > gpuSubmissionHeight) break;
            submissionTotalHeight += shelf.height;
        }
        int numSubmissionsX =
            (submissionTotalWidth + gpuSubmissionWidth - 1) / gpuSubmissionWidth;
        int numSubmissionsY =
            (submissionTotalHeight + gpuSubmissionHeight - 1) / gpuSubmissionHeight;

        for (int numY = 0; numY < numSubmissionsY; numY++)
        {
            for (int numX = 0; numX < numSubmissionsX; numX++)
            {
                Vec2u srcIndex(numX * gpuSubmissionWidth, numY * gpuSubmissionHeight);
                Utils::Copy(temp, srcIndex, virtualTextureWidth, virtualTextureWidth,
                            debugSrc[submissionIndex], Vec2u(0, 0), gpuSubmissionWidth,
                            gpuSubmissionHeight, gpuSubmissionHeight, gpuSubmissionWidth,
                            gpuBytesPerPixel);

                CopyBlockCompressedResultsToDisk(blockCompressedResults);
                SubmitBlockCompressionCommandsToGPU();

                numSubmissions++;
                submissionIndex = numSubmissions & 1;
            }
        }
    }

    for (int levelIndex = 0; levelIndex < maxLevel; levelIndex++)
    {
        u32 gpuHeight               = 0;
        int currentHorizontalOffset = 0;

        Assert(levelIndex < images[handles[0].faceIndex].Length());

        u32 currentShelfHeight = 0;

        std::vector<u8 *> blockCompressedResults;

        for (int handleIndex = 0; handleIndex < numFaces; handleIndex++)
        {
            FaceHandle &handle = handles[handleIndex];
            int faceIndex      = handle.faceIndex;
            if (levelIndex >= images[faceIndex].Length())
            {
                faceInfos[handleIndex].size = Vec2u(0);
                continue;
            }

            const Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
            Assert(!f.isSubface());

            PaddedImage &currentFaceImg = images[faceIndex][levelIndex];

            int cmpHeight = Max(currentFaceImg.height, currentFaceImg.width);
            int cmpWidth  = Min(currentFaceImg.height, currentFaceImg.width);

            if (currentHorizontalOffset + cmpWidth > gpuSubmissionWidth)
            {
                currentHorizontalOffset = 0;
                gpuHeight += currentShelfHeight;
                currentShelfHeight = 0;
            }

            currentShelfHeight = Max(currentShelfHeight, (u32)cmpHeight);

            if (currentShelfHeight + gpuHeight > gpuSubmissionHeight)
            {
                CopyBlockCompressedResultsToDisk(blockCompressedResults);
                SubmitBlockCompressionCommandsToGPU();

                numSubmissions++;
                submissionIndex         = numSubmissions & 1;
                gpuHeight               = 0;
                currentHorizontalOffset = 0;
            }

            if (levelIndex == 0)
            {
                for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
                {
                    // Add edge borders
                    int aeid         = f.adjedge(edgeIndex);
                    int neighborFace = f.adjface(edgeIndex);
                    int rot          = (edgeIndex - aeid + 2) & 3;

                    if (neighborFace == -1)
                    {
                        int stop = 5;
                    }
                    faceMetadata[faceIndex].neighborFaces[edgeIndex] = neighborFace;
                    faceMetadata[faceIndex].rotate |= rot << (2 * edgeIndex);
                }
            }

#if 0
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
#endif

            u32 dstStride = gpuSubmissionWidth * GetFormatSize(baseFormat);
            u32 offset =
                dstStride * gpuHeight + currentHorizontalOffset * GetFormatSize(baseFormat);
            Assert(offset <
                   gpuSubmissionWidth * gpuSubmissionHeight * GetFormatSize(baseFormat));

            // Rotate so that all textures are taller than wide
            PaddedImage img = currentFaceImg;
            bool rotate     = false;
            if (currentFaceImg.width > currentFaceImg.height)
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
                faceMetadata[faceIndex].rotate |= 0x80000000;
            }

            Utils::Copy(img.contents, img.strideNoBorder, debugSrc[submissionIndex] + offset,
                        dstStride, img.GetPaddedHeight(), img.strideNoBorder);

            if (levelIndex == 0)
            {
                faceMetadata[faceIndex].log2Width  = img.log2Width;
                faceMetadata[faceIndex].log2Height = img.log2Height;

                FaceInfo faceInfo        = {};
                faceInfo.srcOffset       = Vec2u(currentHorizontalOffset, gpuHeight);
                faceInfo.size            = Vec2u(img.width, img.height);
                faceInfo.submissionIndex = numSubmissions;
                faceInfo.faceIndex       = faceIndex;

                faceInfos.Push(faceInfo);
            }
            else
            {
                Assert(faceInfos[handleIndex].faceIndex == faceIndex);
                faceInfos[handleIndex].srcOffset = Vec2u(currentHorizontalOffset, gpuHeight);
                faceInfos[handleIndex].size      = Vec2u(img.width, img.height);
                faceInfos[handleIndex].submissionIndex = numSubmissions;
            }

            currentHorizontalOffset += img.width;
        }

        CopyBlockCompressedResultsToDisk(blockCompressedResults);
        SubmitBlockCompressionCommandsToGPU();

        numSubmissions++;
        submissionIndex = numSubmissions & 1;

        CopyBlockCompressedResultsToDisk(blockCompressedResults);

        const u32 numPagesX = gpuSubmissionWidth >> PAGE_SHIFT;
        const u32 numPagesY = gpuSubmissionHeight >> PAGE_SHIFT;

        u32 numPages = numSubmissions * numPagesX * numPagesY;

        u32 sqrtNumPages = std::sqrt(numPages) + 1;

        if (levelIndex == 0)
        {
            textureMetadata->virtualSqrtNumPages = NextPowerOfTwo(sqrtNumPages);
        }

        const u32 maxSqrtNumPages = textureMetadata->virtualSqrtNumPages >> levelIndex;
        sqrtNumPages              = Min(sqrtNumPages, maxSqrtNumPages);
        u32 totalNumpages         = Sqr(sqrtNumPages);
        u32 levelBlockWidth       = sqrtNumPages * pageBlockWidth;
        u32 levelTexelWidth       = sqrtNumPages * pageTexelWidth;
        u32 totalSize             = Sqr(levelBlockWidth) * bytesPerBlock;

        currentPageOffset += totalNumpages;
        textureMetadata->mipPageOffsets[levelIndex] = currentPageOffset;

        // From submissions, shell fill the pages
        u8 *dst         = (u8 *)AllocateSpace(&builder, totalSize);
        u32 currentpage = 0;
        u32 currentSqrt = 0;

        u8 *temp = PushArray(scratch.temp.arena, u8, totalSize);

        const u32 blockShift = GetBlockShift(blockFormat);

        u32 currentDstHorizontalOffset = 0;
        u32 currentDstTotalHeight      = 0;
        currentShelfHeight             = 0;

        StaticArray<bool> blockCopied(scratch.temp.arena,
                                      gpuOutputWidth * gpuOutputHeight * numSubmissions,
                                      gpuOutputWidth * gpuOutputHeight * numSubmissions);
        // Copy to square texture
        for (int i = 0; i < numFaces; i++)
        {
            FaceInfo &faceInfo = faceInfos[i];
            if (faceInfo.size.x == 0 && faceInfo.size.y == 0)
            {
                continue;
            }
            u8 *blockCompressedResult = blockCompressedResults[faceInfo.submissionIndex];

            u32 texelWidth  = faceInfo.size.x;
            u32 texelHeight = faceInfo.size.y;
            u32 blockWidth  = texelWidth >> blockShift;
            u32 blockHeight = texelHeight >> blockShift;
            if (currentDstHorizontalOffset + texelWidth > levelTexelWidth)
            {
                currentDstHorizontalOffset = 0;
                currentDstTotalHeight += currentShelfHeight;
                currentShelfHeight = 0;

                Assert(currentDstTotalHeight < levelTexelWidth);
            }

            currentShelfHeight = Max(currentShelfHeight, texelHeight);

            u32 dstStride = levelBlockWidth * bytesPerBlock;
            Vec2u dstOffset(0);
            if (levelIndex == 0)
            {
                dstOffset          = Vec2u(currentDstHorizontalOffset, currentDstTotalHeight);
                faceInfo.dstOffset = dstOffset;
                dstOffset          = dstOffset >> blockShift;
            }
            else
            {
                dstOffset = faceInfo.dstOffset >> (blockShift + (u32)levelIndex);
            }

            Vec2u block     = faceInfo.srcOffset >> blockShift;
            u32 lookupIndex = faceInfo.submissionIndex * gpuOutputWidth * gpuOutputHeight +
                              block.y * gpuOutputWidth + block.x;
            if (blockCopied[lookupIndex])
            {
                continue;
            }

            blockCopied[lookupIndex] = true;

            blockWidth  = Max(blockWidth, 1u);
            blockHeight = Max(blockHeight, 1u);

            // Copy face data to square texture
            Utils::Copy(blockCompressedResult, block, gpuOutputWidth, gpuOutputHeight, temp,
                        dstOffset, levelBlockWidth, levelBlockWidth, blockHeight, blockWidth,
                        bytesPerBlock);

            currentDstHorizontalOffset += texelWidth;
        }

        // Copy pages to disk
        u32 dstOffset = 0;
        for (int pageY = 0; pageY < sqrtNumPages; pageY++)
        {
            for (int pageX = 0; pageX < sqrtNumPages; pageX++)
            {
                Vec2u srcIndex(pageX * pageBlockWidth, pageY * pageBlockWidth);
                Utils::Copy(temp, srcIndex, levelBlockWidth, levelBlockWidth, dst + dstOffset,
                            Vec2u(0, 0), pageBlockWidth, pageBlockWidth, pageBlockWidth,
                            pageBlockWidth, bytesPerBlock);
                dstOffset += pageByteSize;
            }
        }

        if (levelIndex == 0)
        {
            for (int i = 0; i < numFaces; i++)
            {
                FaceInfo &faceInfo = faceInfos[i];

                faceMetadata[faceInfo.faceIndex].offsetX = faceInfo.dstOffset.x;
                faceMetadata[faceInfo.faceIndex].offsetY = faceInfo.dstOffset.y;
            }
        }

        numSubmissions  = 0;
        submissionIndex = 0;
    }

    MemoryCopy(outFaceMetadata, faceMetadata.data, numFaces * sizeof(FaceMetadata2));

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
        u32 writeIndex = writeOffset & (capacity - 1);
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
    u32 writeIndex = writeOffset & (capacity - 1);
    u32 numToEnd   = capacity - writeIndex;
    MemoryCopy(entries.data + writeIndex, vals, sizeof(T) * Min(num, numToEnd));
    if (num > numToEnd)
    {
        MemoryCopy(entries.data, vals + numToEnd, sizeof(T) * (num - numToEnd));
    }
    writeOffset += num;
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

VirtualTextureManager::VirtualTextureManager(Arena *arena, u32 virtualTextureWidth,
                                             u32 virtualTextureHeight,
                                             u32 physicalTextureWidth,
                                             u32 physicalTextureHeight, u32 numPools,
                                             VkFormat format)
    : format(format), updateRequestRingBuffer(arena, maxCopies),
      feedbackRingBuffer(arena, maxFeedback)
{
    string shaderName = "../src/shaders/update_page_tables.spv";
    string data       = OS_ReadFile(arena, shaderName);
    shader            = device->CreateShader(ShaderStage::Compute, "update page tables", data);

    descriptorSetLayout = {};
    descriptorSetLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    push     = PushConstant(ShaderStage::Compute, 0, sizeof(PageTableUpdatePushConstant));
    pipeline = device->CreateComputePipeline(&shader, &descriptorSetLayout, &push);

    ImageLimits limits = device->GetImageLimits();

    // Allocate page table
    const u32 numMips =
        Log2Int(Max(virtualTextureWidth, virtualTextureHeight) >> PAGE_SHIFT) + 1;
    numVirtPagesWide     = virtualTextureWidth >> PAGE_SHIFT;
    u32 numVirtPagesHigh = virtualTextureHeight >> PAGE_SHIFT;
    ImageDesc pageTableDesc(ImageType::Type2D, numVirtPagesWide, numVirtPagesHigh, 1, numMips,
                            1, VK_FORMAT_R32_UINT, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    pageTable = device->CreateImage(pageTableDesc);
    for (int i = 0; i < numMips; i++)
    {
        device->CreateSubresource(&pageTable, i, 1);
        int bindlessIndex = device->BindlessStorageIndex(&pageTable, i);
        if (i == 0) bindlessPageTableStartIndex = bindlessIndex;
        Assert(bindlessIndex == bindlessPageTableStartIndex + i);
    }

    // Allocate cpu page table
    cpuPageTable = StaticArray<StaticArray<u32>>(arena, numMips, numMips);
    for (int i = 0; i < numMips; i++)
    {
        const u32 mipNumVirtPagesWide = numVirtPagesWide >> i;
        const u32 mipNumVirtPagesHigh = numVirtPagesHigh >> i;
        cpuPageTable[i] = StaticArray<u32>(arena, mipNumVirtPagesWide * mipNumVirtPagesHigh,
                                           mipNumVirtPagesWide * mipNumVirtPagesHigh);
        for (int pageY = 0; pageY < mipNumVirtPagesHigh; pageY++)
        {
            for (int pageX = 0; pageX < mipNumVirtPagesWide; pageX++)
            {
                cpuPageTable[i][pageY * mipNumVirtPagesWide + pageX] = ~0u;
            }
        }
    }

    Assert(IsPow2(physicalTextureWidth) && IsPow2(physicalTextureHeight));
    numPhysPagesWidth  = physicalTextureWidth / PAGE_WIDTH;
    numPhysPagesHeight = physicalTextureHeight / PAGE_WIDTH;

    {
        numPools = Min(numPools, limits.maxNumLayers);
        Assert(physicalTextureWidth <= limits.max2DImageDim);
        Assert(physicalTextureHeight <= limits.max2DImageDim);
        physicalTextureWidth  = Min(physicalTextureWidth, limits.max2DImageDim);
        physicalTextureHeight = Min(physicalTextureHeight, limits.max2DImageDim);

        // Allocate physical page pool + sentinels
        mipSentinels = StaticArray<Sentinel>(arena, numMips);

        const u32 numPhysicalPages =
            2 * numMips + numPools * numPhysPagesWidth * numPhysPagesHeight;
        physicalPages = StaticArray<PhysicalPage>(arena, numPhysicalPages);

        for (int i = 0; i < numPools; i++)
        {
            for (int y = 0; y < numPhysPagesHeight; y++)
            {
                for (int x = 0; x < numPhysPagesWidth; x++)
                {
                    PhysicalPage page = {};
                    page.page         = Vec2u(x, y);
                    page.layer        = i;
                    page.prevPage     = -1;
                    page.nextPage     = -1;
                    physicalPages.Push(page);
                }
            }
        }

        u32 pageSentinelIndex = physicalPages.size();
        for (int i = 0; i < numMips; i++)
        {
            int headIndex = pageSentinelIndex++;
            int tailIndex = pageSentinelIndex++;

            PhysicalPage head;
            head.nextPage = tailIndex;
            head.prevPage = -1;
            PhysicalPage tail;
            tail.nextPage = -1;
            tail.prevPage = headIndex;

            physicalPages.push_back(head);
            physicalPages.push_back(tail);

            Sentinel sentinel;
            sentinel.head = headIndex;
            sentinel.tail = tailIndex;

            mipSentinels.push_back(sentinel);
        }

        freePage = 0;
        // Allocate texture arrays
        ImageDesc poolDesc(ImageType::Array2D, physicalTextureWidth, physicalTextureHeight, 1,
                           1, numPools, format, MemoryUsage::GPU_ONLY,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        gpuPhysicalPool = device->CreateImage(
            poolDesc, -1, QueueFlag_Copy | QueueFlag_Compute | QueueFlag_Graphics);
    }

    // Instantiate streaming system
    {
        uploadBuffers =
            FixedArray<StaticArray<u8>, numPendingSubmissions>(numPendingSubmissions);
        uploadBuffers[0] = StaticArray<u8>(arena, maxUploadSize);
        uploadBuffers[1] = StaticArray<u8>(arena, maxUploadSize);

        uploadCopyCommands = FixedArray<StaticArray<BufferImageCopy>, numPendingSubmissions>(
            numPendingSubmissions);
        uploadCopyCommands[0] = StaticArray<BufferImageCopy>(arena, maxCopies);
        uploadCopyCommands[1] = StaticArray<BufferImageCopy>(arena, maxCopies);

        uploadSemaphores = FixedArray<Semaphore, numPendingSubmissions>(numPendingSubmissions);
        uploadSemaphores[0] = device->CreateSemaphore();
        uploadSemaphores[1] = device->CreateSemaphore();

        writeSubmission.store(0);

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

        readSubmission.store(0);

        updateRequestMutex.count.store(0);

        feedbackBuffers =
            FixedArray<TransferBuffer, numPendingSubmissions>(numPendingSubmissions);
        feedbackBuffers[0] =
            device->GetReadbackBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, megabytes(8));
        feedbackBuffers[1] =
            device->GetReadbackBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, megabytes(8));

        feedbackMutex.count.store(0);
    }

    // callback
    Scheduler::Counter counter = {};
    scheduler.Schedule(&counter, [&](int jobID) { Callback(); });
}

Vec2u VirtualTextureManager::AllocateVirtualPages(Arena *arena, string filename,
                                                  const TextureMetadata &metadata,
                                                  const Vec2u &virtualSize, u8 *contents,
                                                  u32 &index)
{
    Vec2u virtualPage(0);

    // TODO: actually implement this, using some form of sorting

    // for (int columnIndex = 0; columnIndex < allocationColumns.size(); columnIndex++)
    // {
    //     AllocationColumn &column = allocationColumns[columnIndex];
    //     Assert(virtualSize.x <= column.numPagesWide);
    //     if (column.currentPageHeight + virtualSize.y < column.maxPageHeight)
    //     {
    //         virtualPage = Vec2u(column.numPagesX, column.currentPageHeight);
    //         baseVirtualPages.push_back(virtualPage);
    //         index = baseVirtualPages.size() - 1;
    //
    //         return virtualPage;
    //     }
    // }

    TextureInfo texInfo;
    texInfo.filename        = PushStr8Copy(arena, filename);
    texInfo.baseVirtualPage = virtualPage;
    texInfo.metadata        = metadata;
    texInfo.contents        = contents;
    textureInfo.push_back(texInfo);
    index = textureInfo.size() - 1;

    return virtualPage;
}

// void VirtualTextureManager::PinPages(CommandBuffer *cmd, Vec2u baseVirtualPage,
//                                      FaceMetadata2 *faceMetadatas, int numFaces,
//                                      TextureMetadata &metadata, u8 *contents)
// {
//     int numPages = 0;
//     for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
//     {
//         FaceMetadata2 &faceMetadata = faceMetadatas[faceIndex];
//         int maxLevel                = Min(faceMetadata.log2Width, faceMetadata.log2Height);
//
//     }
// }

void VirtualTextureManager::AllocatePhysicalPages(CommandBuffer *cmd, Vec2u baseVirtualPage,
                                                  TextureMetadata &metadata, u8 *contents)
{
#if 0
    ScratchArena scratch;

    const u32 blockShift     = GetBlockShift(format);
    const u32 bytesPerBlock  = GetFormatSize(format);
    const u32 pageBlockWidth = PAGE_WIDTH >> blockShift;
    const u32 pageSize       = pageBlockWidth * pageBlockWidth * bytesPerBlock;
    const u32 pagesPerRow    = 4096 / PAGE_WIDTH;

    Assert(metadata.numLevels);
    const u32 totalNumPages = metadata.mipPageOffsets[metadata.numLevels - 1];
    StaticArray<PageTableUpdateRequest> updateRequests(scratch.temp.arena, totalNumPages);
    StaticArray<BufferImageCopy> copies(scratch.temp.arena, totalNumPages);
    GPUBuffer transferBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, totalNumPages * pageSize, MemoryUsage::CPU_TO_GPU);
    u8 *mappedPtr    = (u8 *)transferBuffer.mappedPtr;
    u32 bufferOffset = 0;

    for (u32 levelIndex = 0; levelIndex < metadata.numLevels; levelIndex++)
    {
        u32 levelNumPages = levelIndex == 0 ? metadata.mipPageOffsets[0]
                                            : metadata.mipPageOffsets[levelIndex] -
                                                  metadata.mipPageOffsets[levelIndex - 1];
        u32 levelOffset   = levelIndex == 0 ? 0 : metadata.mipPageOffsets[levelIndex - 1];

        u32 sqrtLevelNumPages = std::sqrt(levelNumPages);
        updateRequests.size_  = 0;

        for (u32 pageIndex = 0; pageIndex < levelNumPages; pageIndex++)
        {
            Assert(freePool < pools.Length());

            // Calculate address of virtual page
            const Vec2u virtualPage =
                (baseVirtualPage >> levelIndex) +
                Vec2u(pageIndex % sqrtLevelNumPages, pageIndex / sqrtLevelNumPages);

            // Allocate page from pool
            Assert(freePool != InvalidPool && freePool < pools.Length());
            u32 poolIndex          = freePool;
            PhysicalPagePool &pool = pools[poolIndex];
            Vec2u pageLocation     = pool.pagePool[pool.freePage];
            pool.freePage++;
            if (pool.freePage >= pool.pagePool.Length()) freePool++;

            // Create page table update request
            Assert(pageLocation.x < 256 && pageLocation.y < 256);

            u32 packed =
                PackPageTableEntry(pageLocation.x, pageLocation.y, levelIndex, poolIndex);
            PageTableUpdateRequest pageTableUpdateRequest;
            pageTableUpdateRequest.virtualPage = virtualPage;
            pageTableUpdateRequest.packed      = packed;
            updateRequests.push_back(pageTableUpdateRequest);

            // Create buffer image copy command
            BufferImageCopy copy = {};
            copy.bufferOffset    = bufferOffset;
            copy.baseLayer       = poolIndex;
            copy.layerCount      = 1;
            copy.offset = Vec3i(pageLocation.x << PAGE_SHIFT, pageLocation.y << PAGE_SHIFT, 0);
            copy.extent = Vec3u(PAGE_WIDTH, PAGE_WIDTH, 1);

            copies.push_back(copy);

            // Copy to transfer buffer
            u32 pageOffset = (levelOffset + pageIndex) * pageSize;
            u8 *pageStart  = contents + pageOffset;
            MemoryCopy(mappedPtr + bufferOffset, pageStart, pageSize);

            bufferOffset += pageSize;
        }

        PageTableUpdatePushConstant pc;
        pc.numRequests = levelNumPages;

        TransferBuffer pageTableUpdateRequestBuffer =
            cmd->SubmitBuffer(updateRequests.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              sizeof(PageTableUpdateRequest) * pc.numRequests);

        DescriptorSet ds = descriptorSetLayout.CreateDescriptorSet();
        ds.Bind(0, &pageTableUpdateRequestBuffer.buffer).Bind(1, &pageTable, levelIndex);
        cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        cmd->PushConstants(&push, &pc, descriptorSetLayout.pipelineLayout);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                descriptorSetLayout.pipelineLayout);
        cmd->Dispatch((levelNumPages + 63) >> 6, 1, 1);
    }

    cmd->CopyImage(&transferBuffer, &gpuPhysicalPool, copies.data, copies.Length());
#endif
}

void VirtualTextureManager::ClearPageTable(CommandBuffer *cmd)
{
    cmd->ClearImage(&pageTable, ~0u);
}

///////////////////////////////////////
// Streaming/Feedback
//

// Executes on main thread
void VirtualTextureManager::Update(CommandBuffer *computeCmd, CommandBuffer *transferCmd,
                                   CommandBuffer *transitionCmd, QueueType computeCmdQueue)
{
    // Update page table
    ScratchArena scratch;

    u32 currentBuffer = device->GetCurrentBuffer();

    // Synchronize
    uploadSemaphores[currentBuffer].signalValue++;
    transitionCmd->SignalOutsideFrame(uploadSemaphores[currentBuffer]);
    transferCmd->Wait(uploadSemaphores[currentBuffer]);

    uploadSemaphores[currentBuffer].signalValue++;
    transferCmd->SignalOutsideFrame(uploadSemaphores[currentBuffer]);
    computeCmd->Wait(uploadSemaphores[currentBuffer]);

    // Write last frame's feedback buffer
    if (device->frameCount >= 2)
    {
        u32 numFeedbackRequests = ((u32 *)feedbackBuffers[currentBuffer].mappedPtr)[0];
        Vec2u *feedbackRequests =
            PushArrayNoZero(scratch.temp.arena, Vec2u, numFeedbackRequests);

        BeginMutex(&feedbackMutex);
        feedbackRingBuffer.WriteWithOverwrite(
            (Vec2u *)((u32 *)feedbackBuffers[currentBuffer].mappedPtr + 1),
            numFeedbackRequests);
        EndMutex(&feedbackMutex);
    }

    u32 numRequests                        = 0;
    PageTableUpdateRequest *updateRequests = updateRequestRingBuffer.SynchronizedRead(
        &updateRequestMutex, scratch.temp.arena, numRequests);

    if (numRequests)
    {
        Assert(updateRequests);

        TransferBuffer *pageTableRequestBuffer = &pageTableRequestBuffers[currentBuffer];
        MemoryCopy(pageTableRequestBuffer->mappedPtr, updateRequests,
                   sizeof(PageTableUpdateRequest) * numRequests);
        computeCmd->SubmitTransfer(pageTableRequestBuffer);

        // WAR for page table (only execution dependency), RAW for transfer buffer
        // (memory + execution dependency)
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->Barrier(&pageTable, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_WRITE_BIT);
        computeCmd->FlushBarriers();

        PageTableUpdatePushConstant pc;
        pc.numRequests                 = numRequests;
        pc.bindlessPageTableStartIndex = bindlessPageTableStartIndex;

        DescriptorSet updatePageTableDescriptorSet = descriptorSetLayout.CreateDescriptorSet();
        updatePageTableDescriptorSet.Bind(0, &pageTableRequestBuffer->buffer);
        computeCmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        computeCmd->PushConstants(&push, &pc, descriptorSetLayout.pipelineLayout);
        computeCmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE,
                                       &updatePageTableDescriptorSet,
                                       descriptorSetLayout.pipelineLayout);
        computeCmd->Dispatch((numRequests + 63) >> 6, 1, 1);

        // RAW for page table
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->Barrier(&pageTable, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->FlushBarriers();
    }

    u64 readIndex = readSubmission.load(std::memory_order_relaxed);

    u64 writeIndex = writeSubmission.load(std::memory_order_acquire);
    Assert(readIndex <= writeIndex);
    if (readIndex == writeIndex) return;

    // Update physical texture with new entries
    u32 readDoubleBufferIndex = readIndex & 1;

    StaticArray<BufferImageCopy> &writtenCopyCommands =
        uploadCopyCommands[readDoubleBufferIndex];
    u32 numCopies = writtenCopyCommands.size();

    if (numCopies)
    {
        StaticArray<u8> &uploadBuffer = uploadBuffers[readDoubleBufferIndex];
        // Queue family transfer to async copy queue
        transitionCmd->Barrier(&gpuPhysicalPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_SHADER_READ_BIT,
                               VK_ACCESS_2_NONE, computeCmdQueue, QueueType_Copy);
        transferCmd->Barrier(&gpuPhysicalPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_NONE,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT, computeCmdQueue, QueueType_Copy);
        transitionCmd->FlushBarriers();
        transferCmd->FlushBarriers();

        // Copy data to staging buffer
        GPUBuffer *uploadDeviceBuffer = &uploadDeviceBuffers[currentBuffer];
        MemoryCopy(uploadDeviceBuffer->mappedPtr, uploadBuffer.data, uploadBuffer.size());
        transferCmd->CopyImage(uploadDeviceBuffer, &gpuPhysicalPool, writtenCopyCommands.data,
                               numCopies);

        // Queue family transfer back to source queue
        transferCmd->Barrier(&gpuPhysicalPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_NONE,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_NONE, QueueType_Copy,
                             computeCmdQueue);
        computeCmd->Barrier(&gpuPhysicalPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_NONE,
                            VK_ACCESS_2_SHADER_READ_BIT, QueueType_Copy, computeCmdQueue);
        transferCmd->FlushBarriers();
        computeCmd->FlushBarriers();
    }

    // Make sure all memory writes are visible
    readSubmission.store(readIndex + 1, std::memory_order_release);
}

void VirtualTextureManager::UnlinkLRU(int pageIndex)
{
    PhysicalPage &page = physicalPages[pageIndex];
    Assert(page.prevPage != -1 && page.nextPage != -1);

    int prevPage                     = page.prevPage;
    int nextPage                     = page.nextPage;
    physicalPages[prevPage].nextPage = nextPage;
    physicalPages[nextPage].prevPage = prevPage;
}

void VirtualTextureManager::LinkLRU(int index, int mip)
{
    int headPage                  = mipSentinels[mip].head;
    int nextPage                  = physicalPages[headPage].nextPage;
    physicalPages[index].nextPage = nextPage;
    physicalPages[index].prevPage = headPage;

    physicalPages[nextPage].prevPage = index;
    physicalPages[headPage].nextPage = index;
}

// Executes on virtual texture thread
void VirtualTextureManager::Callback()
{
    struct SortKey
    {
        u32 sortKey;
        u32 requestIndex;
    };

    struct FeedbackRequest
    {
        Vec2u packedFeedbackInfo;
        u32 count;
        u32 diff;
    };

    const u32 blockShift    = GetBlockShift(format);
    const u32 bytesPerBlock = GetFormatSize(format);

    const int pageTexelWidth = PAGE_WIDTH;
    const int pageBlockWidth = pageTexelWidth >> blockShift;
    const int pageByteSize   = Sqr(pageBlockWidth) * bytesPerBlock;

    for (;;)
    {
        ScratchArena scratch;
        // Get feedback requests

        u32 numFeedbackRequests = 0;
        Vec2u *feedbackRequests = feedbackRingBuffer.SynchronizedRead(
            &feedbackMutex, scratch.temp.arena, numFeedbackRequests);
        u32 numNonResidentFeedback = 0;

        StaticArray<FeedbackRequest> compactedFeedbackRequests(scratch.temp.arena,
                                                               numFeedbackRequests);

        u32 hashMapSize = NextPowerOfTwo(numFeedbackRequests);
        HashIndex pageHashMap(scratch.temp.arena, hashMapSize, hashMapSize);

        // Compact feedback
        for (int requestIndex = 0; requestIndex < numFeedbackRequests; requestIndex++)
        {
            Vec2u &feedbackRequest = feedbackRequests[requestIndex];
            u32 virtualPageX       = feedbackRequest.x & 0xffff;
            u32 virtualPageY       = feedbackRequest.x >> 16;
            u32 textureIndex       = feedbackRequest.y & 0x00ffffff;
            u32 mipLevel           = feedbackRequest.y >> 24;

            u64 packed = ((u64)mipLevel << 56u) | ((u64)textureIndex << 32u) |
                         (virtualPageY << 16u) | virtualPageX;

            i32 hash = (i32)MixBits(packed);

            bool found = false;
            for (int index = pageHashMap.FirstInHash(hash); index != -1;
                 index     = pageHashMap.NextInHash(index))
            {
                FeedbackRequest &fr = compactedFeedbackRequests[index];

                if (fr.packedFeedbackInfo == feedbackRequest)
                {
                    compactedFeedbackRequests[index].count++;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                u32 index = compactedFeedbackRequests.size();
                FeedbackRequest request;
                request.packedFeedbackInfo = feedbackRequest;
                request.count              = 1;
                compactedFeedbackRequests.push_back(request);
                Assert(index == 0 ||
                       request.packedFeedbackInfo !=
                           compactedFeedbackRequests[index - 1].packedFeedbackInfo);
                pageHashMap.AddInHash(hash, index);
            }
        }

        // Print("Num feedback: %u, num compacted: %u\n", numFeedbackRequests,
        //       compactedFeedbackRequests.size());

        // If requested page is resident Update LRU. Otherwise
        for (int requestIndex = 0; requestIndex < compactedFeedbackRequests.size();
             requestIndex++)
        {
            FeedbackRequest fr    = compactedFeedbackRequests[requestIndex];
            Vec2u feedbackRequest = fr.packedFeedbackInfo;
            u32 virtualPageX      = feedbackRequest.x & 0xffff;
            u32 virtualPageY      = feedbackRequest.x >> 16;
            u32 textureIndex      = feedbackRequest.y & 0x00ffffff;
            u32 mipLevel          = feedbackRequest.y >> 24;

            TextureInfo &texInfo = textureInfo[textureIndex];

            const u32 numMipVirtPagesWide = numVirtPagesWide >> mipLevel;
            u32 packedPageTableEntry =
                cpuPageTable[mipLevel][virtualPageY * numMipVirtPagesWide + virtualPageX];

            Vec4u pageTableEntry = UnpackPageTableEntry(packedPageTableEntry);
            u32 physicalPageX    = pageTableEntry.x;
            u32 physicalPageY    = pageTableEntry.y;
            u32 mip              = pageTableEntry.z;
            u32 layer            = pageTableEntry.w;

            // Move to head of LRU if requested page is already resident
            if (packedPageTableEntry != ~0u && mip == mipLevel)
            {
                int pageIndex = layer * Sqr(numPhysPagesWidth) +
                                physicalPageY * numPhysPagesWidth + physicalPageX;

                UnlinkLRU(pageIndex);
                LinkLRU(pageIndex, mip);
            }
            // Otherwise, page needs to be mapped
            else
            {
                Assert(mip - mipLevel);
                fr.diff                                             = mip - mipLevel;
                compactedFeedbackRequests[numNonResidentFeedback++] = fr;
            }
        }
        compactedFeedbackRequests.size() = numNonResidentFeedback;

        // Wait until the transfer buffer is read on the GPU before writing new data
        u64 writeIndex = writeSubmission.load(std::memory_order_relaxed);
        for (;;)
        {
            u64 readIndex = readSubmission.load(std::memory_order_acquire);
            if (readIndex + numPendingSubmissions > writeIndex) break;
            std::this_thread::yield();
        }

        u32 writeDoubleBufferIndex = writeIndex & 1;

        StaticArray<BufferImageCopy> copies(scratch.temp.arena, numNonResidentFeedback);

        StaticArray<PageTableUpdateRequest> evictRequests(scratch.temp.arena,
                                                          numNonResidentFeedback);

        StaticArray<PageTableUpdateRequest> mapRequests(scratch.temp.arena,
                                                        numNonResidentFeedback);

        auto &uploadBuffer  = uploadBuffers[writeDoubleBufferIndex];
        uploadBuffer.size() = 0;

        // Evict to make space for new entries while populating feedback buffer
        for (FeedbackRequest fr : compactedFeedbackRequests)
        {
            Vec2u feedbackRequest = fr.packedFeedbackInfo;
            u32 virtualPageX      = feedbackRequest.x & 0xffff;
            u32 virtualPageY      = feedbackRequest.x >> 16;
            u32 textureIndex      = feedbackRequest.y & 0x00ffffff;
            u32 mipLevel          = feedbackRequest.y >> 24;

            TextureInfo &texInfo = textureInfo[textureIndex];
            Assert(mipLevel < texInfo.metadata.numLevels);
            TextureMetadata &textureMetadata = texInfo.metadata;
            u32 levelOffset = mipLevel == 0 ? 0 : textureMetadata.mipPageOffsets[mipLevel - 1];
            u32 levelNumPages     = mipLevel == 0
                                        ? textureMetadata.mipPageOffsets[0]
                                        : textureMetadata.mipPageOffsets[mipLevel] -
                                          textureMetadata.mipPageOffsets[mipLevel - 1];
            u32 sqrtLevelNumPages = std::sqrt(levelNumPages);

            // If there are still blank entries in the physical pools, then map those

            u32 pageIndex = ~0u;
            if (freePage < physicalPages.size())
            {
                pageIndex          = freePage;
                PhysicalPage &page = physicalPages[pageIndex];
                freePage++;
            }
            else
            {
                // Otherwise, evict old entires
                for (int levelIndex = 0; levelIndex < mipSentinels.size(); levelIndex++)
                {
                    Sentinel &sentinel = mipSentinels[levelIndex];
                    if (physicalPages[sentinel.tail].prevPage == sentinel.head) continue;
                    pageIndex                  = physicalPages[sentinel.tail].prevPage;
                    PhysicalPage &physicalPage = physicalPages[pageIndex];
                    Vec2u virtualPage          = physicalPage.virtualPage;

                    // TODO: need to handle mapping to coarser mips
                    // Replace previously mapped entry with a coarser mip
                    Vec2u coarserVirtualPage = virtualPage >> 1u;
                    u32 mipNumVertPagesWide  = numVirtPagesWide >> levelIndex;
                    u32 evictPackedEntry =
                        cpuPageTable[levelIndex + 1]
                                    [coarserVirtualPage.y * (mipNumVertPagesWide >> 1) +
                                     coarserVirtualPage.x];

                    cpuPageTable[levelIndex][virtualPage.y * mipNumVertPagesWide +
                                             virtualPage.x] = evictPackedEntry;

                    UnlinkLRU(pageIndex);

                    PageTableUpdateRequest evictRequest;
                    evictRequest.virtualPage = virtualPage;
                    evictRequest.mipLevel    = levelIndex;
                    evictRequest.packed      = evictPackedEntry;
                    evictRequests.push_back(evictRequest);

                    break;
                }
            }

            Assert(pageIndex != ~0u);

            const Vec2u globalVirtualPage =
                (texInfo.baseVirtualPage >> mipLevel) + Vec2u(virtualPageX, virtualPageY);

            PhysicalPage &page = physicalPages[pageIndex];
            page.virtualPage   = globalVirtualPage;

            u32 mapPackedEntry =
                PackPageTableEntry(page.page.x, page.page.y, mipLevel, page.layer);
            // Update CPU page table
            cpuPageTable[mipLevel][globalVirtualPage.y * (numVirtPagesWide >> mipLevel) +
                                   globalVirtualPage.x] = mapPackedEntry;

            LinkLRU(pageIndex, mipLevel);

            BufferImageCopy copy = {};
            copy.bufferOffset    = uploadBuffer.size();
            copy.baseLayer       = page.layer;
            copy.layerCount      = 1;
            copy.offset = Vec3i(page.page.x << PAGE_SHIFT, page.page.y << PAGE_SHIFT, 0);
            copy.extent = Vec3u(PAGE_WIDTH, PAGE_WIDTH, 1);

            copies.push_back(copy);

            u32 virtualPage = virtualPageY * sqrtLevelNumPages + virtualPageX;
            u32 pageOffset  = (levelOffset + virtualPage) * pageByteSize;
            MemoryCopy(uploadBuffer.data + uploadBuffer.size(), texInfo.contents + pageOffset,
                       pageByteSize);
            uploadBuffer.size() += pageByteSize;

            PageTableUpdateRequest mapRequest;
            mapRequest.virtualPage = Vec2u(virtualPageX, virtualPageY);
            mapRequest.mipLevel    = mipLevel;
            mapRequest.packed      = mapPackedEntry;
            mapRequests.push_back(mapRequest);
        }

        // Write evict requests if present
        if (evictRequests.size())
        {
            updateRequestRingBuffer.SynchronizedWrite(&updateRequestMutex, evictRequests.data,
                                                      evictRequests.size());
        }

        // Write copy commands
        auto &copyCommands = uploadCopyCommands[writeDoubleBufferIndex];
        Assert(copies.size() < copyCommands.capacity);
        MemoryCopy(copyCommands.data, copies.data, sizeof(BufferImageCopy) * copies.size());
        copyCommands.size() = copies.size();

        writeSubmission.store(writeIndex + 1, std::memory_order_release);

        // Write map requests
        if (mapRequests.size())
        {
            updateRequestRingBuffer.SynchronizedWrite(&updateRequestMutex, mapRequests.data,
                                                      mapRequests.size());
        }
    }
}

} // namespace rt
