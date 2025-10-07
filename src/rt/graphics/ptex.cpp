#include "../base.h"
#include "../containers.h"
#include "../debug.h"
#include "../memory.h"
#include "../bit_packing.h"
#include "../string.h"
#include "../radix_sort.h"
#include "../parallel.h"
// #include "../scene.h"
// #include "../integrate.h"
#include "vulkan.h"
#include "ptex.h"
#include <Ptexture.h>
#include <PtexReader.h>
#include <atomic>
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

void InitializePtex(u32 maxFiles, u64 maxMem)
{
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
    Ptex::PtexPtr<Ptex::PtexFaceData> d(ptx->getData(faceID, fi.res));
    Ptex::Res tileRes = d->tileRes();

    int u       = fi.res.u();
    int v       = fi.res.v();
    int ntilesu = fi.res.ntilesu(tileRes);
    int ntilesv = fi.res.ntilesv(tileRes);

    u32 bytesPerChannel = Ptex::DataSize(ptx->dataType());
    u32 bytesPerPixel   = numChannels * bytesPerChannel;
    int stride          = (u + 2 * borderSize) * bytesPerPixel;
    int size            = stride * (v + 2 * borderSize);
    u8 *data   = aChan == -1 ? PushArray(temp.arena, u8, size) : PushArray(arena, u8, size);
    int rowlen = (u * bytesPerPixel);

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
    char *buffer            = (char *)data;

    if (d->isConstant())
    {
        // fill dest buffer with pixel value
        // Ptex::PtexUtils::fill(d->getData(), buffer, stride, resu, resv, _pixelsize);
    }
    else if (d->isTiled())
    {
        // loop over tiles
        Ptex::Res tileres = d->tileRes();
        for (u32 i = 1; i < 4; i++)
        {
            Ptex::Res res(fi.res.ulog2 - i, fi.res.vlog2 - i);
            Ptex::PtexPtr<Ptex::PtexFaceData> test(ptx->getData(faceID, res));

            int testU = -1;
            int testV = -1;
            if (test->isTiled())
            {
                testU = test->tileRes().u();
                testV = test->tileRes().v();
            }
            int stop = 5;
        }
        int ntilesu      = fi.res.ntilesu(tileres);
        int ntilesv      = fi.res.ntilesv(tileres);
        int tileures     = tileres.u();
        int tilevres     = tileres.v();
        int tilerowlen   = bytesPerPixel * tileures;
        int tile         = 0;
        char *dsttilerow = (char *)buffer;
        for (int i = 0; i < ntilesv; i++)
        {
            char *dsttile = dsttilerow;
            for (int j = 0; j < ntilesu; j++)
            {
                Ptex::PtexPtr<Ptex::PtexFaceData> t(d->getTile(tile++));
                if (t->isConstant())
                    Ptex::PtexUtils::fill(t->getData(), dsttile, stride, tileures, tilevres,
                                          bytesPerPixel);
                else
                    Ptex::PtexUtils::copy(t->getData(), tilerowlen, dsttile, stride, tilevres,
                                          tilerowlen);
                dsttile += tilerowlen;
            }
            dsttilerow += stride * tilevres;
        }
    }
    else
    {
        // Ptex::PtexUtils::copy(d->getData(), rowlen, buffer, stride, resv, rowlen);
    }

    ptx->getData(faceID, (char *)result.GetContentsRelativeIndex({0, 0}), stride);

    Assert(IsPow2(u) && IsPow2(v));

    // if (flip)
    // {
    //     data += rowlen * (img.h - 1);
    //     stride = -rowlen;
    // }

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

    Print("starting conversion for %S\n", filename);

    if (OS_FileExists(outFilename))
    {
        Print("skipping conversion for %S\n", filename);
        return;
    }

    size_t maxMem = gigabytes(4);

    if (Contains(outFilename, "displacement")) return;

    Ptex::String error;
    Ptex::PtexTexture *t = cache->get((char *)filename.str, error);
    Assert(t);

    int numFaces = t->numFaces();
    StaticArray<Ptex::FaceInfo> ptexFaceInfos(scratch.temp.arena, numFaces);
    for (int i = 0; i < numFaces; i++)
    {
        const Ptex::FaceInfo &f = t->getFaceInfo(i);
        Assert(!f.isSubface());
        ptexFaceInfos.push_back(f);
    }

    t->release();

    const Vec2u uvTable[] = {
        Vec2u(0, 0),
        Vec2u(1, 0),
        Vec2u(1, 1),
        Vec2u(0, 1),
    };

    const u32 bytesPerPixel    = 3;
    const u32 gpuBytesPerPixel = 4;
    const u32 borderSize       = 4;

    const u32 texelWidthPerPage      = 128;
    const u32 log2TexelWidth         = Log2Int(texelWidthPerPage);
    const u32 totalTexelWidthPerPage = 128 + 2 * borderSize;

    const VkFormat baseFormat  = VK_FORMAT_R8G8B8A8_UNORM;
    const VkFormat blockFormat = VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    const u32 bytesPerTexel    = GetFormatSize(baseFormat);
    const u32 bytesPerBlock    = GetFormatSize(blockFormat);
    const u32 blockSize        = GetBlockSize(blockFormat);
    const u32 log2BlockSize    = Log2Int(blockSize);

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
            Ptex::String error;

            Ptex::PtexTexture *t = cache->get((char *)filename.str, error);
            Assert(t);
            PaddedImage img = PtexToImg(arena, t, i, 0);
            t->release();

            // Assert(img.log2Width >= 2 && img.log2Height >= 2);
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

            while (depth < levels)
            {
                Vec2u scale(width == 1 ? 1u : 2u, height == 1 ? 1u : 2u);
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

    TIMED_FUNCTION(miscF);

    // NOTE: pack multiple face textures/mips into one texture atlas that gets submitted to the
    // GPU
    const u32 gpuSubmissionWidth = 4096;
    const u32 gpuOutputWidth     = gpuSubmissionWidth >> log2BlockSize;

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
        gpuSubmissionWidth * gpuSubmissionWidth * GetFormatSize(baseFormat);
    const u32 outputSize = gpuOutputWidth * gpuOutputWidth * GetFormatSize(blockFormat);
    // Allocate GPU resources
    {
        ImageDesc blockCompressedImageDesc(
            ImageType::Type2D, gpuSubmissionWidth, gpuSubmissionWidth, 1, 1, 1, baseFormat,
            MemoryUsage::GPU_ONLY,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_TILING_OPTIMAL);
        gpuSrcImages[0] = device->CreateImage(blockCompressedImageDesc);
        gpuSrcImages[1] = device->CreateImage(blockCompressedImageDesc);

        ImageDesc uavDesc(ImageType::Type2D, gpuOutputWidth, gpuOutputWidth, 1, 1, 1,
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

        readbackBuffers[0] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, outputSize,
                                                  MemoryUsage::GPU_TO_CPU);
        readbackBuffers[1] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, outputSize,
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

    SortHandles<FaceHandle, false>(handles, numFaces);

    StringBuilderMapped builder(outFilename);

    u64 textureMetadataOffset = AllocateSpace(&builder, sizeof(TextureMetadata));
    u64 faceMetadataOffset    = AllocateSpace(&builder, sizeof(FaceMetadata2) * numFaces);

    TextureMetadata textureMetadata;
    textureMetadata.numFaces = numFaces;

    StaticArray<FaceMetadata2> faceMetadatas(scratch.temp.arena, numFaces, numFaces);

    const int pageTexelWidth         = PAGE_WIDTH;
    const int pageBlockWidth         = pageTexelWidth >> log2BlockSize;
    const int pageByteSize           = Sqr(pageBlockWidth) * bytesPerBlock;
    const int submissionNumSqrtPages = gpuOutputWidth / pageBlockWidth;

    const int pageStride = pageBlockWidth * bytesPerBlock;

    struct SubmissionInfo
    {
        u32 numSubmissionX;
        u32 numSubmissionY;

        u32 outputPageWidth;
        u32 outputPageHeight;
    };

    std::vector<SubmissionInfo> submissionInfos;

    auto CopyBlockCompressedResultsToDisk = [&](u8 *dst, u32 mipSqrtNumPages) {
        // Write to face images to disk
        if (numSubmissions > 0)
        {
            const SubmissionInfo &s = submissionInfos[numSubmissions - 1];
            u32 numSubmissionX      = s.numSubmissionX;
            u32 numSubmissionY      = s.numSubmissionY;
            u32 outputPageWidth     = s.outputPageWidth;
            u32 outputPageHeight    = s.outputPageHeight;

            int lastSubmissionIndex   = (submissionIndex - 1) & 1;
            GPUBuffer *readbackBuffer = &readbackBuffers[lastSubmissionIndex];
            device->Wait(semaphores[lastSubmissionIndex]);

            descriptorSets[lastSubmissionIndex].Reset();

            MemoryZero(debugSrc[lastSubmissionIndex], submissionSize);

            // Copy tiles out to disk
            u32 srcOffset = 0;

            u32 pageStride = pageByteSize * mipSqrtNumPages;
            Assert(outputPageWidth <= submissionNumSqrtPages &&
                   outputPageHeight <= submissionNumSqrtPages);

            for (int pageY = 0; pageY < outputPageHeight; pageY++)
            {
                u32 dstOffset =
                    (numSubmissionY * submissionNumSqrtPages + pageY) * pageStride +
                    numSubmissionX * submissionNumSqrtPages * pageByteSize;
                for (int pageX = 0; pageX < outputPageWidth; pageX++)
                {
                    Vec2u srcIndex = Vec2u(pageX, pageY) * (u32)pageBlockWidth;
                    Utils::Copy(readbackBuffers[lastSubmissionIndex].mappedPtr, srcIndex,
                                gpuOutputWidth, gpuOutputWidth, dst + dstOffset, Vec2u(0, 0),
                                pageBlockWidth, pageBlockWidth, pageBlockWidth, pageBlockWidth,
                                bytesPerBlock);

                    dstOffset += pageByteSize;
                }
            }
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
        copy.extent          = Vec3u(gpuSubmissionWidth, gpuSubmissionWidth, 1);

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
            cmd->Dispatch((gpuSubmissionWidth + 7) >> 3, (gpuSubmissionWidth + 7) >> 3, 1);
        }

        // Copy from uav to buffer
        {
            copy.extent = Vec3u(gpuOutputWidth, gpuOutputWidth, 1);
            cmd->Barrier(uavImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            cmd->FlushBarriers();

            cmd->CopyImageToBuffer(readbackBuffer, uavImage, &copy, 1);
        }

        // Submit command buffer
        semaphores[submissionIndex].signalValue++;
        cmd->SignalOutsideFrame(semaphores[submissionIndex]);
        device->SubmitCommandBuffer(cmds[submissionIndex], false, true);
        cmds[submissionIndex] = device->BeginCommandBuffer(QueueType_Compute);
        cmds[submissionIndex]->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    };

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

    int totalTextureArea = 0;
    u32 maxLevel         = 0;
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        PaddedImage &currentFaceImg = images[faceIndex][0];
        totalTextureArea += currentFaceImg.width * currentFaceImg.height;
        maxLevel = Max(maxLevel, images[faceIndex].Length());

        if (currentFaceImg.width == 1 || currentFaceImg.height == 1)
        {
            int stop = 5;
        }
        // Assert(currentFaceImg.width != 1 && currentFaceImg.height != 1);
    }

    textureMetadata.numLevels = maxLevel;

    const int estimateSqrtNumPages      = std::sqrt(totalTextureArea >> (2 * PAGE_SHIFT)) + 1;
    const int estimateVirtualTexelWidth = estimateSqrtNumPages * pageTexelWidth;
    Assert(Sqr(estimateVirtualTexelWidth) >= totalTextureArea);

    int currentTotalHeight = 0;
    {
        PaddedImage &firstFaceImg   = images[handles[0].faceIndex][0];
        int currentHorizontalOffset = 0;
        int currentShelfHeight      = Min(firstFaceImg.width, firstFaceImg.height);
        for (int handleIndex = 0; handleIndex < numFaces; handleIndex++)
        {
            FaceHandle &handle          = handles[handleIndex];
            int faceIndex               = handle.faceIndex;
            PaddedImage &currentFaceImg = images[faceIndex][0];
            int cmpWidth                = Max(currentFaceImg.width, currentFaceImg.height);
            int cmpHeight               = Min(currentFaceImg.width, currentFaceImg.height);

            if (cmpHeight != currentShelfHeight ||
                currentHorizontalOffset + cmpWidth > estimateVirtualTexelWidth)
            {
                currentHorizontalOffset = 0;
                currentTotalHeight += currentShelfHeight;
                currentShelfHeight = cmpHeight;
            }
            currentHorizontalOffset += cmpWidth;
        }
        currentTotalHeight += currentShelfHeight;
    }

    const int virtualTextureWidth       = Max(currentTotalHeight, estimateVirtualTexelWidth);
    const u32 sqrtNumPages              = (virtualTextureWidth + PAGE_WIDTH - 1) >> PAGE_SHIFT;
    textureMetadata.virtualSqrtNumPages = NextPowerOfTwo(sqrtNumPages);

    for (int levelIndex = 0; levelIndex < maxLevel; levelIndex++)
    {
        const u32 mipSqrtNumPages =
            ((virtualTextureWidth >> levelIndex) + PAGE_WIDTH - 1) >> PAGE_SHIFT;
        u32 levelTexelWidth = mipSqrtNumPages * pageTexelWidth;
        u32 levelBlockWidth = mipSqrtNumPages * pageBlockWidth;
        u32 totalOutputSize = Sqr(levelBlockWidth) * bytesPerBlock;

        int numSqrtSubmissions =
            (mipSqrtNumPages * PAGE_WIDTH + gpuSubmissionWidth - 1) / gpuSubmissionWidth;
        int totalNumGPUSubmissions = Sqr(numSqrtSubmissions);

        struct FaceSubmissionInfo
        {
            Vec2u start;
            Vec2u end;
            Vec2u dstStart;
            int faceIndex;
        };
        StaticArray<StaticArray<FaceSubmissionInfo>> faceSubmissionInfos(
            scratch.temp.arena, totalNumGPUSubmissions, totalNumGPUSubmissions);
        for (int i = 0; i < totalNumGPUSubmissions; i++)
        {
            faceSubmissionInfos[i] =
                StaticArray<FaceSubmissionInfo>(scratch.temp.arena, numFaces);
        }

        currentPageOffset += Sqr(mipSqrtNumPages);
        PaddedImage &firstFaceImg                  = images[handles[0].faceIndex][levelIndex];
        textureMetadata.mipPageOffsets[levelIndex] = currentPageOffset;
        int currentHorizontalOffset                = 0;
        int currentShelfHeight = Min(firstFaceImg.width, firstFaceImg.height);
        int currentTotalHeight = 0;
        for (int handleIndex = 0; handleIndex < numFaces; handleIndex++)
        {
            FaceHandle &handle = handles[handleIndex];
            int faceIndex      = handle.faceIndex;

            if (levelIndex >= images[faceIndex].Length()) continue;

            PaddedImage &currentFaceImg = images[faceIndex][levelIndex];

            int cmpWidth  = Max(currentFaceImg.width, currentFaceImg.height);
            int cmpHeight = Min(currentFaceImg.width, currentFaceImg.height);

            if (cmpHeight != currentShelfHeight ||
                currentHorizontalOffset + cmpWidth > levelTexelWidth)
            {
                currentHorizontalOffset = 0;
                currentTotalHeight += currentShelfHeight;
                currentShelfHeight = cmpHeight;
            }
            Assert(currentTotalHeight + currentShelfHeight <= levelTexelWidth);

            // Rotate face
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
                faceMetadatas[faceIndex].rotate |= 0x80000000;

                currentFaceImg = img;
            }

            Vec2u dstLoc = Vec2u(currentHorizontalOffset, currentTotalHeight);
            faceMetadatas[faceIndex].mipOffsets[levelIndex] = dstLoc;

            if (levelIndex == 0)
            {
                faceMetadatas[faceIndex].log2Width  = img.log2Width;
                faceMetadatas[faceIndex].log2Height = img.log2Height;
                const Ptex::FaceInfo &f             = ptexFaceInfos[faceIndex];
                for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
                {
                    // Add edge borders
                    int aeid         = f.adjedge(edgeIndex);
                    int neighborFace = f.adjface(edgeIndex);
                    int rot          = (edgeIndex - aeid + 2) & 3;

                    faceMetadatas[faceIndex].neighborFaces[edgeIndex] = neighborFace;
                    faceMetadatas[faceIndex].rotate |= rot << (2 * edgeIndex);
                }
            }

            Assert(img.width <= gpuSubmissionWidth && img.height <= gpuSubmissionWidth);
            Vec2u locationInSubmission = dstLoc & (gpuSubmissionWidth - 1);
            Vec2u firstChunkMaxSize    = Vec2u(gpuSubmissionWidth) - locationInSubmission;

            FaceSubmissionInfo faceSubmissionInfo;
            faceSubmissionInfo.faceIndex = faceIndex;
            faceSubmissionInfo.start     = Vec2u(0);
            faceSubmissionInfo.end      = Min(Vec2u(img.width, img.height), firstChunkMaxSize);
            faceSubmissionInfo.dstStart = locationInSubmission;

            Vec2u submissionXY          = dstLoc / gpuSubmissionWidth;
            int faceSubmissionInfoIndex = submissionXY.y * numSqrtSubmissions + submissionXY.x;

            faceSubmissionInfos[faceSubmissionInfoIndex].push_back(faceSubmissionInfo);

            // If a face overlaps multiple 4096x4096 regions, add to those submissions
            for (int i = 1; i < 4; i++)
            {
                Vec2u offset = Vec2u(img.width, img.height);
                Vec2u end    = locationInSubmission + offset;
                if (end.x > gpuSubmissionWidth || end.y > gpuSubmissionWidth)
                {
                    Vec2u overlapStart((i & 1) ? faceSubmissionInfo.end.x : 0,
                                       (i >> 1) ? faceSubmissionInfo.end.y : 0);
                    Vec2u overlapEnd((i & 1) ? img.width : faceSubmissionInfo.end.x,
                                     (i >> 1) ? img.height : faceSubmissionInfo.end.y);
                    Vec2u dstStart((i & 1) ? 0 : locationInSubmission.x,
                                   (i >> 1) ? 0 : locationInSubmission.y);

                    FaceSubmissionInfo overlapFaceSubmissionInfo;
                    overlapFaceSubmissionInfo.faceIndex = faceIndex;
                    overlapFaceSubmissionInfo.start     = overlapStart;
                    overlapFaceSubmissionInfo.end       = overlapEnd;
                    overlapFaceSubmissionInfo.dstStart  = dstStart;

                    int overlapFaceSubmissionInfoIndex =
                        (submissionXY.y + (i >> 1)) * numSqrtSubmissions + submissionXY.x +
                        (i & 1);
                    faceSubmissionInfos[overlapFaceSubmissionInfoIndex].push_back(
                        overlapFaceSubmissionInfo);
                }
            }

            currentHorizontalOffset += cmpWidth;
        }

        if (levelIndex == 0)
        {
            // Pinning
            std::vector<Vec3u> pinnedPages;
            pinnedPages.reserve(sqrtNumPages);

            std::vector<std::vector<int>> pinnedFaces;
            pinnedFaces.reserve(sqrtNumPages);

            HashIndex hashMap(scratch.temp.arena);
            for (int handleIndex = 0; handleIndex < numFaces; handleIndex++)
            {
                FaceHandle &handle = handles[handleIndex];
                int faceIndex      = handle.faceIndex;
                u32 levelIndex     = images[faceIndex].Length() - 1;
                Vec2u page         = Vec2u(faceMetadatas[faceIndex].mipOffsets[levelIndex].x,
                                           faceMetadatas[faceIndex].mipOffsets[levelIndex].y) >>
                             u32(PAGE_SHIFT);

                Vec3u pinnedPage(page, (u32)levelIndex);

                Assert(pinnedPage.x < 65536 && pinnedPage.y < 65536);

                int hash = (int)MixBits(((u64)pinnedPage.z << 32) | ((u64)pinnedPage.y << 16) |
                                        (u64)pinnedPage.x);

                bool found = false;
                for (int pinnedPageIndex = hashMap.FirstInHash(hash); pinnedPageIndex != -1;
                     hashMap.NextInHash(pinnedPageIndex))
                {
                    if (pinnedPages[pinnedPageIndex] == pinnedPage)
                    {
                        pinnedFaces[pinnedPageIndex].push_back(faceIndex);
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    pinnedFaces.emplace_back();
                    pinnedFaces.back().push_back(faceIndex);
                    pinnedPages.push_back(pinnedPage);
                    hashMap.AddInHash(hash, pinnedPages.size() - 1);
                }
            }

            u32 *pinnedPageOffsets =
                PushArrayNoZero(scratch.temp.arena, u32, pinnedPages.size() + 1);
            u32 *pinnedPageOffsets1 = &pinnedPageOffsets[1];
            u32 total               = 0;
            for (u32 pinnedPageIndex = 0; pinnedPageIndex < pinnedPages.size();
                 pinnedPageIndex++)
            {
                pinnedPageOffsets1[pinnedPageIndex] = total;
                u32 count                           = pinnedFaces[pinnedPageIndex].size();
                total += count;
            }
            int *pinnedPageData = PushArrayNoZero(scratch.temp.arena, int, total);
            for (u32 pinnedPageIndex = 0; pinnedPageIndex < pinnedPages.size();
                 pinnedPageIndex++)
            {
                for (int pinnedFace : pinnedFaces[pinnedPageIndex])
                {
                    pinnedPageData[pinnedPageOffsets1[pinnedPageIndex]++] = pinnedFace;
                }
            }

            textureMetadata.numPinnedPages = (u32)pinnedPages.size();

            u64 pinnedPagesOffset =
                AllocateSpace(&builder, sizeof(Vec3u) * textureMetadata.numPinnedPages);
            Vec3u *outPinnedPages = (Vec3u *)GetMappedPtr(&builder, pinnedPagesOffset);
            MemoryCopy(outPinnedPages, pinnedPages.data(),
                       sizeof(Vec3u) * textureMetadata.numPinnedPages);

            u64 pinnedPagesOffsetsOffset =
                AllocateSpace(&builder, sizeof(u32) * (textureMetadata.numPinnedPages + 1));
            void *ptr = GetMappedPtr(&builder, pinnedPagesOffsetsOffset);
            MemoryCopy(ptr, pinnedPageOffsets,
                       sizeof(u32) * (textureMetadata.numPinnedPages + 1));

            u64 pinnedPagesDataOffset = AllocateSpace(&builder, sizeof(int) * total);
            ptr                       = GetMappedPtr(&builder, pinnedPagesDataOffset);
            MemoryCopy(ptr, pinnedPageData, sizeof(int) * total);
        }

        // Submit to GPU to calculate mip and block compress, then write to disk
        u64 dstTileDataOffset = AllocateSpace(&builder, totalOutputSize);
        u8 *dst               = (u8 *)GetMappedPtr(&builder, dstTileDataOffset);

        numSubmissions  = 0;
        submissionIndex = 0;
        submissionInfos.clear();

        for (int numY = 0; numY < numSqrtSubmissions; numY++)
        {
            for (int numX = 0; numX < numSqrtSubmissions; numX++)
            {
                StaticArray<FaceSubmissionInfo> &faceSubmissionInfo =
                    faceSubmissionInfos[numY * numSqrtSubmissions + numX];

                for (int i = 0; i < faceSubmissionInfo.size(); i++)
                {
                    FaceSubmissionInfo faceInfo = faceSubmissionInfo[i];
                    int faceIndex               = faceInfo.faceIndex;
                    FaceMetadata2 faceMetadata  = faceMetadatas[faceIndex];
                    PaddedImage &img            = images[faceIndex][levelIndex];

                    Vec2u faceSize = faceInfo.end - faceInfo.start;

                    Utils::Copy(img.contents, faceInfo.start, img.width, img.height,
                                debugSrc[submissionIndex], faceInfo.dstStart,
                                gpuSubmissionWidth, gpuSubmissionWidth, faceSize.y, faceSize.x,
                                img.bytesPerPixel);
                }

                Vec2u srcIndex(numX * gpuSubmissionWidth, numY * gpuSubmissionWidth);

                u32 width =
                    Min(gpuSubmissionWidth, levelTexelWidth - numX * gpuSubmissionWidth);
                u32 height =
                    Min(gpuSubmissionWidth, levelTexelWidth - numY * gpuSubmissionWidth);

                CopyBlockCompressedResultsToDisk(dst, mipSqrtNumPages);
                SubmitBlockCompressionCommandsToGPU();

                u32 outputPageWidth  = (width + PAGE_WIDTH - 1) >> PAGE_SHIFT;
                u32 outputPageHeight = (height + PAGE_WIDTH - 1) >> PAGE_SHIFT;

                SubmissionInfo submissionInfo;
                submissionInfo.numSubmissionX   = numX;
                submissionInfo.numSubmissionY   = numY;
                submissionInfo.outputPageWidth  = outputPageWidth;
                submissionInfo.outputPageHeight = outputPageHeight;
                submissionInfos.push_back(submissionInfo);

                numSubmissions++;
                submissionIndex = numSubmissions & 1;
            }
        }
        CopyBlockCompressedResultsToDisk(dst, mipSqrtNumPages);
    }

    TextureMetadata *outTextureMetadata =
        (TextureMetadata *)GetMappedPtr(&builder, textureMetadataOffset);
    FaceMetadata2 *outFaceMetadata =
        (FaceMetadata2 *)GetMappedPtr(&builder, faceMetadataOffset);
    MemoryCopy(outTextureMetadata, &textureMetadata, sizeof(TextureMetadata));
    MemoryCopy(outFaceMetadata, faceMetadatas.data, numFaces * sizeof(FaceMetadata2));

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

    Print("conversion finished for %S\n", filename);
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
    numToRead     = num == 0 ? numToRead : Min(numToRead, num);
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

VirtualTextureManager::VirtualTextureManager(Arena *arena, u32 maxSize, u32 slabSize,
                                             VkFormat format)
    : format(format), slabAllocInfoRingBuffer(arena, 1u << 20u), slabSize(slabSize)
{
    string shaderName = "../src/shaders/update_page_tables.spv";
    string data       = OS_ReadFile(arena, shaderName);
    shader            = device->CreateShader(ShaderStage::Compute, "update page tables", data);

    string clearPageTableName = "../src/shaders/clear_page_table.spv";
    string clearPageTableData = OS_ReadFile(arena, clearPageTableName);
    Shader clearPageTableShader =
        device->CreateShader(ShaderStage::Compute, "clear page table", clearPageTableData);

    descriptorSetLayout = {};
    descriptorSetLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    descriptorSetLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    push     = PushConstant(ShaderStage::Compute, 0, sizeof(PageTableUpdatePushConstant));
    pipeline = device->CreateComputePipeline(&shader, &descriptorSetLayout, &push);

    clearPageTableLayout = {};
    clearPageTableLayout.AddBinding(0, DescriptorType::StorageImage,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
    clearPageTablePipeline =
        device->CreateComputePipeline(&clearPageTableShader, &clearPageTableLayout);

    // ImageLimits limits = device->GetImageLimits();

    // Allocate page table
    pageHashTableBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               sizeof(Vec4u) * MAX_LEVEL * (1u << 20u));

    // Allocate cpu page table
    cpuPageHashTable =
        StaticArray<HashTableEntry>(arena, MAX_LEVEL * (1u << 20u), MAX_LEVEL * (1u << 20u));
    MemorySet(cpuPageHashTable.data, 0xff, sizeof(HashTableEntry) * MAX_LEVEL * (1u << 20u));

    gpuSubmissionStates = StaticArray<GPUSubmissionState>(arena, 1024);
    for (u32 i = 0; i < 1024; i++)
    {
        GPUSubmissionState state    = {};
        state.arena                 = ArenaAlloc();
        state.status                = SubmissionStatus::None;
        state.semaphore             = device->CreateSemaphore();
        state.semaphore.signalValue = 1;
        state.uploadBuffer.size     = {};
    }

    // Slabs
    {
        u32 numSlabs = maxSize / slabSize;
        slabs        = StaticArray<Slab>(arena, numSlabs);
        caches       = StaticArray<StaticArray<int>>(arena, log2MaxTileDim + 1);
        for (int i = 0; i <= log2MaxTileDim; i++)
        {
            caches.Push(StaticArray<int>(arena, log2MaxTileDim + 1));
            for (int j = 0; j <= log2MaxTileDim; j++)
            {
                caches[i].Push(-1);
            }
        }

        texArena = ArenaAlloc();

        lruHead = 0;
        lruTail = 1;

        HashTableEntry entry      = {};
        entry.lruNext             = lruTail;
        cpuPageHashTable[lruHead] = entry;
        entry.lruPrev             = lruHead;
        cpuPageHashTable[lruTail] = entry;
    }

    // Instantiate streaming system
    {
        feedbackBuffers =
            FixedArray<TransferBuffer, numPendingSubmissions>(numPendingSubmissions);
        feedbackBuffers[0] =
            device->GetReadbackBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, megabytes(8));
        feedbackBuffers[1] =
            device->GetReadbackBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, megabytes(8));
    }
}

u32 VirtualTextureManager::AllocateVirtualPages(Arena *arena, string filename)
{
    TextureInfo texInfo;
    texInfo.filename = PushStr8Copy(arena, filename);
    textureInfo.push_back(texInfo);

    return textureInfo.size() - 1;
}

bool VirtualTextureManager::AllocateMemory(int logWidth, int logHeight, SlabAllocInfo &info)
{
    int indexX = logWidth;
    int indexY = logHeight;

    int slabHead  = caches[indexX][indexY];
    int slabIndex = slabHead;

    while (slabIndex != -1)
    {
        Slab &slab = slabs[slabIndex];
        if (slab.freeList.Length())
        {
            u32 entryIndex = slab.freeList.Pop();
            Vec3u offset   = slab.entries[entryIndex];

            SlabAllocInfo result;
            result.offset     = offset.xy;
            result.layerIndex = offset.z;
            result.slabIndex  = slabIndex;
            result.entryIndex = entryIndex;

            info = result;

            return true;
        }
        slabIndex = slab.next;
    }
    Assert(slabIndex == -1);

    int blockShift   = GetBlockShift(format);
    u32 formatSize   = GetFormatSize(format);
    int newLogWidth  = Max(logWidth - blockShift, 0);
    int newLogHeight = Max(logHeight - blockShift, 0);

    u32 sizeX = Max((1u << newLogWidth), 16u);
    u32 sizeY = Max((1u << newLogHeight), 16u);

    u32 entrySizeX = 1u << newLogWidth;
    u32 entrySizeY = 1u << newLogHeight;

    if (slabs.Length() == slabs.capacity) return false;

    slabIndex = slabs.Length();

    u32 numLayers  = slabSize / (sizeX * sizeY * formatSize);
    u32 entriesX   = sizeX / (1u << logWidth);
    u32 entriesY   = sizeY / (1u << logHeight);
    u32 numEntries = entriesX * entriesY * numLayers;

    Slab slab     = slabs[slabIndex];
    slab.entries  = StaticArray<Vec3u>(texArena, numEntries);
    slab.freeList = StaticArray<int>(texArena, numEntries);

    ImageDesc desc(
        ImageType::Type2D, sizeX, sizeY, 1, 1, numLayers, format, MemoryUsage::GPU_ONLY,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_OPTIMAL);

    slab.image = device->CreateImage(desc);
    slab.bindlessIndex =
        device->BindlessIndex(&slab.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    for (u32 layer = 0; layer < numLayers; layer++)
    {
        for (u32 entryY = 0; entryY < entriesY; entryY++)
        {
            for (u32 entryX = 0; entryX < entriesX; entryX++)
            {
                Vec3u offset(entryX * entrySizeX, entryY * entrySizeY, layer);
                slab.entries.Push(offset);
                slab.freeList.Push(slab.entries.Length() - 1);
            }
        }
    }
    slab.next              = slabHead;
    caches[indexX][indexY] = slabIndex;

    u32 freeListIndex = slab.freeList.Pop();
    Vec3u offset      = slab.entries[freeListIndex];

    SlabAllocInfo result;
    result.offset     = offset.xy;
    result.layerIndex = offset.z;
    result.slabIndex  = slabIndex;
    result.entryIndex = freeListIndex;

    info = result;
    slabs.Push(slab);

    return true;
}

u64 VirtualTextureManager::CalculateHash(u32 textureIndex, u32 faceIndex, u32 mipLevel,
                                         u32 tileIndex)
{
    u64 bitsToHash = (u64)textureIndex | ((u64)tileIndex << 16) | ((u64)faceIndex << 32) |
                     ((u64)mipLevel << 60);
    u64 hash = MixBits(bitsToHash);
    return hash;
}

u32 VirtualTextureManager::UpdateHash(HashTableEntry entry, u64 hash)
{
    u64 hashIndex = hash % cpuPageHashTable.Length();
    for (;;)
    {
        if (cpuPageHashTable[hashIndex].textureIndex == ~0u)
        {
            cpuPageHashTable[hashIndex] = entry;
            return hashIndex;
        }
        hashIndex = (hashIndex + 1) % cpuPageHashTable.Length();
    }
}

int VirtualTextureManager::GetInHash(u32 textureIndex, u32 tileIndex, u32 faceID, u32 mipLevel)
{
    u64 hash      = CalculateHash(textureIndex, faceID, mipLevel, tileIndex);
    u64 hashIndex = hash % cpuPageHashTable.Length();
    for (;;)
    {
        if (cpuPageHashTable[hashIndex].textureIndex != ~0u)
        {
            HashTableEntry &testEntry = cpuPageHashTable[hashIndex];
            if (testEntry.faceID == faceID && testEntry.textureIndex == textureIndex &&
                testEntry.mipLevel == mipLevel && testEntry.tileIndex == tileIndex)
            {
                return hashIndex;
            }
        }
        else
        {
            return -1;
        }
        hashIndex = (hashIndex + 1) % cpuPageHashTable.Length();
    }
}

#if 0
void VirtualTextureManager::PinPages(CommandBuffer *cmd, u32 textureIndex, Vec3u *pinnedPages,
                                     u32 *offsets, u32 *data, u32 numPages)
{
    ScratchArena scratch;

    const u32 blockShift     = GetBlockShift(format);
    const u32 bytesPerBlock  = GetFormatSize(format);
    const u32 pageBlockWidth = PAGE_WIDTH >> blockShift;
    const u32 pageSize       = pageBlockWidth * pageBlockWidth * bytesPerBlock;

    TextureInfo &texInfo      = textureInfo[textureIndex];
    u8 *contents              = texInfo.contents;
    TextureMetadata &metadata = texInfo.metadata;

    StaticArray<PageTableUpdateRequest> updateRequests(scratch.temp.arena, numPages);
    StaticArray<BufferImageCopy> copies(scratch.temp.arena, numPages);
    GPUBuffer transferBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, numPages * pageSize, MemoryUsage::CPU_TO_GPU);
    u8 *mappedPtr    = (u8 *)transferBuffer.mappedPtr;
    u32 bufferOffset = 0;

    for (u32 pageIndex = 0; pageIndex < numPages; pageIndex++)
    {
        Vec3u &pinnedPage = pinnedPages[pageIndex];
        u32 levelIndex    = pinnedPage.z;

        u32 levelNumPages = levelIndex == 0 ? metadata.mipPageOffsets[0]
                                            : metadata.mipPageOffsets[levelIndex] -
                                                  metadata.mipPageOffsets[levelIndex - 1];
        u32 levelOffset   = levelIndex == 0 ? 0 : metadata.mipPageOffsets[levelIndex - 1];

        u32 sqrtLevelNumPages = std::sqrt(levelNumPages);

        // Allocate page from pool
        u32 freePageIndex = freePage;
        Assert(freePageIndex < physicalPages.size());
        PhysicalPage &page = physicalPages[freePageIndex];
        freePage++;

        // Create packed page table entry
        for (u32 faceDataIndex = offsets[pageIndex]; faceDataIndex < offsets[pageIndex + 1];
             faceDataIndex++)
        {
            u32 face           = data[faceDataIndex];
            Vec2u offsetInPage = texInfo.faceMetadata[face].mipOffsets[levelIndex];
            u32 packed         = PackPageTableEntry(page.page.x, page.page.y, offsetInPage.x,
                                                    offsetInPage.y, page.layer);
            u64 bitsToHash = (u64)textureIndex | ((u64)levelIndex << 16) | ((u64)face << 32);
            u64 hash       = MixBits(bitsToHash);

            PageTableUpdateRequest request;
            request.packed    = Vec3u(textureIndex | (levelIndex << 16), face, packed);
            request.hashIndex = UpdateHash(request.packed, hash);
            updateRequests.push_back(request);
        }

        // Create buffer image copy command
        BufferImageCopy copy = {};
        copy.bufferOffset    = bufferOffset;
        copy.baseLayer       = page.layer;
        copy.layerCount      = 1;
        copy.offset          = Vec3i(page.page.x << PAGE_SHIFT, page.page.y << PAGE_SHIFT, 0);
        copy.extent          = Vec3u(PAGE_WIDTH, PAGE_WIDTH, 1);

        copies.push_back(copy);

        // Copy to transfer buffer
        u32 flattenedVirtualPage = pinnedPage.y * sqrtLevelNumPages + pinnedPage.x;
        u32 pageOffset           = (levelOffset + flattenedVirtualPage) * pageSize;

        MemoryCopy(mappedPtr + bufferOffset, contents + pageOffset, pageSize);

        bufferOffset += pageSize;
    }

    PageTableUpdatePushConstant pc;
    pc.numRequests                 = numPages;
    pc.bindlessPageTableStartIndex = bindlessPageTableStartIndex;

    if (sizeof(PageTableUpdateRequest) * pc.numRequests >
        pageTableUpdateRequestUploadBuffer.size)
    {
        if (pageTableUpdateRequestUploadBuffer.size)
        {
            device->DestroyBuffer(&pageTableUpdateRequestUploadBuffer);
            device->DestroyBuffer(&pageTableUpdateRequestBuffer);
        }
        pageTableUpdateRequestUploadBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sizeof(PageTableUpdateRequest) * pc.numRequests,
            MemoryUsage::CPU_TO_GPU);
        pageTableUpdateRequestBuffer =
            device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 sizeof(PageTableUpdateRequest) * pc.numRequests);
    }

    MemoryCopy(pageTableUpdateRequestUploadBuffer.mappedPtr, updateRequests.data,
               sizeof(PageTableUpdateRequest) * pc.numRequests);
    BufferToBufferCopy bufferCopy = {};
    bufferCopy.size               = sizeof(PageTableUpdateRequest) * pc.numRequests;
    cmd->CopyBuffer(&pageTableUpdateRequestBuffer, &pageTableUpdateRequestUploadBuffer,
                    &bufferCopy, 1);
    // TransferBuffer pageTableUpdateRequestBuffer =
    //     cmd->SubmitBuffer(updateRequests.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    //                       sizeof(PageTableUpdateRequest) * pc.numRequests);

    DescriptorSet ds = descriptorSetLayout.CreateDescriptorSet();
    ds.Bind(0, &pageTableUpdateRequestBuffer);
    ds.Bind(1, &pageHashTableBuffer);
    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();
    cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmd->PushConstants(&push, &pc, descriptorSetLayout.pipelineLayout);
    cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                            descriptorSetLayout.pipelineLayout);
    cmd->Dispatch((pc.numRequests + 63) >> 6, 1, 1);

    cmd->CopyImage(&transferBuffer, &gpuPhysicalPool, copies.data, copies.Length());
}
#endif

void VirtualTextureManager::ClearTextures(CommandBuffer *cmd)
{
    cmd->ClearBuffer(&pageHashTableBuffer, ~0u);
}

///////////////////////////////////////
// Streaming/Feedback
//

Vec4u VirtualTextureManager::PackHashTableEntry(HashTableEntry &entry)
{
    Vec4u packed;
    packed.x = entry.textureIndex | (entry.tileIndex << 16u);
    packed.y = entry.faceID | (entry.mipLevel << 28u);
    packed.z = slabs[entry.slabIndex].bindlessIndex;
    packed.w = entry.offset.z | (entry.offset.x << 12u) | (entry.offset.y << 16u);
    return packed;
}
// Executes on main thread
void VirtualTextureManager::Update(CommandBuffer *computeCmd)
{
    // Update page table
    ScratchArena scratch;
    u32 currentBuffer = device->frameCount & 1;

    u32 numFeedbackRequests = ((u32 *)feedbackBuffers[currentBuffer].mappedPtr)[0];
    Vec2u *feedbackRequests = PushArrayNoZero(scratch.temp.arena, Vec2u, numFeedbackRequests);

    // Process feedback requests
    int textureFeedbackRequest = TIMED_CPU_RANGE_BEGIN();
    StaticArray<FeedbackRequest> compactedFeedbackRequests(scratch.temp.arena,
                                                           numFeedbackRequests);
    u32 hashMapSize = NextPowerOfTwo(numFeedbackRequests);
    HashIndex pageHashMap(scratch.temp.arena, hashMapSize, hashMapSize);

    // Compact feedback
    for (int requestIndex = 0; requestIndex < numFeedbackRequests; requestIndex++)
    {
        Vec2u &feedbackRequest = feedbackRequests[requestIndex];
        u32 textureIndex       = BitFieldExtractU32(feedbackRequest.x, 16, 0);
        u32 tileIndex          = BitFieldExtractU32(feedbackRequest.x, 16, 16);
        u32 faceID             = BitFieldExtractU32(feedbackRequest.y, 28, 0);
        u32 mipLevel           = BitFieldExtractU32(feedbackRequest.y, 4, 28);

        u64 packed = ((u64)feedbackRequest.y << 32u) | ((u64)feedbackRequest.x);
        i32 hash   = (i32)MixBits(packed);

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
            Assert(index == 0 || request.packedFeedbackInfo !=
                                     compactedFeedbackRequests[index - 1].packedFeedbackInfo);
            pageHashMap.AddInHash(hash, index);
        }
    }

    u32 numCompacted           = compactedFeedbackRequests.size();
    u32 numNonResidentFeedback = 0;

    int lruUsed = -1;
    // If requested page is resident Update LRU. Otherwise
    for (int requestIndex = 0; requestIndex < compactedFeedbackRequests.size(); requestIndex++)
    {
        FeedbackRequest fr    = compactedFeedbackRequests[requestIndex];
        Vec2u feedbackRequest = fr.packedFeedbackInfo;
        u32 textureIndex      = BitFieldExtractU32(feedbackRequest.x, 16, 0);
        u32 tileIndex         = BitFieldExtractU32(feedbackRequest.x, 16, 16);
        u32 faceID            = BitFieldExtractU32(feedbackRequest.y, 28, 0);
        u32 mipLevel          = BitFieldExtractU32(feedbackRequest.y, 4, 28);

        TextureInfo &texInfo = textureInfo[textureIndex];
        int entryIndex       = GetInHash(textureIndex, tileIndex, faceID, mipLevel);

        // Move to head of LRU if requested page is already resident
        if (entryIndex != -1)
        {
            HashTableEntry &entry = cpuPageHashTable[entryIndex];
            lruUsed               = lruUsed == -1 ? entryIndex : lruUsed;
            if (entry.lruNext != -1 && entry.lruPrev != -1)
            {
                UnlinkLRU(entryIndex);
                LinkLRU(entryIndex);
            }
        }
        // Otherwise, page needs to be mapped
        else
        {
            compactedFeedbackRequests[numNonResidentFeedback++] = fr;
        }
    }
    compactedFeedbackRequests.size() = numNonResidentFeedback;

    Print("Num feedback: %u, num compacted: %u, num non resident: %u\n", numFeedbackRequests,
          numCompacted, numNonResidentFeedback);

    Array<PageTableUpdateRequest> updateRequests(scratch.temp.arena, numNonResidentFeedback);
    StaticArray<SlabAllocInfo> newSlabAllocInfo(scratch.temp.arena, numNonResidentFeedback);

    for (FeedbackRequest fr : compactedFeedbackRequests)
    {
        Vec2u feedbackRequest = fr.packedFeedbackInfo;
        u32 textureIndex      = BitFieldExtractU32(feedbackRequest.x, 16, 0);
        u32 tileIndex         = BitFieldExtractU32(feedbackRequest.x, 16, 16);
        u32 faceID            = BitFieldExtractU32(feedbackRequest.y, 28, 0);
        u32 mipLevel          = BitFieldExtractU32(feedbackRequest.y, 4, 28);

        TextureInfo &texInfo = textureInfo[textureIndex];
        Ptex::String error;
        Ptex::PtexTexture *t = cache->get((char *)texInfo.filename.str, error);
        Assert(t);
        Ptex::FaceInfo fi = t->getFaceInfo(faceID);

        Assert(mipLevel >= fi.res.ulog2 || mipLevel >= fi.res.vlog2);
        int log2Width  = Clamp((int)fi.res.ulog2 - (int)mipLevel, 0, (int)log2MaxTileDim);
        int log2Height = Clamp((int)fi.res.vlog2 - (int)mipLevel, 0, (int)log2MaxTileDim);
        int minDim     = Min(log2Width, log2Height);
        int maxDim     = Max(log2Width, log2Height);

        SlabAllocInfo info;
        bool allocated = AllocateMemory(maxDim, minDim, info);

        int entryIndex = -1;
        if (!allocated)
        {
            for (;;)
            {
                int entryToEvict = cpuPageHashTable[lruTail].lruPrev;
                if (entryToEvict == lruUsed)
                {
                    ErrorExit(0, "Ran out of entries in the LRU\n");
                    break;
                }

                UnlinkLRU(entryToEvict);
                HashTableEntry &entry = cpuPageHashTable[entryToEvict];
                Slab &slab            = slabs[entry.slabIndex];
                slab.freeList.Push(entry.slabEntryIndex);

                if (slab.log2Width == maxDim && slab.log2Height == minDim)
                {
                    PageTableUpdateRequest evictRequest;
                    evictRequest.packed    = Vec4u(~0u);
                    evictRequest.hashIndex = entryToEvict;
                    updateRequests.Push(evictRequest);

                    entryIndex = entryToEvict;
                    allocated  = AllocateMemory(maxDim, minDim, info);
                    Assert(allocated);
                    break;
                }
            }
        }
        info.feedbackRequest = feedbackRequest;
        newSlabAllocInfo.Push(info);
    }

    // TODO:
    // 1. submit the feedback requests + their destination addresuse to a ring buffer
    // 2. start worker threads that read some amount of requests, load the data requested and
    // submit. small allocations are not submitted and instead written back to another ring
    // buffer accessed by the host. a semaphore is also written back.
    // 3. after submitting the data, the threads write back small allocation data
    // 4. in main loop, go through the ring buffer, and if the semaphore is signaled
    // process the small allocations, and also update the gpu hash table for all mapped entries
    // 5. if there isn't enough space to put on ring buffer, don't write the entries

    // TODO: this stalls  if the ring buffer is full...
    slabAllocInfoRingBuffer.SynchronizedWrite(&slabInfoMutex, newSlabAllocInfo.data,
                                              newSlabAllocInfo.Length());

    for (;;)
    {
        BeginMutex(&submissionMutex);
        u32 submissionIndex               = submissionReadIndex % gpuSubmissionStates.Length();
        GPUSubmissionState &state         = gpuSubmissionStates[submissionIndex];
        SubmissionStatus submissionStatus = state.status;
        EndMutex(&submissionMutex);

        if (submissionStatus == SubmissionStatus::None) break;
        else if (submissionStatus == SubmissionStatus::Empty)
        {
            BeginMutex(&submissionMutex);
            submissionReadIndex++;
            EndMutex(&submissionMutex);
            continue;
        }

        state.status = SubmissionStatus::Empty;
        u64 semValue =
            device->GetSemaphoreValue(gpuSubmissionStates[submissionIndex].semaphore);

        if (state.semaphore.signalValue == semValue)
        {
            state.semaphore.signalValue++;

            // Update hash table
            for (SlabAllocInfo &info : state.slabAllocInfos)
            {
                Vec2u fr = info.feedbackRequest;
                u64 hash = MixBits((u64)fr.x | ((u64)fr.y));

                u32 textureIndex = BitFieldExtractU32(fr.x, 16, 0);
                u32 tileIndex    = BitFieldExtractU32(fr.x, 16, 16);
                u32 faceID       = BitFieldExtractU32(fr.y, 28, 0);
                u32 mipLevel     = BitFieldExtractU32(fr.y, 4, 28);

                HashTableEntry entry(textureIndex, faceID, tileIndex, mipLevel, info.slabIndex,
                                     info.entryIndex, Vec3u(info.offset, info.layerIndex));

                u32 hashIndex = UpdateHash(entry, hash);
                LinkLRU(hashIndex);

                PageTableUpdateRequest request;
                request.packed    = PackHashTableEntry(entry);
                request.hashIndex = hashIndex;
                updateRequests.Push(request);

                Slab &slab = slabs[info.slabIndex];
                computeCmd->Barrier(
                    &slab.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT, QueueType_Ignored, QueueType_Ignored, 0, ~0u,
                    info.layerIndex, 1);
            }

            // Allocate small entries
            for (SmallAllocation &info : state.smallAllocs)
            {
                Slab &slab = slabs[info.info.slabIndex];
                // TODO
                if (slab.image.lastLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                {
                    computeCmd->Barrier(&slab.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        VK_PIPELINE_STAGE_2_NONE,
                                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_NONE,
                                        VK_ACCESS_2_TRANSFER_WRITE_BIT);
                }
            }
            computeCmd->FlushBarriers();

            device->DestroyBuffer(&state.uploadBuffer);
            GPUBuffer &uploadBuffer = state.uploadBuffer;
            u32 maxSmallUploadSize =
                state.smallAllocs.Length() * 8 * 16 * GetFormatSize(format);
            uploadBuffer = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                maxSmallUploadSize, MemoryUsage::CPU_TO_GPU);

            u32 bufferOffset = 0;
            for (SmallAllocation &info : state.smallAllocs)
            {
                Slab &slab = slabs[info.info.slabIndex];
                MemoryCopy((u8 *)uploadBuffer.mappedPtr + bufferOffset, info.buffer,
                           info.size);

                BufferImageCopy copy = {};
                copy.bufferOffset    = bufferOffset;
                copy.baseLayer       = info.info.layerIndex;
                copy.layerCount      = 1;
                copy.offset          = Vec3i(info.info.offset.x, info.info.offset.y, 0);
                copy.extent          = Vec3u(info.width, info.height, 1);
                computeCmd->CopyImage(&uploadBuffer, &slab.image, &copy, 1);

                bufferOffset += info.size;
            }

            Print("num small allocs: %u, size: %u\n", state.smallAllocs.Length(),
                  bufferOffset);

            for (SmallAllocation &info : state.smallAllocs)
            {
                Slab &slab = slabs[info.info.slabIndex];
                Vec2u fr   = info.info.feedbackRequest;

                u64 hash = MixBits((u64)fr.x | ((u64)fr.y));

                u32 textureIndex = BitFieldExtractU32(fr.x, 16, 0);
                u32 tileIndex    = BitFieldExtractU32(fr.x, 16, 16);
                u32 faceID       = BitFieldExtractU32(fr.y, 28, 0);
                u32 mipLevel     = BitFieldExtractU32(fr.y, 4, 28);

                HashTableEntry entry(textureIndex, faceID, tileIndex, mipLevel,
                                     info.info.slabIndex, info.info.entryIndex,
                                     Vec3u(info.info.offset, info.info.layerIndex));

                u32 hashIndex = UpdateHash(entry, hash);
                LinkLRU(hashIndex);

                PageTableUpdateRequest request;
                request.packed    = PackHashTableEntry(entry);
                request.hashIndex = hashIndex;
                updateRequests.Push(request);

                if (slab.image.lastLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                {
                    computeCmd->Barrier(
                        &slab.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                }
            }
            computeCmd->FlushBarriers();

            ArenaClear(state.arena);

            BeginMutex(&submissionMutex);
            submissionReadIndex++;
            EndMutex(&submissionMutex);
        }
        else
        {
            break;
        }
    }

    if (updateRequests.Length() * sizeof(PageTableUpdateRequest) > requestUploadBuffer.size)
    {
        if (requestUploadBuffer.size)
        {
            device->DestroyBuffer(&requestUploadBuffer);
            device->DestroyBuffer(&requestBuffer);
        }
        requestUploadBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sizeof(PageTableUpdateRequest) * updateRequests.Length(), MemoryUsage::CPU_TO_GPU);
        requestBuffer =
            device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 sizeof(PageTableUpdateRequest) * updateRequests.Length());

        BufferToBufferCopy copy = {};
        copy.size               = sizeof(PageTableUpdateRequest) * updateRequests.Length();
        MemoryCopy(requestUploadBuffer.mappedPtr, updateRequests.data, copy.size);

        computeCmd->CopyBuffer(&requestBuffer, &requestUploadBuffer, &copy, 1);

        computeCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->FlushBarriers();

        Print("Evicting %u entries\n", updateRequests.Length());

        PageTableUpdatePushConstant pc;
        pc.numRequests = updateRequests.Length();

        computeCmd->StartBindingCompute(pipeline, &descriptorSetLayout)
            .Bind(&requestBuffer)
            .Bind(&pageHashTableBuffer)
            .PushConstants(&push, &pc)
            .End();
        computeCmd->Dispatch((pc.numRequests + 63) >> 6, 1, 1);
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->FlushBarriers();
    }

    TIMED_RANGE_END(textureFeedbackRequest);

    u32 totalNumRequests = newSlabAllocInfo.Length();

    // Dispatch worker threads to upload data to GPU
    u32 maxThreads = OS_NumProcessors();
    u32 numThreads = Min(maxThreads, totalNumRequests / 1024);
    for (u32 i = 0; i < numThreads; i++)
    {
        scheduler.Schedule([&](u32 jobID) {
            for (;;)
            {
                BeginMutex(&submissionMutex);
                if (submissionWriteIndex + 1 >=
                    submissionReadIndex + gpuSubmissionStates.Length())
                {
                    EndMutex(&submissionMutex);
                    return;
                }
                u32 submissionIndex               = submissionWriteIndex++;
                GPUSubmissionState &state         = gpuSubmissionStates[submissionIndex];
                SubmissionStatus submissionStatus = state.status;
                Assert(submissionStatus == SubmissionStatus::Empty);
                EndMutex(&submissionMutex);

                u32 num = 1024;
                SlabAllocInfo *infos =
                    slabAllocInfoRingBuffer.SynchronizedRead(&slabInfoMutex, state.arena, num);

                if (!infos) return;

                state.status         = SubmissionStatus::Pending;
                state.smallAllocs    = StaticArray<SmallAllocation>(state.arena, num);
                state.slabAllocInfos = StaticArray<SlabAllocInfo>(state.arena, num);

                ScratchArena scratch;
                ChunkedLinkedList<u8> data(scratch.temp.arena);

                CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Copy);

                struct GPUTextureSubmitInfo
                {
                    u32 slabIndex;
                    u32 layerIndex;
                    u32 bufferOffset;
                    u32 width;
                    u32 height;
                };

                StaticArray<GPUTextureSubmitInfo> submitInfos(scratch.temp.arena, num);
                for (u32 requestIndex = 0; requestIndex < num; requestIndex++)
                {
                    SlabAllocInfo &slabAllocInfo = infos[requestIndex];
                    Vec2u fr                     = slabAllocInfo.feedbackRequest;

                    u32 textureIndex = BitFieldExtractU32(fr.x, 16, 0);
                    u32 tileIndex    = BitFieldExtractU32(fr.x, 16, 16);
                    u32 faceID       = BitFieldExtractU32(fr.y, 28, 0);
                    u32 mipLevel     = BitFieldExtractU32(fr.y, 4, 28);

                    TextureInfo &texInfo = textureInfo[textureIndex];

                    Ptex::String error;
                    Ptex::PtexTexture *t = cache->get((char *)texInfo.filename.str, error);
                    Ptex::FaceInfo fi    = t->getFaceInfo(faceID);

                    Assert(mipLevel <= fi.res.ulog2);
                    Assert(mipLevel <= fi.res.vlog2);

                    int log2Width  = Max((int)fi.res.ulog2 - (int)mipLevel, 0);
                    int log2Height = Max((int)fi.res.vlog2 - (int)mipLevel, 0);
                    Ptex::Res res(log2Width, log2Height);

                    Assert(t);
                    Ptex::PtexPtr<Ptex::PtexFaceData> d(t->getData(faceID, res));

                    u32 numChannels     = t->numChannels();
                    u32 bytesPerChannel = Ptex::DataSize(t->dataType());
                    u32 bytesPerPixel   = numChannels * bytesPerChannel;
                    int stride          = fi.res.u() * bytesPerPixel;
                    int rowlen          = stride;
                    int size            = stride * fi.res.v();

                    char *buffer = PushArrayNoZero(scratch.temp.arena, char, size);

                    u32 numTilesU = Max(res.u() >> 7u, 1);
                    u32 numTilesV = Max(res.v() >> 7u, 1);

                    u32 offsetU = 128u * (tileIndex % numTilesU);
                    u32 offsetV = 128u * (tileIndex / numTilesU);

                    int dataResU = res.u();
                    int dataResV = res.v();

                    if (d->isConstant())
                    {
                        Ptex::PtexUtils::fill(d->getData(), buffer, stride, res.u(), res.v(),
                                              bytesPerPixel);
                    }
                    else if (d->isTiled())
                    {
                        // loop over tiles
                        Ptex::Res tileres = d->tileRes();
                        int ntilesu       = res.ntilesu(tileres);
                        int ntilesv       = res.ntilesv(tileres);
                        int tileures      = tileres.u();
                        int tilevres      = tileres.v();

                        u32 ptexTileIndex =
                            ntilesu * (offsetV / tilevres) + offsetU / tileures;

                        dataResU = tileures;
                        dataResV = tilevres;

                        int tilerowlen   = bytesPerPixel * tileures;
                        int tile         = 0;
                        char *dsttilerow = (char *)buffer;
                        char *dsttile    = dsttilerow;

                        Ptex::PtexPtr<Ptex::PtexFaceData> t(d->getTile(ptexTileIndex));
                        if (t->isConstant())
                            Ptex::PtexUtils::fill(t->getData(), dsttile, stride, tileures,
                                                  tilevres, bytesPerPixel);
                        else
                            Ptex::PtexUtils::copy(t->getData(), tilerowlen, dsttile, stride,
                                                  tilevres, tilerowlen);
                    }
                    else
                    {
                        Ptex::PtexUtils::copy(d->getData(), rowlen, buffer, stride, res.v(),
                                              rowlen);
                    }

                    if (dataResU >= maxTileDim || dataResV >= maxTileDim)
                    {
                        int dstStride = Min(stride, int(maxTileDim * bytesPerPixel));
                        int vres      = Min(dataResV, int(maxTileDim));

                        char *newBuffer =
                            PushArrayNoZero(scratch.temp.arena, char, dstStride *vres);

                        Utils::Copy(buffer, stride, newBuffer, dstStride, vres, dstStride);

                        stride = dstStride;
                        rowlen = dstStride;

                        dataResU = Min(dataResU, (int)maxTileDim);
                        dataResV = Min(dataResV, (int)maxTileDim);
                        buffer   = newBuffer;
                    }

                    PaddedImage currentFaceImg;
                    currentFaceImg.contents      = 0;
                    currentFaceImg.width         = dataResU;
                    currentFaceImg.height        = dataResV;
                    currentFaceImg.log2Width     = Log2Int(dataResU);
                    currentFaceImg.log2Height    = Log2Int(dataResV);
                    currentFaceImg.bytesPerPixel = GetFormatSize(format);
                    currentFaceImg.strideNoBorder =
                        currentFaceImg.width * currentFaceImg.bytesPerPixel;
                    currentFaceImg.borderSize       = 0;
                    currentFaceImg.strideWithBorder = currentFaceImg.strideNoBorder;

                    u32 bufferOffset = data.Length();
                    if (t->alphaChannel() == -1)
                    {
                        currentFaceImg.contents =
                            PushArrayNoZero(scratch.temp.arena, u8, size);
                        u32 dstBytesPerPixel = GetFormatSize(format);

                        auto *node = data.AddNode(size);

                        u32 dstOffset = 0;
                        u32 srcOffset = 0;
                        u8 alpha      = 255;
                        for (int i = 0; i < currentFaceImg.GetPaddedWidth() *
                                                currentFaceImg.GetPaddedHeight();
                             i++)
                        {
                            MemoryCopy(currentFaceImg.contents + dstOffset, buffer + srcOffset,
                                       bytesPerPixel);
                            MemorySet(currentFaceImg.contents + dstOffset + bytesPerPixel,
                                      alpha, bytesPerChannel);
                            dstOffset += dstBytesPerPixel;
                            srcOffset += bytesPerPixel;
                        }
                        bytesPerPixel = dstBytesPerPixel;
                    }
                    else
                    {
                        currentFaceImg.contents = (u8 *)buffer;
                    }

                    if (dataResU < dataResV)
                    {
                        PaddedImage img = currentFaceImg;
                        img.contents    = PushArray(scratch.temp.arena, u8, size);
                        Swap(img.width, img.height);
                        Swap(img.log2Width, img.log2Height);
                        img.strideNoBorder   = img.width * img.bytesPerPixel;
                        img.strideWithBorder = img.GetPaddedWidth() * img.bytesPerPixel;
                        img.WriteRotated(currentFaceImg, Vec2u(0, 0), Vec2u(0, 0), 1,
                                         currentFaceImg.GetPaddedHeight(),
                                         currentFaceImg.GetPaddedWidth(),
                                         currentFaceImg.GetPaddedWidth(),
                                         currentFaceImg.GetPaddedHeight(), Vec2u(0, 0));

                        currentFaceImg = img;
                    }
                    t->release();

                    if (currentFaceImg.height < 16)
                    {
                        char *out = PushArrayNoZero(state.arena, char, size);
                        MemoryCopy(out, currentFaceImg.contents, size);

                        SmallAllocation smallAlloc;
                        smallAlloc.info   = state.slabAllocInfos[requestIndex];
                        smallAlloc.buffer = (u8 *)out;
                        smallAlloc.size   = size;
                        smallAlloc.width  = currentFaceImg.width;
                        smallAlloc.height = currentFaceImg.height;

                        state.smallAllocs.Push(smallAlloc);
                    }
                    else
                    {
                        auto *node = data.AddNode(size);
                        MemoryCopy(node->values, currentFaceImg.contents, size);

                        GPUTextureSubmitInfo submitInfo;
                        submitInfo.slabIndex    = slabAllocInfo.slabIndex;
                        submitInfo.layerIndex   = slabAllocInfo.layerIndex;
                        submitInfo.bufferOffset = bufferOffset;
                        submitInfo.width        = currentFaceImg.width;
                        submitInfo.height       = currentFaceImg.height;

                        submitInfos.Push(submitInfo);
                    }
                }

                for (GPUTextureSubmitInfo &info : submitInfos)
                {
                    Slab &slab = slabs[info.slabIndex];
                    cmd->Barrier(&slab.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                 VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 QueueType_Ignored, QueueType_Ignored, 0, ~0u, info.layerIndex,
                                 1);
                }
                cmd->FlushBarriers();

                GPUBuffer &uploadBuffer = state.uploadBuffer;
                if (uploadBuffer.size)
                {
                    device->DestroyBuffer(&uploadBuffer);
                }
                uploadBuffer = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    data.Length(), MemoryUsage::CPU_TO_GPU);
                data.Flatten((u8 *)uploadBuffer.mappedPtr);

                for (GPUTextureSubmitInfo &info : submitInfos)
                {
                    BufferImageCopy copy = {};
                    copy.bufferOffset    = info.bufferOffset;
                    copy.baseLayer       = info.layerIndex;
                    copy.layerCount      = 1;
                    copy.offset          = Vec3i(0);
                    copy.extent          = Vec3u(info.width, info.height, 1);
                    cmd->CopyImage(&uploadBuffer, &slabs[info.slabIndex].image, &copy, 1);
                }

                cmd->SignalOutsideFrame(state.semaphore);
                device->SubmitCommandBuffer(cmd, false, true);
            }
        });
    }

#if 0
    u64 readIndex  = readSubmission.load(std::memory_order_relaxed);
    u64 writeIndex = writeSubmission.load(std::memory_order_acquire);
    Assert(readIndex <= writeIndex);
    if (readIndex == writeIndex) return;

    u64 semaphoreValue = device->GetSemaphoreValue(submissionSemaphore[readIndex]);
    if (readIndex >= semaphoreValue - 1) return;

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
        computeCmd->FlushBarriers();
    }

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
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
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
        if (uploadDeviceBuffer->size < uploadBuffer.size())
        {
            if (uploadDeviceBuffer->size)
            {
                device->DestroyBuffer(uploadDeviceBuffer);
            }
            *uploadDeviceBuffer =
                device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, uploadBuffer.size(),
                                     MemoryUsage::CPU_TO_GPU);
        }

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
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_NONE,
                            VK_ACCESS_2_SHADER_READ_BIT, QueueType_Copy, computeCmdQueue);
        transferCmd->FlushBarriers();
        computeCmd->FlushBarriers();
    }

    // Make sure all memory writes are visible
    readSubmission.store(readIndex + 1, std::memory_order_release);
#endif
}

void VirtualTextureManager::UnlinkLRU(int entryIndex)
{
    HashTableEntry &entry = cpuPageHashTable[entryIndex];
    Assert(entry.lruPrev != -1 && entry.lruNext != -1);

    int prevEntry                       = entry.lruPrev;
    int nextEntry                       = entry.lruNext;
    cpuPageHashTable[prevEntry].lruNext = nextEntry;
    cpuPageHashTable[nextEntry].lruPrev = prevEntry;
}

void VirtualTextureManager::LinkLRU(int index)
{
    int nextEntry                   = cpuPageHashTable[lruHead].lruNext;
    cpuPageHashTable[index].lruNext = nextEntry;
    cpuPageHashTable[index].lruPrev = lruHead;

    cpuPageHashTable[nextEntry].lruPrev = index;
    cpuPageHashTable[index].lruNext     = index;
}

#if 0
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

        // Wait until the transfer buffer is read on the GPU before writing new data
        u64 writeIndex = writeSubmission.load(std::memory_order_relaxed);
        u64 readIndex  = readSubmission.load(std::memory_order_acquire);
        if (readIndex + numPendingSubmissions <= writeIndex || numFeedbackRequests == 0)
        {
            std::this_thread::yield();
            continue;
        }

        u32 writeDoubleBufferIndex = writeIndex & 1;

        StaticArray<BufferImageCopy> copies(scratch.temp.arena, numNonResidentFeedback);

        StaticArray<PageTableUpdateRequest> evictRequests(scratch.temp.arena,
                                                          numNonResidentFeedback);

        StaticArray<PageTableUpdateRequest> mapRequests(scratch.temp.arena,
                                                        numNonResidentFeedback);

        auto &uploadBuffer  = uploadBuffers[writeDoubleBufferIndex];
        uploadBuffer.size() = 0;

        if (freePage >= physicalPages.size())
        {
            Assert(0);
            Print("Evicting %u pages\n", numNonResidentFeedback);
        }

        // Evict to make space for new entries while populating feedback buffer
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Copy);
        for (FeedbackRequest fr : compactedFeedbackRequests)
        {
            Vec2u feedbackRequest = fr.packedFeedbackInfo;
            u32 virtualPageX      = feedbackRequest.x & 0xffff;
            u32 virtualPageY      = feedbackRequest.x >> 16;
            u32 textureIndex      = feedbackRequest.y & 0x00ffffff;
            u32 mipLevel          = feedbackRequest.y >> 24;
            u32 faceID            = 0;
            u32 tileIndex         = 0;

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
            // TODO IMPORTANT

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
                pageIndex                  = physicalPages[lruTail].prevPage;
                PhysicalPage &physicalPage = physicalPages[pageIndex];
                Vec2u virtualPage          = physicalPage.virtualPage;
                u32 level                  = physicalPage.level;

                // Replace previously mapped entry with a coarser mip

                UnlinkLRU(pageIndex);
            }

            Assert(pageIndex != ~0u);

            PhysicalPage &page = physicalPages[pageIndex];
            page.virtualPage   = 0;
            page.level         = mipLevel;

            u32 mapPackedEntry = 0;
            // PackPageTableEntry(page.page.x, page.page.y, mipLevel, page.layer);
            // Update CPU page table

            LinkLRU(pageIndex);

            // Get the ptex data
            // benefits:
            // 1. no offline texture processing
            // 2. downsides: no block compression... so cache fills up faster
            // note: #2 may not matter. #1 is a definite benefit...

            // copies.push_back(copy);

            // u32 virtualPage = virtualPageY * sqrtLevelNumPages + virtualPageX;
            // u32 pageOffset  = (levelOffset + virtualPage) * pageByteSize;
            MemoryCopy(uploadBuffer.data + uploadBuffer.size(), texInfo.contents + pageOffset,
                       pageByteSize);
            uploadBuffer.size() += pageByteSize;

            PageTableUpdateRequest mapRequest;
            mapRequest.hashIndex = 0;
            mapRequest.packed    = mapPackedEntry;
            mapRequests.push_back(mapRequest);
        }
        Semaphore &sem  = submissionSemaphore[writeDoubleBufferIndex];
        sem.signalValue = writeIndex + 1;
        cmd->SignalOutsideFrame(sem);
        device->SubmitCommandBuffer(cmd, false, true);

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
#endif

} // namespace rt
