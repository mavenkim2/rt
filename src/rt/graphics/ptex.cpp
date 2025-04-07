#include "../base.h"
#include "../memory.h"
#include "../string.h"
#include "../scene.h"
#include "../integrate.h"
#include "vulkan.h"
#include "ptex.h"
#include <Ptexture.h>
#include <PtexReader.h>
#include <cstring>

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
                               u32 stride, u32 borderSize, u32 bytesPerPixel)
{
    Assert(p.x < width + 2 * borderSize && p.y < height + 2 * borderSize);
    return (u8 *)contents + stride * p.y + p.x * bytesPerPixel;
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
    return (u8 *)Utils::GetContentsAbsoluteIndex(contents, p, width, height, strideWithBorder,
                                                 borderSize, bytesPerPixel);
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
void PaddedImage::WriteRotatedBorder(PaddedImage &other, Vec2u srcStart, Vec2u dstStart,
                                     int edgeIndex, int rotate, int srcVLen, int srcRowLen,
                                     int dstVLen, int dstRowLen, Vec2u scale)
{
    int uStart = srcStart.x;
    int vStart = srcStart.y;

    Assert(bytesPerPixel == other.bytesPerPixel);

    u8 *src       = other.GetContentsRelativeIndex(srcStart);
    u32 srcStride = other.strideWithBorder;
    ScratchArena scratch;

    // How this works.
    // 1. if rotate > 0, take the src image and rotate the border by rotate into a temporary
    // buffer
    // 2. copy from the temp buffer to the destination
    int pixelsToWrite = Max(1 << scale.x, 1 << scale.y);
    if (rotate != 0)
    {
        int duplicateStep = (edgeIndex & 1) ? dstRowLen * bytesPerPixel : 1;

        u32 size       = dstVLen * dstRowLen * bytesPerPixel;
        u8 *tempBuffer = PushArrayNoZero(scratch.temp.arena, u8, size);
        for (int v = 0; v < srcVLen; v++)
        {
            for (int u = 0; u < srcRowLen; u++)
            {
                // int newU = u - ((edgeIndex & 1) ? 0 : (dstDim - srcDim));
                // int newV = v - ((edgeIndex & 1) ? (dstDim - srcDim) : 0);

                Vec2i dstAddress;
                switch (rotate)
                {
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
                    default: Assert(0);
                }

                Vec2i dstPos(dstAddress);
                dstPos.x <<= scale.x;
                dstPos.y <<= scale.y;

                Vec2u srcPos(u, v);
                srcPos += srcStart;

                u32 offset = dstPos.y * dstRowLen * bytesPerPixel + dstPos.x;

                for (int i = 0; i < pixelsToWrite; i++)
                {
                    Assert(offset < size);
                    MemoryCopy(tempBuffer + offset, other.GetContentsRelativeIndex(srcPos),
                               bytesPerPixel);
                    offset += duplicateStep;
                }
            }
        }
        src           = tempBuffer;
        srcStride     = dstRowLen * bytesPerPixel;
        pixelsToWrite = 1;
    }

    if (pixelsToWrite > 1)
    {
        // Have to copy one pixel at a time in the horizontal case
        int writeSize, dstSkip, dstStride;
        if (edgeIndex & 1)
        {
            Assert(srcRowLen == dstRowLen);
            writeSize = srcRowLen * bytesPerPixel;
            dstSkip   = strideWithBorder;
            dstStride = pixelsToWrite * strideWithBorder;
        }
        else
        {
            writeSize = bytesPerPixel;
            dstSkip   = writeSize;
            dstStride = strideWithBorder;
        }

        const char *sptr = (const char *)src;
        char *dptr       = (char *)GetContentsAbsoluteIndex(dstStart);
        for (const char *end = sptr + srcVLen * other.strideWithBorder; sptr != end;)
        {
            char *writeDst      = dptr;
            const char *readSrc = sptr;
            for (const char *rowend = readSrc + srcRowLen * bytesPerPixel; readSrc != rowend;)
            {
                for (int i = 0; i < pixelsToWrite; i++)
                {
                    MemoryCopy(writeDst, readSrc, writeSize);
                    writeDst += dstSkip;
                }
                readSrc += writeSize;
            }
            sptr += other.strideWithBorder;
            dptr += dstStride;
        }
    }
    else
    {
        Utils::Copy(src, srcStride, GetContentsAbsoluteIndex(dstStart), strideWithBorder,
                    dstVLen, dstRowLen * bytesPerPixel);
    }
}

PaddedImage PtexToImg(Arena *arena, Ptex::PtexTexture *ptx, int faceID, int borderSize,
                      bool flip)
{
    Assert(faceID >= 0 && faceID < ptx->numFaces());

    u32 numChannels = ptx->numChannels();
    u32 aChan       = ptx->alphaChannel();

    Ptex::FaceInfo fi = ptx->getFaceInfo(faceID);

    int u = fi.res.u();
    int v = fi.res.v();

    Assert(IsPow2(u) && IsPow2(v));

    u32 bytesPerPixel = numChannels * Ptex::DataSize(ptx->dataType());
    int stride        = (u + 2 * borderSize) * bytesPerPixel;
    int size          = stride * (v + 2 * borderSize);
    u8 *data          = PushArray(arena, u8, size);
    int rowlen        = (u * bytesPerPixel);
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

    return result;
}

Vec3f GammaToLinear(const u8 rgb[3])
{
    return Vec3f(Pow(rgb[0] / 255.f, 2.2f), Pow(rgb[1] / 255.f, 2.2f),
                 Pow(rgb[2] / 255.f, 2.2f));
}

void LinearToGamma(const Vec3f &rgb, u8 *out)
{
    out[0] = u8(Clamp(Pow(rgb.x * 255.f, 1 / 2.2f), 0.f, 255.f));
    out[1] = u8(Clamp(Pow(rgb.y * 255.f, 1 / 2.2f), 0.f, 255.f));
    out[2] = u8(Clamp(Pow(rgb.z * 255.f, 1 / 2.2f), 0.f, 255.f));
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

void Convert(string filename)
{
    ScratchArena scratch;
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

    const u32 bytesPerBlock      = 8;
    const u32 blockSize          = GetBlockSize(VK_FORMAT_BC1_RGB_UNORM_BLOCK);
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

        inputBinding  = bcLayout.AddBinding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                            VK_SHADER_STAGE_COMPUTE_BIT);
        outputBinding = bcLayout.AddBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            VK_SHADER_STAGE_COMPUTE_BIT);
        bcLayout.AddImmutableSamplers();
        pipeline    = device->CreateComputePipeline(&shader, &bcLayout);
        initialized = true;
    }

    // Create all face images and mip maps
    StaticArray<PaddedImage> images(scratch.temp.arena, numFaces, numFaces);
    // StaticArray<StaticArray<PaddedImage>> images(scratch.temp.arena, numFaces, numFaces);
    // StaticArray<int> numLevels(scratch.temp.arena, numFaces, numFaces);
    StaticArray<GPUImage> tempUavs(scratch.temp.arena, numFaces, numFaces);
    StaticArray<GPUBuffer> blockCompressedImages(scratch.temp.arena, numFaces, numFaces);
    StaticArray<TransferBuffer> gpuSrcImages(scratch.temp.arena, numFaces, numFaces);

    std::vector<Tile> tiles;

    struct TileMetadata
    {
        u32 offset;
        u32 count;
    };
    StaticArray<TileMetadata> metaData(scratch.temp.arena, numFaces, numFaces);

    for (int i = 0; i < numFaces; i++)
    {
        PaddedImage img = PtexToImg(scratch.temp.arena, t, i, borderSize, false);
        images[i]       = img;
    }
#if 0
    for (int i = 0; i < numFaces; i++)
    {
        PaddedImage img = PtexToImg(scratch.temp.arena, t, i, borderSize, false);
        int levels      = Max(Max(img.log2Width, img.log2Height), 1);
        images[i]       = StaticArray<PaddedImage>(scratch.temp.arena, levels, levels);
        numLevels[i]    = levels;
        images[i][0]    = img;

        // Generate mip maps
        Assert(IsPow2(img.width) && IsPow2(img.height));

        int width = img.width;
        width >>= 1;
        int height = img.height >> 1;

        PaddedImage inPaddedImage = img;
        int depth                 = 1;

        Vec2u scale = 2u;

        while (depth < levels)
        {
            PaddedImage outPaddedImage;
            outPaddedImage.width            = width;
            outPaddedImage.height           = height;
            outPaddedImage.log2Width        = Log2Int(width);
            outPaddedImage.log2Height       = Log2Int(height);
            outPaddedImage.bytesPerPixel    = img.bytesPerPixel;
            outPaddedImage.strideNoBorder   = width * img.bytesPerPixel;
            outPaddedImage.borderSize       = borderSize;
            outPaddedImage.strideWithBorder = (width + 2 * borderSize) * img.bytesPerPixel;
            outPaddedImage.contents =
                PushArrayNoZero(scratch.temp.arena, u8,
                                outPaddedImage.strideWithBorder * (height + 2 * borderSize));
            for (u32 v = 0; v < height; v++)
            {
                for (u32 u = 0; u < width; u++)
                {
                    Vec2u xy = scale * Vec2u(u, v);
                    Vec2u zw = xy + scale - 1u;

                    Vec3f topleft = GammaToLinear(inPaddedImage.GetContentsRelativeIndex(xy));
                    Vec3f topright =
                        GammaToLinear(inPaddedImage.GetContentsRelativeIndex({zw.x, xy.y}));
                    Vec3f bottomleft =
                        GammaToLinear(inPaddedImage.GetContentsRelativeIndex({xy.x, zw.y}));
                    Vec3f bottomright =
                        GammaToLinear(inPaddedImage.GetContentsRelativeIndex({zw.x, zw.y}));

                    Vec3f avg = (topleft + topright + bottomleft + bottomright) / .25f;

                    LinearToGamma(avg, outPaddedImage.GetContentsRelativeIndex({u, v}));
                }
            }

            images[i][depth] = outPaddedImage;
            inPaddedImage    = outPaddedImage;

            u32 prevWidth = width;
            width         = Max(width >> 1, 1);
            scale[0]      = prevWidth / width;

            u32 prevHeight = height;
            height         = Max(height >> 1, 1);
            scale[1]       = prevHeight / height;

            depth++;
        }
    }
#endif

    CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Compute);
    // Add borders to all images
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        const Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
        Assert(!f.isSubface());

        PaddedImage &currentFaceImg = images[faceIndex];

        for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
        {
            // Add edge borders
            int aeid         = f.adjedge(edgeIndex);
            int neighborFace = f.adjface(edgeIndex);
            int rot          = (edgeIndex - aeid + 2) & 3;

            Vec2u dstBaseSize(images[faceIndex].log2Width, images[faceIndex].log2Height);
            Vec2u srcBaseSize(images[neighborFace].log2Width, images[neighborFace].log2Height);

            int dstCompareDim = (edgeIndex & 1) ? dstBaseSize.y : dstBaseSize.x;
            int srcCompareDim = (aeid & 1) ? srcBaseSize.y : srcBaseSize.x;

            int srcBaseDepth = srcCompareDim - dstCompareDim;
            int srcDepthIndex =
                Clamp(srcBaseDepth, 0,
                      Max(images[neighborFace].log2Width, images[neighborFace].log2Height));

            PaddedImage neighborFaceImg = images[neighborFace];

            Vec2u start;
            Vec2u scale;
            int vRes;
            int rowLen;
            int s = Max(-srcBaseDepth, 0);

            if (edgeIndex == e_bottom)
            {
                start = Vec2u(borderSize, 0);
            }
            else if (edgeIndex == e_right)
            {
                start = Vec2u(currentFaceImg.width + borderSize, borderSize);
            }
            else if (edgeIndex == e_top)
            {
                start = Vec2u(borderSize, currentFaceImg.height + borderSize);
            }
            else if (edgeIndex == e_left)
            {
                start = Vec2u(0, borderSize);
            }

            Vec2u srcStart;
            if (aeid == e_bottom)
            {
                srcStart = Vec2u(0, 0);
            }
            else if (aeid == e_right)
            {
                srcStart = Vec2u(neighborFaceImg.width - borderSize, 0);
            }
            else if (aeid == e_top)
            {
                srcStart = Vec2u(0, neighborFaceImg.height - borderSize);
            }
            else if (aeid == e_left)
            {
                srcStart = Vec2u(0, 0);
            }

            scale.y       = (edgeIndex & 1) ? s : 0;
            scale.x       = (edgeIndex & 1) ? 0 : s;
            int srcVRes   = (aeid & 1) ? neighborFaceImg.height : borderSize;
            int srcRowLen = (aeid & 1) ? borderSize : neighborFaceImg.width;
            int dstVRes   = (edgeIndex & 1) ? currentFaceImg.height : borderSize;
            int dstRowLen = (edgeIndex & 1) ? borderSize : currentFaceImg.width;

            currentFaceImg.WriteRotatedBorder(neighborFaceImg, srcStart, start, edgeIndex, rot,
                                              srcVRes, srcRowLen, dstVRes, dstRowLen, scale);

            // Add corner borders
            int afid                 = faceIndex;
            aeid                     = edgeIndex;
            const Ptex::FaceInfo *af = &f;

            const int MaxValence = 10;
            int cfaceId[MaxValence];
            int cedgeId[MaxValence];
            const Ptex::FaceInfo *cface[MaxValence];

            int numCorners = 0;
            for (int i = 0; i < MaxValence; i++)
            {
                int prevFace = afid;
                afid         = af->adjface(aeid);
                aeid         = (af->adjedge(aeid) + 1) & 3;

                if (afid < 0 || (afid == faceIndex && aeid == edgeIndex))
                {
                    numCorners = i - 2;
                    break;
                }

                af         = &reader->getFaceInfo(afid);
                cfaceId[i] = afid;
                cedgeId[i] = aeid;
                cface[i]   = af;

                Assert(!af->isSubface());
            }

            if (numCorners == 1)
            {
                int rotate = (edgeIndex - cedgeId[1] + 2) & 3;

                PaddedImage &cornerImg = images[cfaceId[1]];
                Vec2u srcStart = uvTable[cedgeId[1]] * Vec2u(cornerImg.width - borderSize,
                                                             cornerImg.height - borderSize);
                Vec2u dstStart =
                    uvTable[edgeIndex] * Vec2u(currentFaceImg.width + borderSize,
                                               currentFaceImg.height + borderSize);

                currentFaceImg.WriteRotatedBorder(cornerImg, srcStart, dstStart, cedgeId[1],
                                                  rotate, borderSize, borderSize, borderSize,
                                                  borderSize, scale);
            }
            else if (numCorners > 1)
            {
                // TODO: what do I do here?
            }
            else
            {
                // valence 2 or 3, ignore corner face (just adjust weight)
            }
        }

        // Convert image from 3 to 4 channels
        u32 totalGpuSize = currentFaceImg.strideWithBorder * currentFaceImg.GetPaddedHeight() /
                           bytesPerPixel * gpuBytesPerPixel;
        u8 *contents  = PushArray(scratch.temp.arena, u8, totalGpuSize);
        u32 dstOffset = 0;
        u32 srcOffset = 0;
        for (int i = 0; i < currentFaceImg.GetPaddedWidth() * currentFaceImg.GetPaddedHeight();
             i++)
        {
            Assert(dstOffset < totalGpuSize);
            MemoryCopy(contents + dstOffset, currentFaceImg.contents + srcOffset,
                       bytesPerPixel);
            dstOffset += gpuBytesPerPixel;
            srcOffset += bytesPerPixel;
        }

        // Block compress the entire face texture
        ImageDesc blockCompressedImageDesc(ImageType::Type2D, currentFaceImg.GetPaddedWidth(),
                                           currentFaceImg.GetPaddedHeight(), 1, 1, 1,
                                           VK_FORMAT_R8G8B8A8_UNORM, MemoryUsage::GPU_ONLY,
                                           VK_IMAGE_USAGE_SAMPLED_BIT |
                                               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                           VK_IMAGE_TILING_OPTIMAL);

        TransferBuffer blockCompressedImage =
            cmd->SubmitImage(contents, blockCompressedImageDesc);

        u32 outputBlockWidth  = currentFaceImg.GetPaddedWidth() / blockSize;
        u32 outputBlockHeight = currentFaceImg.GetPaddedHeight() / blockSize;
        ImageDesc uavDesc(ImageType::Type2D, outputBlockWidth, outputBlockHeight, 1, 1, 1,
                          VK_FORMAT_R32G32_UINT, MemoryUsage::GPU_ONLY,
                          VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_TILING_OPTIMAL);

        GPUImage uavImage = device->CreateImage(uavDesc);

        ImageDesc readbackDesc(ImageType::Type2D, currentFaceImg.GetPaddedWidth(),
                               currentFaceImg.GetPaddedHeight(), 1, 1, 1,
                               VK_FORMAT_BC1_RGB_UNORM_BLOCK, MemoryUsage::GPU_TO_CPU,
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_OPTIMAL);

        GPUBuffer readbackBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            outputBlockWidth * outputBlockHeight * bytesPerBlock, MemoryUsage::GPU_TO_CPU);

        cmd->Barrier(&blockCompressedImage.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->Barrier(&uavImage, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->FlushBarriers();

        gpuSrcImages[faceIndex]          = blockCompressedImage;
        tempUavs[faceIndex]              = uavImage;
        blockCompressedImages[faceIndex] = readbackBuffer;

        DescriptorSet set = bcLayout.CreateDescriptorSet();
        set.Bind(inputBinding, &blockCompressedImage.image);
        set.Bind(outputBinding, &blockCompressedImages[faceIndex]);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &set, bcLayout.pipelineLayout);

        cmd->Barrier(&tempUavs[faceIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        BufferImageCopy copy = {};
        copy.layerCount      = 1;
        copy.extent =
            Vec3u(currentFaceImg.GetPaddedWidth(), currentFaceImg.GetPaddedHeight(), 1);
        cmd->CopyImageToBuffer(&blockCompressedImages[faceIndex], &tempUavs[faceIndex], copy);
    }

    Semaphore semaphore   = device->CreateGraphicsSemaphore();
    semaphore.signalValue = 1;
    cmd->Signal(semaphore);
    device->SubmitCommandBuffer(cmd);

    device->Wait(semaphore);

    // Discretize into tiles
    // If one of the dimensions is larger than 128, but the texel area is <= 128x128,
    // then pack it into one tile
    // If the texel area is less than 128x128, then the current level and all of the
    // remaining mips are packed into a single texture
    u32 vLen   = totalBlocksPerPage;
    u32 rowLen = totalBlocksPerPage * bytesPerBlock;
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        u32 offset          = (u32)tiles.size();
        GPUBuffer *gpuImage = &blockCompressedImages[faceIndex];

        // for (int levelIndex = 0; levelIndex < numLevels[faceIndex]; levelIndex++)
        // {
        // PaddedImage *image = &images[faceIndex][levelIndex];
        PaddedImage *image = &images[faceIndex];

        // Write all remaining mip levels to the same tile
        // TODO: write all the mips at a given level to consecutive tiles,
        // instead of writing all the levels for a face to consecutive tiles
        // if (image->width <= texelWidthPerTile && image->height <= texelWidthPerTile)
        // {
        //     Tile tile;
        //     tile.contents    = PushArray(scratch.temp.arena, u8, vLen * rowLen);
        //     u32 xTexelOffset = 0;
        //     u32 yTexelOffset = 0;
        //     for (; levelIndex < numLevels[faceIndex]; levelIndex++)
        //     {
        //         image         = &images[faceIndex][levelIndex];
        //         u32 mipVLen   = vLen >> levelIndex;
        //         u32 mipRowLen = rowLen >> levelIndex;
        //
        //         u32 dstOffset = ((yTexelOffset >> log2BlockSize) * totalBlocksPerTileX +
        //                          (xTexelOffset >> log2BlockSize)) *
        //                         blockSize;
        //
        //         Utils::Copy(image->contents, image->strideWithBorder,
        //                     tile.contents + dstOffset, totalBlocksPerTileX * blockSize,
        //                     mipVLen, mipRowLen);
        //         xOffset += image->width;
        //
        //         if (xOffset >= totalTexelsPerTileX)
        //         {
        //             Assert(xOffset == totalTexelsPerTileX);
        //             xOffset = 0;
        //             yOffset = totalTexelsPerTileY / 2;
        //         }
        //     }
        //     break;
        // }

        // One dimension is < texelsPerTile
        if (Min(image->width, image->height) < texelWidthPerPage &&
            Max(image->width, image->height) > texelWidthPerPage)
        {
            u32 numTiles = (image->width * image->height) >> (log2TexelWidth * 2);

            u32 remainingDim = Max(image->width, image->height);
            u32 xTexelOffset = 0;
            u32 yTexelOffset = 0;

            u32 xSrcOffset = 0;
            u32 ySrcOffset = 0;

            Tile tile;
            tile.contents = PushArray(scratch.temp.arena, u8, vLen * rowLen);
            for (u32 i = 0; i < numTiles; i++)
            {
                u32 subTileWidth =
                    Min(totalTexelWidthPerPage, (u32)image->width + 2 * borderSize);
                u32 subTileHeight =
                    Min(totalTexelWidthPerPage, (u32)image->height + 2 * borderSize);

                Utils::Copy(Utils::GetContentsAbsoluteIndex(
                                gpuImage->mappedPtr,
                                {xSrcOffset >> log2BlockSize, ySrcOffset >> log2BlockSize},
                                image->GetPaddedWidth() >> log2BlockSize,
                                image->GetPaddedHeight() >> log2BlockSize, rowLen,
                                blockBorderSize, bytesPerBlock),
                            (image->GetPaddedWidth() >> log2BlockSize) * bytesPerBlock,
                            Utils::GetContentsAbsoluteIndex(
                                tile.contents,
                                {xTexelOffset >> log2BlockSize, yTexelOffset >> log2BlockSize},
                                totalBlocksPerPage, totalBlocksPerPage, rowLen,
                                blockBorderSize, bytesPerBlock),
                            rowLen, subTileHeight >> log2BlockSize,
                            (subTileWidth >> log2BlockSize) * bytesPerBlock);

                if (image->width > texelWidthPerPage)
                {
                    // total tile size is 136 by 136
                    xSrcOffset += texelWidthPerPage + borderSize;
                    xTexelOffset = 0;
                    yTexelOffset += image->height + borderSize;
                    // Allocate new tile
                    if (yTexelOffset == texelWidthPerPage + borderSize)
                    {
                        tiles.push_back(tile);
                        tile.contents = PushArray(scratch.temp.arena, u8, vLen * rowLen);
                        yTexelOffset  = 0;
                    }
                }
                else if (image->height > texelWidthPerPage)
                {
                    ySrcOffset += texelWidthPerPage + borderSize;
                    yTexelOffset = 0;
                    xTexelOffset += image->width + borderSize;
                    // Allocate new tile
                    if (xTexelOffset == texelWidthPerPage + borderSize)
                    {
                        tiles.push_back(tile);
                        tile.contents = PushArray(scratch.temp.arena, u8, vLen * rowLen);
                        xTexelOffset  = 0;
                    }
                }
            }
        }
        // Both dimensions >= texelsPerTile
        else
        {

            u32 numTilesX = image->width / texelWidthPerPage;
            u32 numTilesY = image->height / texelWidthPerPage;

            for (int tileY = 0; tileY < numTilesY; tileY++)
            {
                for (int tileX = 0; tileX < numTilesX; tileX++)
                {
                    Vec2u tileStart(totalBlocksPerPage * tileX, totalBlocksPerPage * tileY);

                    Tile tile;
                    tile.contents   = PushArray(scratch.temp.arena, u8, vLen * rowLen);
                    tile.parentFace = faceIndex;
                    Utils::Copy(Utils::GetContentsAbsoluteIndex(
                                    gpuImage->mappedPtr, tileStart,
                                    image->GetPaddedWidth() >> log2BlockSize,
                                    image->GetPaddedHeight() >> log2BlockSize, rowLen,
                                    blockBorderSize, bytesPerBlock),
                                (image->GetPaddedWidth() >> log2BlockSize) * bytesPerBlock,
                                tile.contents, rowLen, vLen, rowLen);

                    tiles.push_back(tile);
                }
            }
        }
        u32 size = (u32)tiles.size() - offset;

        metaData[faceIndex] = {offset, size};
    }

    // Write metadata and tiles to disk
    string outFilename =
        PushStr8F(scratch.temp.arena, "%S.tiles", RemoveFileExtension(filename));
    StringBuilderMapped builder(outFilename);

    PutData(&builder, &numFaces, sizeof(numFaces));
    PutData(&builder, metaData.data, sizeof(TileMetadata) * metaData.Length());
    for (auto &tile : tiles)
    {
        PutData(&builder, tile.contents, vLen * rowLen);
    }

    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        device->DestroyBuffer(&gpuSrcImages[faceIndex].stagingBuffer);
        device->DestroyImage(&gpuSrcImages[faceIndex].image);
        device->DestroyImage(&tempUavs[faceIndex]);
        device->DestroyBuffer(&blockCompressedImages[faceIndex]);
    }

    // Output tiles to disk
    t->release();
}

// two lookups: one that converts uvs into a virtual address (multiply uv by scale and add
// offset), then use this address to look up the page table
//
// number of pages x and y per face.
// base virtual address for entire texture
// per face data that specifies the offset in virtual address space and the size of the face
// tex in each dimension?
// if you think about it uv is 2 4 byte floats, and this is 2 4 byte uints...

// virtual address = base virtual address + offset + floor(v * numPagesY * numPagesX) + floor(u
// * numPagesX)

// virtual address -> physical address lookup directly?

VirtualTextureManager::VirtualTextureManager(Arena *arena, u32 totalNumPages,
                                             u32 inPageWidthPerPool, u32 inTexelWidthPerPage,
                                             u32 borderSize, VkFormat format)
    : format(format)
{
    Assert(IsPow2(pageWidthPerPool) && IsPow2(texelWidthPerPage));
    ImageLimits limits = device->GetImageLimits();

    texelWidthPerPage   = inTexelWidthPerPage + 2 * borderSize;
    pageWidthPerPool    = inPageWidthPerPool;
    u32 numPagesPerPool = Sqr(pageWidthPerPool);
    u32 numPools        = (totalNumPages + numPagesPerPool - 1) / numPagesPerPool;

    u32 texelWidthPerPool =
        NextPowerOfTwo(Min(texelWidthPerPage * pageWidthPerPool, limits.max2DImageDim));
    pageWidthPerPool = texelWidthPerPool / texelWidthPerPage;
    pageWidthPerPool = 1u << Log2Int(pageWidthPerPool);
    numPagesPerPool  = Sqr(pageWidthPerPool);

    maxNumLayers         = 1u << Log2Int(limits.maxNumLayers);
    u32 numTextureArrays = (numPools + maxNumLayers - 1) / maxNumLayers;
    Assert(numTextureArrays == 1);

    pools            = StaticArray<PhysicalPagePool>(arena, numPools);
    pageRanges       = StaticArray<BlockRange>(arena, totalNumPages);
    gpuPhysicalPools = StaticArray<GPUImage>(arena, numTextureArrays);

    // Allocate texture arrays
    for (int i = 0; i < numTextureArrays; i++)
    {
        ImageDesc poolDesc(ImageType::Type2D, texelWidthPerPool, texelWidthPerPool, 1, 1,
                           numPools, format, MemoryUsage::GPU_ONLY,
                           VK_IMAGE_USAGE_SAMPLED_BIT);
        gpuPhysicalPools.Push(device->CreateImage(poolDesc));
    }

    // Allocate physical page pool
    for (int i = 0; i < numPools; i++)
    {
        PhysicalPagePool pool;
        pool.freePages = StaticArray<u32>(arena, numPagesPerPool);
        pool.prevFree  = i == 0 ? InvalidPool : i - 1;
        pool.nextFree  = i == numPools - 1 ? InvalidPool : i + 1;
        for (int j = 0; j < numPagesPerPool; j++)
        {
            pool.freePages.Push(numPagesPerPool - 1 - j);
        }
        pools.Push(pool);
    }
    completelyFreePool = 0;
    partiallyFreePool  = InvalidPool;

    // Allocate block ranges
    pageRanges.Push(BlockRange(AllocationStatus::Free, 0, totalNumPages, InvalidRange,
                               InvalidRange, InvalidRange, InvalidRange));

    freeRange = 0;
}

void VirtualTextureManager::AllocateVirtualPages(PhysicalPageAllocation *allocations,
                                                 u32 numPages)
{
    u32 freeIndex = freeRange;

    u32 leftover   = ~0u;
    u32 allocIndex = ~0u;

    // Find best fit
    while (freeIndex != InvalidRange)
    {
        BlockRange &range = pageRanges[freeIndex];
        u32 numFreePages  = range.onePastEndPage - range.startPage;
        if (numFreePages >= numPages && numFreePages - numPages < leftover)
        {
            allocIndex = freeIndex;
            leftover   = numFreePages - numPages;
        }
        freeIndex = range.nextFree;
    }

    Assert(leftover != ~0u && allocIndex != ~0u);

    // Allocate pages
    BlockRange &range = pageRanges[allocIndex];
    Assert(range.status == AllocationStatus::Free);

    u32 rightRangeIndex = pageRanges.Length();

    BlockRange leftRange(AllocationStatus::Allocated, range.startPage,
                         range.startPage + numPages, range.prevRange, rightRangeIndex,
                         InvalidRange, InvalidRange);
    BlockRange rightRange(AllocationStatus::Free, range.startPage + numPages,
                          range.onePastEndPage, allocIndex, range.nextRange, range.prevFree,
                          range.nextFree);

    if (range.prevFree == InvalidRange) freeRange = rightRangeIndex;
    else pageRanges[range.prevFree].nextFree = allocIndex;

    if (range.nextFree != InvalidRange) pageRanges[range.nextFree].prevFree = allocIndex;

    pageRanges[allocIndex] = leftRange;
    pageRanges.Push(rightRange);

    // Next, allocate physical pages
    for (u32 i = 0; i < numPages; i++)
    {
        u32 freeIndex =
            partiallyFreePool == InvalidPool ? completelyFreePool : partiallyFreePool;
        bool isFullyFree = partiallyFreePool == InvalidPool;

        u32 poolAllocIndex = ~0u;
        u32 pageAllocIndex = ~0u;
        while (freeIndex != InvalidPool)
        {
            PhysicalPagePool &pool = pools[freeIndex];
            if (pool.freePages.Length())
            {
                poolAllocIndex = freeIndex;
                pageAllocIndex = pool.freePages.Pop();
                if (isFullyFree)
                {
                    UnlinkFreeList(pools, allocIndex, completelyFreePool, InvalidPool);
                    LinkFreeList(pools, allocIndex, partiallyFreePool, InvalidPool);
                }
                else if (pool.freePages.Length() == 0)
                {
                    UnlinkFreeList(pools, allocIndex, partiallyFreePool, InvalidPool);
                }
                break;
            }
            freeIndex = pool.nextFree;
        }
        ErrorExit(allocIndex != ~0u, "Ran out of memory");
        allocations[i].poolIndex = poolAllocIndex;
        allocations[i].pageIndex = pageAllocIndex;
    }
}

void VirtualTextureManager::AllocatePhysicalPages(CommandBuffer *cmd, Tile *tiles,
                                                  PhysicalPageAllocation *allocations,
                                                  u32 numPages)
{
    u32 pageSize  = Sqr(texelWidthPerPage >> GetBlockShift(format)) * GetFormatSize(format);
    u32 allocSize = pageSize * numPages;

    TransferBuffer buf = device->GetStagingBuffer(allocSize);
    ScratchArena scratch;

    u64 offset              = 0;
    BufferImageCopy *copies = PushArray(scratch.temp.arena, BufferImageCopy, numPages);

    u32 log2PageWidthPerPool = Log2Int(pageWidthPerPool);
    for (int i = 0; i < numPages; i++)
    {
        Assert(allocations[i].poolIndex < maxNumLayers);
        MemoryCopy((u8 *)buf.mappedPtr + offset, tiles[i].contents, pageSize);

        copies[i].bufferOffset = offset;
        copies[i].baseLayer    = allocations[i].poolIndex & (maxNumLayers - 1);
        copies[i].layerCount   = 1;

        u32 pageX        = allocations[i].pageIndex & (pageWidthPerPool - 1);
        u32 pageY        = allocations[i].pageIndex >> log2PageWidthPerPool;
        copies[i].offset = Vec3i(pageX * texelWidthPerPage, pageY * texelWidthPerPage, 0);
        copies[i].extent = Vec3u(texelWidthPerPage, texelWidthPerPage, 0);

        offset += pageSize;
    }
    cmd->CopyImage(&buf, &gpuPhysicalPools[0], copies, numPages);
}

} // namespace rt
