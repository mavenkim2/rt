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
    dst = Utils::GetContentsAbsoluteIndex(dst, dstIndex, srcWidth, srcHeight, dstStride,
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
                    MemoryCopy(tempBuffer + dstOffset, other.GetContentsRelativeIndex(srcPos),
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

    // if (!Contains(outFilename, "mountainb0001_geo")) return;

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

        inputBinding =
            bcLayout.AddBinding(0, DescriptorType::SampledImage, VK_SHADER_STAGE_COMPUTE_BIT);
        outputBinding =
            bcLayout.AddBinding(1, DescriptorType::StorageImage, VK_SHADER_STAGE_COMPUTE_BIT);
        bcLayout.AddImmutableSamplers();
        pipeline    = device->CreateComputePipeline(&shader, &bcLayout);
        initialized = true;
    }

    // Create all face images and mip maps
    StaticArray<StaticArray<PaddedImage>> images(scratch.temp.arena, numFaces, numFaces);
    // StaticArray<int> numLevels(scratch.temp.arena, numFaces, numFaces);
    StaticArray<GPUImage> tempUavs(scratch.temp.arena, numFaces, numFaces);
    StaticArray<GPUBuffer> blockCompressedImages(scratch.temp.arena, numFaces, numFaces);
    StaticArray<TransferBuffer> gpuSrcImages(scratch.temp.arena, numFaces, numFaces);
    StaticArray<DescriptorSet> descriptorSets(scratch.temp.arena, numFaces, numFaces);

    std::vector<Tile> tiles;

    StaticArray<TileMetadata> metaData(scratch.temp.arena, numFaces, numFaces);

    for (int i = 0; i < numFaces; i++)
    {
        PaddedImage img = PtexToImg(scratch.temp.arena, t, i, borderSize);
        int levels      = Max(Max(img.log2Width, img.log2Height), 1);
        images[i]       = StaticArray<PaddedImage>(scratch.temp.arena, levels, levels);
        // numLevels[i]    = levels;
        images[i][0] = img;

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
            PaddedImage outPaddedImage = GenerateMips(scratch.temp.arena, inPaddedImage, width,
                                                      height, scale, borderSize);

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

    auto GetNeighborFaceImage = [&](PaddedImage &currentFaceImg, u32 neighborFace, u32 rot,
                                    Vec2u &scale) {
        Vec2i dstBaseSize(currentFaceImg.log2Width, currentFaceImg.log2Height);
        const Vec2i srcBaseSize(images[neighborFace][0].log2Width,
                                images[neighborFace][0].log2Height);

        if (rot & 1) Swap(dstBaseSize[0], dstBaseSize[1]);
        Vec2i srcBaseDepth = srcBaseSize - dstBaseSize;
        int depthIndex     = Max(0, Min(srcBaseDepth.x, srcBaseDepth.y));

        PaddedImage neighborFaceImg = images[neighborFace][depthIndex];
        u32 log2Width               = neighborFaceImg.log2Width;
        u32 log2Height              = neighborFaceImg.log2Height;

        // reduce u
        if (log2Width > dstBaseSize.x)
        {
            for (int i = 0; i < log2Width - dstBaseSize.x; i++)
            {
                u32 width = neighborFaceImg.width;
                width >>= 1;
                neighborFaceImg = GenerateMips(scratch.temp.arena, neighborFaceImg, width,
                                               neighborFaceImg.height, {2, 1}, borderSize);
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
                                 height, {1, 2}, borderSize);
            }
        }
        scale = Vec2u(Max(Vec2i(0), -srcBaseDepth));
        return neighborFaceImg;
    };

    CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Compute);
    cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    // Add borders to all images
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        const Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
        Assert(!f.isSubface());

        PaddedImage &currentFaceImg = images[faceIndex][0];

        for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
        {
            // Add edge borders
            int aeid         = f.adjedge(edgeIndex);
            int neighborFace = f.adjface(edgeIndex);
            int rot          = (edgeIndex - aeid + 2) & 3;

            Vec2u scale;
            PaddedImage neighborFaceImg =
                GetNeighborFaceImage(currentFaceImg, neighborFace, rot, scale);

            // NOTE: Scale is only used for the base layer, when the src image has smaller
            // dimensions than the dst image. For mip levels, the above reductions are used
            // to make the src and dst image have equal dimensions
            Vec2u start;
            int vRes;
            int rowLen;

            start.x = (edgeIndex == e_left ? 0 : borderSize) +
                      (edgeIndex == e_right ? currentFaceImg.width : 0);
            start.y = (edgeIndex == e_bottom ? 0 : borderSize) +
                      (edgeIndex == e_top ? currentFaceImg.height : 0);

            Vec2u srcStart;
            srcStart.x = aeid == e_right ? neighborFaceImg.width - borderSize : 0;
            srcStart.y = aeid == e_top ? neighborFaceImg.height - borderSize : 0;

            int srcVRes   = (aeid & 1) ? neighborFaceImg.height : (borderSize >> scale.y);
            int srcRowLen = (aeid & 1) ? (borderSize >> scale.x) : neighborFaceImg.width;
            int dstVRes   = (edgeIndex & 1) ? currentFaceImg.height : borderSize;
            int dstRowLen = (edgeIndex & 1) ? borderSize : currentFaceImg.width;

            currentFaceImg.WriteRotatedBorder(neighborFaceImg, srcStart, start, edgeIndex, rot,
                                              srcVRes, srcRowLen, dstVRes, dstRowLen,
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

                Vec2u srcStart = uvTable[cedgeId[1]] * Vec2u(cornerImg.width - borderSize,
                                                             cornerImg.height - borderSize);
                Vec2u dstStart =
                    uvTable[edgeIndex] * Vec2u(currentFaceImg.width + borderSize,
                                               currentFaceImg.height + borderSize);

                currentFaceImg.WriteRotatedBorder(
                    cornerImg, srcStart, dstStart, cedgeId[1], cornerRotate,
                    borderSize >> cornerScale.y, borderSize >> cornerScale.x, borderSize,
                    borderSize, (cornerRotate & 1) ? cornerScale.yx() : cornerScale);
            }
            else if (numCorners > 1)
            {
                // TODO: what do I do here?
            }
            else
            {
            }
        }

        // Block compress the entire face texture
        ImageDesc blockCompressedImageDesc(ImageType::Type2D, currentFaceImg.GetPaddedWidth(),
                                           currentFaceImg.GetPaddedHeight(), 1, 1, 1,
                                           VK_FORMAT_R8G8B8A8_UNORM, MemoryUsage::GPU_ONLY,
                                           VK_IMAGE_USAGE_SAMPLED_BIT |
                                               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                           VK_IMAGE_TILING_OPTIMAL);

        TransferBuffer blockCompressedImage =
            cmd->SubmitImage(currentFaceImg.contents, blockCompressedImageDesc);

        u32 outputBlockWidth  = currentFaceImg.GetPaddedWidth() >> log2BlockSize;
        u32 outputBlockHeight = currentFaceImg.GetPaddedHeight() >> log2BlockSize;
        ImageDesc uavDesc(ImageType::Type2D, outputBlockWidth, outputBlockHeight, 1, 1, 1,
                          VK_FORMAT_R32G32_UINT, MemoryUsage::GPU_ONLY,
                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          VK_IMAGE_TILING_OPTIMAL);

        GPUImage uavImage = device->CreateImage(uavDesc);

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

        DescriptorSet set = bcLayout.CreateNewDescriptorSet();
        set.Bind(inputBinding, &gpuSrcImages[faceIndex].image);
        set.Bind(outputBinding, &tempUavs[faceIndex]);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &set, bcLayout.pipelineLayout);
        cmd->Dispatch((outputBlockWidth + 7) >> 3, (outputBlockHeight + 7) >> 3, 1);

        cmd->Barrier(&tempUavs[faceIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        cmd->FlushBarriers();

        descriptorSets[faceIndex] = set;

        BufferImageCopy copy = {};
        copy.layerCount      = 1;
        copy.extent          = Vec3u(outputBlockWidth, outputBlockHeight, 1);
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

        u32 currentTexelsPerPage = texelWidthPerPage;
        PaddedImage *image       = &images[faceIndex][0];

        u32 numTilesX = Max(image->width / texelWidthPerPage, 1u);
        u32 numTilesY = Max(image->height / texelWidthPerPage, 1u);

        for (int tileY = 0; tileY < numTilesY; tileY++)
        {
            for (int tileX = 0; tileX < numTilesX; tileX++)
            {
                Vec2u tileStart(blocksPerPage * tileX, blocksPerPage * tileY);

                Tile tile;
                tile.contents   = PushArray(scratch.temp.arena, u8, vLen * rowLen);
                tile.parentFace = faceIndex;
                Utils::Copy(
                    Utils::GetContentsAbsoluteIndex(gpuImage->mappedPtr, tileStart,
                                                    image->GetPaddedWidth() >> log2BlockSize,
                                                    image->GetPaddedHeight() >> log2BlockSize,
                                                    rowLen, bytesPerBlock),
                    (image->GetPaddedWidth() >> log2BlockSize) * bytesPerBlock, tile.contents,
                    rowLen, Min(vLen, image->GetPaddedHeight() >> log2BlockSize),
                    Min(rowLen, (image->GetPaddedWidth() >> log2BlockSize) * bytesPerBlock));

                tiles.push_back(tile);
            }
        }
        u32 size = (u32)tiles.size() - offset;

        metaData[faceIndex] = {offset, image->log2Width, image->log2Height};
    }

    // Pack each subsequent mip level into 128x128 tiles. Each subsequent mip level is
    // split into (128 >> level) x (128 >> level) tiles, and packed in row-major order into a
    // 128x128 tile until space runs out.
    //
    // At runtime: The virtual page offset is shifted down by the number of dimensions * the
    // current level to get the virtual index used to index the page table. The bottom dim *
    // level bits determine the location within the 128x128 page.
    // e.g. for the first mip level,
    // if the calculated offset is 7, we shift down by 2 to get 1. This is the virtual address
    // we use to look up in the page table. Once we get the physical page, the lower 2 bits (3)
    // means our start address in the page is (0.5, 0.5), i.e. the bottom right 64 x 64 tile
#if 0
    for (int levelIndex = 1; levelIndex < numLevels[faceIndex]; levelIndex++)
    {
        u32 currentLevelBlocksPerPage = (texelWidthPerPage >> levelIndex) >> log2BlockSize;

        u32 currentTileOffset  = 0;
        u32 maxSubTilesPerTile = 1u << (levelIndex * 2);

        u32 tileWidth     = texelWidthPerPage + ((2 * borderSize) << levelIndex);
        u32 subTileVLen   = levelVLen >> levelIndex;
        u32 subTileRowLen = tileVLen * bytesPerBlock;

        Tile tile;
        tile.contents = PushArray(scratch.temp.arena, u8, levelVLen * levelRowLen);

        // 136, 144, 160, 192, 256, 384
        // idea: tiles for higher mip levels are going to be bigger to preserve
        // bit shift logic in shader.
        // use a separate layer/texture? that seems to make the most sense

        // TODO: how do I handle the lowest mip levels?
        for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
        {
            PaddedImage *image = &images[faceIndex][levelIndex];

            u32 numTilesX = Max(image->width / (texelWidthPerPage >> levelIndex), 1u);
            u32 numTilesY = Max(image->height / (texelWidthPerPage >> levelIndex), 1u);

            for (int tileY = 0; tileY < numTilesY; tileY++)
            {
                for (int tileX = 0; tileX < numTilesX; tileX++)
                {
                    Vec2u srcStart(currentLevelBlocksPerPage * tileX,
                                   currentLevelBlocksPerPage * tileY);

                    u32 dstTileX = currentTileOffset & ((1u << levelIndex) - 1);
                    u32 dstTileY = currenTileOffset >> levelIndex;
                    Vec2u dstStart(currentLevelBlocksPerPage * dstTileX,
                                   currentLevelBlocksPerPage * dstTileY);

                    Utils::Copy(gpuImage->mappedPtr, srcStart,
                                image->GetPaddedWidth() >> log2BlockSize,
                                image->GetPaddedHeight() >> log2BlockSize, tile.contents,
                                dstStart, tileWidth, tileHeight, subTileVLen, subTileRowLen,
                                bytesPerBlock);

                    currentTileOffset++;
                    if (currentTileOffset == maxSubTilesPerTile)
                    {
                        currenTileOffset = 0;
                        tiles.push_back(tile);
                        tile.contents =
                            PushArray(scratch.temp.arena, u8, levelVLen * levelRowLen);
                    }
                }
            }
        }
    }
#endif

    // Write metadata and tiles to disk
    StringBuilderMapped builder(outFilename);

    u32 numTiles = (u32)tiles.size();
    PutData(&builder, &numFaces, sizeof(numFaces));
    PutData(&builder, &numTiles, sizeof(numTiles));
    PutData(&builder, metaData.data, sizeof(TileMetadata) * metaData.Length());
    for (auto &tile : tiles)
    {
        u32 vLen   = totalBlocksPerPage;
        u32 rowLen = totalBlocksPerPage * bytesPerBlock;
        PutData(&builder, tile.contents, vLen * rowLen);
    }

    OS_UnmapFile(builder.ptr);
    OS_ResizeFile(builder.filename, builder.totalSize);

    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        device->DestroyBuffer(&gpuSrcImages[faceIndex].stagingBuffer);
        device->DestroyImage(&gpuSrcImages[faceIndex].image);
        device->DestroyImage(&tempUavs[faceIndex]);
        device->DestroyBuffer(&blockCompressedImages[faceIndex]);
        device->DestroyPool(descriptorSets[faceIndex].pool);
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

    texelWidthPerPage   = inTexelWidthPerPage + 2 * borderSize;
    pageWidthPerPool    = inPageWidthPerPool;
    u32 numPagesPerPool = Sqr(pageWidthPerPool);
    u32 numPools        = (totalNumPages + numPagesPerPool - 1) / numPagesPerPool;

    u32 texelWidthPerPool = Min(texelWidthPerPage * pageWidthPerPool, limits.max2DImageDim);
    pageWidthPerPool      = texelWidthPerPool / texelWidthPerPage;
    pageWidthPerPool      = 1u << Log2Int(pageWidthPerPool);
    numPagesPerPool       = Sqr(pageWidthPerPool);

    maxNumLayers         = 1u << Log2Int(limits.maxNumLayers);
    u32 numTextureArrays = (numPools + maxNumLayers - 1) / maxNumLayers;
    Assert(numTextureArrays == 1);

    Assert(IsPow2(pageWidthPerPool));
    pools            = StaticArray<PhysicalPagePool>(arena, numPools);
    pageRanges       = StaticArray<BlockRange>(arena, totalNumPages);
    gpuPhysicalPools = StaticArray<GPUImage>(arena, numTextureArrays);

    // Allocate texture arrays
    for (int i = 0; i < numTextureArrays; i++)
    {
        ImageDesc poolDesc(ImageType::Array2D, texelWidthPerPool, texelWidthPerPool, 1, 1,
                           numPools, format, MemoryUsage::GPU_ONLY,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        gpuPhysicalPools.Push(device->CreateImage(poolDesc));
    }

    // Allocate page table
    pageTableBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u32) * totalNumPages);

    // ways of doing this:
    // 1. have multiple page sizes; e.g. 128x128 for highest, 64x64 for next level, etc.
    // this complicates page management, but makes lookup easy
    // 2. only have one page size and pack multiple faces into the same page. makes page
    // managemnt trivial but complicates the lookup/ requires more information for the lookup

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

u32 VirtualTextureManager::AllocateVirtualPages(u32 numPages)
{
    Assert(numPages);
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

    return allocIndex;
}

void VirtualTextureManager::AllocatePhysicalPages(CommandBuffer *cmd, u8 *contents,
                                                  u32 allocIndex)
{
    const BlockRange &range = pageRanges[allocIndex];
    const u32 numPages      = range.onePastEndPage - range.startPage;
    u32 pageSize  = Sqr(texelWidthPerPage >> GetBlockShift(format)) * GetFormatSize(format);
    u32 allocSize = pageSize * numPages;

    TransferBuffer buf = device->GetStagingBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, allocSize);
    ScratchArena scratch;

    u64 offset              = 0;
    u64 requestOffset       = 0;
    BufferImageCopy *copies = PushArray(scratch.temp.arena, BufferImageCopy, numPages);

    PageTableUpdateRequest *requests =
        PushArray(scratch.temp.arena, PageTableUpdateRequest, numPages);

    u32 log2PageWidthPerPool = Log2Int(pageWidthPerPool);
    for (int i = 0; i < numPages; i++)
    {
        // First allocate page
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
                    UnlinkFreeList(pools, poolAllocIndex, completelyFreePool, InvalidPool);
                    LinkFreeList(pools, poolAllocIndex, partiallyFreePool, InvalidPool);
                }
                else if (pool.freePages.Length() == 0)
                {
                    UnlinkFreeList(pools, poolAllocIndex, partiallyFreePool, InvalidPool);
                }
                break;
            }
            freeIndex = pool.nextFree;
        }

        ErrorExit(allocIndex != ~0u, "Ran out of memory");
        Assert(poolAllocIndex < maxNumLayers);

        requests[i] = PageTableUpdateRequest{range.startPage + i,
                                             (poolAllocIndex << (2u * 7u)) | pageAllocIndex};
        MemoryCopy((u8 *)buf.mappedPtr + offset, contents, pageSize);
        contents += pageSize;
        copies[i].bufferOffset = offset;
        copies[i].baseLayer    = poolAllocIndex & (maxNumLayers - 1);
        copies[i].layerCount   = 1;

        u32 pageX        = pageAllocIndex & (pageWidthPerPool - 1);
        u32 pageY        = pageAllocIndex >> log2PageWidthPerPool;
        copies[i].offset = Vec3i(pageX * texelWidthPerPage, pageY * texelWidthPerPage, 0);
        copies[i].extent = Vec3u(texelWidthPerPage, texelWidthPerPage, 1);

        offset += pageSize;
        requestOffset += sizeof(PageTableUpdateRequest);
    }

    PageTableUpdatePushConstant pc;
    pc.numRequests = numPages;

    cmd->SubmitTransfer(&buf);
    TransferBuffer pageTableUpdateRequestBuffer =
        cmd->SubmitBuffer(requests, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          sizeof(PageTableUpdateRequest) * numPages);

    DescriptorSet ds = descriptorSetLayout.CreateDescriptorSet();
    ds.Bind(0, &pageTableUpdateRequestBuffer.buffer).Bind(1, &pageTableBuffer);
    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();
    cmd->CopyImage(&buf, &gpuPhysicalPools[0], copies, numPages);
    cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmd->PushConstants(&push, &pc, descriptorSetLayout.pipelineLayout);
    cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                            descriptorSetLayout.pipelineLayout);
    cmd->Dispatch((numPages + 63) >> 6, 1, 1);
}

} // namespace rt
