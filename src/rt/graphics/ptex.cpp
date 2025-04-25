#include "../base.h"
#include "../memory.h"
#include "../bit_packing.h"
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
    Assert(borderSize == other.borderSize);

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
    FixedArray<StaticArray<PaddedImage>, MAX_LEVEL> images;
    StaticArray<u32> faceStartLevel(scratch.temp.arena, numFaces, numFaces);

    StaticArray<GPUImage> tempUavs(scratch.temp.arena, numFaces, numFaces);
    StaticArray<GPUBuffer> blockCompressedImages(scratch.temp.arena, numFaces, numFaces);
    StaticArray<TransferBuffer> gpuSrcImages(scratch.temp.arena, numFaces, numFaces);
    StaticArray<DescriptorSet> descriptorSets(scratch.temp.arena, numFaces, numFaces);

    std::vector<Tile> tiles;

    StaticArray<TileMetadata> metaData(scratch.temp.arena, numFaces, numFaces);

    // Initialize image array
    for (int i = 0; i < MAX_LEVEL; i++)
    {
        images[i] = StaticArray<PaddedImage>(scratch.temp.arena, numFaces, numFaces);
    }

    // Generate mips for faces
    for (int i = 0; i < numFaces; i++)
    {
        PaddedImage img = PtexToImg(scratch.temp.arena, t, i, borderSize);
        int levels      = Max(Max(img.log2Width, img.log2Height), 1);

        u32 minDim            = Min(img.log2Width, img.log2Height);
        int startLevel        = Max(MAX_LEVEL - (int)minDim, 0);
        images[startLevel][i] = img;
        faceStartLevel[i]     = startLevel;

        // Generate mip maps
        Assert(IsPow2(img.width) && IsPow2(img.height));

        int width = img.width;
        width >>= 1;
        int height = img.height >> 1;

        PaddedImage inPaddedImage = img;
        int depth                 = startLevel + 1;

        Vec2u scale = 2u;

        while (depth < MAX_LEVEL)
        {
            u32 mipBorderSize          = depth < MAX_COMPRESSED_LEVEL ? borderSize : 1;
            PaddedImage outPaddedImage = GenerateMips(scratch.temp.arena, inPaddedImage, width,
                                                      height, scale, mipBorderSize);

            images[depth][i] = outPaddedImage;
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

    auto GetNeighborFaceImage = [&](PaddedImage &currentFaceImg, u32 neighborFace,
                                    u32 levelIndex, u32 rot, Vec2u &scale) {
        Vec2i dstBaseSize(currentFaceImg.log2Width, currentFaceImg.log2Height);

        u32 startLevel                         = faceStartLevel[neighborFace];
        u32 neighborLevelIndex                 = Max(startLevel, levelIndex);
        const PaddedImage &baseNeighborFaceImg = images[neighborLevelIndex][neighborFace];
        const Vec2i srcBaseSize(baseNeighborFaceImg.log2Width, baseNeighborFaceImg.log2Height);

        if (rot & 1) Swap(dstBaseSize[0], dstBaseSize[1]);
        Vec2i srcBaseDepth = srcBaseSize - dstBaseSize;
        int depthIndex     = Max(0, (int)levelIndex + Min(srcBaseDepth.x, srcBaseDepth.y));

        PaddedImage neighborFaceImg = images[depthIndex][neighborFace];
        u32 log2Width               = neighborFaceImg.log2Width;
        u32 log2Height              = neighborFaceImg.log2Height;

        u32 mipBorderSize = GetBorderSize(levelIndex);
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

    TileFileHeader header = {};
    header.numFaces       = numFaces;

    for (int levelIndex = 0; levelIndex < MAX_LEVEL; levelIndex++)
    {
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Compute);
        cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        Semaphore semaphore = device->CreateGraphicsSemaphore();

        // Add borders to all images
        for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
        {
            if (faceStartLevel[faceIndex] > levelIndex) continue;

            const Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
            Assert(!f.isSubface());

            PaddedImage &currentFaceImg = images[levelIndex][faceIndex];

            for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
            {
                // Add edge borders
                int aeid         = f.adjedge(edgeIndex);
                int neighborFace = f.adjface(edgeIndex);
                int rot          = (edgeIndex - aeid + 2) & 3;

                Vec2u scale;
                PaddedImage neighborFaceImg =
                    GetNeighborFaceImage(currentFaceImg, neighborFace, levelIndex, rot, scale);

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
                Assert(aeid != e_right || mipBorderSize > neighborFaceImg.width);
                Assert(aeid != e_top || mipBorderSize > neighborFaceImg.height);
                srcStart.x =
                    aeid == e_right ? neighborFaceImg.width - (mipBorderSize >> scale.x) : 0;
                srcStart.y =
                    aeid == e_top ? neighborFaceImg.height - (mipBorderSize >> scale.y) : 0;

                Assert(!(mipBorderSize == 1 && scale.y != 0 && scale.x != 0));
                int srcVRes = (aeid & 1) ? neighborFaceImg.height : (mipBorderSize >> scale.y);
                int srcRowLen =
                    (aeid & 1) ? (mipBorderSize >> scale.x) : neighborFaceImg.width;
                int dstVRes   = (edgeIndex & 1) ? currentFaceImg.height : mipBorderSize;
                int dstRowLen = (edgeIndex & 1) ? mipBorderSize : currentFaceImg.width;

                currentFaceImg.WriteRotatedBorder(neighborFaceImg, srcStart, start, edgeIndex,
                                                  rot, srcVRes, srcRowLen, dstVRes, dstRowLen,
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
                    PaddedImage cornerImg = GetNeighborFaceImage(
                        currentFaceImg, cfaceId[1], levelIndex, cornerRotate, cornerScale);

                    u32 cornerMipBorderSize = cornerImg.borderSize;
                    Assert(cornerMipBorderSize == currentFaceImg.borderSize);
                    Vec2u srcStart =
                        uvTable[cedgeId[1]] * Vec2u(cornerImg.width - cornerMipBorderSize,
                                                    cornerImg.height - cornerMipBorderSize);
                    Vec2u dstStart = uvTable[edgeIndex] *
                                     Vec2u(currentFaceImg.width + cornerMipBorderSize,
                                           currentFaceImg.height + cornerMipBorderSize);

                    currentFaceImg.WriteRotatedBorder(
                        cornerImg, srcStart, dstStart, cedgeId[1], cornerRotate,
                        cornerMipBorderSize >> cornerScale.y,
                        cornerMipBorderSize >> cornerScale.x, cornerMipBorderSize,
                        cornerMipBorderSize,
                        (cornerRotate & 1) ? cornerScale.yx() : cornerScale);
                }
                else if (numCorners > 1)
                {
                    // TODO: what do I do here?
                }
                else
                {
                }
            }

            if (levelIndex < MAX_COMPRESSED_LEVEL)
            {
                // Block compress the entire face texture
                ImageDesc blockCompressedImageDesc(
                    ImageType::Type2D, currentFaceImg.GetPaddedWidth(),
                    currentFaceImg.GetPaddedHeight(), 1, 1, 1, baseFormat,
                    MemoryUsage::GPU_ONLY,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_TILING_OPTIMAL);

                TransferBuffer blockCompressedImage =
                    cmd->SubmitImage(currentFaceImg.contents, blockCompressedImageDesc);

                u32 outputBlockWidth  = currentFaceImg.GetPaddedWidth() >> log2BlockSize;
                u32 outputBlockHeight = currentFaceImg.GetPaddedHeight() >> log2BlockSize;
                ImageDesc uavDesc(ImageType::Type2D, outputBlockWidth, outputBlockHeight, 1, 1,
                                  1, VK_FORMAT_R32G32_UINT, MemoryUsage::GPU_ONLY,
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                  VK_IMAGE_TILING_OPTIMAL);

                GPUImage uavImage = device->CreateImage(uavDesc);

                GPUBuffer readbackBuffer =
                    device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         outputBlockWidth * outputBlockHeight * bytesPerBlock,
                                         MemoryUsage::GPU_TO_CPU);

                cmd->Barrier(
                    &blockCompressedImage.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                cmd->Barrier(&uavImage, VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT);
                cmd->FlushBarriers();

                gpuSrcImages[faceIndex]          = blockCompressedImage;
                tempUavs[faceIndex]              = uavImage;
                blockCompressedImages[faceIndex] = readbackBuffer;

                DescriptorSet set = bcLayout.CreateNewDescriptorSet();
                set.Bind(inputBinding, &gpuSrcImages[faceIndex].image);
                set.Bind(outputBinding, &tempUavs[faceIndex]);
                cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &set,
                                        bcLayout.pipelineLayout);
                cmd->Dispatch((outputBlockWidth + 7) >> 3, (outputBlockHeight + 7) >> 3, 1);

                cmd->Barrier(&tempUavs[faceIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd->FlushBarriers();

                descriptorSets[faceIndex] = set;

                BufferImageCopy copy = {};
                copy.layerCount      = 1;
                copy.extent          = Vec3u(outputBlockWidth, outputBlockHeight, 1);
                cmd->CopyImageToBuffer(&blockCompressedImages[faceIndex], &tempUavs[faceIndex],
                                       copy);

                runningCount++;
                // Run block compression concurrently with border copying
                if (runningCount == flushSize)
                {
                    if (semaphore.signalValue) cmd->Wait(semaphore);
                    semaphore.signalValue++;
                    cmd->Signal(semaphore);
                    device->SubmitCommandBuffer(cmd);

                    cmd = device->BeginCommandBuffer(QueueType_Compute);
                    cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                }
            }
            else
            {
                GPUBuffer buf;
                buf.mappedPtr                    = currentFaceImg.contents;
                blockCompressedImages[faceIndex] = buf;
            }
        }

        if (semaphore.signalValue) cmd->Wait(semaphore);
        semaphore.signalValue++;
        cmd->Signal(semaphore);
        device->SubmitCommandBuffer(cmd);

        device->Wait(semaphore);

        // Pack each subsequent mip level into variable sized tiles. At each subsequent mip
        // level, the tile size increases (136, 144, 160, 192, etc.). This allows the offset
        // variable to be used for all mip levels. At level 1, the texelWidth is 64, including
        // the border is 72. Thus, 4 sub tiles can fit into a 144x144 texel tile. At level 2,
        // 16 sub tiles can fit. Sub tiles are packed in row major order.
        //
        // At MAX_COMPRESSED_LEVEL, the border size switches to 1, block compression is no
        // longer used, and bilinear filtering is used instead of catmull-rom. This prevents
        // intractably large tile sizes at high mip levels.
        //
        // At runtime: The virtual page offset is shifted down by the number of dimensions *
        // the current level to get the virtual index used to index the page table. The
        // page table is a Texture1D with mips. The bottom dim * level bits determine the
        // location within the 128x128 page. e.g. for the first mip level, if the calculated
        // offset is 7, we shift down by 2 to get 1. This is the virtual address we use to look
        // up in the page table, at mip level 1. Once we get the physical page, the lower 2
        // bits (3) means our start address in the page is (0.5, 0.5), i.e. the bottom right
        // 72 x 72 tile

        u32 currentTileOffset  = 0;
        u32 maxSubTilesPerTile = 1u << (levelIndex * 2);

        u32 mipBorderSize = levelIndex < MAX_COMPRESSED_LEVEL ? borderSize : 1;
        u32 tileWidth     = GetTileTexelWidth(levelIndex);

        u32 levelLog2BlockSize = levelIndex < MAX_COMPRESSED_LEVEL ? log2BlockSize : 0;
        u32 levelBytesPerBlock =
            levelIndex < MAX_COMPRESSED_LEVEL ? bytesPerBlock : bytesPerTexel;

        u32 currentLevelBlocksPerPage =
            Max((texelWidthPerPage >> levelIndex) >> levelLog2BlockSize, 1u);
        u32 tileBlockWidth = tileWidth >> levelLog2BlockSize;
        u32 subTileVLen    = tileBlockWidth >> levelIndex;
        u32 subTileRowLen  = subTileVLen * levelBytesPerBlock;

        u32 tileSize = tileBlockWidth * tileBlockWidth * levelBytesPerBlock;
        Tile tile;
        tile.contents = PushArray(scratch.temp.arena, u8, tileSize);

        header.tileSizes[levelIndex] = tileSize;
        header.offsets[levelIndex]   = (u32)tiles.size();
        for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
        {
            if (faceStartLevel[faceIndex] > levelIndex) continue;
            u32 offset = (u32)tiles.size();

            GPUBuffer *gpuImage = &blockCompressedImages[faceIndex];

            PaddedImage *image = &images[levelIndex][faceIndex];

            u32 numTilesX = Max(image->width / Max(texelWidthPerPage >> levelIndex, 1u), 1u);
            u32 numTilesY = Max(image->height / Max(texelWidthPerPage >> levelIndex, 1u), 1u);

            for (int tileY = 0; tileY < numTilesY; tileY++)
            {
                for (int tileX = 0; tileX < numTilesX; tileX++)
                {
                    Vec2u srcStart(currentLevelBlocksPerPage * tileX,
                                   currentLevelBlocksPerPage * tileY);

                    u32 dstTileX = currentTileOffset & ((1u << levelIndex) - 1);
                    u32 dstTileY = currentTileOffset >> levelIndex;
                    Vec2u dstStart(subTileVLen * dstTileX, subTileVLen * dstTileY);

                    Utils::Copy(gpuImage->mappedPtr, srcStart,
                                image->GetPaddedWidth() >> levelLog2BlockSize,
                                image->GetPaddedHeight() >> levelLog2BlockSize, tile.contents,
                                dstStart, tileBlockWidth, tileBlockWidth, subTileVLen,
                                subTileRowLen, levelBytesPerBlock);

                    currentTileOffset++;
                    if (currentTileOffset == maxSubTilesPerTile)
                    {
                        currentTileOffset = 0;
                        tiles.push_back(tile);
                        tile.contents = PushArray(scratch.temp.arena, u8, tileSize);
                    }
                }
            }
            u32 size = (u32)tiles.size() - offset;

            if (metaData[faceIndex].log2Width == 0)
                metaData[faceIndex] = {offset, levelIndex, image->log2Width,
                                       image->log2Height};
        }
        header.sizes[levelIndex] = (u32)tiles.size() - header.offsets[levelIndex];
        for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
        {
            device->DestroyBuffer(&gpuSrcImages[faceIndex].stagingBuffer);
            device->DestroyImage(&gpuSrcImages[faceIndex].image);
            device->DestroyImage(&tempUavs[faceIndex]);
            device->DestroyBuffer(&blockCompressedImages[faceIndex]);
            device->DestroyPool(descriptorSets[faceIndex].pool);
        }
    }

    // Write metadata and tiles to disk
    StringBuilderMapped builder(outFilename);

    u32 numTiles = (u32)tiles.size();
    PutData(&builder, &header, sizeof(header));
    PutData(&builder, metaData.data, sizeof(TileMetadata) * metaData.Length());
    for (int levelIndex = 0; levelIndex < MAX_LEVEL; levelIndex++)
    {
        int offset = header.offsets[levelIndex];
        for (int tileIndex = offset; tileIndex < offset + header.sizes[levelIndex];
             tileIndex++)
        {
            PutData(&builder, tiles[tileIndex].contents, header.tileSizes[levelIndex]);
        }
    }

    OS_UnmapFile(builder.ptr);
    OS_ResizeFile(builder.filename, builder.totalSize);

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

VirtualTextureManager::VirtualTextureManager(Arena *arena, u32 numVirtualPages,
                                             u32 numPhysicalPages, int numLevels,
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
    descriptorSetLayout.AddBinding(1, DescriptorType::StorageImage,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    push     = PushConstant(ShaderStage::Compute, 0, sizeof(PageTableUpdatePushConstant));
    pipeline = device->CreateComputePipeline(&shader, &descriptorSetLayout, &push);

    ImageLimits limits = device->GetImageLimits();

    levelInfo = StaticArray<LevelInfo>(arena, numLevels, numLevels);
    // Allocate page table

    ImageDesc pageTableDesc(ImageType::Type1D, numVirtualPages, 1, 1, 1, numLevels,
                            VK_FORMAT_R32_UINT, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    pageTable = device->CreateImage(pageTableDesc);

    {
        Assert(IsPow2(inPageWidthPerPool));
        pageWidthPerPool = inPageWidthPerPool;

        u32 numPagesPerPool = Sqr(pageWidthPerPool);
        u32 numPools        = (numPhysicalPages + numPagesPerPool - 1) / numPagesPerPool;

        maxNumLayers         = 1u << Log2Int(limits.maxNumLayers);
        u32 numTextureArrays = (numPools + maxNumLayers - 1) / maxNumLayers;
        Assert(numTextureArrays == 1);

        // Allocate physical page pool
        pools = StaticArray<PhysicalPagePool>(arena, numPools);
        for (int i = 0; i < numPools; i++)
        {
            PhysicalPagePool pool;
            pool.ranges    = StaticArray<BlockRange>(arena, numPagesPerPool);
            pool.freeRange = 0;
            pool.freePages = numPagesPerPool;
            pool.prevFree  = i == 0 ? InvalidPool : i - 1;
            pool.nextFree  = i == numPools - 1 ? InvalidPool : i + 1;
            pools.Push(pool);
        }

        // Allocate block ranges
        const u32 invalidRange = BlockRange::InvalidRange;
        pageRanges.Push(BlockRange(AllocationStatus::Free, 0, numVirtualPages, invalidRange,
                                   invalidRange, invalidRange, invalidRange));

        completelyFreePool = 0;
        partiallyFreePool  = InvalidPool;
    }

    for (int levelIndex = 0; levelIndex < numLevels; levelIndex++)
    {
        LevelInfo &level = levelInfo[levelIndex];
        // level.pageTableSubresourceIndex = device->CreateSubresource(&pageTable, levelIndex,
        // 1);
        level.texelWidthPerPage = GetTileTexelWidth(levelIndex);

        u32 texelWidthPerPool = level.texelWidthPerPage * pageWidthPerPool;
        ErrorExit(Min(texelWidthPerPool, limits.max2DImageDim),
                  "Page width per pool is too large.\n");

        // Allocate texture arrays
        ImageDesc poolDesc(ImageType::Array2D, texelWidthPerPool, texelWidthPerPool, 1, 1,
                           numPools, format, MemoryUsage::GPU_ONLY,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        level.gpuPhysicalPool = device->CreateImage(poolDesc);
    }

    pageRanges = StaticArray<BlockRange>(arena, numVirtualPages);
    freeRange  = 0;
}

u32 BlockRange::FindBestFree(const StaticArray<BlockRange> &ranges, u32 freeIndex, u32 num,
                             u32 leftover)
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

void BlockRange::Split(StaticArray<BlockRange> &ranges, u32 index, u32 &freeIndex, u32 num)
{
    BlockRange &range = ranges[index];

    u32 oldOnePastEnd = range.onePastEnd;
    u32 oldNextRange  = range.nextRange;
    u32 oldPrevFree   = range.prevFree;
    u32 oldNextFree   = range.nextFree;
    u32 newRangeIndex = ranges.Length();

    Assert(range.status == AllocationStatus::Free);
    range.status     = AllocationStatus::Allocated;
    range.onePastEnd = range.start + num;

    if (range.onePastEnd != oldOnePastEnd)
    {
        range.nextRange = newRangeIndex;

        UnlinkFreeList(ranges, index, freeIndex, InvalidRange);

        BlockRange newRange(AllocationStatus::Free, range.start + num, oldOnePastEnd, index,
                            oldNextRange, oldPrevFree, oldNextFree);

        ranges.Push(newRange);
    }
}

u32 BlockRange::GetNum() const { return onePastEnd - start; }

// TODO: instead of having multiple tiles per face, the minimum streamable unit is going to
// just be an individual face textures (and its higher mips). additionally, if mips must be
// PHYSICALLY allocated together
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

// this is my idea:
// observation: face textures have have dimensions in the range 64-512 generally. thus, we can
// "reserve" a row for a particularly height dimension, and face textures with the same height
// can go into this row. if a row with a height dimension isn't found, then we create a new one
// if available, OR we can pack it into a row with a larger height dimension if there is no
// space available necessary.

Vec2i CalculateFaceSize(int log2Width, int log2Height, int levelIndex)
{
    int borderSize = GetBorderSize(levelIndex);
    Vec2i allocationSize =
        Vec2i((1u << log2Width) + 2 * borderSize, (1u << log2Height) + 2 * borderSize);
    return allocationSize;
}

u32 VirtualTextureManager::AllocatePhysicalPages2(CommandBuffer *cmd, TileRequest *requests,
                                                  u32 numRequests, TileMetadata *metadata,
                                                  u32 numFaces)
{
    ScratchArena scratch;

    const int InvalidNode = -1;
    const int InvalidRow  = -1;

    Shelf *currentRow     = 0;
    u32 currentLog2Height = ~0u;

    TransferBuffer transferBuffer;
    u32 bufferOffset = 0;

    StaticArray<PageTableUpdateRequest> updateRequests(scratch.temp.arena, numRequests,
                                                       numRequests);

    // TODO: the reqeusts should be radix sorted with key: hhhhwwww (height in MSBs)
    for (int requestIndex = 0; requestIndex < numRequests; requestIndex++)
    {
        const TileRequest &request = requests[requestIndex];
        int faceIndex              = request.faceIndex;

        const TileMetadata &faceMetadata = metadata[faceIndex];
        int startLevel                   = faceMetadata.startLevel;

        Vec2i allocationSize =
            CalculateFaceSize(faceMetadata.log2Width, faceMetadata.log2Height, startLevel);

        // Allocation space for subsequent mip levels to the right
        allocationSize.x +=
            (1u << (faceMetadata.log2Width - 1u)) + 2 * GetBorderSize(startLevel + 1);

        // First find the pool to use
        u32 freePoolIndex =
            partiallyFreePool == InvalidPool ? completelyFreePool : partiallyFreePool;

        PhysicalPagePool *pool = &pools[freePoolIndex];

        currentLog2Height = faceMetadata.log2Height;

        u32 shelfStart = pool->shelfStarts[currentLog2Height];

        u32 blockRangeIndex = BlockRange::InvalidRange;
        int rowIndex        = -1;

        // Find a suitable shelf
        for (;;)
        {
            while (shelfStart == InvalidRow)
            {
                if (pool->totalHeight + allocationSize.y > pool->maxHeight)
                {
                    // Go to next free pool if there is no more space in current pool
                    freePoolIndex = pool->nextFree;
                    Assert(freePoolIndex != InvalidPool);
                    pool = &pools[freePoolIndex];
                }
                else
                {
                    // Allocate a new row in the current pool
                    pool->shelves.emplace_back();
                    Shelf &newRow = pool->shelves.back();
                    newRow.startY = pool->totalHeight;
                    newRow.height = allocationSize.y;
                    newRow.ranges.reserve(pool->maxWidth / allocationSize.x);
                    newRow.ranges.push_back(
                        BlockRange(AllocationStatus::Free, 0, pool->maxWidth));
                    LinkFreeList(pool->shelves, pool->shelves.size() - 1,
                                 pool->shelfStarts[currentLog2Height], InvalidRow);

                    pool->totalHeight += allocationSize.y;
                    shelfStart = pool->shelfStarts[currentLog2Height];
                }
            }

            // Find suitable span in row
            Shelf *row = &pool->shelves[shelfStart];
            u32 rangeIndex =
                BlockRange::FindBestFree(row->ranges, row->freeRange, allocationSize.x);

            if (rangeIndex == ~0u)
            {
                u32 leftover = ~0u;

                // First try to fit into rows with larger height
                for (int testDimIndex = faceMetadata.log2Height;
                     testDimIndex < pool->shelfStarts.Length(); testDimIndex++)
                {
                    int freeRow = pool->shelfStarts[testDimIndex];
                    while (freeRow != InvalidRow)
                    {
                        Shelf *testRow     = &pool->shelves[freeRow];
                        u32 testRangeIndex = BlockRange::FindBestFree(
                            testRow->ranges, testRow->freeRange, allocationSize.x, leftover);
                        if (testRangeIndex != BlockRange::InvalidRange)
                        {
                            rowIndex        = freeRow;
                            blockRangeIndex = testRangeIndex;
                            leftover =
                                testRow->ranges[testRangeIndex].GetNum() - allocationSize.x;
                        }
                        freeRow = testRow->nextFree;
                    }
                }

                // If a row was found that can fit the current entry
                if (rowIndex != -1)
                {
                    row = pool->shelves[rowIndex];
                    BlockRange::Split(row->ranges, rangeIndex, row->freeRange,
                                      allocationSize.x);
                }
                // Repeat the loop
                else
                {
                    shelfStart = InvalidRow;
                    continue;
                }
            }
            else
            {
                BlockRange::Split(row->ranges, rangeIndex, row->freeRange, allocationSize.x);
                rowIndex        = shelfStart;
                blockRangeIndex = rangeIndex;
            }
            break;
        }

        // Create the buffer image copy commands and page table update requests
        const Shelf &row             = pool->shelves[rowIndex];
        const BlockRange &blockRange = row.ranges[rangeIndex];

        u32 packed       = 0;
        u32 packedOffset = 0;
        packed           = BitFieldPackU32(packed, blockRange.start, packedOffset, 15);
        packed           = BitFieldPackU32(packed, row.startY, packedOffset, 15);
        Assert(pool->layerIndex < 4);
        packed = BitFieldPackU32(packed, pool->layerIndex, packedOffset, 2);

        // TODO: how do I store the current max packed mip without needing another u32?
        updateRequests[requestIndex].faceIndex        = faceIndex;
        updateRequests[requestIndex].packed_x_y_layer = packed;

        Vec3i start = Vec3i(blockRange.start, row.startY, 0);
        Vec2i begin = start.xy;
        for (int levelIndex = 0; levelIndex < request.numLevels; levelIndex++)
        {
            Assert(start.xy - begin.xy < allocationSize);
            BufferImageCopy copy;
            copy.bufferOffset = bufferOffset;
            copy.baseLayer    = pool->layerIndex;
            copy.layerCount   = 1;
            copy.offset       = start;

            Vec2i extent = CalculateFaceSize(faceMetadata.log2Width, faceMetadata.log2Height,
                                             request.startLevelIndex + levelIndex);
            copy.extent  = Vec3u(extent.x, extent.y, 1);

            start[levelIndex & 1] += extent[levelIndex & 1];
        }
    }

    PageTableUpdatePushConstant pc;
    pc.numRequests = numRequests;

    cmd->SubmitTransfer(&transferBuffer);
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
    cmd->Dispatch((numPages + 63) >> 6, 1, 1);
}

void VirtualTextureManager::AllocatePhysicalPages(
    CommandBuffer *cmd, const TileFileHeader &header, TileMetadata *metadata,
    TileRequest *tileRequests, int numTileRequests, u32 numFaces, u8 *contents, u32 allocIndex)
{
    const BlockRange &range = pageRanges[allocIndex];
    const u32 numPages      = range.GetNum();

    ScratchArena scratch;

    u64 offset = 0;

    PageTableUpdateRequest *requests =
        PushArray(scratch.temp.arena, PageTableUpdateRequest, numPages);

    u32 log2PageWidthPerPool = Log2Int(pageWidthPerPool);

    u32 leftover = ~0u;

    u32 *allocationRequests = PushArray(scratch.temp.arena, u32, numFaces);

    // Allocate continuous ranges of pages for each face
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        // First allocate page
        u32 freeIndex =
            partiallyFreePool == InvalidPool ? completelyFreePool : partiallyFreePool;
        bool isFullyFree = partiallyFreePool == InvalidPool;

        u32 poolAllocIndex = ~0u;
        u32 pageAllocIndex = ~0u;

        u32 log2Width  = metadata[faceIndex].log2Width;
        u32 log2Height = metadata[faceIndex].log2Height;
        u32 minDim     = Min(log2Width, log2Height);

        // TODO: At the highest mip levels (7+), this value changes. Either I need to clamp,
        // or handle them specially in some way (e.g. have them always be resident?)
        u32 faceNumPages =
            1u << ((log2Width - Min(minDim, 7u)) + (log2Height - Min(minDim, 7u)));

        while (freeIndex != InvalidPool)
        {
            PhysicalPagePool &pool = pools[freeIndex];
            if (pool.freePages >= numPages)
            {
                u32 rangeIndex =
                    BlockRange::FindBestFree(pool.ranges, pool.freeRange, leftover);
                if (rangeIndex != BlockRange::InvalidRange)
                {
                    u32 num = pool.ranges[rangeIndex].GetNum();
                    Assert(num < faceNumPages + leftover);
                    leftover = num - faceNumPages;

                    poolAllocIndex = freeIndex;
                    pageAllocIndex = rangeIndex;

                    BlockRange::Split(pool.ranges, rangeIndex, pool.freeRange, numPages);

                    pool.freePages -= numPages;
                    if (isFullyFree)
                    {
                        UnlinkFreeList(pools, poolAllocIndex, completelyFreePool, InvalidPool);
                        LinkFreeList(pools, poolAllocIndex, partiallyFreePool, InvalidPool);
                    }
                    else if (pool.freeRange == BlockRange::InvalidRange)
                    {
                        UnlinkFreeList(pools, poolAllocIndex, partiallyFreePool, InvalidPool);
                    }
                    break;
                }
                else
                {
                    Print("need to defrag physical pages\n");
                    // TODO: defrag pool
                }
            }
            freeIndex = pool.nextFree;
        }

        ErrorExit(allocIndex != ~0u, "Ran out of memory");
        Assert(poolAllocIndex < maxNumLayers);

        Assert(pageAllocIndex < (1u << (2u * 7u)));
        u32 request                   = (poolAllocIndex << (2u * 7u)) | pageAllocIndex;
        allocationRequests[faceIndex] = request;

        // TODO: this faceIndex needs to be offset by something
        requests[faceIndex] = PageTableUpdateRequest{
            faceIndex,
            pools[poolAllocIndex].ranges[pageAllocIndex].start,
        };
    }

    // Allocate buffer to image copy commands

    // for (int levelIndex = 0; levelIndex < MAX_LEVEL; levelIndex++)
    // {
    //     const LevelInfo &level = levelInfo[levelIndex];
    //     u32 levelPageSize =
    //         Sqr(level.texelWidthPerPage >> GetBlockShift(format)) * GetFormatSize(format);
    //     for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    //     {
    //         if (metadata[faceIndex].startLevel > levelIndex ||
    //             metadata[faceIndex].endLevel < levelIndex)
    //             continue;
    //
    //         BufferImageCopy copy;
    //
    //         MemoryCopy((u8 *)buf.mappedPtr + offset, contents, levelPageSize);
    //         contents += levelPageSize;
    //
    //         u32 request = allocationRequests[faceIndex];
    //
    //         u32 bitShift = (2u * (7u + levelIndex));
    //         u32 bitMask  = (1u << (7u + levelIndex)) - 1u;
    //
    //         copy.bufferOffset = offset;
    //         copy.baseLayer    = request >> bitShift;
    //         copy.layerCount   = 1;
    //
    //         u32 pageX = request & bitMask;
    //         u32 pageY = (request >> (7u + levelIndex)) & bitMask;
    //
    //         copy.offset =
    //             Vec3i(pageX * level.texelWidthPerPage, pageY * level.texelWidthPerPage, 0);
    //         copy.extent = Vec3u(level.texelWidthPerPage, level.texelWidthPerPage, 1);
    //
    //         copies.push_back(copy);
    //         offset += levelPageSize;
    //     }
    // }

    // i think i need a two level mapping: one from face index to something, then from
    // something to a physical page
    u32 log2BlockSize = GetBlockShift(format);
    u32 allocSize     = 0;
    for (int tileRequestIndex = 0; tileRequestIndex < numTileRequests; tileRequestIndex++)
    {
        const TileRequest &request = tileRequests[tileRequestIndex];
        int levelIndex             = request.levelIndex;

        u32 levelLog2BlockSize = levelIndex < MAX_COMPRESSED_LEVEL ? log2BlockSize : 0;
        const LevelInfo &level = levelInfo[levelIndex];
        u32 pageSize =
            Sqr(level.texelWidthPerPage >> levelLog2BlockSize) * GetFormatSize(format);
        allocSize += pageSize;
    }

    TransferBuffer buf = device->GetStagingBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, allocSize);

    u32 levelOffsets[MAX_LEVEL];
    u32 levelOffset = 0;
    u32 dstOffset   = 0;
    FixedArray<StaticArray<BufferImageCopy, MAX_LEVEL>> copies(MAX_LEVEL);

    for (int i = 0; i < MAX_LEVEL; i++)
    {
        levelOffsets[i] = levelOffset;
        levelOffset += header.tileSizes[i] * header.tileCounts[i];
        copies[i] = StaticArray<BufferImageCopy>(scratch.temp.arena, numTileRequests);
    }

    // from the faces and levels, get the tiles I need
    // ensure that the single offset maps to the right tile at every level
    for (int tileRequestIndex = 0; tileRequestIndex < numTileRequests; tileRequestIndex++)
    {
        // TODO: make sure I'm not repeating copies?
        const TileRequest &request = tileRequests[tileRequestIndex];
        u32 allocationRequest      = allocationRequests[request.faceIndex];
        u32 rangeIndex             = allocationRequest & ((1u << (2u * 7u)) - 1u);
        const BlockRange &range    = pageRanges[rangeIndex];

        for (int levelIndex = request.startLevelIndex;
             levelIndex < request.startLevelIndex + request.numLevels; levelIndex++)
        {
            Assert(levelIndex < MAX_LEVEL);

            u32 levelLog2BlockSize = levelIndex < MAX_COMPRESSED_LEVEL ? log2BlockSize : 0;
            const LevelInfo &level = levelInfo[levelIndex];
            u32 pageSize =
                Sqr(level.texelWidthPerPage >> levelLog2BlockSize) * GetFormatSize(format);

            u32 startTile = range.start >> (2u * levelIndex);
            Assert(range.onePastEnd != 0);
            u32 endTile = ((range.onePastEnd - 1) >> (2u * levelIndex)) + 1;

            u32 faceTileLevelOffset = ? ;
            for (u32 tileIndex = startTile; tileIndex < endTile;
                 tileIndex++, faceTileLevelOffset++)
            {
                BufferImageCopy copy;
                const u8 *src = contents + levelOffset[levelIndex] +
                                header.tileSizes[levelIndex] * faceTileLevelOffset;
                MemoryCopy((u8 *)buf.mappedPtr + dstOffset, src, pageSize);
                dstOffset += pageSize;

                u32 request = allocationRequests[faceIndex];

                u32 bitShift = (2u * (7u + levelIndex));
                u32 bitMask  = (1u << (7u + levelIndex)) - 1u;

                copy.bufferOffset = dstOffset;
                copy.baseLayer    = request >> bitShift;
                copy.layerCount   = 1;

                // TODO: one offset doesn't seem to work because at level 7, I can only have
                // one physical page??? the only other way to make this work is if
                // only one mip or a couple of mips are resident at once?
                u32 pageAllocIndex = request & ((1u << (2u * 7u)) - 1u);
                u32 pageX          = request & bitMask;
                u32 pageY          = (request >> (7u + levelIndex)) & bitMask;

                copy.offset =
                    Vec3i(pageX * level.texelWidthPerPage, pageY * level.texelWidthPerPage, 0);
                copy.extent = Vec3u(level.texelWidthPerPage, level.texelWidthPerPage, 1);

                copies[levelIndex].Push(copy);
            }
        }
    }

    PageTableUpdatePushConstant pc;
    pc.numRequests = numFaces;

    cmd->SubmitTransfer(&buf);
    TransferBuffer pageTableUpdateRequestBuffer =
        cmd->SubmitBuffer(requests, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          sizeof(PageTableUpdateRequest) * numFaces);

    DescriptorSet ds = descriptorSetLayout.CreateDescriptorSet();
    ds.Bind(0, &pageTableUpdateRequestBuffer.buffer).Bind(1, &pageTable);
    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // Submit copy commands
    {
        for (int levelIndex = 0; levelIndex < copies.Length(); levelIndex++)
        {
            const LevelInfo &level = levelInfo[levelIndex];
            if (copies[levelIndex].Length())
                cmd->Barrier(&level.gpuPhysicalPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        }
        cmd->FlushBarriers();
        for (int levelIndex = 0; levelIndex < copies.Length(); levelIndex++)
        {
            const LevelInfo &level = levelInfo[levelIndex];
            if (copies[levelIndex].Length())
                cmd->CopyImage(&buf, &level.gpuPhysicalPool, copies[levelIndex].data,
                               copies[levelIndex].Length());
        }
    }

    cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmd->PushConstants(&push, &pc, descriptorSetLayout.pipelineLayout);
    cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                            descriptorSetLayout.pipelineLayout);
    cmd->Dispatch((numPages + 63) >> 6, 1, 1);
}

} // namespace rt
