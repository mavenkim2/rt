#include "base.h"
#include "memory.h"
#include "string.h"
#include "scene.h"
#include "integrate.h"
#include "ptex.h"
#include "vulkan.h"
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
    Assert(p.x < width + 2 * borderSize && p.y < height + 2 * borderSize);
    return contents + strideWithBorder * p.y + p.x;
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
    u8 *data          = PushArrayNoZero(arena, u8, size);
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

PaddedImage **Convert(Arena *arena, PtexTexture *texture, int filterWidth, int &outNumFaces)
{
    // Get every mip level
    TempArena temp = ScratchStart(&arena, 1);

    // Highest mip level
    Ptex::String error;
    Ptex::PtexTexture *t     = cache->get((char *)texture->filename.str, error);
    Ptex::PtexReader *reader = static_cast<Ptex::PtexReader *>(t);
    int numFaces             = reader->numFaces();
    int borderSize           = filterWidth - 1;

    PaddedImage **images = PushArrayNoZero(arena, PaddedImage *, numFaces);
    int *numLevels       = PushArrayNoZero(temp.arena, int, numFaces);
    const int maxDepth   = 10;

    int log2FilterWidth = Log2Int(filterWidth);

    for (int i = 0; i < numFaces; i++)
    {
        PaddedImage img = PtexToImg(arena, t, i, borderSize, false);
        int levels      = Max(Max(img.log2Width, img.log2Height), 1);
        images[i]       = PushArrayNoZero(arena, PaddedImage, levels);
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
            outPaddedImage.contents         = PushArrayNoZero(
                arena, u8, outPaddedImage.strideWithBorder * (height + 2 * borderSize));
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
            width         = Max(width >> 1, filterWidth);
            scale[0]      = prevWidth / width;

            u32 prevHeight = height;
            height         = Max(height >> 1, filterWidth);
            scale[1]       = prevHeight / height;

            depth++;
        }
    }
    // Fill border with adjacent texels based on filter width
    Assert(filterWidth > 0);

    enum EdgeId
    {
        e_bottom, ///< Bottom edge, from UV (0,0) to (1,0)
        e_right,  ///< Right edge, from UV (1,0) to (1,1)
        e_top,    ///< Top edge, from UV (1,1) to (0,1)
        e_left,   ///< Left edge, from UV (0,1) to (0,0)
        e_max,
    };

    // Copy shared borders
    // TODO: instead of doing two copies, I could attempt to pack textures into 128x128
    // tiles, with shared edges always in the same tile? corners could be an issue
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        const Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
        Assert(!f.isSubface());
        for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
        {
            int aeid         = f.adjedge(edgeIndex);
            int neighborFace = f.adjface(edgeIndex);
            int rot          = (edgeIndex - aeid + 2) & 3;

            Vec2u dstBaseSize(images[faceIndex][0].log2Width, images[faceIndex][0].log2Height);
            Vec2u srcBaseSize(images[neighborFace][0].log2Width,
                              images[neighborFace][0].log2Height);

            int dstCompareDim = (edgeIndex & 1) ? dstBaseSize.y : dstBaseSize.x;
            int srcCompareDim = (aeid & 1) ? srcBaseSize.y : srcBaseSize.x;

            // Find corner face
            // (taken from PtexSeparableFilters.cpp)

            for (int depth = 0; depth < numLevels[faceIndex]; depth++)
            {
                int srcBaseDepth = srcCompareDim - dstCompareDim;
                // int srcDepthIndex = Clamp(srcBaseDepth, 0, maxDepth - 1);
                int srcDepthIndex = Clamp(srcBaseDepth, 0, numLevels[neighborFace] - 1);

                PaddedImage &currentFaceImg = images[faceIndex][depth];
                PaddedImage neighborFaceImg = images[neighborFace][srcDepthIndex];

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

                currentFaceImg.WriteRotatedBorder(neighborFaceImg, srcStart, start, edgeIndex,
                                                  rot, srcVRes, srcRowLen, dstVRes, dstRowLen,
                                                  scale);

                dstCompareDim--;
            }
        }
    }

    outNumFaces = numFaces;
    ScratchEnd(temp);
    t->release();
    return images;
}

// steps:
// 1. write borders, including corners
//      - how do I handle
// 2. block compress on gpu
// 3. group into tiles, write to disk
//      - how do I load the right tiles from disk at runtime?

// 4. load the right tiles from disk at runtime, keeping highest mip level
// always resident
// 5. create a page table texture
// 6. upload the tiles to the gpu, and update the page table texture
// 7. on gpu, look at page table texture to find right tiles. do the texture lookup
// + filter
// 8. if tile isn't found, populate feedback buffer
// 9. asynchronously read feedback buffer on cpu. go to step 4
void Convert(string filename)
{
    ScratchArena scratch;
    const Vec2u uvTable[] = {
        Vec2u(0, 0),
        Vec2u(0, 1),
        Vec2u(1, 1),
        Vec2u(0, 1),
    };

    // Highest mip level
    Ptex::String error;
    Ptex::PtexTexture *t     = cache->get((char *)filename.str, error);
    Ptex::PtexReader *reader = static_cast<Ptex::PtexReader *>(t);
    int numFaces             = reader->numFaces();

    PaddedImage *images = PushArrayNoZero(scratch.temp.arena, PaddedImage, numFaces);
    GPUImage *blockCompressedImages = PushArrayNoZero(scratch.temp.arena, GPUImage, numFaces);

    std::vector<Tile> tiles;

    const u32 bytesPerPixel    = 3;
    const u32 gpuBytesPerPixel = 4;
    const u32 borderSize       = 4;

    const u32 texelsPerTileX      = 128;
    const u32 texelsPerTileY      = 128;
    const u32 totalTexelsPerTileX = 128 + 2 * borderSize;
    const u32 totalTexelsPerTileY = 128 + 2 * borderSize;

    const u32 tileStride = texelsPerTileX * bytesPerPixel;

    const u32 bytesPerBlock       = 8;
    const u32 blockSize           = GetBlockSize(VK_FORMAT_BC1_RGB_UNORM_BLOCK);
    const u32 totalBlocksPerTileX = totalTexelsPerTileX / blockSize;
    const u32 totalBlocksPerTileY = totalTexelsPerTileY / blockSize;

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
        pipeline      = device->CreateComputePipeline(&shader, &bcLayout);
        initialized   = true;
    }

    // Create all face images
    for (int i = 0; i < numFaces; i++)
    {
        images[i] = PtexToImg(scratch.temp.arena, t, i, borderSize, false);
    }

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

            if (srcDepthIndex > 0)
            {
                // TODO: handle case where neighbor is larger than current face
                continue;
            }

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
                Vec2u srcStart = uvTable[cedgeId[2]] * Vec2u(cornerImg.width - borderSize,
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
        u8 *contents =
            PushArray(scratch.temp.arena, u8,
                      currentFaceImg.strideWithBorder * currentFaceImg.GetPaddedHeight() /
                          bytesPerPixel * gpuBytesPerPixel);
        u32 dstOffset = 0;
        u32 srcOffset = 0;
        for (int i = 0; i < currentFaceImg.GetPaddedWidth() * currentFaceImg.GetPaddedHeight();
             i++)
        {
            MemoryCopy(contents + dstOffset, currentFaceImg.contents + srcOffset,
                       bytesPerPixel);
            dstOffset += gpuBytesPerPixel;
            srcOffset += bytesPerPixel;
        }

        // Block compress the entire face texture
        ImageDesc blockCompressedImageDesc(
            ImageType::Type2D, currentFaceImg.GetPaddedWidth(),
            currentFaceImg.GetPaddedHeight(), 1, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
            MemoryUsage::GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TILING_OPTIMAL);

        GPUImage blockCompressedImage =
            cmd->SubmitImage(currentFaceImg.contents, blockCompressedImageDesc);

        ImageDesc readbackDesc(ImageType::Type2D, currentFaceImg.GetPaddedWidth() / blockSize,
                               currentFaceImg.GetPaddedHeight() / blockSize, 1, 1, 1,
                               VK_FORMAT_R32G32_UINT, MemoryUsage::GPU_TO_CPU,
                               VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_TILING_OPTIMAL);

        GPUImage readbackImage = device->CreateImage(readbackDesc);

        cmd->Barrier(&blockCompressedImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->Barrier(&readbackImage, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->FlushBarriers();
        blockCompressedImages[faceIndex] = readbackImage;

        DescriptorSet set = bcLayout.CreateDescriptorSet();
        set.Bind(inputBinding, &blockCompressedImage);
        set.Bind(outputBinding, &blockCompressedImages[faceIndex]);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &set, bcLayout.pipelineLayout);
    }

    Semaphore semaphore   = device->CreateGraphicsSemaphore();
    semaphore.signalValue = 1;
    cmd->Signal(semaphore);
    device->SubmitCommandBuffer(cmd);

    device->Wait(semaphore);

    // Discretize into tiles
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        GPUImage *gpuImage = &blockCompressedImages[faceIndex];
        PaddedImage *image = &images[faceIndex];

        u32 numTilesX       = image->width / texelsPerTileX;
        u32 numTilesY       = image->height / texelsPerTileY;
        u32 blockBorderSize = image->borderSize / blockSize;

        u32 vLen   = totalBlocksPerTileY;
        u32 rowLen = totalBlocksPerTileX * bytesPerBlock;

        for (int tileY = 0; tileY < numTilesY; tileY++)
        {
            for (int tileX = 0; tileX < numTilesX; tileX++)
            {
                Vec2u tileStart(blockBorderSize + totalBlocksPerTileX * tileX,
                                blockBorderSize + totalBlocksPerTileY * tileY);

                Tile tile;
                tile.contents   = PushArray(scratch.temp.arena, u8, vLen * rowLen);
                tile.parentFace = faceIndex;
                Utils::Copy(image->GetContentsAbsoluteIndex(tileStart),
                            image->strideWithBorder, tile.contents, rowLen, vLen, rowLen);

                tiles.push_back(tile);
            }
        }
    }

    // Output tiles to disk
    t->release();
}

} // namespace rt
