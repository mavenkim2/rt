#include "base.h"
#include "memory.h"
#include "string.h"
#include "scene.h"
#include "integrate.h"
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

} // namespace Utils

namespace rt
{

Vec2u PtexImage::ConvertRelativeToAbsolute(const Vec2u &p)
{
    return Vec2u(p.x + borderSize, p.y + borderSize);
}

// NOTE: absolute index
u8 *PtexImage::GetContentsAbsoluteIndex(const Vec2u &p)
{
    Assert(p.x < width + 2 * borderSize && p.y < height + 2 * borderSize);
    return contents + strideWithBorder * p.y + p.x;
}

// NOTE: ignores border
u8 *PtexImage::GetContentsRelativeIndex(const Vec2u &p)
{
    Assert(p.x < width && p.y < height);
    return GetContentsAbsoluteIndex(ConvertRelativeToAbsolute(p));
}

// NOTE: start is an absolute index
void PtexImage::WriteRotatedBorder(PtexImage &other, Vec2u srcStart, Vec2u dstStart,
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
    if (rotate != 0)
    {
        int pixelsToWrite = Max(1 << scale.x, 1 << scale.y);
        int duplicateStep = (edgeIndex & 1) ? dstRowLen * bytesPerPixel : 1;
        // duplicateStep *= (((rotate + (edgeIndex & 1)) & 3) >= 2) ? 1 : -1;

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
        src       = tempBuffer;
        srcStride = dstRowLen * bytesPerPixel; // other.strideNoBorder;
    }

    // TODO: need to handle case where there is no rotation, but the src dimension
    // is a fraction of the dst dimension
    Utils::Copy(src, srcStride, GetContentsAbsoluteIndex(dstStart), strideWithBorder, dstVLen,
                dstRowLen * bytesPerPixel);
}

PtexImage PtexToImg(Arena *arena, Ptex::PtexTexture *ptx, int faceID, int borderSize,
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

    PtexImage result;
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

string Convert(Arena *arena, PtexTexture *texture, int filterWidth)
{
    // Get every mip level

    // Highest mip level
    Ptex::String error;
    Ptex::PtexTexture *t     = cache->get((char *)texture->filename.str, error);
    Ptex::PtexReader *reader = static_cast<Ptex::PtexReader *>(t);
    int numFaces             = reader->numFaces();
    int borderSize           = filterWidth - 1;

    PtexImage **images = PushArrayNoZero(arena, PtexImage *, numFaces);
    int *numLevels     = PushArrayNoZero(arena, int, numFaces);
    const int maxDepth = 10;

    int log2FilterWidth = Log2Int(filterWidth);

    for (int i = 0; i < numFaces; i++)
    {
        PtexImage img = PtexToImg(arena, t, i, borderSize, false);
        int levels    = Max(Max(img.log2Width, img.log2Height) - log2FilterWidth, 1);
        images[i]     = PushArrayNoZero(arena, PtexImage, levels);
        numLevels[i]  = levels;
        images[i][0]  = img;

        // Generate mip maps
        Assert(IsPow2(img.width) && IsPow2(img.height));

        int width = img.width;
        width >>= 1;
        int height = img.height >> 1;

        PtexImage inPtexImage = img;
        int depth             = 1;

        Vec2u scale = 2u;

        while (depth < levels)
        {
            PtexImage outPtexImage;
            outPtexImage.width            = width;
            outPtexImage.height           = height;
            outPtexImage.log2Width        = Log2Int(width);
            outPtexImage.log2Height       = Log2Int(height);
            outPtexImage.bytesPerPixel    = img.bytesPerPixel;
            outPtexImage.strideNoBorder   = width * img.bytesPerPixel;
            outPtexImage.borderSize       = borderSize;
            outPtexImage.strideWithBorder = (width + 2 * borderSize) * img.bytesPerPixel;
            outPtexImage.contents         = PushArrayNoZero(
                arena, u8, outPtexImage.strideWithBorder * (height + 2 * borderSize));
            for (u32 v = 0; v < height; v++)
            {
                for (u32 u = 0; u < width; u++)
                {
                    Vec2u xy = scale * Vec2u(u, v);
                    Vec2u zw = xy + scale - 1u;

                    Vec3f topleft = GammaToLinear(inPtexImage.GetContentsRelativeIndex(xy));
                    Vec3f topright =
                        GammaToLinear(inPtexImage.GetContentsRelativeIndex({zw.x, xy.y}));
                    Vec3f bottomleft =
                        GammaToLinear(inPtexImage.GetContentsRelativeIndex({xy.x, zw.y}));
                    Vec3f bottomright =
                        GammaToLinear(inPtexImage.GetContentsRelativeIndex({zw.x, zw.y}));

                    Vec3f avg = (topleft + topright + bottomleft + bottomright) / .25f;

                    LinearToGamma(avg, outPtexImage.GetContentsRelativeIndex({u, v}));
                }
            }

            images[i][depth] = outPtexImage;
            inPtexImage      = outPtexImage;

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

            Vec2u offset;
            if (edgeIndex == e_bottom) offset = Vec2u(borderSize, 0);
            else if (edgeIndex == e_right) offset = Vec2u(2 * borderSize, borderSize);
            else if (edgeIndex == e_top) offset = Vec2u(borderSize, 2 * borderSize);
            else if (edgeIndex == e_left) offset = Vec2u(0, borderSize);

            for (int depth = 0; depth < numLevels[faceIndex]; depth++)
            {
                int srcBaseDepth  = srcCompareDim - dstCompareDim;
                int srcDepthIndex = Clamp(srcBaseDepth, 0, maxDepth - 1);

                PtexImage &currentFaceImg = images[faceIndex][depth];
                PtexImage neighborFaceImg = images[neighborFace][srcDepthIndex];

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

    size_t size = numFaces * reader->_pixelsize;
    string result;
    result.str  = PushArrayNoZero(arena, u8, size);
    result.size = size;
    MemoryCopy(result.str, reader->getConstData(), numFaces * reader->_pixelsize);
    t->release();
    return result;
}
} // namespace rt
