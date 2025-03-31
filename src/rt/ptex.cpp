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
    Assert(p.x < width + borderSize && p.y < height + borderSize);
    return contents + strideWithBorder * p.y + p.x;
}

// NOTE: ignores border
u8 *PtexImage::GetContentsRelativeIndex(const Vec2u &p)
{
    // Assert(p.x < width && p.y < height);
    Assert(p.x < width + borderSize && p.y < height + borderSize);
    return GetContentsAbsoluteIndex(ConvertRelativeToAbsolute(p));
}

// NOTE: start is an absolute index
void PtexImage::WriteRotatedBorder(PtexImage &other, Vec2u dstStart, Vec2u offset, int rotate,
                                   int srcVLen, int srcRowLen, int dstVLen, int dstRowLen,
                                   Vec2i scale)
{
    int uStart = dstStart.x;
    int vStart = dstStart.y;

    int uEnd = dstStart.x + rowLen - 1;
    int vEnd = dstStart.y + vLen - 1;

    Assert(bytesPerPixel == other.bytesPerPixel);

    if (rotate == 0)
    {
        Utils::Copy(other.GetContentsRelativeIndex(dstStart), other.strideWithBorder,
                    GetContentsAbsoluteIndex(dstStart + offset), strideWithBorder, vLen,
                    rowLen);
        return;
    }

    for (int v = vStart; v < vStart + vLen; v++)
    {
        for (int u = uStart; u < uStart + rowLen; u++)
        {
            Vec2u srcAddress;
            switch (rotate)
            {
                case 1:
                {
                    srcAddress.x = height - 1 - v;
                    srcAddress.y = u;
                }
                break;
                case 2:
                {
                    srcAddress.x = width - 1 - u;
                    srcAddress.y = height - 1 - v;
                }
                break;
                case 3:
                {
                    srcAddress.x = v;
                    srcAddress.y = width - 1 - u;
                }
                break;
            }
            Vec2u dstPos(u, v);
            Vec2u srcPos(u >> scale.x, v >> scale.y);
            u8 *dst = GetContentsAbsoluteIndex(dstAddress + offset);
            u8 z[3] = {};
            Assert(bytesPerPixel == 3);
            Assert(memcmp(dst, z, 3) == 0);
            MemoryCopy(dst, other.GetContentsRelativeIndex(srcPos), bytesPerPixel);
        }
    }
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
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        const Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
        Assert(!f.isSubface());
        for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
        {
            int aeid         = f.adjedge(edgeIndex);
            int neighborFace = f.adjface(edgeIndex);
            int rot          = edgeIndex - aeid + 2;

            if (rot != 0)
            {
                int stop = 5;
            }

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
                if (aeid == e_bottom)
                {
                    start  = Vec2u(0, 0);
                    scale  = Vec2i(s, 0);
                    vRes   = borderSize;
                    rowLen = currentFaceImg.width;
                }
                else if (aeid == e_right)
                {
                    start  = Vec2u(currentFaceImg.width - borderSize, 0);
                    scale  = Vec2i(0, s);
                    vRes   = currentFaceImg.height;
                    rowLen = borderSize;
                }
                else if (aeid == e_top)
                {
                    start  = Vec2u(0, currentFaceImg.height - borderSize);
                    scale  = Vec2i(s, 0);
                    vRes   = borderSize;
                    rowLen = currentFaceImg.width;
                }
                else if (aeid == e_left)
                {
                    start  = Vec2u(0, 0);
                    scale  = Vec2i(0, s);
                    vRes   = currentFaceImg.height;
                    rowLen = borderSize;
                }

                currentFaceImg.WriteRotatedBorder(neighborFaceImg, start, offset, rot, vRes,
                                                  rowLen, scale);

                dstCompareDim++;
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
