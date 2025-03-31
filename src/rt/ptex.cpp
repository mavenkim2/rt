#include "base.h"
#include "memory.h"
#include "string.h"
#include "scene.h"
#include "integrate.h"
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
void PtexImage::WriteRotatedBorder(const PtexImage &other, Vec2u start, Vec2u offset, int vLen,
                                   int rowLen, Vec2i scale)
{
    int u = start.x;
    int v = start.y;

    Assert(bytesPerPixel == other.bytesPerPixel);

    if (rotate == 0)
    {
        Utils::Copy(other.contents, other.strideWithBorder, contents,
                    contents.strideWithBorder, vLen, rowLen);
        return;
    }

    for (int v = vStart; v < vStart + vLen; v++)
    {
        for (int u = uStart; u < uStart + rowLen; u++)
        {
            Vec2u dstAddress(u, v);
            switch (rotate)
            {
                case 1:
                {
                    dstAddress.x = width - 1 - dstAddress.x;
                    Swap(dstAddress.x, dstAddress.y);
                }
                break;
                case 2:
                {
                    dstAddress.x = width - 1 - dstAddress.x;
                    dstAddress.y = height - 1 - dstAddress.y;
                }
                break;
                case 3:
                {
                    dstAddress.y = height - 1 - dstAddress.y;
                    Swap(dstAddress.x, dstAddress.y);
                }
                break;
            }
            Assert(newU >= 0 && newV >= 0);
            Vec2u srcPos(u >> scale.x, v >> scale.y);
            u8 *dst    = GetContentsAbsoluteIndex(dstAddress + offset);
            u8 zero[3] = {};
            Assert(bytesPerPixel == 3);
            Assert(memcmp(dst, zero, 3) == 0);
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
    int size          = rowlen * (v + 2 * borderSize);
    u8 *data          = PushArrayNoZero(arena, u8, size);
    int rowlen        = (u * bytesPerPixel) * v;
    // if (flip)
    // {
    //     data += rowlen * (img.h - 1);
    //     stride = -rowlen;
    // }
    ptx->getData(faceID, (char *)data, stride);

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
    const int maxDepth = 10;

    int log2FilterWidth = Log2Int(filterWidth);

    for (int i = 0; i < numFaces; i++)
    {
        PtexImage img = PtexToImg(arena, t, i, borderSize, false);
        images[i]     = PushArrayNoZero(arena, PtexImage, maxDepth);
        images[i][0]  = img;

        // Generate mip maps
        Assert(img.width == img.height);
        Assert(IsPow2(img.width));

        int width     = img.width;
        int srcStride = img.bytesPerPixel * width;
        width >>= 1;
        int height    = img.height >> 1;
        int dstStride = img.bytesPerPixel * width;

        PtexImage inPtexImage = img;
        int depth             = 1;
        while (depth < maxDepth)
        {
            PtexImage outPtexImage;
            outPtexImage.width         = width;
            outPtexImage.height        = height;
            outPtexImage.bytesPerPixel = img.bytesPerPixel;
            outPtexImage.contents      = PushArrayNoZero(arena, u8, dstStride * height);
            for (int v = 0; v < height; v++)
            {
                for (int u = 0; u < width; u++)
                {
                    Vec2u xy = 2u * Vec2u(u, v);
                    Vec2u zw = xy + 1u;

                    Vec3f topleft =
                        GammaToLinear(inPtexImage.contents + xy.y * srcStride + xy.x);
                    Vec3f topright =
                        GammaToLinear(inPtexImage.contents + xy.y * srcStride + zw.x);
                    Vec3f bottomleft =
                        GammaToLinear(inPtexImage.contents + zw.y * srcStride + xy.x);
                    Vec3f bottomright =
                        GammaToLinear(inPtexImage.contents + zw.y * srcStride + zw.x);

                    Vec3f avg = (topleft + topright + bottomleft + bottomright) * .25f;

                    LinearToGamma(avg, outPtexImage.contents + v * dstStride + u);
                }
            }

            if (height == 1 && width == 1) break;

            images[i][depth] = outPtexImage;
            inPtexImage      = outPtexImage;

            srcStride = dstStride;
            width >>= 1;
            height >>= 1;
            dstStride = img.bytesPerPixel * width;
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
        Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
        Assert(!f.isSubface());
        for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
        {
            int aeid         = f.adjedge(edgeIndex);
            int neighborFace = f.adjface(edgeIndex);
            int rot          = aeid - edgeIndex + 2;

            if (rot != 0)
            {
                int stop = 5;
            }

            Vec2u dstBaseSize(images[faceIndex][0].log2Width, images[faceIndex][0].log2Height);
            Vec2u srcBaseSize(images[neighborFace][0].log2Width,
                              images[neighborFace][0].log2Height);

            int dstCompareDim = (edgeIndex & 1) ? dstBaseSize.y : dstBaseSize.x;
            int srcCompareDim;
            if (rot & 1)
            {
                if (edgeIndex & 1) srcCompareDim = srcBaseSize.x;
                else srcCompareDim = srcBaseSize.y;
            }
            else
            {
                if (edgeIndex & 1) srcCompareDim = srcBaseSize.y;
                else srcCompareDim = srcBaseSize.x;
            }

            for (int depth = 0; depth < maxDepth; depth++)
            {
                int srcBaseDepth  = srcCompareDim - dstCompareDim;
                int srcDepthIndex = Clamp(srcBaseDepth, 0, maxDepth - 1);

                PtexImage &currentFaceImg = images[faceIndex][depth];
                PtexImage neighborFaceImg = images[neighborFace][srcDepthIndex];

                Vec2u start;
                Vec2u offset;
                Vec2u scale;
                int vRes;
                int rowLen;
                int scale = Max(-srcBaseDepth, 0);
                if (aeid == e_bottom)
                {
                    start  = Vec2u(0, 0);
                    offset = Vec2u(borderSize, 0);
                    scale  = Vec2i(scale, 0);
                    vRes   = borderSize;
                    rowLen = neighborFaceImg.width;
                }
                else if (aeid == e_right)
                {
                    start  = Vec2u(neighborFaceImg.width - borderSize, 0);
                    offset = Vec2u(2 * borderSize, borderSize);
                    scale  = Vec2i(0, scale);
                    vRes   = neighborFaceImg.height;
                    rowLen = borderSize;
                }
                else if (aeid == e_top)
                {
                    start  = Vec2u(0, neighborFaceImg.height - borderSize);
                    offset = Vec2u(borderSize, 2 * borderSize);
                    scale  = Vec2i(scale, 0);
                    vRes   = borderSize;
                    rowLen = neighborFaceImg.width;
                }
                else if (aeid == e_left)
                {
                    start  = Vec2u(0, 0);
                    offset = Vec2u(0, borderSize);
                    scale  = Vec2i(0, scale);
                    vRes   = neighborFaceImg.height;
                    rowLen = borderSize;
                }

                currentFaceImg.WriteRotatedBorder(neighborFaceImg, start, offset, vRes, rowLen,
                                                  scale);

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

    // for (int levelIndex = 0; levelIndex < texture->_levels.size(); levelIndex++)
    // {
    //     Ptex::PtexReader::Level *level = texture->getLevel(levelIndex);
    //     for (int faceIndex = 0; faceIndex < texture->numFaces; faceIndex++)
    //     {
    //         Ptex::PtexReadere::FaceData * face = getFace(levelIndex, level, faceIndex,
    //     }
    // }
}
} // namespace rt
