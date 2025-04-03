#include "../rt.h"
#include "../math/math_include.h"
#include "../memory.h"
#include "../ptex.h"

namespace rt
{

enum class TileType
{
    Corner,
    Edge,
    Edge,
    Center,
};

enum class EdgeId
{
    Bottom, ///< Bottom edge, from UV (0,0) to (1,0)
    Right,  ///< Right edge, from UV (1,0) to (1,1)
    Top,    ///< Top edge, from UV (1,1) to (0,1)
    Left,   ///< Left edge, from UV (0,1) to (0,0)
    Max,
};

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

struct Tile
{
    u8 *contents;
    TileType type;
    EdgeId edgeId;
    u32 parentFace;
};

} // namespace rt

using namespace rt;

void main(int argc, char *argv[])
{
    // Divide texture into 64x64 tiles
    // Get every mip level
    Arena *arena = ArenaAlloc();

    // Highest mip level
    Ptex::String error;
    Ptex::PtexTexture *t     = cache->get((char *)texture->filename.str, error);
    Ptex::PtexReader *reader = static_cast<Ptex::PtexReader *>(t);
    int numFaces             = reader->numFaces();
    int borderSize           = filterWidth - 1;

    PtexImage *images = PushArrayNoZero(arena, PtexImage *, numFaces);
    int *numLevels    = PushArrayNoZero(temp.arena, int, numFaces);

    int log2FilterWidth = Log2Int(filterWidth);

    // Fill border with adjacent texels based on filter width
    Assert(filterWidth > 0);

    int filterWidth     = 4;
    PtexImage *images   = PushArrayNoZero(arena, PtexImage *, numFaces);
    int log2FilterWidth = Log2Int(filterWidth);

    u32 minDim = pos_inf;

    // Get all the images
    for (int i = 0; i < numFaces; i++)
    {
        PtexImage img = PtexToImg(arena, t, i, borderSize, false);
        images[i]     = img;
        minDim        = Min(img.width, img.height);
    }

    Assert(IsPow2(minDim));

    u32 tileSize = minDim / 2;

    // Essentially, we are packing 128x128 tiles such that edges and shared corners are within
    // the same tile. For example:
    //
    // 1 | 2
    // -----
    // 3 | 4
    // For a "normal" vertex with valence 4, corner texels will be within the same tile.

    // how do I handle extraordinary vertices?

    std::vector<Tile> tiles;

    const u32 totalTileX    = 128;
    const u32 totalTileY    = 128;
    const u32 bytesPerPixel = 3;
    const u32 borderSize    = 4;

    const u32 texelsPerTileX      = 128;
    const u32 texelsPerTileY      = 128;
    const u32 totalTexelsPerTileX = 128 + 2 * borderSize;
    const u32 totalTexelsPerTileY = 128 + 2 * borderSize;

    const u32 tileStride = texelsPerTileX * bytesPerPixel;

    // Divide textures into 128x128 tiles
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        const Ptex::FaceInfo &f = reader->getFaceInfo(faceIndex);
        Assert(!f.isSubface());

        // Add border
        for (int edgeIndex = e_bottom; edgeIndex < e_max; edgeIndex++)
        {
            int aeid = f.adjedge(edgeIndex);
            Assert(IsPow2(dim));

            int aeid         = f.adjedge(edgeIndex);
            int neighborFace = f.adjface(edgeIndex);
            int rot          = (edgeIndex - aeid + 2) & 3;

            Vec2u dstBaseSize(images[faceIndex].log2Width, images[faceIndex].log2Height);
            Vec2u srcBaseSize(images[neighborFace].log2Width, images[neighborFace].log2Height);

            int dstCompareDim = (edgeIndex & 1) ? dstBaseSize.y : dstBaseSize.x;
            int srcCompareDim = (aeid & 1) ? srcBaseSize.y : srcBaseSize.x;

            int srcBaseDepth = srcCompareDim - dstCompareDim;
            // int srcDepthIndex = Clamp(srcBaseDepth, 0, maxDepth - 1);
            int srcDepthIndex = Clamp(srcBaseDepth, 0, numLevels[neighborFace] - 1);

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

            currentFaceImg.WriteRotatedBorder(neighborFaceImg, srcStart, start, edgeIndex, rot,
                                              srcVRes, srcRowLen, dstVRes, dstRowLen, scale);

            dstCompareDim--;
        }
    }

    // Pad tiles with border texels
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        PtexImage *image = &images[faceIndex];

        Assert(image->width >= totalTileX && IsPow2(image->width));
        Assert(image->height >= totalTileY && IsPow2(image->height));
        u32 numTilesX = image->width / totalTileX;
        u32 numTilesY = image->height / totalTileY;

        u32 vLen   = totalTexelsPerTileY;
        u32 rowLen = totalTexelsPerTileX * bytesPerPixel;

        for (int tileY = 0; tileY < numTilesY; tileY++)
        {
            for (int tileX = 0; tileX < numTilesX; tileX++)
            {
                Vec2u tileStart(borderSize + texelsPerTileX * tileX,
                                borderSize + texelsPerTileY * tileY);

                Tile tile;
                tile.contents   = PushArray(arena, u8, vLen * rowLen);
                tile.parentFace = faceIndex;
                Utils::Copy(image->GetContentsAbsoluteIndex(tileStart),
                            image->strideWithBorder, tile.contents, rowLen, vLen, rowLen);

                tiles.push_back(tile);
            }
        }
    }

    outNumFaces = numFaces;
    ScratchEnd(temp);
    t->release();
    return images;
}
