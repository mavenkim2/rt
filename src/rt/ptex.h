#ifndef PTEX_H_
#define PTEX_H_
#include "rt.h"

namespace Utils
{
void Copy(const void *src, int sstride, void *dst, int dstride, int vres, int rowlen);
}

namespace rt
{

struct PtexTexture;
class Ptex::PtexTexture;

struct PtexImage : Image
{
    // Does not include border
    int log2Width;
    int log2Height;
    int strideNoBorder;

    // Includes border
    int borderSize;
    int strideWithBorder;

    Vec2u ConvertRelativeToAbsolute(const Vec2u &p);
    u8 *GetContentsAbsoluteIndex(const Vec2u &p);
    u8 *GetContentsRelativeIndex(const Vec2u &p);

    void WriteRotatedBorder(PtexImage &other, Vec2u srcStart, Vec2u dstStart, int edgeIndex,
                            int rotate, int srcVLen, int srcRowLen, int dstVLen, int dstRowLen,
                            Vec2u scale);
};

PtexImage **Convert(Arena *arena, PtexTexture *texture, int filterWidth, int &numFaces);
string Convert(Arena *arena, PtexTexture *texture, int filterWidth = 4);
} // namespace rt
#endif
