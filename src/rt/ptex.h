#ifndef PTEX_H_
#define PTEX_H_
namespace rt
{

struct PtexTexture;
class Ptex::PtexTexture;

struct PtexImage
{
    u8 *contents;
    // Does not include border
    int width;
    int height;
    int log2Width;
    int log2Height;
    int bytesPerPixel;
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

PtexImage PtexToImg(Arena *arena, Ptex::PtexTexture *ptx, int faceid, bool flip);
string Convert(Arena *arena, PtexTexture *texture, int filterWidth = 4);
} // namespace rt
#endif
