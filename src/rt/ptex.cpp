#include "base.h"
#include "memory.h"
#include "string.h"
#include "scene.h"
#include "integrate.h"
#include <Ptexture.h>
#include <PtexReader.h>
namespace rt
{

// void PtexToImg(Arena *arena, PtexTexture *ptx, Image image, int faceid, bool flip)
// {
//     Assert(faceID >= 0 && faceID < ptx->numFaces());
//
//     image.dt    = ptx->dataType();
//     image.nchan = ptx->numChannels();
//     image.achan = ptx->alphaChannel();
//
//     Ptex::FaceInfo fi = ptx->getFaceInfo(faceid);
//     img.w             = fi.res.u();
//     img.h             = fi.res.v();
//     int rowlen        = img.w * img.nchan * Ptex::DataSize(img.dt);
//     int size          = rowlen * img.h;
//     img.data          = PushArrayNoZero(arena, u8, size);
//     int stride        = rowlen;
//     // if (flip)
//     // {
//     //     data += rowlen * (img.h - 1);
//     //     stride = -rowlen;
//     // }
//     ptx->getData(faceid, (char *)img.data, stride);
//     return 1;
// }

string Convert(Arena *arena, PtexTexture *texture)
{
    // Get every mip level

    // Highest mip level
    Ptex::String error;
    Ptex::PtexTexture *t     = cache->get((char *)texture->filename.str, error);
    Ptex::PtexReader *reader = static_cast<Ptex::PtexReader *>(t);
    int numFaces             = reader->numFaces();

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
