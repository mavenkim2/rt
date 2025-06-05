#include "../base.h"
#include "../thread_statistics.h"
#include "../macros.h"
#include "../template.h"
#include "../math/basemath.h"
#include "../math/simd_include.h"
#include "../math/vec2.h"
#include "../math/vec3.h"
#include "../math/vec4.h"
#include "../math/bounds.h"
#include "../math/matx.h"
#include "../math/math.h"

#include "../platform.h"
#include "../memory.h"
#include "../string.h"
#include "../containers.h"
#include "../thread_context.h"
#include "../hash.h"
#include "../random.h"
#include "../parallel.h"
#include "../graphics/ptex.h"
#include "../graphics/vulkan.h"
#include "../graphics/block_compressor.h"
#include <cstdlib>

namespace rt
{

struct Mesh
{
    Vec3f *p     = 0;
    Vec3f *n     = 0;
    Vec2f *uv    = 0;
    u32 *indices = 0;
    u32 numIndices;
    u32 numVertices;
    u32 numFaces;
};

} // namespace rt

#include "../handles.h"
#include "../scene_load.h"

namespace rt
{

void WriteQuadOBJ(Mesh &mesh, string filename)
{
    StringBuilder builder = {};
    ScratchArena scratch;
    builder.arena = scratch.temp.arena;
    for (int i = 0; i < mesh.numVertices; i++)
    {
        const Vec3f &v = mesh.p[i];
        Put(&builder, "v %f %f %f\n", v.x, v.y, v.z);
    }
    if (mesh.uv)
    {
        for (int i = 0; i < mesh.numVertices; i++)
        {
            const Vec2f &uv = mesh.uv[i];
            Put(&builder, "vt %f %f \n", uv.x, uv.y);
        }
    }
    if (mesh.n)
    {
        for (int i = 0; i < mesh.numVertices; i++)
        {
            const Vec3f &n = mesh.n[i];
            Put(&builder, "vn %f %f %f\n", n.x, n.y, n.z);
        }
    }
    for (int i = 0; i < mesh.numIndices; i += 4)
    {
        Put(&builder, "f ");
        for (int j = 0; j < 4; j++)
        {
            int idx = mesh.indices[i + j] + 1;
            Put(&builder, "%u/", idx);
            if (mesh.uv) Put(&builder, "%u/", idx);
            else Put(&builder, "/");
            Assert(mesh.n);
            Put(&builder, "%u ", idx);
        }
        Put(&builder, "\n");
    }
    WriteFileMapped(&builder, filename);
}

f32 Cross2D(Vec2f a, Vec2f b) { return a.x * b.y - a.y * b.x; }

// https://iquilezles.org/articles/ibilinear/
bool InverseBilinearUV(const Vec2f &uv00, const Vec2f &uv10, const Vec2f &uv11,
                       const Vec2f &uv01, const Vec2f &uv, Vec2f &outUv)
{
    Vec2f base = uv - uv00;
    Vec2f x    = uv00 - uv10 + uv11 - uv01;
    Vec2f y    = uv10 - uv00;
    Vec2f z    = uv01 - uv00;

    f32 a = Cross2D(x, z);
    f32 b = Cross2D(base, x) + Cross2D(y, z);
    f32 c = Cross2D(base, y);

    // Quadratic equation
    // If edges are parallel,  equation is linear
    if (Abs(a) < 0.001)
    {
        f32 t = -c / b;
        f32 s = (base.x - t * z.x) / (t * x.x + y.x);
        outUv = Vec2f(s, t);
    }
    else
    {
        f32 discriminant = b * b - 4 * a * c;
        if (discriminant < 0.f) return false;

        f32 t = 0.5 * (-b + Sqrt(discriminant)) / a;
        f32 s = (base.x - t * z.x) / (t * x.x + y.x);

        if (s < 0.f || s > 1.f || t < 0.f || t < 1.f)
        {
            f32 t = 0.5 * (-b - Sqrt(discriminant)) / a;
            f32 s = (base.x - t * z.x) / (t * x.x + y.x);
        }

        outUv = Vec2f(s, t);
    }
    return true;
}

void ConvertPtexToUVTexture(string textureFilename, string meshFilename)
{
    const VkFormat baseFormat  = VK_FORMAT_R8G8B8A8_UNORM;
    const VkFormat blockFormat = VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    const u32 bytesPerTexel    = GetFormatSize(baseFormat);
    const u32 bytesPerBlock    = GetFormatSize(blockFormat);
    const u32 blockSize        = GetBlockSize(blockFormat);
    const u32 log2BlockSize    = Log2Int(blockSize);

    const u32 gpuSubmissionWidth     = 4096;
    const int pageTexelWidth         = PAGE_WIDTH;
    const int pageBlockWidth         = pageTexelWidth >> log2BlockSize;
    const int pageByteSize           = Sqr(pageBlockWidth) * bytesPerBlock;
    const int submissionNumSqrtPages = gpuSubmissionWidth / pageTexelWidth;

    const int pageStride = pageBlockWidth * bytesPerBlock;

    ScratchArena scratch;

    int numMeshes;
    Mesh *mesh = LoadObjWithWedges(scratch.temp.arena, meshFilename, numMeshes);
    Assert(numMeshes == 1);

    Ptex::String error;
    Ptex::PtexTexture *t = cache->get((char *)textureFilename.str, error);
    Assert(t);

    int numFaces = t->numFaces();

    u32 totalTextureArea = 0;
    for (int i = 0; i < numFaces; i++)
    {
        const Ptex::FaceInfo &f = t->getFaceInfo(i);
        totalTextureArea += f.res.u() * f.res.v();
        Assert(!f.isSubface());
    }

    Assert(totalTextureArea);

    const u32 sqrtNumPages = std::sqrt((totalTextureArea - 1) >> (2 * PAGE_SHIFT)) + 1;
    const int textureWidth = sqrtNumPages * pageTexelWidth;

    int numMips = Log2Int(NextPowerOfTwo(textureWidth)) - PAGE_SHIFT;
    Assert(numMips > 0);

    BlockCompressor blockCompressor(gpuSubmissionWidth, baseFormat, blockFormat, numMips);

    StaticArray<u32> mipOffsets(scratch.temp.arena, numMips);
    StaticArray<u32> levelsSqrtNumPages(scratch.temp.arena, numMips);
    u32 mipOffset = 0;

    for (int i = 0; i < numMips; i++)
    {
        mipOffsets.Push(mipOffset);
        u32 mipSqrtNumPages = ((textureWidth >> i) + PAGE_WIDTH - 1) >> PAGE_SHIFT;
        levelsSqrtNumPages.Push(mipSqrtNumPages);
        mipOffset += Sqr(mipSqrtNumPages) * pageByteSize;
    }

    string outFilename =
        PushStr8F(scratch.temp.arena, "%S.tiles", RemoveFileExtension(textureFilename));
    StringBuilderMapped builder(outFilename);
    u64 dataOffset = AllocateSpace(&builder, mipOffset);
    u8 *dst        = (u8 *)GetMappedPtr(&builder, dataOffset);

    SubmissionInfo submissionInfos[2];

    // TODO
    // 1. when the submission's mip width is less than a page (i.e. 128x128 texels or 32x32
    // blocks), have to copy to a temp buffer and collate between submissions instead of
    // copying to disk
    // 2. how am I going to handle the case where the mip resolution is less than a page for
    // the entire texture?

    // Submission: 4096x4906
    // 1. When the mip resolution is less than 128x128 texels, stop writing pages to disk and
    // instead write blocks to a temp buffer
    // 2. When the resolution is less than 4x4 texels, write the texels to a separate gpu
    // buffer/image
    //
    // Entire texture: N x N
    // 1. collate the blocks in the temp buffer into pages
    // 2. figure out what to do with the texels

    const int avgFacesPerCell = 10;
    int numCells              = numFaces / avgFacesPerCell;

    const int sqrtNumCells = Sqrt(numCells);
    numCells               = Sqr(sqrtNumCells);

    int *offsets  = PushArrayNoZero(scratch.temp.arena, int, numCells + 1);
    int *offsets1 = &offsets[1];

    // Populate grid
    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        int idx0 = mesh->indices[faceIndex * 4 + 0];
        int idx1 = mesh->indices[faceIndex * 4 + 1];
        int idx2 = mesh->indices[faceIndex * 4 + 2];
        int idx3 = mesh->indices[faceIndex * 4 + 3];

        Vec2f uv00 = mesh->uv[idx0];
        Vec2f uv10 = mesh->uv[idx1];
        Vec2f uv11 = mesh->uv[idx2];
        Vec2f uv01 = mesh->uv[idx3];

        Vec2f bbMin = Min(Min(uv00, uv10), Min(uv11, uv01));
        Vec2f bbMax = Max(Max(uv00, uv10), Max(uv11, uv01));

        Vec2i bbiMin = Clamp(Vec2i(bbMin.x * sqrtNumCells, bbMin.y * sqrtNumCells), Vec2i(0),
                             Vec2i(sqrtNumCells - 1));
        Vec2i bbiMax = Clamp(Vec2i(bbMax.x * sqrtNumCells, bbMax.y * sqrtNumCells), Vec2i(0),
                             Vec2i(sqrtNumCells - 1));

        for (int v = bbiMin.y; v < bbiMax.y; v++)
        {
            for (int u = bbiMin.x; u < bbiMax.y; u++)
            {
                int index = v * sqrtNumCells + u;
                Assert(index < numCells);
                offsets1[index]++;
            }
        }
    }

    int sum = 0;
    for (int i = 0; i < numCells; i++)
    {
        int num     = offsets1[i];
        offsets1[i] = sum;
        sum += num;
    }

    int *faceData = PushArrayNoZero(scratch.temp.arena, int, sum);

    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {
        int idx0 = mesh->indices[faceIndex * 4 + 0];
        int idx1 = mesh->indices[faceIndex * 4 + 1];
        int idx2 = mesh->indices[faceIndex * 4 + 2];
        int idx3 = mesh->indices[faceIndex * 4 + 3];

        Vec2f uv00 = mesh->uv[idx0];
        Vec2f uv10 = mesh->uv[idx1];
        Vec2f uv11 = mesh->uv[idx2];
        Vec2f uv01 = mesh->uv[idx3];

        Vec2f bbMin = Min(Min(uv00, uv10), Min(uv11, uv01));
        Vec2f bbMax = Max(Max(uv00, uv10), Max(uv11, uv01));

        Vec2i bbiMin = Clamp(Vec2i(bbMin.x * sqrtNumCells, bbMin.y * sqrtNumCells), Vec2i(0),
                             Vec2i(sqrtNumCells - 1));
        Vec2i bbiMax = Clamp(Vec2i(bbMax.x * sqrtNumCells, bbMax.y * sqrtNumCells), Vec2i(0),
                             Vec2i(sqrtNumCells - 1));

        for (int v = bbiMin.y; v < bbiMax.y; v++)
        {
            for (int u = bbiMin.x; u < bbiMax.y; u++)
            {
                int index = v * sqrtNumCells + u;
                Assert(index < numCells);
                faceData[offsets1[index]++] = faceIndex;
            }
        }
    }

    Ptex::PtexFilter::FilterType fType = Ptex::PtexFilter::FilterType::f_catmullrom;
    Ptex::PtexFilter::Options opts(fType);
    Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(t, opts);
    int nc                   = t->numChannels();

    auto CalculateFaceUV = [&](Vec2f uv, Vec2f &st) {
        // Find the face that contains this uv
        Vec2i gridPos = Vec2i(uv) * sqrtNumCells;
        int gridIndex = gridPos.y * sqrtNumCells + gridPos.x;
        int gridCount = offsets[gridIndex + 1] - offsets[gridIndex];

        for (int faceIndexIndex = 0; faceIndexIndex < gridCount; faceIndexIndex++)
        {
            int faceIndex = faceData[faceIndexIndex + offsets[gridIndex]];

            int idx0 = mesh->indices[faceIndex * 4 + 0];
            int idx1 = mesh->indices[faceIndex * 4 + 1];
            int idx2 = mesh->indices[faceIndex * 4 + 2];
            int idx3 = mesh->indices[faceIndex * 4 + 3];

            Vec2f uv00 = mesh->uv[idx0];
            Vec2f uv10 = mesh->uv[idx1];
            Vec2f uv11 = mesh->uv[idx2];
            Vec2f uv01 = mesh->uv[idx3];

            if (InverseBilinearUV(uv00, uv10, uv11, uv01, uv, st)) return faceIndex;
        }
        return -1;
    };

    u32 sqrtNumSubmissions = (textureWidth + gpuSubmissionWidth - 1) / gpuSubmissionWidth;

    for (u32 submissionY = 0; submissionY < sqrtNumSubmissions; submissionY++)
    {
        for (u32 submissionX = 0; submissionX < sqrtNumSubmissions; submissionX++)
        {
            Vec2u start = Vec2u(submissionX, submissionY) * gpuSubmissionWidth;
            u32 vExtent =
                Min(gpuSubmissionWidth, textureWidth - submissionY * gpuSubmissionWidth);
            u32 uExtent =
                Min(gpuSubmissionWidth, textureWidth - submissionX * gpuSubmissionWidth);

            ScratchArena submissionScratch;

            u8 *temp = PushArrayNoZero(submissionScratch.temp.arena, u8,
                                       blockCompressor.submissionSize);

            for (u32 v = 0; v < vExtent; v++)
            {
                for (u32 u = 0; u < uExtent; u++)
                {
                    Vec2f uv = Vec2f(start + Vec2u(u, v)) / (f32)textureWidth;

                    Vec2f st;
                    int faceIndex = CalculateFaceUV(uv, st);
                    Assert(faceIndex != -1);

                    Vec2f uv_du = Vec2f(u + 1, v) / (f32)textureWidth;
                    Vec2f st_du;
                    int faceIndex_du = CalculateFaceUV(uv_du, st_du);
                    if (faceIndex != faceIndex_du)
                    {
                        uv_du        = Vec2f(u - 1, v) / (f32)textureWidth;
                        faceIndex_du = CalculateFaceUV(uv_du, st_du);
                        if (faceIndex != faceIndex_du) st_du = st;
                    }

                    Vec2f uv_dv = Vec2f(u, v + 1) / (f32)textureWidth;
                    Vec2f st_dv;
                    int faceIndex_dv = CalculateFaceUV(uv_dv, st_dv);
                    if (faceIndex != faceIndex_dv)
                    {
                        uv_dv        = Vec2f(u, v - 1) / (f32)textureWidth;
                        faceIndex_dv = CalculateFaceUV(uv_dv, st_dv);
                        if (faceIndex != faceIndex_dv) st_dv = st;
                    }

                    Vec2f ds = st_du - st;
                    Vec2f dt = st_dv - st;

                    f32 out[3] = {};
                    filter->eval(out, 0, nc, faceIndex, st.x, st.y, ds.x, ds.y, dt.x, dt.y);
                }
            }

            u32 width =
                Min(gpuSubmissionWidth, textureWidth - submissionX * gpuSubmissionWidth);
            u32 height =
                Min(gpuSubmissionWidth, textureWidth - submissionY * gpuSubmissionWidth);

            SubmissionInfo submissionInfo;
            submissionInfo.numSubmissionX = submissionX;
            submissionInfo.numSubmissionY = submissionY;
            submissionInfo.width          = width;
            submissionInfo.height         = height;

            submissionInfos[blockCompressor.submissionIndex] = submissionInfo;

            blockCompressor.SubmitBlockCompressedCommands(temp);
            blockCompressor.CopyBlockCompressedResultsToDisk([&](u8 *src,
                                                                 int lastSubmissionIndex) {
                u32 gpuOutputWidth = blockCompressor.gpuOutputWidth;

                const SubmissionInfo &s = submissionInfos[lastSubmissionIndex];
                u32 numSubmissionX      = s.numSubmissionX;
                u32 numSubmissionY      = s.numSubmissionY;
                u32 width               = s.width;
                u32 height              = s.height;

                // Copy tiles out to disk

                int submissionNumMips = Min(Log2Int(width), Log2Int(height)) - 2;
                for (int i = 0; i < submissionNumMips; i++)
                {
                    u32 mipPageHeight = (height + PAGE_WIDTH - 1) >> PAGE_SHIFT;
                    u32 mipPageWidth  = (width + PAGE_WIDTH - 1) >> PAGE_SHIFT;
                    u32 pageStride    = pageByteSize * levelsSqrtNumPages[i];

                    u32 srcOffset     = 0;
                    u32 dstBaseOffset = mipOffsets[i];
                    for (int pageY = 0; pageY < mipPageHeight; pageY++)
                    {
                        u32 dstOffset =
                            dstBaseOffset +
                            (numSubmissionY * submissionNumSqrtPages + pageY) * pageStride +
                            numSubmissionX * submissionNumSqrtPages * pageByteSize;
                        for (int pageX = 0; pageX < mipPageWidth; pageX++)
                        {
                            Vec2u srcIndex = Vec2u(pageX, pageY) * (u32)pageBlockWidth;
                            Utils::Copy(src + srcOffset, srcIndex, gpuOutputWidth,
                                        gpuOutputWidth, dst + dstOffset, Vec2u(0, 0),
                                        pageBlockWidth, pageBlockWidth, pageBlockWidth,
                                        pageBlockWidth, bytesPerBlock);

                            dstOffset += pageByteSize;
                        }
                    }

                    srcOffset += Sqr(gpuSubmissionWidth >> i) * bytesPerBlock;
                    height >>= 1;
                    width >>= 1;
                }
            });
        }
    }

    filter->release();
    t->release();
}

} // namespace rt

using namespace rt;

int main(int argc, char **argv)
{
    Arena *arena = ArenaAlloc();

    InitThreadContext(arena, "[Main Thread]", 1);
    OS_Init();
    u32 numProcessors     = OS_NumProcessors();
    threadLocalStatistics = PushArray(arena, ThreadStatistics, numProcessors);
    scheduler.Init(numProcessors);

    threadLocalStatistics  = PushArray(arena, ThreadStatistics, numProcessors);
    threadMemoryStatistics = PushArray(arena, ThreadMemoryStatistics, numProcessors);

    if (argc != 2)
    {
        printf("You must pass i na file or a directory. Aborting... \n");
        return 1;
    }

    StringBuilder builder = {};
    builder.arena         = arena;
    string filename       = Str8C(argv[1]);

    bool fileExists      = OS_FileExists(filename);
    bool directoryExists = OS_DirectoryExists(filename);

    std::vector<string> files;

    ScratchArena scratch;

    if (fileExists)
    {
        files.push_back(filename);
    }
    else
    {

        std::vector<string> directoryStack;
        directoryStack.push_back(filename);
        while (!directoryStack.empty())
        {
            string directory = directoryStack.back();
            directoryStack.pop_back();

            OS_FileProperties properties;
            OS_FileIter fileItr =
                OS_DirectoryIterStart(directory, OS_FileIterFlag_SkipHiddenFiles);
            for (; !OS_DirectoryIterNext(scratch.temp.arena, &fileItr, &properties);)
            {
                if (properties.isDirectory) directoryStack.push_back(properties.name);
                else
                {
                    if (GetFileExtension(properties.name) == "obj")
                        files.push_back(properties.name);
                }
            }
        }
    }

    InitializePtex(1, gigabytes(1));

    Scheduler::Counter counter = {};
    for (string file : files)
    {
        int numMeshes;
        Mesh *meshes = LoadObjWithWedges(arena, file, numMeshes);

        string f = RemoveFileExtension(file);

        ParallelFor(0, numMeshes, 1, [&](int jobID, int start, int count) {
            for (int i = start; i < start + count; i++)
            {
                ScratchArena scratch;
                string filename    = PushStr8F(scratch.temp.arena, "%S_temp_%u.obj", f, i);
                string outFilename = PushStr8F(scratch.temp.arena, "%S_%u.obj", f, i);
                WriteQuadOBJ(meshes[i], filename);

                string cmd = PushStr8F(scratch.temp.arena, "UnWrapConsole3.exe \"%S\" \"%S\"",
                                       filename, outFilename);
                system((const char *)cmd.str);
            }
        });
    }
}
