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

void ConvertPtexToUVTexture(string textureFilename, string meshFilename)
{
    ScratchArena scratch;
    int numMeshes;
    Mesh *mesh = LoadObjWithWedges(scratch.temp.arena, meshFilename, numMeshes);
    Assert(numMeshes == 1);

    Ptex::String error;
    Ptex::PtexTexture *t = cache->get((char *)textureFilename.str, error);
    Assert(t);

    int numFaces = t->numFaces();
    StaticArray<Ptex::FaceInfo> ptexFaceInfos(scratch.temp.arena, numFaces);
    for (int i = 0; i < numFaces; i++)
    {
        const Ptex::FaceInfo &f = t->getFaceInfo(i);
        Assert(!f.isSubface());
        ptexFaceInfos.push_back(f);
    }

    // Create uv grid
    [128][128];

    // Populate grid

    // uv = uv0 + duv10 * bary.x + duv20 * bary.y
    // uv = (1, 0) * bary.x + (1, 1) * bary.y

    // Square texture dimension
    int textureWidth;

    for (int v = 0; v < textureWidth; v++)
    {
        for (int u = 0; u < textureWidth; u++)
        {
            Vec2f uv = Vec2f(u, v) / textureWidth;

            // Find the face that contains this uv
            int faceIndex;

            int idx0 = mesh->indices[faceIndex * 4 + 0];
            int idx1 = mesh->indices[faceIndex * 4 + 1];
            int idx2 = mesh->indices[faceIndex * 4 + 2];
            int idx3 = mesh->indices[faceIndex * 4 + 3];

            Vec2f uv00 = mesh->uv[idx0];
            Vec2f uv10 = mesh->uv[idx1];
            Vec2f uv11 = mesh->uv[idx2];
            Vec2f uv01 = mesh->uv[idx3];

            // Inverse bilinear
            // https://iquilezles.org/articles/ibilinear/
            Vec2f base = uv - uv00;
            Vec2f x    = uv00 - uv10 + uv11 - uv01;
            Vec2f y    = uv10 - uv00;
            Vec2f z    = uv01 - uv00;

            f32 a = Cross2D(x, z);
            f32 b = Cross2D(base, x) + Cross2D(y, z);
            f32 c = Cross2D(base, y);

            // Quadratic equation
            // If edges are parallel,  equation is linear
            Vec2f faceUv;
            if (Abs(a) < 0.001)
            {
                f32 t  = -c / b;
                f32 s  = (base.x - t * z.x) / (t * x.x + y.x);
                faceUv = Vec2f(s, t);
            }
            else
            {
                f32 discriminant = b * b - 4 * a * c;
                Assert(discriminant >= 0.f);
                f32 t = 0.5 * (-b + Sqrt(discriminant)) / a;
                f32 s = (base.x - t * z.x) / (t * x.x + y.x);

                if (s < 0.f || s > 1.f || t < 0.f || t < 1.f)
                {
                    f32 t = 0.5 * (-b - Sqrt(discriminant)) / a;
                    f32 s = (base.x - t * z.x) / (t * x.x + y.x);
                }

                faceUv = Vec2f(s, t);
            }
        }
    }

    for (int faceIndex = 0; faceIndex < numFaces; faceIndex++)
    {

        Vec2i texel00 = Vec2i(uv00 * (f32)textureWidth);
        Vec2i texel10 = Vec2i(uv10 * (f32)textureWidth);
        Vec2i texel11 = Vec2i(uv11 * (f32)textureWidth);
        Vec2i texel01 = Vec2i(uv01 * (f32)textureWidth);

        // Get the rectangle in the texture represented by this face
        const auto &faceInfo = ptexFaceInfos[faceIndex];
    }
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
