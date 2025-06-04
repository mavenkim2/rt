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
