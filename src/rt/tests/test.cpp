#include "../base.h"
#include "../math/matx.h"
#include "../scene.h"
#include "../sampler.h"
#include "../lights.h"
#include "../spectrum.h"
#include "../template.h"
#include "../integrate.h"
namespace rt
{
TriangleMesh *GenerateMesh(Arena *arena, u32 count, f32 min = -100.f, f32 max = 100.f)
{
    arena->align       = 64;
    TriangleMesh *mesh = PushStruct(arena, TriangleMesh);
    mesh->p            = PushArray(arena, Vec3f, count);
    mesh->numVertices  = count;
    mesh->indices      = PushArray(arena, u32, count);
    mesh->numIndices   = count;

    for (u32 i = 0; i < count; i++)
    {
        mesh->indices[i] = RandomInt(0, count);
    }

    for (u32 i = 0; i < count / 3; i++)
    {
        mesh->p[i * 3]     = RandomVec3(min, max);
        mesh->p[i * 3 + 1] = RandomVec3(min, max);
        mesh->p[i * 3 + 2] = RandomVec3(min, max);
    }
    return mesh;
}

PrimRefCompressed *GenerateAOSData(Arena *arena, TriangleMesh *mesh, u32 numFaces,
                                   Bounds &geomBounds, Bounds &centBounds)
{
    arena->align = 64;
    PrimRefCompressed *refs =
        PushArray(arena, PrimRefCompressed, u32(numFaces * GROW_AMOUNT) + 1);
    for (u32 i = 0; i < numFaces; i++)
    {
        u32 i0 = mesh->indices[i * 3];
        u32 i1 = mesh->indices[i * 3 + 1];
        u32 i2 = mesh->indices[i * 3 + 2];

        PrimRefCompressed *prim = &refs[i];
        Vec3f v0                = mesh->p[i0];
        Vec3f v1                = mesh->p[i1];
        Vec3f v2                = mesh->p[i2];

        Vec3f min = Min(Min(v0, v1), v2);
        Vec3f max = Max(Max(v0, v1), v2);

        Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
        Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);
        Lane4F32::StoreU(prim->min, -mins);
        prim->maxX = max.x;
        prim->maxY = max.y;
        prim->maxZ = max.z;
        // prim->m256 = Lane8F32(-mins, maxs);

        prim->primID = i;

        geomBounds.Extend(mins, maxs);
        centBounds.Extend((maxs + mins)); //* 0.5f);
    }
    return refs;
}

PrimRefCompressed *GenerateQuadData(Arena *arena, QuadMesh *mesh, u32 numFaces,
                                    Bounds &geomBounds, Bounds &centBounds)
{
    arena->align = 64;
    PrimRefCompressed *refs =
        PushArray(arena, PrimRefCompressed, u32(numFaces * GROW_AMOUNT) + 1);
    for (u32 i = 0; i < numFaces; i++)
    {
        PrimRefCompressed *prim = &refs[i];
        Vec3f v0                = mesh->p[4 * i + 0];
        Vec3f v1                = mesh->p[4 * i + 1];
        Vec3f v2                = mesh->p[4 * i + 2];
        Vec3f v3                = mesh->p[4 * i + 3];

        Vec3f min = Min(Min(v0, v1), Min(v2, v3));
        Vec3f max = Max(Max(v0, v1), Min(v2, v3));

        Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
        Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);
        Lane4F32::StoreU(prim->min, -mins);
        prim->maxX = max.x;
        prim->maxY = max.y;
        prim->maxZ = max.z;

        prim->primID = i;

        geomBounds.Extend(mins, maxs);
        centBounds.Extend((maxs + mins)); //* 0.5f);
    }
    return refs;
}

void AOSSBVHBuilderTest(Arena *arena, TriangleMesh *mesh)
{
    arena->align = 64;

    const u32 numFaces = mesh->numIndices / 3;
    u32 numProcessors  = OS_NumProcessors();
    Arena **arenas     = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16); // ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
    }

    PerformanceCounter counter = OS_StartCounter();
    Bounds geomBounds;
    Bounds centBounds;
    PrimRefCompressed *refs = GenerateAOSData(arena, mesh, numFaces, geomBounds, centBounds);

    BuildSettings settings;
    settings.intCost = 0.3f;

    RecordAOSSplits record;
    record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    record.SetRange(0, numFaces, u32(numFaces * GROW_AMOUNT));

    BVHNodeN bvh = BuildQuantizedTriSBVH(settings, arenas, mesh, refs, record);
    f32 time     = OS_GetMilliseconds(counter);
    printf("num faces: %u\n", numFaces);
    printf("Build time: %fms\n", time);

    f64 totalMiscTime     = 0;
    u64 numNodes          = 0;
    u64 totalNodeMemory   = 0;
    u64 totalRecordMemory = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        totalNodeMemory += threadMemoryStatistics[i].totalBVHMemory;
        totalRecordMemory += threadMemoryStatistics[i].totalRecordMemory;
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
        numNodes += threadLocalStatistics[i].misc;
    }
    printf("total time: %fms \n", totalMiscTime);
    printf("num nodes: %llu\n", numNodes);               // num nodes: %llu\n", numNodes);
    printf("node kb: %llu\n", totalNodeMemory / 1024);   // num nodes: %llu\n", numNodes);
    printf("record kb: %llu", totalRecordMemory / 1024); // num nodes: %llu\n", numNodes);
}

void QuadSBVHBuilderTest(Arena *arena, QuadMesh *mesh)
{
    arena->align = 64;

    const u32 numFaces = mesh->numVertices / 4;
    u32 numProcessors  = OS_NumProcessors();
    Arena **arenas     = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16); // ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
    }

    PerformanceCounter counter = OS_StartCounter();
    Bounds geomBounds;
    Bounds centBounds;
    PrimRefCompressed *refs = GenerateQuadData(arena, mesh, numFaces, geomBounds, centBounds);

    BuildSettings settings;
    settings.intCost = 0.3f;

    RecordAOSSplits record;
    record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    record.SetRange(0, numFaces, u32(numFaces * GROW_AMOUNT));

    BVHNodeN bvh = BuildQuantizedQuadSBVH(settings, arenas, mesh, refs, record);
    f32 time     = OS_GetMilliseconds(counter);
    printf("num faces: %u\n", numFaces);
    printf("Build time: %fms\n", time);

    f64 totalMiscTime     = 0;
    u64 numNodes          = 0;
    u64 totalNodeMemory   = 0;
    u64 totalRecordMemory = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        totalNodeMemory += threadMemoryStatistics[i].totalBVHMemory;
        totalRecordMemory += threadMemoryStatistics[i].totalRecordMemory;
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
        numNodes += threadLocalStatistics[i].misc;
    }
    printf("total time: %fms \n", totalMiscTime);
    printf("num nodes: %llu\n", numNodes);               // num nodes: %llu\n", numNodes);
    printf("node kb: %llu\n", totalNodeMemory / 1024);   // num nodes: %llu\n", numNodes);
    printf("record kb: %llu", totalRecordMemory / 1024); // num nodes: %llu\n", numNodes);
}

void PartialRebraidBuilderTest(Arena *arena)
{
    TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16); // ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
    }

    string meshDirectory = "data/island/pbrt-v4/meshes/";
    string instanceFile  = "data/island/pbrt-v4/instances.data";
    string transformFile = "data/island/pbrt-v4/transforms.data";
    PerformanceCounter counter;
    counter = OS_StartCounter();
    u64 numScenes;
    Scene2 *scenes =
        InitializeScene(arenas, meshDirectory, instanceFile, transformFile, numScenes);
    printf("scene initialization + blas build time: %fms\n", OS_GetMilliseconds(counter));
    Print("scene initialization + blas build time: %fms\n", OS_GetMilliseconds(counter));

    RecordAOSSplits record;
    counter         = OS_StartCounter();
    BRef *buildRefs = GenerateBuildRefs(scenes, 0, numScenes, temp.arena, record);
    printf("time to generate build refs: %fms\n", OS_GetMilliseconds(counter));

    BuildSettings settings;
    settings.intCost = 0.3f;

    counter       = OS_StartCounter();
    BVHNodeN node = BuildTLASQuantized(settings, arenas, &scenes[0], buildRefs, record);
    printf("time to generate tlas: %fms\n", OS_GetMilliseconds(counter));
    Print("time to generate tlas: %fms\n", OS_GetMilliseconds(counter));

    f64 totalMiscTime     = 0;
    u64 numNodes          = 0;
    u64 totalNodeMemory   = 0;
    u64 totalRecordMemory = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        totalNodeMemory += threadMemoryStatistics[i].totalBVHMemory;
        totalRecordMemory += threadMemoryStatistics[i].totalRecordMemory;
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
        numNodes += threadLocalStatistics[i].misc;
    }
    printf("total misc time: %fms \n", totalMiscTime);
    printf("num nodes: %llu\n", numNodes);
    Print("num nodes: %llu\n", numNodes);
    printf("node kb: %llu\n", totalNodeMemory / 1024);
    printf("record kb: %llu", totalRecordMemory / 1024);
    ScratchEnd(temp);
}

void PartitionFix()
{
    Arena *arena = ArenaAlloc();
    string data  = OS_ReadFile(arena, "build/parallel_mid.txt");
    Tokenizer tokenizer;
    tokenizer.input  = data;
    tokenizer.cursor = tokenizer.input.str;
    bool *testArray  = PushArray(arena, bool, 291916);
    u32 numLines     = 0;
    while (!EndOfBuffer(&tokenizer))
    {
        // string line = ReadLine(&tokenizer);
        // u32 number  = 0;
        // u32 i       = 0;
        // while (line.str[i] != ' ')
        // {
        //     number *= 10;
        //     number += (line.str[i++] - 48);
        // }
        // if (testArray[number])
        // {
        //     // Assert(!"don't think this should happen\n");
        //     string word      = SkipToNextWord(SkipToNextWord(line));
        //     Split::Type type = (Split::Type)(word.str[0] - 48);
        //     Assert(type == Split::Type::Object);
        // }
        // testArray[number] = true;
        numLines++;
        SkipToNextLine(&tokenizer);
    }
    printf("num lines %u\n", numLines);
}

#if 0
void VolumeRenderingTest(Arena *arena, string filename)
{
    // TODO: this is hardcoded from disney-cloud.pbrt. Update the scene reader to handle participating media
    // Sampler
    const u32 width      = 1280;
    const u32 height     = 720;
    const u32 spp        = 1024;
    const u32 maxDepth   = 100;
    const f32 lensRadius = 0.f;
    Vec2f filterRadius(0.5f, 0.5f);
    f32 focalLength = 0.f;
    // Camera matrix
    Vec3f pCamera = Vec3f(648.064, -82.473, -63.856);
    Vec3f look    = Vec3f(6.021, 100.043, -43.679);
    Vec3f up      = Vec3f(0.273, 0.962, -0.009);
    // AffineSpace camera           = AffineSpace::LookAt(pCamera,
    //
    //

    // NOTE: render space is just world space centered at the camera
    // Vec3f up(0.f, 0.f, 1.f);
    Vec3f f = Normalize(pCamera - look);
    Vec3f s = Normalize(Cross(up, f));
    Vec3f u = Cross(f, s);

    Mat4 cameraFromRender(f.x, f.y, f.z, 0.f,
                          s.x, s.y, s.z, 0.f,
                          u.x, u.y, u.z, 0.f,
                          0.f, 0.f, 0.f, 1.f);

    Mat4 renderFromCamera = Inverse(cameraFromRender);
    Mat4 NDCFromCamera    = Mat4::Perspective(Radians(31.07), 1280.f / 720.f);
    // maps to raster coordinates
    Mat4 rasterFromNDC = Scale(Vec3f(width, -i32(height), 1.f)) * Scale(Vec3f(1.f / 2.f, 1.f / 2.f, 1.f)) *
                         Translate(Vec3f(1.f, -1.f, 0.f));
    Mat4 rasterFromCamera = rasterFromNDC * NDCFromCamera;
    Mat4 cameraFromRaster = Inverse(rasterFromCamera);

    // Sun
    Vec3f rgbL(2.6, 2.5, 2.3);
    RGBIlluminantSpectrum spec(*RGBColorSpace::sRGB, rgbL);
    DistantLight sun(Vec3f(-0.5826f, -0.77660f, -0.2717f), &spec);

    // Light
    rgbL = Vec3f(0.03, 0.07, 0.23);
    RGBIlluminantSpectrum spec2(*RGBColorSpace::sRGB, rgbL);
    UniformInfiniteLight light(&spec2);
    Scene2 scene;

    // Media
    AffineSpace transform = AffineSpace::Identity();

    f32 lambdas[] = {200.f, 900.f};
    f32 valuesA[] = {0.f, 0.f};
    f32 valuesS[] = {1.f, 1.f};

    ConstantSpectrum cAbs(0.f);
    ConstantSpectrum cScatter(1.f);
    f32 scale = 4.f;
    f32 g     = .877;

    NanoVDBVolume volumes[] = {
        NanoVDBVolume("data/wdas_cloud_quarter.nvdb", &transform, &cAbs, &cScatter, g, scale),
    };

    scene.volumes.Set(volumes, ArrayLength(volumes));

    //////////////////////////////
    // Primitives

    // Disk
    AffineSpace cameraTranslate = AffineSpace::Translate(-pCamera);
    AffineSpace diskTransform   = AffineSpace::Translate(0.f, -1000.f, 0.f) * AffineSpace::Scale(2000.f) *
                                AffineSpace::Rotate(Vec3f(1.f, 0.f, 0.f), -PI / 2.f);
    diskTransform                    = cameraTranslate * diskTransform;
    AffineSpace diskObjectFromRender = Inverse(diskTransform);
    Disk disk(&diskObjectFromRender);

    AffineSpace boxTransform = AffineSpace::Translate(-9.984, 73.008, -42.64) * AffineSpace::Scale(206.544, 140.4, 254.592);
    boxTransform             = cameraTranslate * boxTransform;
    scene.affineTransforms   = &boxTransform;

    // Bounding box
#if 0
    QuadMesh mesh;
    Vec3f p[] = {
        // front
        Vec3f(-0.5f, -0.5f, -0.5f),
        Vec3f(0.5f, -0.5f, -0.5f),
        Vec3f(0.5f, -0.5f, 0.5f),
        Vec3f(-0.5f, -0.5f, 0.5f),
        // right
        Vec3f(0.5f, -0.5f, -0.5f),
        Vec3f(0.5f, 0.5f, -0.5f),
        Vec3f(0.5f, 0.5f, 0.5f),
        Vec3f(0.5f, -0.5f, 0.5f),
        // back
        Vec3f(0.5f, 0.5f, -0.5f),
        Vec3f(-0.5f, 0.5f, -0.5f),
        Vec3f(-0.5f, 0.5f, 0.5f),
        Vec3f(0.5f, 0.5f, 0.5f),
        // left
        Vec3f(-0.5f, 0.5f, -0.5f),
        Vec3f(-0.5f, -0.5f, -0.5f),
        Vec3f(-0.5f, -0.5f, 0.5f),
        Vec3f(-0.5f, 0.5f, 0.5f),
        // top
        Vec3f(-0.5f, -0.5f, 0.5f),
        Vec3f(0.5f, -0.5f, 0.5f),
        Vec3f(0.5f, 0.5f, 0.5f),
        Vec3f(-0.5f, 0.5f, 0.5f),
        // bottom
        Vec3f(-0.5f, 0.5f, -0.5f),
        Vec3f(0.5f, 0.5f, -0.5f),
        Vec3f(0.5f, -0.5f, -0.5f),
        Vec3f(-0.5f, -0.5f, -0.5f),
    };
    mesh.p           = p;
    mesh.numVertices = 24;
    scene.primitives.Set(&mesh, 1);
#endif
    scene.primitives.Set(&disk, 1);

    const Scene2::PrimitiveIndices indices[2][1] = {
        {},
        {Scene2::PrimitiveIndices(0, 0, 0)},
    };
    const Scene2::PrimitiveIndices *primIndices[2] = {indices[0], indices[1]};
    scene.primIndices                              = primIndices;

    BuildLightPDF(&scene);

    scene.aggregate.Build(arena, &scene);

}
#endif

void TriangleMeshBVHTest(Arena *arena)
{
    // DONE:
    // - make the materials polymorphic so that the integrator can access them
    // - fix the compressed leaf intersector
    // - instantiate the water material and attach the triangle mesh to it
    // - have the intersector handle the case where there are no geomIDs (only primIDs)
    // - make sure traversal code works
    // - add material index when intersecting
    // - make sure the environment map works properly and returns the right radiances
    // - make sure i'm calculating the final rgb value correctly for each pixel
    // - make sure i'm shooting the right camera rays

    // TODO:
    // - have to handle the handedness scaling factor
    // - fix material evaluation

    // once the ocean is rendered
    // - need to support a bvh with quad/triangle mesh instances
    // - load the scene description and properly instantiate lights/materials/textures
    // - render the scene with all quad meshes, then add support for the bspline curves
    // - change the bvh build process to support N-wide leaves (need to change the sah to
    // account for this)

    // once moana is rendered
    // - ray differentials

    scene_        = PushStruct(arena, Scene2);
    Scene2 *scene = GetScene();
    // TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16); // ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
    }

    // Camera
    u32 width  = 1920;
    u32 height = 804;
    Vec3f pCamera(-1139.0159, 23.286734, 1479.7947);
    Vec3f look(244.81433, 238.80714, 560.3801);
    Vec3f up(-0.107149, .991691, .07119);

    Mat4 cameraFromRender = LookAt(pCamera, look, up) * Translate(pCamera);

    Mat4 renderFromCamera = Inverse(cameraFromRender);
    // TODO: going to have to figure out how to handle this automatically
    Mat4 NDCFromCamera = Mat4::Perspective2(Radians(69.50461), 2.386946);
    // maps to raster coordinates
    Mat4 rasterFromNDC = Scale(Vec3f(f32(width), -f32(height), 1.f)) *
                         Scale(Vec3f(1.f / 2.f, 1.f / 2.f, 1.f)) *
                         Translate(Vec3f(1.f, -1.f, 0.f));
    Mat4 rasterFromCamera = rasterFromNDC * NDCFromCamera;
    Mat4 cameraFromRaster = Inverse(rasterFromCamera);

    // ocean mesh
    TriangleMesh mesh =
        LoadPLY(arena, "../data/island/pbrt-v4/osOcean/osOcean_geometry_00001.ply");
    u32 numFaces = mesh.numIndices / 3;
    // convert to "render space" (i.e. world space centered around the camera)
    for (u32 i = 0; i < mesh.numVertices; i++)
    {
        mesh.p[i] -= pCamera;
    }
    Bounds geomBounds;
    Bounds centBounds;
    PrimRefCompressed *refs = GenerateAOSData(arena, &mesh, numFaces, geomBounds, centBounds);

    // environment map
    f32 sceneRadius             = 0.5f * Max(geomBounds.maxP[0] - geomBounds.minP[0],
                                             Max(geomBounds.maxP[1] - geomBounds.minP[1],
                                                 geomBounds.maxP[2] - geomBounds.minP[2]));
    AffineSpace renderFromLight = AffineSpace::Scale(-1, 1, 1) *
                                  AffineSpace::Rotate(Vec3f(-1, 0, 0), Radians(90)) *
                                  AffineSpace::Rotate(Vec3f(0, 0, 1), Radians(65));
    renderFromLight = AffineSpace::Translate(-pCamera) * renderFromLight;

    f32 scale = 1.f / SpectrumToPhotometric(RGBColorSpace::sRGB->illuminant);
    Assert(scale == 0.00935831666f);
    ImageInfiniteLight infLight(
        arena, LoadFile("../data/island/pbrt-v4/textures/islandsunVIS-equiarea.png"),
        &renderFromLight, RGBColorSpace::sRGB, sceneRadius, scale);
    scene->lights.Set<ImageInfiniteLight>(&infLight, 1);

    BuildSettings settings;
    settings.intCost = 0.3f;

    RecordAOSSplits record;
    record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    record.SetRange(0, numFaces, u32(numFaces * GROW_AMOUNT));

    BVHNodeN bvh          = BuildQuantizedTriSBVH(settings, arenas, &mesh, refs, record);
    scene->nodePtr        = bvh;
    scene->triangleMeshes = &mesh;
    scene->numTriMeshes   = 1;

    ConstantTexture<1> ct(0.f);
    ConstantSpectrum spec(1.1f);
    DielectricMaterialBase mat(DielectricMaterialConstant(ct, spec), NullShader());
    scene->materials.Set<DielectricMaterialBase>(&mat, 1);
    Scene2::PrimitiveIndices ids[] = {
        Scene2::PrimitiveIndices(LightHandle(), MaterialHandle(MT_DielectricMaterial, 0)),
    };
    scene->primIndices = ids;
    // PerformanceCounter counter = OS_StartCounter();
    // f32 time                   = OS_GetMilliseconds(counter);

    RenderParams2 params;
    params.cameraFromRaster = cameraFromRaster;
    params.renderFromCamera = renderFromCamera;
    params.width            = width;
    params.height           = height;
    params.filterRadius     = Vec2f(0.5f);
    params.spp              = 8;
    params.maxDepth         = 10;
    params.lensRadius       = 0.003125;
    params.focalLength      = 1675.3383;

    Render(arena, params);
}

} // namespace rt
