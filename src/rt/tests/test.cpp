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
void WhiteFurnaceTest(Arena *arena, Options *options = 0)
{
    scene_       = PushStruct(arena, Scene);
    Scene *scene = GetScene();
    // ScenePrimitives *scenePrims     = PushStruct(arena, ScenePrimitives);
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16);
    }

    PerformanceCounter counter = OS_StartCounter();

    // Camera
    u32 width  = 1920;
    u32 height = 804;
    Vec3f pCamera(0.f, -10.f, 0.f);
    Vec3f look(0.f, 1.f, 0.f);
    Vec3f up(0.f, 0.f, 1.f);

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

    AffineSpace renderFromWorld = AffineSpace::Translate(-pCamera);
    AffineSpace worldFromRender = AffineSpace::Translate(pCamera);

    // MSDielectricMaterial1 mat(MSDielectricMaterial(0.3f, 0.3f, 1.2f), NullShader());
    // scene->materials.Set<MSDielectricMaterial1>(&mat, 1);

    // ConstantTexture ct(0.3f);
    // ConstantSpectrum spec(1.2f);
    // DielectricMaterialBase dielMat(DielectricMaterialConstant(ct, spec), NullShader());
    // scene->materials.Set<DielectricMaterialBase>(&dielMat, 1);

    PrimitiveIndices *ids = PushStructConstruct(arena, PrimitiveIndices)(
        LightHandle(), MaterialHandle(MaterialType::DielectricMaterialBase, 0));
    scene->scene.primIndices = ids;

    Vec3f p[] = {
        // Front
        // Vec3f(-1.f, -1.f, -1.f),
        // Vec3f(1.f, -1.f, -1.f),
        // Vec3f(1.f, -1.f, 1.f),
        // Vec3f(-1.f, -1.f, 1.f),
        // // // Right
        // Vec3f(1.f, -1.f, -1.f),
        // Vec3f(1.f, 1.f, -1.f),
        // Vec3f(1.f, 1.f, 1.f),
        // Vec3f(1.f, -1.f, 1.f),
        // // // Back
        // Vec3f(1.f, 1.f, -1.f),
        // Vec3f(-1.f, 1.f, -1.f),
        // Vec3f(-1.f, 1.f, 1.f),
        // Vec3f(1.f, 1.f, 1.f),
        // // // Left
        // Vec3f(-1.f, 1.f, -1.f),
        // Vec3f(-1.f, -1.f, -1.f),
        // Vec3f(-1.f, -1.f, 1.f),
        // Vec3f(-1.f, 1.f, 1.f),
        // Top
        Vec3f(-1.f, -1.f, 1.f), Vec3f(1.f, -1.f, 1.f), Vec3f(1.f, 1.f, 1.f),
        Vec3f(-1.f, 1.f, 1.f),
        // Bottom
        // Vec3f(-1.f, -1.f, -1.f),
        // Vec3f(1.f, -1.f, -1.f),
        // Vec3f(1.f, 1.f, -1.f),
        // Vec3f(-1.f, 1.f, -1.f),

    };
    AffineSpace t = renderFromWorld * AffineSpace::Rotate(Vec3f(1, 0, 0), Radians(45.f));
    for (u32 i = 0; i < ArrayLength(p); i++)
    {
        p[i] = TransformP(t, p[i]);
    }

    Mesh mesh;
    mesh.p           = p;
    mesh.numVertices = ArrayLength(p);
    mesh.numFaces    = 1;

    BuildSettings settings;

    scene->scene.primitives    = &mesh;
    scene->scene.numPrimitives = 1;
    BuildQuadBVH(arenas, settings, &scene->scene);

    f32 scale = 1.f / SpectrumToPhotometric(RGBColorSpace::sRGB->illuminant);
    ConstantSpectrum spec2(1.f);
    UniformInfiniteLight infLight(&spec2, scale);
    scene->lights.Set<UniformInfiniteLight>(&infLight, 1);
    scene->numLights = 1;

    f32 time = OS_GetMilliseconds(counter);
    printf("setup time: %fms\n", time);

    RenderParams2 params;
    params.cameraFromRaster = cameraFromRaster;
    params.renderFromCamera = renderFromCamera;
    params.width            = width;
    params.height           = height;
    params.filterRadius     = Vec2f(0.5f);
    params.spp              = 64;
    params.maxDepth         = 10;
    params.lensRadius       = 0.003125;
    params.focalLength      = 1675.3383;

    if (options)
    {
        if (options->pixelX != -1 && options->pixelY != -1)
        {
            params.pixelMin = Vec2u(options->pixelX, options->pixelY);
            params.pixelMax = params.pixelMin + Vec2u(1, 1);
        }
    }

    counter = OS_StartCounter();
    Render(arena, params);
    time = OS_GetMilliseconds(counter);
    printf("total render time: %fms\n", time);

    f64 totalMiscTime = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
    }
    printf("total misc time: %fms \n", totalMiscTime);
}
#endif

void TestRender(Arena *arena, Options *options = 0)
{
    // TODO:
    // - add area lights (making sure they cannot be intersected, but are sampled properly)
    // - render the diffuse transmission materials properly
    // - load the scene description and properly instantiate lights/materials/textures

    // - render the scene with all quad meshes, then add support for the bspline curves

    // once moana is rendered
    // - ray differentials

    scene_       = PushStruct(arena, Scene);
    Scene *scene = GetScene();
    // ScenePrimitives *scenePrims     = PushStruct(arena, ScenePrimitives);
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16);
    }

    Arena **tempArenas = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        tempArenas[i] = ArenaAlloc(16);
    }

    PerformanceCounter counter = OS_StartCounter();

    // Camera

    u32 width  = 1920;
    u32 height = 804;

    // beach cam
    // Vec3f pCamera(-510.523907, 87.308744, 181.770197);
    // Vec3f look(152.465305, 30.939795, -72.727517);
    // Vec3f up(0.073871, 0.996865, -0.028356);

    f32 lensRadius = .003125;
    // f32 focalDistance = 712.391212;
    // f32 fov           = 54.43222;
    f32 aspectRatio = 2.386946;

    // base cam
    Vec3f pCamera(-1139.0159, 23.286734, 1479.7947);
    Vec3f look(244.81433, 238.80714, 560.3801);
    Vec3f up(-0.107149, .991691, .07119);
    f32 fov           = 69.50461;
    f32 focalDistance = 1675.3383;

    Mat4 cameraFromRender = LookAt(pCamera, look, up) * Translate(pCamera);

    Mat4 renderFromCamera = Inverse(cameraFromRender);
    // TODO: going to have to figure out how to handle this automatically
    Mat4 NDCFromCamera = Mat4::Perspective2(Radians(fov), aspectRatio);

    // maps to raster coordinates
    Mat4 rasterFromNDC = Scale(Vec3f(f32(width), -f32(height), 1.f)) *
                         Scale(Vec3f(1.f / 2.f, 1.f / 2.f, 1.f)) *
                         Translate(Vec3f(1.f, -1.f, 0.f));
    Mat4 rasterFromCamera = rasterFromNDC * NDCFromCamera;
    Mat4 cameraFromRaster = Inverse(rasterFromCamera);

    AffineSpace renderFromWorld = AffineSpace::Translate(-pCamera);
    AffineSpace worldFromRender = AffineSpace::Translate(pCamera);

    LoadScene(arenas, tempArenas, "../data/island/pbrt-v4/", options->filename, NDCFromCamera,
              height, &renderFromWorld);

    // environment map
#if 1
    Bounds bounds              = scene->scene.GetBounds();
    f32 sceneRadius            = Length(ToVec3f(bounds.Centroid() - bounds.maxP));
    AffineSpace worldFromLight = AffineSpace::Scale(-1, 1, 1) *
                                 AffineSpace::Rotate(Vec3f(-1, 0, 0), Radians(90)) *
                                 AffineSpace::Rotate(Vec3f(0, 0, 1), Radians(65));
    AffineSpace renderFromLight = worldFromLight;

    f32 scale = 1.f / SpectrumToPhotometric(RGBColorSpace::sRGB->illuminant);
    Assert(scale == 0.00935831666f);
    ImageInfiniteLight infLight(
        arena, LoadFile("../data/island/pbrt-v4/textures/islandsunVIS-equiarea.png"),
        &renderFromLight, RGBColorSpace::sRGB, sceneRadius, scale);
    scene->lights.Set<ImageInfiniteLight>(&infLight, 1);
    scene->numLights = 1;
#endif

    // f32 scale = 1.f / SpectrumToPhotometric(RGBColorSpace::sRGB->illuminant);
    // ConstantSpectrum spec2(1.f);
    // UniformInfiniteLight infLight(&spec2, scale);
    // scene->lights.Set<UniformInfiniteLight>(&infLight, 1);
    // scene->numLights = 1;

    f32 time = OS_GetMilliseconds(counter);
    printf("setup time: %fms\n", time);
    f64 totalMiscTime            = 0;
    u64 totalCompressedNodeCount = 0;
    u64 totalNodeCount           = 0;
    u64 totalBVHMemory           = 0;
    u64 totalShapeMemory         = 0;
    u64 totalNumSpatialSplits    = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        totalCompressedNodeCount += threadLocalStatistics[i].misc;
        totalNodeCount += threadLocalStatistics[i].misc2;
        totalBVHMemory += threadMemoryStatistics[i].totalBVHMemory;
        totalShapeMemory += threadMemoryStatistics[i].totalShapeMemory;
        totalNumSpatialSplits += threadLocalStatistics[i].misc3;
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
    }
    printf("total misc time: %fms \n", totalMiscTime);
    printf("total c node#: %llu \n", totalCompressedNodeCount);
    printf("total node#: %llu \n", totalNodeCount);
    printf("total bvh bytes: %llu \n", totalBVHMemory);
    printf("total shape bytes: %llu \n", totalShapeMemory);
    printf("total # spatial splits: %llu\n", totalNumSpatialSplits);

    RenderParams2 params;
    params.cameraFromRaster = cameraFromRaster;
    params.renderFromCamera = renderFromCamera;
    params.width            = width;
    params.height           = height;
    params.filterRadius     = Vec2f(0.5f);
    params.spp              = 64;
    params.maxDepth         = 10;
    params.lensRadius       = lensRadius;
    params.focalLength      = focalDistance;

    if (options)
    {
        if (options->pixelX != -1 && options->pixelY != -1)
        {
            params.pixelMin = Vec2u(options->pixelX, options->pixelY);
            params.pixelMax = params.pixelMin + Vec2u(1, 1);
        }
    }

    shadingThreadState_ = PushArray(arena, ShadingThreadState, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        ShadingThreadState *state = &shadingThreadState_[i];
        state->rayStates          = RayStateList(arenas[i]);
        state->rayFreeList        = RayStateFreeList(arenas[i]);

        state->rayQueue.handler = RayIntersectionHandler;
        state->shadingQueues[(u32)MaterialTypes::Interface].handler =
            ShadingQueueHandler<NullMaterial>;
        state->shadingQueues[(u32)MaterialTypes::Diffuse].handler =
            ShadingQueueHandler<DiffuseMaterial>;
        state->shadingQueues[(u32)MaterialTypes::DiffuseTransmission].handler =
            ShadingQueueHandler<DiffuseTransmissionMaterial>;
        state->shadingQueues[(u32)MaterialTypes::CoatedDiffuse].handler =
            ShadingQueueHandler<CoatedDiffuseMaterial>;
        state->shadingQueues[(u32)MaterialTypes::Dielectric].handler =
            ShadingQueueHandler<DielectricMaterial>;
    }

    counter = OS_StartCounter();
    Render(arena, params);
    // RenderSIMD(arena, params);
    time = OS_GetMilliseconds(counter);
    printf("total render time: %fms\n", time);
}

template <typename T>
T Func(const T &x)
{
    T x2 = x * x;
    T x3 = x2 * x;
    return T(3.f) * x3 - T(2.f) * x2 + T(1.f);
}

template <typename T>
T Func2(const T &x, const T &y)
{
    T x2 = x * x;
    T y2 = y * y;
    return y2 / x2;
}

template <typename T>
T Func3(const T &x)
{
    return x * x * x;
}

template <typename T>
T Func4(const T &x, const T &y)
{
    T x2 = x * x;
    T y2 = y * y;
    return y2 * x2;
}

template <typename T>
inline Vec3<T> ReflectTest(const Vec3<T> &v, const Vec3<T> &norm)
{
    return -v + 2 * Dot(v, norm) * norm;
}

template <i32 N>
using Vec3df = Vec3<Dual<f32, N>>;
void DualTest()
{
#if 1
    Vec3df<1> dual(Dual<f32, 1>(5.f, 1.f), Dual<f32, 1>(3.f, 1.f), Dual<f32, 1>(2.f, 1.f));
    auto result = Func(dual);

    Dual<f32, 2> dual1(5.f, 1.f, 0.f);
    Dual<f32, 2> dual2(3.f, 0.f, 1.f);

    auto result2 = Func2(dual1, dual2);

    HyperDual<f32, 1> dual3(5.f, 1.f, 1.f);
    auto result3 = Func3(dual3);

    HyperDual<f32, 2> dual4(5.f, 1.f, 1.f, 0.f, 0.f);
    HyperDual<f32, 2> dual5(3.f, 0.f, 0.f, 1.f, 1.f);
    auto result4 = Func4(dual4, dual5);
#else
    // Ray2 ray;
    // ray.pxOffset = si.p + dpdx;
    // ray.pyOffset = si.p + dpdy;

    // Compute differential reflected directions
    // TODO: see if you get the same result with duals as with igehy's formulation
    // it should work, right?

#endif

    int stop = 5;
}

} // namespace rt
