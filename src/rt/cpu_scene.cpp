#include "cpu_scene.h"
#include "integrate.h"
#include "platform.h"
#include "scene.h"
#include "simd_integrate.h"
#include "win32.h"

namespace rt
{

void BuildAllSceneBVHs(RenderParams2 *params, ScenePrimitives **scenes, int numScenes,
                       int maxDepth)
{
    for (int depth = maxDepth; depth >= 0; depth--)
    {
        ParallelFor(0, numScenes, 1, [&](int jobID, int start, int count) {
            for (int i = start; i < start + count; i++)
            {
                if (scenes[i]->depth.load(std::memory_order_acquire) == depth)
                    scenes[i]->BuildSceneBVHs(params->arenas, params->NDCFromCamera,
                                              params->cameraFromRender, params->height);
            }
        });
    }
}

void ComputeEdgeRates(ScenePrimitives *scene, const AffineSpace &transform,
                      const Vec4f *planes, TessellationStyle style)
{
    switch (scene->geometryType)
    {
        case GeometryType::Instance:
        {
            Instance *instances = (Instance *)scene->primitives;
            ParallelFor(
                0, scene->numPrimitives, PARALLEL_THRESHOLD, PARALLEL_THRESHOLD,
                [&](int jobID, int start, int count) {
                    for (int i = start; i < start + count; i++)
                    {
                        const Instance &instance = instances[i];
                        AffineSpace t =
                            transform * scene->affineTransforms[instance.transformIndex];
                        ComputeEdgeRates(scene->childScenes[instance.id], t, planes, style);
                    }
                });
        }
        break;
        case GeometryType::CatmullClark:
        {
            Mesh *controlMeshes = (Mesh *)scene->primitives;
            ParallelFor(0, scene->numPrimitives, PARALLEL_THRESHOLD, PARALLEL_THRESHOLD,
                        [&](int jobID, int start, int count) {
                            for (int i = start; i < start + count; i++)
                            {
                                TessellationParams &params = scene->tessellationParams[i];

                                switch (style)
                                {
                                    case TessellationStyle::ClosestInstance:
                                    {
                                        BeginRMutex(&params.mutex);
                                        Bounds bounds  = Transform(transform, params.bounds);
                                        Vec3f centroid = ToVec3f(bounds.Centroid());
                                        f64 currentMinDistance = params.currentMinDistance;
                                        EndRMutex(&params.mutex);

                                        // NOTE: this skips the far plane test
                                        int result =
                                            IntersectFrustumAABB<1>(planes, &bounds, 5);

                                        Vec3<f64> centroidDouble(
                                            (f64)centroid.x, (f64)centroid.y, (f64)centroid.z);
                                        f64 distance = Length(centroidDouble);

                                        // Camera is at origin in this coordinate space
                                        if (result && distance < currentMinDistance)
                                        {
                                            BeginWMutex(&params.mutex);
                                            if (distance < params.currentMinDistance)
                                            {
                                                params.transform          = transform;
                                                params.currentMinDistance = distance;
                                            }
                                            EndWMutex(&params.mutex);
                                        }
                                    }
                                    break;
                                    case TessellationStyle::PerInViewInstancePerEdge:
                                    {
                                        Assert(0);
                                    }
                                    break;
                                }
                            }
                        });
        }
        break;
        default:
        {
        }
        break;
    }
}

void Render(RenderParams2 *params, int numScenes, Image *envMap)
{
    ScenePrimitives **scenes = GetScenes();
    Scene *scene             = GetScene();
    AffineSpace space        = AffineSpace::Identity();

    Vec4f planes[6];
    ExtractPlanes(planes, params->NDCFromCamera * params->cameraFromRender);

    ComputeEdgeRates(&scene->scene, space, planes, TessellationStyle::ClosestInstance);

    int maxDepth = 0;
    for (int i = 0; i < numScenes; i++)
    {
        maxDepth = Max(maxDepth, scenes[i]->depth.load(std::memory_order_acquire));
    }

    BuildAllSceneBVHs(params, scenes, numScenes, maxDepth);

    f64 totalMiscTime            = 0;
    u64 totalCompressedNodeCount = 0;
    u64 totalNodeCount           = 0;
    u64 totalBVHMemory           = 0;
    u64 totalShapeMemory         = 0;
    u64 totalInstanceMemory      = 0;
    u64 totalNumSpatialSplits    = 0;
    u64 maxEdgeFactor            = 0;
    for (u32 i = 0; i < OS_NumProcessors(); i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        totalCompressedNodeCount += threadLocalStatistics[i].misc;
        totalNodeCount += threadLocalStatistics[i].misc2;
        totalBVHMemory += threadMemoryStatistics[i].totalBVHMemory;
        totalShapeMemory += threadMemoryStatistics[i].totalShapeMemory;
        totalInstanceMemory += threadMemoryStatistics[i].totalInstanceMemory;
        totalNumSpatialSplits += threadLocalStatistics[i].misc3;
        maxEdgeFactor = Max(maxEdgeFactor, threadLocalStatistics[i].misc4);
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
    }
    printf("total misc time: %fms \n", totalMiscTime);
    printf("total c node#: %llu \n", totalCompressedNodeCount);
    printf("total node#: %llu \n", totalNodeCount);
    printf("total bvh bytes: %llu \n", totalBVHMemory);
    printf("total shape bytes: %llu \n", totalShapeMemory);
    printf("total instance bytes: %llu\n", totalInstanceMemory);
    printf("total # spatial splits: %llu\n", totalNumSpatialSplits);
    printf("max edge factor:  %llu\n", maxEdgeFactor);

    PerformanceCounter counter = OS_StartCounter();
    RenderSIMD(params->arenas, *params);
    u64 time = OS_GetMilliseconds(counter);
    printf("total render time: %fms\n", time);
}

} // namespace rt
