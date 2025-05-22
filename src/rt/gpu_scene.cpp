#include "bit_packing.h"
#include "camera.h"
#include "dgfs.h"
#include "debug.h"
#include "graphics/vulkan.h"
#include "integrate.h"
#include "gpu_scene.h"
#include "math/simd_base.h"
#include "string.h"
#include "memory.h"
#include "parallel.h"
#include "platform.h"
#include "shader_interop/as_shaderinterop.h"
#include "shader_interop/dense_geometry_shaderinterop.h"
#include "shader_interop/gpu_scene_shaderinterop.h"
#include "shader_interop/hit_shaderinterop.h"
#include "shader_interop/ray_shaderinterop.h"
#include "shader_interop/debug_shaderinterop.h"
#include "shader_interop/virtual_textures_shaderinterop.h"
#include "../third_party/nvapi/nvapi.h"
#include "nvapi.h"
#include "scene.h"
#include "win32.h"
#include "graphics/ptex.h"

namespace rt
{

void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          MaterialHashMap *materialHashMap, Mesh &mesh,
                          ChunkedLinkedList<Mesh, MemoryType_Shape> &shapes,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices,
                          ChunkedLinkedList<Light *, MemoryType_Light> &lights)
{
    PrimitiveIndices &primIndices = indices.AddBack();

    MaterialHandle materialHandle;
    LightHandle lightHandle;
    Texture *alphaTexture   = 0;
    DiffuseAreaLight *light = 0;

    // TODO: handle instanced mesh lights
    AffineSpace *transform = 0;

    // Check for material
    if (Advance(&tokenizer, "m "))
    {
        Assert(materialHashMap);
        string materialName      = ReadWord(&tokenizer);
        const MaterialNode *node = materialHashMap->Get(materialName);
        materialHandle           = node->handle;
    }

    if (Advance(&tokenizer, "transform "))
    {
        u32 transformIndex = ReadInt(&tokenizer);
        SkipToNextChar(&tokenizer);
    }

    // Check for area light
    if (Advance(&tokenizer, "a "))
    {
        ErrorExit(type == GeometryType::QuadMesh, "Only quad area lights supported for now\n");
        Assert(transform);

        DiffuseAreaLight *areaLight =
            ParseAreaLight(arena, &tokenizer, transform, sceneID, shapes.totalCount - 1);
        lightHandle      = LightHandle(LightClass::DiffuseAreaLight, lights.totalCount);
        lights.AddBack() = areaLight;
        light            = areaLight;
    }

    // Check for alpha
    if (Advance(&tokenizer, "alpha "))
    {
        Texture *alphaTexture = ParseTexture(arena, &tokenizer, directory);

        // TODO: this is also a hack: properly evaluate whether the alpha is
        // always 0
        if (lightHandle)
        {
            light->type = LightType::DeltaPosition;
        }
    }
    primIndices = PrimitiveIndices(lightHandle, materialHandle, alphaTexture);
}

// what's next?
// 1. restir di
// 3. clas
//      - are memory savings possible with this? it seems like not really, and that this
//      just speeds up rebuilds for dynamic/adaptively tessellated geometry. not really
//      what I need.
//      - on blackwell there's memory savings
//      - could not get it to run
// 4. actual bsdfs and brdfs
// 5. add other parts of the scene, with actual instancing
// 6. disney bsdf
// 7. recycle memory
// 8. textures
StaticArray<VkAabbPositionsKHR> CreateAABBForNTriangles(Arena *arena, ClusterBuilder &builder,
                                                        int numBlocks)
{
    u32 N          = TRIANGLES_PER_LEAF;
    u32 shift      = LOG2_TRIANGLES_PER_LEAF;
    u32 totalAabbs = MAX_CLUSTER_TRIANGLES >> shift;
    StaticArray<VkAabbPositionsKHR> aabbs(arena, totalAabbs * numBlocks);
    PrimRef *refs = builder.primRefs;
    for (int index = 0; index < builder.threadClusters.Length(); index++)
    {
        for (auto *node = builder.threadClusters[index].l.first; node != 0; node = node->next)
        {
            for (int i = 0; i < node->count; i++)
            {
                u32 numTriangles        = 0;
                RecordAOSSplits &record = node->values[i];

                int numAabbs = (record.count + N - 1) >> shift;
                for (int j = 0; j < numAabbs; j++)
                {
                    VkAabbPositionsKHR aabb;
                    aabb.minX         = pos_inf;
                    aabb.minY         = pos_inf;
                    aabb.minZ         = pos_inf;
                    aabb.maxX         = neg_inf;
                    aabb.maxY         = neg_inf;
                    aabb.maxZ         = neg_inf;
                    int startRefIndex = record.start + N * j;
                    for (int subIndex = 0; subIndex < Min(N, record.count - startRefIndex);
                         subIndex++)
                    {
                        int refIndex = startRefIndex + subIndex;
                        aabb.minX    = Min(aabb.minX, -refs[refIndex].minX);
                        aabb.minY    = Min(aabb.minY, -refs[refIndex].minY);
                        aabb.minZ    = Min(aabb.minZ, -refs[refIndex].minZ);
                        aabb.maxX    = Max(aabb.maxX, refs[refIndex].maxX);
                        aabb.maxY    = Max(aabb.maxY, refs[refIndex].maxY);
                        aabb.maxZ    = Max(aabb.maxZ, refs[refIndex].maxZ);
                    }

                    aabbs.Push(aabb);
                }

                VkAabbPositionsKHR nullAabb = {};
                nullAabb.minX               = f32(NaN);
                for (int remaining = 0; remaining < totalAabbs - numAabbs; remaining++)
                {
                    aabbs.Push(nullAabb);
                }
            }
        }
    }
    return aabbs;
}

void BuildAllSceneBVHs(RenderParams2 *params, ScenePrimitives **scenes, int numScenes,
                       int maxDepth, Image *envMap)
{
    NvAPI_Initialize();
    // Compile shaders
    Shader shader;
    Shader decodeShader;
    Shader fillInstanceShader;

    RayTracingShaderGroup groups[3];
    Arena *arena = params->arenas[GetThreadIndex()];
    {
        string raygenShaderName       = "../src/shaders/render_raytrace_rgen.spv";
        string missShaderName         = "../src/shaders/render_raytrace_miss.spv";
        string hitShaderName          = "../src/shaders/render_raytrace_hit.spv";
        string intersectionShaderName = "../src/shaders/render_raytrace_dgf_intersect.spv";

        string rgenData      = OS_ReadFile(arena, raygenShaderName);
        string missData      = OS_ReadFile(arena, missShaderName);
        string hitData       = OS_ReadFile(arena, hitShaderName);
        string intersectData = OS_ReadFile(arena, intersectionShaderName);

        Shader raygenShader = device->CreateShader(ShaderStage::Raygen, "raygen", rgenData);
        Shader missShader   = device->CreateShader(ShaderStage::Miss, "miss", missData);
        Shader hitShader    = device->CreateShader(ShaderStage::Hit, "hit", hitData);
        Shader isectShader =
            device->CreateShader(ShaderStage::Intersect, "intersect", intersectData);

        groups[0].shaders[0] = raygenShader;
        groups[0].numShaders = 1;
        groups[0].stage[0]   = ShaderStage::Raygen;

        groups[1].shaders[0] = missShader;
        groups[1].numShaders = 1;
        groups[1].stage[0]   = ShaderStage::Miss;

        groups[2].shaders[0] = hitShader;
        groups[2].stage[0]   = ShaderStage::Hit;
#ifdef USE_PROCEDURAL_CLUSTER_INTERSECTION
        groups[2].shaders[1] = isectShader;
        groups[2].stage[1]   = ShaderStage::Intersect;
        groups[2].numShaders = 2;
        groups[2].type       = RayTracingShaderGroupType::Procedural;
#else
        groups[2].numShaders = 1;
        groups[2].type       = RayTracingShaderGroupType::Triangle;
#endif

        string shaderName = "../src/shaders/render_raytrace.spv";
        string shaderData = OS_ReadFile(arena, shaderName);
        shader = device->CreateShader(ShaderStage::Compute, "pathtrace", shaderData);

        string decodeShaderName = "../src/shaders/decode_dgf_clusters.spv";
        string decodeShaderData = OS_ReadFile(arena, decodeShaderName);
        decodeShader =
            device->CreateShader(ShaderStage::Compute, "decode clusters", decodeShaderData);

        string fillInstanceShaderName = "../src/shaders/fill_instance_descs.spv";
        string fillShaderData         = OS_ReadFile(arena, fillInstanceShaderName);
        fillInstanceShader =
            device->CreateShader(ShaderStage::Compute, "fill instances", fillShaderData);
    }

    // Compile pipelines
#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
    DescriptorSetLayout fillLayout = {};
    int addressesBinding =
        fillLayout.AddBinding(0, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    int instancesBinding =
        fillLayout.AddBinding(1, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline fillInstancePipeline =
        device->CreateComputePipeline(&fillInstanceShader, &fillLayout);

    DescriptorSetLayout decodeLayout = {};
    PushConstant decodeConstants;
    decodeConstants.stage  = ShaderStage::Compute;
    decodeConstants.size   = sizeof(DecodePushConstant);
    decodeConstants.offset = 0;

    int indexBufferBinding =
        decodeLayout.AddBinding(0, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    int vertexBufferBinding =
        decodeLayout.AddBinding(1, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    int buildClasDescsBinding =
        decodeLayout.AddBinding(2, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    int decodeGlobalsBinding =
        decodeLayout.AddBinding(3, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    int clusterDataBinding =
        decodeLayout.AddBinding(4, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline decodePipeline =
        device->CreateComputePipeline(&decodeShader, &decodeLayout, &decodeConstants);
#endif

    // Build clusters
    // TODO: make sure to NOT make gpumeshes
    ScratchArena sceneScratch;

    Bounds *bounds = PushArrayNoZero(sceneScratch.temp.arena, Bounds, numScenes);

    for (int depth = maxDepth; depth >= 0; depth--)
    {
        ParallelFor(0, numScenes, 1, [&](int jobID, int start, int count) {
            for (int i = start; i < start + count; i++)
            {
                if (scenes[i]->depth == depth)
                {
                    bounds[i] = GetSceneBounds(scenes[i]);
                }
            }
        });
    }

    Bounds sceneBounds;
    for (int i = 0; i < numScenes; i++)
    {
        sceneBounds.Extend(bounds[i]);
    }

    struct BLASSceneGPUInfo
    {
        DenseGeometryBuildData data;
        GPUBuffer aabbBuffer;
        GPUBuffer dgfBuffer;
        GPUBuffer headerBuffer;
        u32 aabbLength;
        u8 *dgfByteBuffer;

        PackedDenseGeometryHeader *headers;
        u32 **firstUses;
        u32 **reuses;
        VkAabbPositionsKHR *positions;
        TriangleStripType **types;
        u32 **debugFaceIDs;
        u32 **debugIndices;
        u32 **debugRestartCount;
        u32 **debugRestartHighBit;

        u32 numHeaders = 0;
    };

    // u32 totalNumClusters = 0;
    // u32 numVertices      = 0;
    // u32 numTriangles     = 0;

    StaticArray<ScenePrimitives *> blasScenes(arena, numScenes);
    StaticArray<ScenePrimitives *> tlasScenes(arena, numScenes);

    for (int i = 0; i < numScenes; i++)
    {
        ScenePrimitives *scene = scenes[i];

        if (scene->geometryType == GeometryType::Instance)
        {
            tlasScenes.Push(scene);
            continue;
        }

        RecordAOSSplits record(neg_inf);

        Mesh *meshes = (Mesh *)scene->primitives;
        if (scene->geometryType == GeometryType::QuadMesh ||
            scene->geometryType == GeometryType::CatmullClark)
        {
            for (int j = 0; j < scene->numPrimitives; j++)
            {
                meshes[j] = ConvertQuadToTriangleMesh(sceneScratch.temp.arena, meshes[j]);
            }
            scene->geometryType = GeometryType::TriangleMesh;
        }
        blasScenes.Push(scene);
    }

    Scene *rootScene = GetScene();

    StaticArray<string> tiledPtexFilenames(sceneScratch.temp.arena,
                                           rootScene->ptexTextures.size());
    StaticArray<StaticArray<FaceMetadata2>> ptexInfo(sceneScratch.temp.arena,
                                                     rootScene->ptexTextures.size());

    struct GPUTextureInfo
    {
        u8 *packedFaceData;
        int packedDataSize;
        int minLog2Dim;
        int numVirtualOffsetBits;
        int numFaceDimBits;
        int numFaceIDBits;
        u32 virtualAddressIndex;
    };
    StaticArray<GPUTextureInfo> gpuTextureInfo(sceneScratch.temp.arena,
                                               rootScene->ptexTextures.size(),
                                               rootScene->ptexTextures.size());

    VirtualTextureManager virtualTextureManager(
        sceneScratch.temp.arena, 131072, 131072, PHYSICAL_POOL_NUM_PAGES_WIDE * PAGE_WIDTH,
        PHYSICAL_POOL_NUM_PAGES_WIDE * PAGE_WIDTH, 1, VK_FORMAT_BC1_RGB_UNORM_BLOCK);

    CommandBuffer *tileCmd          = device->BeginCommandBuffer(QueueType_Compute);
    Semaphore tileSubmitSemaphore   = device->CreateSemaphore();
    tileSubmitSemaphore.signalValue = 1;
    tileCmd->TransferWriteBarrier(&virtualTextureManager.pageTable);
    tileCmd->FlushBarriers();
    virtualTextureManager.ClearTextures(tileCmd);

    for (int i = 0; i < rootScene->ptexTextures.size(); i++)
    {
        PtexTexture &ptexTexture = rootScene->ptexTextures[i];
        string filename          = PushStr8F(sceneScratch.temp.arena, "%S.tiles",
                                             RemoveFileExtension(ptexTexture.filename));

        if (!Contains(filename, "mountainb0001_geo") || Contains(filename, "displacement"))
            continue;

        Tokenizer tokenizer;
        tokenizer.input  = OS_ReadFile(sceneScratch.temp.arena, filename);
        tokenizer.cursor = tokenizer.input.str;

        TextureMetadata fileHeader;
        GetPointerValue(&tokenizer, &fileHeader);

        FaceMetadata2 *metaData = (FaceMetadata2 *)tokenizer.cursor;
        Advance(&tokenizer, sizeof(FaceMetadata2) * fileHeader.numFaces);

        auto array = StaticArray<FaceMetadata2>(sceneScratch.temp.arena, fileHeader.numFaces,
                                                fileHeader.numFaces);
        MemoryCopy(array.data, metaData, sizeof(FaceMetadata2) * fileHeader.numFaces);
        tiledPtexFilenames.Push(filename);
        ptexInfo.Push(array);

        u32 allocIndex        = 0;
        Vec2u baseVirtualPage = virtualTextureManager.AllocateVirtualPages(
            sceneScratch.temp.arena, filename, fileHeader,
            Vec2u(fileHeader.virtualSqrtNumPages, fileHeader.virtualSqrtNumPages),
            tokenizer.cursor, allocIndex);

        OS_UnmapFile(tokenizer.input.str);

        // Pack ptex face metadata into FaceData bitstream
        u32 maxVirtualOffset = neg_inf;

        int minLog2Dim = 16;
        int maxLog2Dim = 0;

        for (int faceIndex = 0; faceIndex < fileHeader.numFaces; faceIndex++)
        {
            maxVirtualOffset = Max(maxVirtualOffset, metaData[faceIndex].offsetX);
            maxVirtualOffset = Max(maxVirtualOffset, metaData[faceIndex].offsetY);

            Assert(metaData[faceIndex].log2Width < 9 && metaData[faceIndex].log2Height < 9);
            Assert(Max(metaData[faceIndex].log2Width, metaData[faceIndex].log2Height) > 4);
            minLog2Dim = Min(minLog2Dim, metaData[faceIndex].log2Width);
            maxLog2Dim = Max(maxLog2Dim, metaData[faceIndex].log2Width);
            minLog2Dim = Min(minLog2Dim, metaData[faceIndex].log2Height);
            maxLog2Dim = Max(maxLog2Dim, metaData[faceIndex].log2Height);
        }

        u32 numVirtualOffsetBits = Log2Int(maxVirtualOffset) + 1;
        u32 numFaceDimBits =
            (minLog2Dim == maxLog2Dim ? 0 : Log2Int(Max(maxLog2Dim - minLog2Dim, 1)) + 1);
        u32 numFaceIDBits = Log2Int(fileHeader.numFaces) + 1;

        u32 faceDataBitStreamBitSize =
            (2 * (numVirtualOffsetBits + numFaceDimBits) + 4 * numFaceIDBits + 9) *
            fileHeader.numFaces;
        u32 faceDataBitStreamSize = (faceDataBitStreamBitSize + 7) >> 3;

        u8 *faceDataStream    = PushArray(sceneScratch.temp.arena, u8, faceDataBitStreamSize);
        u32 faceDataBitOffset = 0;
        for (int faceIndex = 0; faceIndex < fileHeader.numFaces; faceIndex++)
        {
            FaceMetadata2 &faceMetadata = metaData[faceIndex];
            WriteBits((u32 *)faceDataStream, faceDataBitOffset, faceMetadata.offsetX,
                      numVirtualOffsetBits);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset, faceMetadata.offsetY,
                      numVirtualOffsetBits);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset,
                      faceMetadata.log2Width - minLog2Dim, numFaceDimBits);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset,
                      faceMetadata.log2Height - minLog2Dim, numFaceDimBits);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset,
                      faceMetadata.rotate & ((1u << 8u) - 1u), 8);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset, faceMetadata.rotate >> 31, 1);
            for (int edgeIndex = 0; edgeIndex < 4; edgeIndex++)
            {
                int neighborFace = faceMetadata.neighborFaces[edgeIndex];
                u32 writeNeighborFace =
                    neighborFace == -1 ? (~0u >> (32 - numFaceIDBits)) : (u32)neighborFace;
                WriteBits((u32 *)faceDataStream, faceDataBitOffset, writeNeighborFace,
                          numFaceIDBits);
            }
        }
        Assert(faceDataBitOffset == faceDataBitStreamBitSize);
        gpuTextureInfo[i].packedFaceData       = faceDataStream;
        gpuTextureInfo[i].packedDataSize       = faceDataBitStreamSize;
        gpuTextureInfo[i].minLog2Dim           = minLog2Dim;
        gpuTextureInfo[i].numVirtualOffsetBits = numVirtualOffsetBits;
        gpuTextureInfo[i].numFaceDimBits       = numFaceDimBits;
        gpuTextureInfo[i].numFaceIDBits        = numFaceIDBits;
        gpuTextureInfo[i].virtualAddressIndex  = allocIndex;
    }

    tileCmd->SignalOutsideFrame(tileSubmitSemaphore);
    tileCmd->Barrier(&virtualTextureManager.gpuPhysicalPool,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    tileCmd->Barrier(&virtualTextureManager.pageTable,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    tileCmd->FlushBarriers();
    device->SubmitCommandBuffer(tileCmd);

    StaticArray<BLASSceneGPUInfo> blasSceneInfo(arena, blasScenes.Length());

    for (int i = 0; i < blasScenes.Length(); i++)
    {
        ScenePrimitives *scene       = blasScenes[i];
        scene->semaphore             = device->CreateSemaphore();
        scene->semaphore.signalValue = 1;

        BLASSceneGPUInfo info;
        info.data.Init();
        DenseGeometryBuildData &data = info.data;

        CommandBuffer *dgfTransferCmd = device->BeginCommandBuffer(QueueType_Copy);

        RecordAOSSplits record;
        PrimRef *refs = ParallelGenerateMeshRefs<GeometryType::TriangleMesh>(
            sceneScratch.temp.arena, scene, record, false);

        ClusterBuilder builder(arena, scene, refs);
        builder.BuildClusters(record, true);

        builder.CreateDGFs(scene, &data, (Mesh *)scene->primitives, scene->numPrimitives,
                           sceneBounds);

#ifdef USE_PROCEDURAL_CLUSTER_INTERSECTION
        // Convert primrefs to aabbs, submit to GPU

        StaticArray<VkAabbPositionsKHR> aabbs =
            CreateAABBForNTriangles(sceneScratch.temp.arena, builder, data.numBlocks);

        info.aabbLength = aabbs.Length();
        info.aabbBuffer =
            dgfTransferCmd
                ->SubmitBuffer(
                    aabbs.data,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    aabbs.Length() * sizeof(VkAabbPositionsKHR))
                .buffer;

        // Combine all blocks together
        info.dgfByteBuffer =
            PushArrayNoZero(sceneScratch.temp.arena, u8, data.byteBuffer.Length());
        data.byteBuffer.Flatten(info.dgfByteBuffer);
        info.dgfBuffer =
            dgfTransferCmd
                ->SubmitBuffer(info.dgfByteBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               data.byteBuffer.Length())
                .buffer;

        // Upload headers
        info.headers = PushArrayNoZero(sceneScratch.temp.arena, PackedDenseGeometryHeader,
                                       data.headers.Length());
        data.headers.Flatten(info.headers);
        info.numHeaders = data.headers.Length();
        info.headerBuffer =
            dgfTransferCmd
                ->SubmitBuffer(info.headers, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               sizeof(PackedDenseGeometryHeader) * data.headers.Length())
                .buffer;

        // Debug
        info.types =
            PushArrayNoZero(sceneScratch.temp.arena, TriangleStripType *, data.numBlocks);
        // firstUses    = PushArrayNoZero(sceneScratch.temp.arena, u32 *,
        // data.numBlocks); reuses       = PushArrayNoZero(sceneScratch.temp.arena, u32 *,
        // data.numBlocks);
        info.debugFaceIDs = PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        info.debugIndices = PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        info.debugRestartHighBit =
            PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        info.debugRestartCount =
            PushArrayNoZero(sceneScratch.temp.arena, u32 *, data.numBlocks);
        info.positions = aabbs.data;
        int c          = 0;
        for (auto *node = data.types.first; node != 0; node = node->next)
        {
            info.types[c++] = node->values;
        }
        c = 0;
        for (auto *node = data.debugFaceIDs.first; node != 0; node = node->next)
        {
            info.debugFaceIDs[c++] = node->values;
        }
        // c = 0;
        // for (auto *node = data.firstUse.first; node != 0; node = node->next)
        // {
        //     firstUses[c++] = node->values;
        // }
        // c = 0;
        // for (auto *node = data.reuse.first; node != 0; node = node->next)
        // {
        //     reuses[c++] = node->values;
        // }
        c = 0;
        for (auto *node = data.debugIndices.first; node != 0; node = node->next)
        {
            info.debugIndices[c++] = node->values;
        }
        c = 0;
        for (auto *node = data.debugRestartHighBitPerDword.first; node != 0; node = node->next)
        {
            info.debugRestartHighBit[c++] = node->values;
        }
        c = 0;
        for (auto *node = data.debugRestartCountPerDword.first; node != 0; node = node->next)
        {
            info.debugRestartCount[c++] = node->values;
        }
#endif

        dgfTransferCmd->SignalOutsideFrame(scene->semaphore);
        device->SetName(&info.dgfBuffer, "DGF Buffer");
        device->SubmitCommandBuffer(dgfTransferCmd);

        blasSceneInfo.Push(info);
        // Send offsets buffer and dense geometry info
        ReleaseArenaArray(builder.arenas);
    }

#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
    CommandBuffer *computeCmd = device->BeginCommandBuffer(QueueType_Compute);
    computeCmd->WaitOn(dgfTransferCmd);

    device->SetName(&dgfBuffer, "DGF Buffer");
    device->SubmitCommandBuffer(dgfTransferCmd);

    // Decode layout pipeline description
    u32 buildClasDescsSize = sizeof(BuildClasDesc) * totalNumClusters;

    GPUBuffer indexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        megabytes(256));
    GPUBuffer vertexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        megabytes(256));
    GPUBuffer buildClasDescs = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        buildClasDescsSize);
    GPUBuffer globalsDesc = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * 4);
    GPUBuffer clusterData = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(ClusterData) * totalNumClusters);

    // Decode clusters and set up clas calls
    {
        computeCmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, decodePipeline);
        DescriptorSet descriptorSet = decodeLayout.CreateDescriptorSet();
        descriptorSet.Bind(indexBufferBinding, &indexBuffer)
            .Bind(vertexBufferBinding, &vertexBuffer)
            .Bind(buildClasDescsBinding, &buildClasDescs)
            .Bind(decodeGlobalsBinding, &globalsDesc)
            .Bind(clusterDataBinding, &clusterData)
            .Bind(denseGeometryBufferBinding, &dgfBuffer)
            .Bind(packedDenseGeometryHeaderBufferBinding, &headerBuffer);

        u64 indexBufferAddress  = device->GetDeviceAddress(indexBuffer.buffer);
        u64 vertexBufferAddress = device->GetDeviceAddress(vertexBuffer.buffer);

        DecodePushConstant pc;
        pc.numHeaders                      = numHeaders;
        pc.indexBufferBaseAddressLowBits   = indexBufferAddress & 0xffffffff;
        pc.indexBufferBaseAddressHighBits  = (indexBufferAddress >> 32u) & 0xffffffff;
        pc.vertexBufferBaseAddressLowBits  = vertexBufferAddress & 0xffffffff;
        pc.vertexBufferBaseAddressHighBits = (vertexBufferAddress >> 32u) & 0xffffffff;

        computeCmd->PushConstants(&decodeConstants, &pc, decodeLayout.pipelineLayout);
        computeCmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &descriptorSet,
                                       decodeLayout.pipelineLayout);
        computeCmd->Dispatch(totalNumClusters, 1, 1);

        computeCmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                            VK_ACCESS_2_SHADER_WRITE_BIT,
                            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        computeCmd->FlushBarriers();
    }

    // Build the CLAS
    GPUBuffer clusterAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * totalNumClusters);
    GPUBuffer clusterAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * totalNumClusters);
    {
        computeCmd->BuildCLAS(&clusterAccelAddresses, &clusterAccelSizes, &clusterAccelSizes,
                              &globalsDesc, sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX,
                              &GetScene()->scene, totalNumClusters, numTriangles, numVertices);
    }

    // TODO: Compact the CLAS over BLAS
    {
    }

    // Set up bottom level input
    u32 numBlas                  = 1;
    GPUBuffer blasAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizeof(u64) * numBlas);

    ClusterBottomLevelInfo bli = {};
    bli.clusterReferencesCount = totalNumClusters;
    bli.clusterReferences      = device->GetDeviceAddress(clusterAccelAddresses.buffer);

    GPUBuffer bottomLevelInfo = computeCmd->SubmitBuffer(
        &bli,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizeof(ClusterBottomLevelInfo) * numBlas);

    // Build the BLAS over CLAS
    {
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        computeCmd->FlushBarriers();
        computeCmd->BuildClusterBLAS(&bottomLevelInfo, &blasAccelAddresses, &globalsDesc,
                                     sizeof(u32) * GLOBALS_BLAS_COUNT_INDEX, totalNumClusters);
    }

    // Build the TLAS over BLAS
    GPUAccelerationStructure tlas = {};
    {
        // First fill out instance info on GPU
        Instance instance          = {};
        ScenePrimitives *baseScene = &GetScene()->scene;
        GPUBuffer instanceData     = computeCmd->CreateTLASInstances(
            &instance, 1, &params->renderFromWorld, &baseScene);

        computeCmd->Barrier(&instanceData, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                            VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->FlushBarriers();

        computeCmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, fillInstancePipeline);
        DescriptorSet descriptorSet = fillLayout.CreateDescriptorSet();
        descriptorSet.Bind(addressesBinding, &blasAccelAddresses)
            .Bind(instancesBinding, &instanceData);

        computeCmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &descriptorSet,
                                       fillLayout.pipelineLayout);
        computeCmd->Dispatch(1, 1, 1);

        // Build the TLAS
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                            VK_ACCESS_2_SHADER_WRITE_BIT,
                            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        computeCmd->FlushBarriers();
        tlas = computeCmd->BuildTLAS(&instanceData, 1);
    }
    Semaphore tlasSemaphore   = device->CreateSemaphore();
    tlasSemaphore.signalValue = 1;
    computeCmd->Signal(tlasSemaphore);
    device->SubmitCommandBuffer(computeCmd);

#endif

    Swapchain swapchain = device->CreateSwapchain(params->window, VK_FORMAT_R8G8B8A8_SRGB,
                                                  params->width, params->height);

    PushConstant pushConstant;
    pushConstant.stage  = ShaderStage::Compute; // ShaderStage::Raygen | ShaderStage::Miss;
    pushConstant.offset = 0;
    pushConstant.size   = sizeof(RayPushConstant);

    Semaphore submitSemaphore = device->CreateSemaphore();
    // Transfer data to GPU
    GPUScene gpuScene;
    gpuScene.cameraFromRaster = params->cameraFromRaster;
    gpuScene.renderFromCamera = params->renderFromCamera;
    gpuScene.lightFromRender  = params->lightFromRender;
    gpuScene.dxCamera         = params->dxCamera;
    gpuScene.lensRadius       = params->lensRadius;
    gpuScene.dyCamera         = params->dyCamera;
    gpuScene.focalLength      = params->focalLength;
    gpuScene.height           = params->height;
    gpuScene.fov              = params->fov;

    ShaderDebugInfo shaderDebug;

    RTBindingData bindingData;
    bindingData.materialIndex = 0;

    CommandBuffer *transferCmd = device->BeginCommandBuffer(QueueType_Copy);
    GPUBuffer bindingDataBuffer =
        transferCmd
            ->SubmitBuffer(&bindingData, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(RTBindingData))
            .buffer;

    ImageDesc gpuEnvMapDesc(ImageType::Type2D, envMap->width, envMap->height, 1, 1, 1,
                            VK_FORMAT_R8G8B8A8_SRGB, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_TILING_OPTIMAL);

    GPUImage gpuEnvMap = transferCmd->SubmitImage(envMap->contents, gpuEnvMapDesc).image;
    transferCmd->Barrier(&gpuEnvMap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transferCmd->FlushBarriers();

    submitSemaphore.signalValue = 1;
    transferCmd->SignalOutsideFrame(submitSemaphore);

    ImageDesc targetUavDesc(ImageType::Type2D, params->width, params->height, 1, 1, 1,
                            VK_FORMAT_R16G16B16A16_SFLOAT, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_IMAGE_TILING_OPTIMAL);
    GPUImage images[2] = {
        device->CreateImage(targetUavDesc),
        device->CreateImage(targetUavDesc),
    };

    // Create descriptor set layout and pipeline
    // VkShaderStageFlags flags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
    // |
    //                            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    //                            VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    VkShaderStageFlags flags   = VK_SHADER_STAGE_COMPUTE_BIT;
    DescriptorSetLayout layout = {};
    layout.pipelineLayout      = nullptr;
    int accelBindingIndex      = layout.AddBinding((u32)RTBindings::Accel,
                                                   DescriptorType::AccelerationStructure, flags);
    int imageBindingIndex =
        layout.AddBinding((u32)RTBindings::Image, DescriptorType::StorageImage, flags);

    int sceneBindingIndex =
        layout.AddBinding((u32)RTBindings::Scene, DescriptorType::UniformBuffer, flags);

    int bindingDataBindingIndex = layout.AddBinding((u32)RTBindings::RTBindingData,
                                                    DescriptorType::StorageBuffer, flags);

    int gpuMaterialBindingIndex =
        layout.AddBinding((u32)RTBindings::GPUMaterial, DescriptorType::StorageBuffer, flags);

    int pageTableBindingIndex =
        layout.AddBinding((u32)RTBindings::PageTable, DescriptorType::SampledImage, flags);

    int physicalPagesBindingIndex =
        layout.AddBinding((u32)RTBindings::PhysicalPages, DescriptorType::SampledImage, flags);

    int shaderDebugIndex = layout.AddBinding((u32)RTBindings::ShaderDebugInfo,
                                             DescriptorType::UniformBuffer, flags);

    // int clusterDataIndex = layout.AddBinding(8, DescriptorType::StorageBuffer, flags);
    // int vertexDataIndex  = layout.AddBinding(9, DescriptorType::StorageBuffer, flags);
    // int indexDataIndex   = layout.AddBinding(10, DescriptorType::StorageBuffer, flags);

    int feedbackBufferIndex = layout.AddBinding(11, DescriptorType::StorageBuffer, flags);

    // TODO: what is this?
    int counterIndex = layout.AddBinding(8, DescriptorType::StorageBuffer, flags);
    int nvApiIndex =
        layout.AddBinding(NVAPI_SLOT, DescriptorType::StorageBuffer, VK_SHADER_STAGE_ALL);

    layout.AddImmutableSamplers();

    // RayTracingState rts = device->CreateRayTracingPipeline(groups, ArrayLength(groups),
    //                                                        &pushConstant, &layout, 2, true);
    VkPipeline pipeline = device->CreateComputePipeline(&shader, &layout, &pushConstant);

#ifdef USE_PROCEDURAL_CLUSTER_INTERSECTION
    Semaphore tlasSemaphore = device->CreateSemaphore();

    GPUAccelerationStructure tlas = {};
    // TODO: new command buffers have to wait on ones from the previous depth
    // also this doesn't work if there's actually a nontrivial TLAS

    CommandBuffer *allCommandBuffer = device->BeginCommandBuffer(QueueType_Graphics);

    StaticArray<GPUAccelerationStructurePayload> uncompactedPayloads(sceneScratch.temp.arena,
                                                                     blasScenes.Length());
    for (int i = 0; i < blasScenes.Length(); i++)
    {
        ScratchArena scratch;
        ScenePrimitives *scene = blasScenes[i];

        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics);
        device->BeginEvent(cmd, "BLAS Build");
        int bvhCount = 0;

        Semaphore semaphore   = device->CreateSemaphore();
        semaphore.signalValue = 1;
        cmd->Wait(scene->semaphore);
        cmd->SignalOutsideFrame(semaphore);

        GPUAccelerationStructurePayload payload =
            cmd->BuildCustomBLAS(&blasSceneInfo[i].aabbBuffer, blasSceneInfo[i].aabbLength);
        uncompactedPayloads.Push(payload);

        // Compact
        QueryPool queryPool = cmd->GetCompactionSizes(&payload);
        device->SubmitCommandBuffer(cmd);
        device->EndEvent(cmd);

        allCommandBuffer->Wait(semaphore);
        semaphore.signalValue = 2;
        allCommandBuffer->Signal(semaphore);
        scene->gpuBVH = allCommandBuffer->CompactAS(queryPool, &payload);

        device->DestroyBuffer(&payload.scratch);
    }

    // Set the instance ids of each scene
    int runningTotal = 0;
    for (int i = 0; i < blasScenes.Length(); i++)
    {
        ScenePrimitives *scene = blasScenes[i];
        int bindlessHeaders    = device->BindlessStorageIndex(&blasSceneInfo[i].headerBuffer);
        Assert(bindlessHeaders == 2 * runningTotal);

        int bindlessData = device->BindlessStorageIndex(&blasSceneInfo[i].dgfBuffer);
        Assert(bindlessData == 2 * runningTotal + 1);
        scene->gpuInstanceID = runningTotal++;
    }

    // Populate GPU materials
    StaticArray<GPUMaterial> gpuMaterials(sceneScratch.temp.arena,
                                          rootScene->materials.Length());
    for (int i = 0; i < rootScene->materials.Length(); i++)
    {
        GPUMaterial material = rootScene->materials[i]->ConvertToGPU();
        int index            = rootScene->materials[i]->ptexReflectanceIndex;
        if (index != -1)
        {
            GPUTextureInfo &textureInfo = gpuTextureInfo[index];
            if (textureInfo.numVirtualOffsetBits != 0)
            {
                VirtualTextureManager::TextureInfo &texInfo =
                    virtualTextureManager.textureInfo[textureInfo.virtualAddressIndex];
                material.baseVirtualPage      = texInfo.baseVirtualPage;
                material.textureIndex         = textureInfo.virtualAddressIndex;
                material.minLog2Dim           = textureInfo.minLog2Dim;
                material.numVirtualOffsetBits = textureInfo.numVirtualOffsetBits;
                material.numFaceDimBits       = textureInfo.numFaceDimBits;
                material.numFaceIDBits        = textureInfo.numFaceIDBits;

                TransferBuffer faceDataBuffer = transferCmd->SubmitBuffer(
                    textureInfo.packedFaceData, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    textureInfo.packedDataSize);
                int bindlessFaceDataIndex =
                    device->BindlessStorageIndex(&faceDataBuffer.buffer);

                material.bindlessFaceDataIndex = bindlessFaceDataIndex;
            }
        }
        gpuMaterials.Push(material);
    }
    GPUBuffer materialBuffer =
        transferCmd
            ->SubmitBuffer(gpuMaterials.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(GPUMaterial) * gpuMaterials.Length())
            .buffer;

    if (tlasScenes.Length() != 0)
    {
        Assert(tlasScenes.Length() == 1);
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        for (int i = 0; i < tlasScenes.Length(); i++)
        {
            ScenePrimitives *scene = tlasScenes[i];
            device->BeginEvent(allCommandBuffer, "TLAS Build");
            Instance *instances                     = (Instance *)scene->primitives;
            GPUAccelerationStructurePayload payload = allCommandBuffer->BuildTLAS(
                instances, scene->numPrimitives, scene->affineTransforms, scene->childScenes);

            scene->gpuBVH = payload.as;
            tlas          = scene->gpuBVH;

            tlasSemaphore.signalValue = 1;
            allCommandBuffer->SignalOutsideFrame(tlasSemaphore);
        }
        allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                  VK_ACCESS_2_SHADER_READ_BIT);
        device->SubmitCommandBuffer(allCommandBuffer);
    }
    else
    {
        Assert(numScenes == 1);

        ScenePrimitives *scene = blasScenes[0];
        allCommandBuffer->Wait(scene->semaphore);
        tlasSemaphore.signalValue = 1;
        allCommandBuffer->SignalOutsideFrame(tlasSemaphore);
        Instance instance          = {};
        ScenePrimitives *baseScene = &GetScene()->scene;
        GPUBuffer tlasBuffer =
            allCommandBuffer
                ->CreateTLASInstances(&instance, 1, &params->renderFromWorld, &baseScene)
                .buffer;
        allCommandBuffer->Barrier(&tlasBuffer,
                                  VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                  VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        allCommandBuffer->FlushBarriers();
        tlas = allCommandBuffer->BuildTLAS(&tlasBuffer, 1).as;

        device->SubmitCommandBuffer(allCommandBuffer);
    }

    device->SubmitCommandBuffer(transferCmd);

#endif

    f32 frameDt = 1.f / 60.f;
    int envMapBindlessIndex;

    ViewCamera camera = {};
    camera.position   = params->pCamera;
    camera.forward    = Normalize(params->look - params->pCamera);
    camera.right      = Normalize(Cross(camera.forward, params->up));

    camera.pitch = ArcSin(camera.forward.y);
    camera.yaw   = -Atan2(camera.forward.z, camera.forward.x);

    Vec3f baseForward = camera.forward;
    Vec3f baseRight   = camera.right;

    TransferBuffer sceneTransferBuffers[2] = {
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene)),
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene))};

    TransferBuffer shaderDebugBuffers[2] = {
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene)),
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene))};

    bool mousePressed = false;
    OS_Key keys[4]    = {
        OS_Key_D,
        OS_Key_A,
        OS_Key_W,
        OS_Key_S,
    };
    int dir[4] = {};

    Vec3f cameraStart = params->pCamera;

    GPUBuffer counterBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u32));
    GPUBuffer nvapiBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 256);

    for (;;)
    {
        ScratchArena frameScratch;

        MSG message;
        while (PeekMessageW(&message, 0, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        f32 frameTime = OS_NowSeconds();

        ChunkedLinkedList<OS_Event> &events = OS_GetEvents();

        Vec2i dMouseP(0.f);

        for (auto *node = events.first; node != 0; node = node->next)
        {
            for (int i = 0; i < node->count; i++)
            {
                if (node->values[i].key == OS_Mouse_R)
                {
                    mousePressed = node->values[i].type == OS_EventType_KeyPressed;
                    dMouseP[0]   = node->values[i].mouseMoveX;
                    dMouseP[1]   = node->values[i].mouseMoveY;
                }
                else if (node->values[i].type == OS_EventType_MouseMove)
                {
                    dMouseP[0] = node->values[i].mouseMoveX;
                    dMouseP[1] = node->values[i].mouseMoveY;
                }
                for (int keyIndex = 0; keyIndex < ArrayLength(keys); keyIndex++)
                {
                    if (node->values[i].key == keys[keyIndex])
                    {
                        dir[keyIndex] = node->values[i].type == OS_EventType_KeyPressed;
                    }
                }
            }
        }

        if (!mousePressed)
        {
            dMouseP[0] = 0;
            dMouseP[1] = 0;
            MemoryZero(dir, sizeof(dir));
        }

        events.Clear();

        // Input
        {
            f32 speed = 5000.f;

            f32 rotationSpeed = 0.001f * PI;
            camera.RotateCamera(dMouseP, rotationSpeed);

            camera.position += (dir[2] - dir[3]) * camera.forward * speed * frameDt;
            camera.position += (dir[0] - dir[1]) * camera.right * speed * frameDt;

            camera.forward.x = Cos(camera.yaw) * Cos(camera.pitch);
            camera.forward.y = Sin(camera.pitch);
            camera.forward.z = -Sin(camera.yaw) * Cos(camera.pitch);
            camera.forward   = Normalize(camera.forward);

            camera.right = Normalize(Cross(camera.forward, params->up));

            LinearSpace3f axis;
            axis.e[2] = -camera.forward;
            f32 d = camera.forward.x * camera.forward.x + camera.forward.z * camera.forward.z;
            Assert(d != 0);
            if (d == 0)
            {
                axis.e[0][0] = 1.f;
                axis.e[0][1] = 0.f;
                axis.e[0][2] = 0.f;
            }
            else
            {
                f32 invSqrt  = 1 / Sqrt(d);
                axis.e[0][0] = -camera.forward.z * invSqrt;
                axis.e[0][1] = 0.f;
                axis.e[0][2] = camera.forward.x * invSqrt;
            }
            axis.e[1] = Cross(axis.e[2], axis.e[0]);
            axis      = Transpose(axis);

            AffineSpace cameraFromRender =
                AffineSpace(axis, Vec3f(0)) * Translate(cameraStart - camera.position);

            gpuScene.renderFromCamera = Inverse(cameraFromRender);
            OS_GetMousePos(params->window, shaderDebug.mousePos.x, shaderDebug.mousePos.y);
        }
        u32 dispatchDimX =
            (params->width + PATH_TRACE_NUM_THREADS_X - 1) / PATH_TRACE_NUM_THREADS_X;
        u32 dispatchDimY =
            (params->height + PATH_TRACE_NUM_THREADS_Y - 1) / PATH_TRACE_NUM_THREADS_Y;
        gpuScene.dispatchDimX = dispatchDimX;
        gpuScene.dispatchDimY = dispatchDimY;

        device->BeginFrame();

        u32 frame       = device->GetCurrentBuffer();
        GPUImage *image = &images[frame];
        string cmdBufferName =
            PushStr8F(frameScratch.temp.arena, "Graphics Cmd %u", device->frameCount);
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics, cmdBufferName);
        debugState.BeginFrame(cmd);

        if (device->frameCount == 0)
        {
            cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            cmd->Wait(submitSemaphore);
            cmd->Wait(tileSubmitSemaphore);
            device->Wait(tlasSemaphore);

            for (int i = 0; i < uncompactedPayloads.Length(); i++)
            {
                auto &payload = uncompactedPayloads[i];
                device->DestroyAccelerationStructure(&payload.as);
            }
            for (int i = 0; i < blasSceneInfo.Length(); i++)
            {
                device->DestroyBuffer(&blasSceneInfo[i].aabbBuffer);
            }
            envMapBindlessIndex = device->BindlessIndex(&gpuEnvMap);
        }

        u32 currentBuffer = device->GetCurrentBuffer();

        // Virtual texture system
        cmdBufferName = PushStr8F(frameScratch.temp.arena, "Virtual Texture Async Copy Cmd %u",
                                  device->frameCount);
        CommandBuffer *virtualTextureCopyCmd =
            device->BeginCommandBuffer(QueueType_Copy, cmdBufferName);
        cmdBufferName = PushStr8F(frameScratch.temp.arena, "Virtual Texture Transition Cmd %u",
                                  device->frameCount);
        CommandBuffer *transitionCmd =
            device->BeginCommandBuffer(QueueType_Graphics, cmdBufferName);
        virtualTextureManager.Update(cmd, virtualTextureCopyCmd, transitionCmd,
                                     QueueType_Graphics);
        device->SubmitCommandBuffer(transitionCmd);
        device->SubmitCommandBuffer(virtualTextureCopyCmd);

        cmd->ClearBuffer(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer);

        RayPushConstant pc;
        pc.envMap   = envMapBindlessIndex;
        pc.frameNum = (u32)device->frameCount;
        pc.width    = envMap->width;
        pc.height   = envMap->height;

        MemoryCopy(sceneTransferBuffers[currentBuffer].mappedPtr, &gpuScene, sizeof(GPUScene));
        cmd->SubmitTransfer(&sceneTransferBuffers[currentBuffer]);
        MemoryCopy(shaderDebugBuffers[currentBuffer].mappedPtr, &shaderDebug,
                   sizeof(ShaderDebugInfo));
        cmd->SubmitTransfer(&shaderDebugBuffers[currentBuffer]);

        VkPipelineStageFlags2 flags   = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
        cmd->Barrier(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer, flags,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->Barrier(&sceneTransferBuffers[currentBuffer].buffer, flags,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->Barrier(&shaderDebugBuffers[currentBuffer].buffer, flags,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->Barrier(image, VK_IMAGE_LAYOUT_GENERAL, flags, VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->FlushBarriers();
        cmd->BindPipeline(bindPoint, pipeline); // rts.pipeline);
        DescriptorSet descriptorSet = layout.CreateDescriptorSet();
        descriptorSet.Bind(accelBindingIndex, &tlas.as)
            .Bind(imageBindingIndex, image)
            .Bind(sceneBindingIndex, &sceneTransferBuffers[currentBuffer].buffer)
            .Bind(bindingDataBindingIndex, &bindingDataBuffer)
            .Bind(gpuMaterialBindingIndex, &materialBuffer)
            .Bind(shaderDebugIndex, &shaderDebugBuffers[currentBuffer].buffer)
            .Bind(pageTableBindingIndex, &virtualTextureManager.pageTable)
            .Bind(physicalPagesBindingIndex, &virtualTextureManager.gpuPhysicalPool)
            .Bind(feedbackBufferIndex,
                  &virtualTextureManager.feedbackBuffers[currentBuffer].buffer)
            .Bind(counterIndex, &counterBuffer)
            .Bind(nvApiIndex, &nvapiBuffer);
#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
        .Bind(clusterDataIndex, &clusterData)
            .Bind(vertexDataIndex, &vertexBuffer)
            .Bind(indexDataIndex, &indexBuffer);
#endif
        // .Bind(aabbIndex, &aabbBuffer);

        // cmd->BindDescriptorSets(bindPoint, &descriptorSet, rts.layout);
        cmd->BindDescriptorSets(bindPoint, &descriptorSet, layout.pipelineLayout);

        cmd->PushConstants(&pushConstant, &pc, /*rts.layout);*/ layout.pipelineLayout);
        // cmd->TraceRays(&rts, params->width, params->height, 1);

        int beginIndex = TIMED_GPU_RANGE_BEGIN(cmd, "ray trace");
        cmd->Dispatch(dispatchDimX, dispatchDimY, 1);
        TIMED_RANGE_END(beginIndex);

        // Copy feedback from device to host
        CommandBuffer *transferCmd =
            device->BeginCommandBuffer(QueueType_Copy, "feedback copy cmd");
        transferCmd->WaitOn(cmd);
        transferCmd->CopyBuffer(
            &virtualTextureManager.feedbackBuffers[currentBuffer].stagingBuffer,
            &virtualTextureManager.feedbackBuffers[currentBuffer].buffer);
        device->SubmitCommandBuffer(transferCmd, true);

        debugState.EndFrame(cmd);
        device->CopyFrameBuffer(&swapchain, cmd, image);
        device->EndFrame(QueueFlag_Copy | QueueFlag_Graphics);

        debugState.PrintDebugRecords();

        // Wait until new update
        f32 endWorkFrameTime = OS_NowSeconds();
        f32 timeElapsed      = endWorkFrameTime - frameTime;

        if (timeElapsed < frameDt)
        {
            u32 msTimeToSleep = (u32)(1000.f * (frameDt - timeElapsed));
            if (msTimeToSleep > 0)
            {
                OS_Sleep(msTimeToSleep);
            }
        }

        while (timeElapsed < frameDt)
        {
            timeElapsed = OS_NowSeconds() - frameTime;
        }
    }
}
} // namespace rt
