#include "base.h"
#include "bit_packing.h"
#include "camera.h"
#include "debug.h"
#include "graphics/render_graph.h"
#include "graphics/vulkan.h"
#include "integrate.h"
#include "gpu_scene.h"
#include "lights.h"
#include "math/simd_base.h"
#include "radix_sort.h"
#include "random.h"
#include "shader_interop/hierarchy_traversal_shaderinterop.h"
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
#include "scene/scene.h"
#include "win32.h"
#include "graphics/ptex.h"
#include "virtual_geometry/virtual_geometry_manager.h"
#include "photon_mapping/photon_mapping.h"
#include "volume.h"

#include "gpu/path_guiding.h"

// TODO: temp
#include "gpu/cuda/cuda_device.h"

#include "../../third_party/streamline/include/sl.h"
#include "../../third_party/oidn/include/OpenImageDenoise/oidn.hpp"

namespace rt
{

using PFun_oidnNewDevice    = OIDNDevice(OIDNDeviceType deviceType);
using PFun_oidnCommitDevice = void(OIDNDevice device);
using PFun_oidnGetDeviceInt = int(OIDNDevice device, const char *name);
using PFun_oidnNewFilter    = OIDNFilter(OIDNDevice device, const char *name);
using PFun_oidnNewSharedBufferFromWin32Handle =
    OIDNBuffer(__cdecl)(OIDNDevice device, OIDNExternalMemoryTypeFlag flag, void *handle,
                        const void *name, size_t byteSize);
using PFun_oidnSetFilterImage    = void(OIDNFilter filter, const char *name, OIDNBuffer buffer,
                                     OIDNFormat format, size_t width, size_t height,
                                     size_t byteOffset, size_t pixelByteStride,
                                     size_t rowByteStride);
using PFun_oidnCommitFilter      = void(OIDNFilter filter);
using PFun_oidnExecuteFilter     = void(OIDNFilter filter);
using PFun_oidnSetFilterBool     = void(OIDNFilter filter, const char *name, bool value);
using PFun_oidnSetFilterInt      = void(OIDNFilter filter, const char *name, int value);
using PFun_oidnReleaseBuffer     = void(OIDNBuffer buffer);
using PFun_oidnRemoveFilterImage = void(OIDNFilter filter, const char *name);
using PFun_oidnUpdateFilterData  = void(OIDNFilter filter, const char *name);

static void FlattenInstances(ScenePrimitives *currentScene, const AffineSpace &transform,
                             Array<Instance> &instances, Array<AffineSpace> &transforms)
{
    Instance *sceneInstances = (Instance *)currentScene->primitives;
    for (u32 i = 0; i < currentScene->numPrimitives; i++)
    {
        Instance &instance = sceneInstances[i];
        AffineSpace newTransform =
            transform * currentScene->affineTransforms[instance.transformIndex];
        ScenePrimitives *nextScene = currentScene->childScenes[instance.id];

        if (nextScene->geometryType == GeometryType::Instance)
        {
            FlattenInstances(nextScene, newTransform, instances, transforms);
        }
        else
        {
            u32 resourceID          = nextScene->sceneIndex;
            Instance instance       = {};
            instance.transformIndex = transforms.Length();
            instance.id             = resourceID;

            transforms.Push(newTransform);
            instances.Push(instance);
        }
    }
}

// https://alextardif.com/TAA.html
static float Halton(uint32_t i, uint32_t b)
{
    float f = 1.0f;
    float r = 0.0f;

    while (i > 0)
    {
        f /= static_cast<float>(b);
        r = r + f * static_cast<float>(i % b);
        i = static_cast<uint32_t>(floorf(static_cast<float>(i) / static_cast<float>(b)));
    }

    return r;
}

static float Mitchell1D(float x)
{
    const float b = 1.f / 3.f;
    const float c = 1.f / 3.f;
    x             = Abs(x);
    if (x <= 1)
        return ((12 - 9 * b - 6 * c) * x * x * x + (-18 + 12 * b + 6 * c) * x * x +
                (6 - 2 * b)) *
               (1.f / 6.f);
    else if (x <= 2)
        return ((-b - 6 * c) * x * x * x + (6 * b + 30 * c) * x * x + (-12 * b - 48 * c) * x +
                (8 * b + 24 * c)) *
               (1.f / 6.f);
    else return 0;
}

static float MitchellEvaluate(Vec2f p, Vec2f radius)
{
    float result = Mitchell1D(2.f * p.x / radius.x) * Mitchell1D(2.f * p.y / radius.y);
    return result;
}

static float Gaussian(float x, float mu = 0, float sigma = 1)
{
    return 1 / Sqrt(2 * PI * sigma * sigma) * FastExp(-Sqr(x - mu) / (2 * sigma * sigma));
}

static float GaussianEvaluate(Vec2f p, float sigma, float expX, float expY)
{
    return (Max(0.f, Gaussian(p.x, 0, sigma) - expX) *
            Max(0.f, Gaussian(p.y, 0, sigma) - expY));
}

void Render(RenderParams2 *params, int numScenes, Image *envMap)
{
    CUDADevice cudDevice;
    string path = "../src/rt/gpu/kernel.cubin";
    cudDevice.RegisterModule(path);
    PathGuiding guiding(&cudDevice);
    guiding.Update();

    Arena *arena             = params->arenas[GetThreadIndex()];
    ScenePrimitives **scenes = GetScenes();

    PerformanceCounter counter = OS_StartCounter();
    // Compile shaders
    Shader prepareIndirectShader;

    Shader testShader;
    Shader mvecShader;
    Shader accumulateShader;
    Shader copyDenoisedShader;

    RayTracingShaderGroup groups[3];
    {
        string raygenShaderName = "../src/shaders/render_raytrace_rgen.spv";
        string missShaderName   = "../src/shaders/render_raytrace_miss.spv";
        string hitShaderName    = "../src/shaders/render_raytrace_hit.spv";

        string rgenData = OS_ReadFile(arena, raygenShaderName);
        string missData = OS_ReadFile(arena, missShaderName);
        string hitData  = OS_ReadFile(arena, hitShaderName);

        Shader raygenShader = device->CreateShader(ShaderStage::Raygen, "raygen", rgenData);
        Shader missShader   = device->CreateShader(ShaderStage::Miss, "miss", missData);
        Shader hitShader    = device->CreateShader(ShaderStage::Hit, "hit", hitData);

        groups[0].shaders[0] = raygenShader;
        groups[0].numShaders = 1;
        groups[0].stage[0]   = ShaderStage::Raygen;

        groups[1].shaders[0] = missShader;
        groups[1].numShaders = 1;
        groups[1].stage[0]   = ShaderStage::Miss;

        groups[2].shaders[0] = hitShader;
        groups[2].stage[0]   = ShaderStage::Hit;
        groups[2].numShaders = 1;
        groups[2].type       = RayTracingShaderGroupType::Triangle;

        string prepareIndirectName = "../src/shaders/prepare_indirect.spv";
        string prepareIndirectData = OS_ReadFile(arena, prepareIndirectName);
        prepareIndirectShader = device->CreateShader(ShaderStage::Compute, "prepare indirect",
                                                     prepareIndirectData);

        string testShaderName = "../src/shaders/test.spv";
        string testShaderData = OS_ReadFile(arena, testShaderName);
        testShader = device->CreateShader(ShaderStage::Compute, "test shader", testShaderData);

        string mvecShaderName = "../src/shaders/calculate_motion_vectors.spv";
        string mvecShaderData = OS_ReadFile(arena, mvecShaderName);
        mvecShader = device->CreateShader(ShaderStage::Compute, "mvec shader", mvecShaderData);

        string accumulateShaderName = "../src/shaders/accumulate_frames.spv";
        string accumulateShaderData = OS_ReadFile(arena, accumulateShaderName);
        accumulateShader = device->CreateShader(ShaderStage::Compute, "accumulate shader",
                                                accumulateShaderData);

        string copyDenoisedShaderName = "../src/shaders/copy_denoised_output.spv";
        string copyDenoisedShaderData = OS_ReadFile(arena, copyDenoisedShaderName);
        copyDenoisedShader = device->CreateShader(ShaderStage::Compute, "copy denoised",
                                                  copyDenoisedShaderData);
    }

    // Compile pipelines

    // prepare indirect
    DescriptorSetLayout prepareIndirectLayout = {};
    prepareIndirectLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);
    VkPipeline prepareIndirectPipeline = device->CreateComputePipeline(
        &prepareIndirectShader, &prepareIndirectLayout, 0, "prepare indirect");

    DescriptorSetLayout testLayout = {};
    testLayout.AddBinding(0, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    testLayout.AddBinding(1, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    testLayout.AddBinding((u32)RTBindings::ClusterPageData, DescriptorType::StorageBuffer,
                          VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline testPipeline =
        device->CreateComputePipeline(&testShader, &testLayout, 0, "test");

    DescriptorSetLayout mvecLayout = {};
    mvecLayout.AddBinding(0, DescriptorType::SampledImage, VK_SHADER_STAGE_COMPUTE_BIT);
    mvecLayout.AddBinding(1, DescriptorType::StorageImage, VK_SHADER_STAGE_COMPUTE_BIT);
    mvecLayout.AddBinding(2, DescriptorType::UniformBuffer, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline mvecPipeline =
        device->CreateComputePipeline(&mvecShader, &mvecLayout, 0, "mvec");

    DescriptorSetLayout accumulateLayout = {};
    accumulateLayout.AddBinding(0, DescriptorType::SampledImage, VK_SHADER_STAGE_COMPUTE_BIT);
    accumulateLayout.AddBinding(1, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    accumulateLayout.AddBinding(2, DescriptorType::SampledImage, VK_SHADER_STAGE_COMPUTE_BIT);
    accumulateLayout.AddBinding(3, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);

    PushConstant accumulatePC;
    accumulatePC.size   = sizeof(NumPushConstant);
    accumulatePC.offset = 0;
    accumulatePC.stage  = ShaderStage::Compute;

    VkPipeline accumulatePipeline = device->CreateComputePipeline(
        &accumulateShader, &accumulateLayout, &accumulatePC, "accumulate pipeline");

    DescriptorSetLayout copyDenoisedLayout = {};
    copyDenoisedLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    copyDenoisedLayout.AddBinding(1, DescriptorType::StorageImage,
                                  VK_SHADER_STAGE_COMPUTE_BIT);

    VkPipeline copyDenoisedPipeline = device->CreateComputePipeline(
        &copyDenoisedShader, &copyDenoisedLayout, 0, "copy denoised pipeline");

    // device->GetDLSSTargetDimensions(targetWidth, targetHeight);
    u32 targetWidth  = 2560;
    u32 targetHeight = 1440;
    // targetWidth  = 1920;
    // targetHeight = 1080;

    Swapchain swapchain = device->CreateSwapchain(params->window, VK_FORMAT_R8G8B8A8_SRGB,
                                                  params->width, params->height);

    PushConstant pushConstant;
    pushConstant.stage  = ShaderStage::Raygen | ShaderStage::Miss;
    pushConstant.offset = 0;
    pushConstant.size   = sizeof(RayPushConstant);

    Semaphore submitSemaphore = device->CreateSemaphore();

    Mat4 rasterFromNDC = Scale(Vec3f(f32(targetWidth), -f32(targetHeight), 1.f)) *
                         Scale(Vec3f(1.f / 2.f, 1.f / 2.f, 1.f)) *
                         Translate(Vec3f(1.f, -1.f, 0.f));
    Mat4 rasterFromCamera = rasterFromNDC * params->NDCFromCamera;
    Mat4 cameraFromRaster = Inverse(rasterFromCamera);

    // Transfer data to GPU
    GPUScene gpuScene;
    gpuScene.clipFromRender   = params->NDCFromCamera * params->cameraFromRender;
    gpuScene.cameraFromRaster = cameraFromRaster;
    gpuScene.renderFromCamera = params->renderFromCamera;
    gpuScene.cameraFromRender = params->cameraFromRender;
    gpuScene.lightFromRender  = params->lightFromRender;
    gpuScene.lodScale         = 0.5f * params->NDCFromCamera[1][1] * targetHeight;
    gpuScene.dxCamera         = params->dxCamera;
    gpuScene.lensRadius       = params->lensRadius;
    gpuScene.dyCamera         = params->dyCamera;
    gpuScene.focalLength      = params->focalLength;
    gpuScene.width            = targetWidth;
    gpuScene.height           = targetHeight;
    gpuScene.fov              = params->fov;
    gpuScene.p22              = params->NDCFromCamera[2][2];
    gpuScene.p23              = params->NDCFromCamera[3][2];
    gpuScene.cameraBase       = params->pCamera;

    ShaderDebugInfo shaderDebug;

    RenderGraph *graph = PushStructConstruct(arena, RenderGraph)();
    SetRenderGraph(graph);
    RenderGraph *rg = GetRenderGraph();

    CommandBuffer *transferCmd = device->BeginCommandBuffer(QueueType_Copy);

    ImageDesc gpuEnvMapDesc(ImageType::Type2D, envMap->width, envMap->height, 1, 1, 1,
                            VK_FORMAT_R8G8B8A8_SRGB, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_TILING_OPTIMAL);

    TransferBuffer gpuEnvMapTransfer =
        transferCmd->SubmitImage(envMap->contents, gpuEnvMapDesc);
    GPUImage gpuEnvMap = gpuEnvMapTransfer.image;
    ImageDesc depthBufferDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                              VK_FORMAT_R32_SFLOAT, MemoryUsage::GPU_ONLY,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_IMAGE_TILING_OPTIMAL);
    GPUImage depthBuffer = device->CreateImage(depthBufferDesc);
    ResourceHandle depthBufferHandle =
        rg->RegisterExternalResource("depth buffer", &depthBuffer);

    ImageDesc targetUavDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                            VK_FORMAT_R16G16B16A16_SFLOAT, MemoryUsage::GPU_ONLY,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle image = rg->CreateImageResource("target image", targetUavDesc);

    GPUBuffer accumulatedImage =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             12 * targetWidth * targetHeight, MemoryUsage::EXTERNAL);
    ResourceHandle accumulatedImageHandle =
        rg->RegisterExternalResource("accumulate", &accumulatedImage);

    // GPUBuffer imageOut =
    //     device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    //                          12 * targetWidth * targetHeight, MemoryUsage::EXTERNAL);
    // ResourceHandle imageOutHandle = rg->RegisterExternalResource("image out", &imageOut);
    u32 imageOutSize              = 12 * targetWidth * targetHeight;
    ResourceHandle imageOutHandle = rg->CreateBufferResource(
        "image out", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, imageOutSize, MemoryUsage::EXTERNAL);

    ImageDesc uavDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                      VK_FORMAT_R16G16B16A16_SFLOAT, MemoryUsage::GPU_ONLY,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle finalImageHandle = rg->CreateImageResource("final image", uavDesc);

    GPUBuffer normals =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             12 * targetWidth * targetHeight, MemoryUsage::EXTERNAL);
    ResourceHandle normalsHandle = rg->RegisterExternalResource("normals", &normals);
    // ResourceHandle normalsHandle =
    //     rg->CreateBufferResource("normals", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    //                              12 * targetWidth * targetHeight, MemoryUsage::EXTERNAL);

    ImageDesc albedoDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                         VK_FORMAT_R8G8B8A8_UNORM, MemoryUsage::GPU_ONLY,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle albedoHandle = rg->CreateImageResource("albedo", albedoDesc);

    GPUBuffer accumulatedAlbedo =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             12 * targetWidth * targetHeight, MemoryUsage::EXTERNAL);
    ResourceHandle accumulatedAlbedoHandle =
        rg->RegisterExternalResource("accumulated albedo", &accumulatedAlbedo);

    ImageDesc motionVectorDesc(ImageType::Type2D, targetWidth, targetHeight, 1, 1, 1,
                               VK_FORMAT_R32G32_SFLOAT, MemoryUsage::GPU_ONLY,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_TILING_OPTIMAL);
    ResourceHandle motionVectorBuffer =
        rg->CreateImageResource("motion vector image", motionVectorDesc);

    // Set up open image denoise
    HMODULE module = LoadLibraryA("../../src/third_party/oidn/bin/OpenImageDenoise.dll");
    Assert(module);

    PFun_oidnNewDevice *oidnNewDevice_p =
        (PFun_oidnNewDevice *)GetProcAddress(module, "oidnNewDevice");
    PFun_oidnCommitDevice *oidnCommitDevice_p =
        (PFun_oidnCommitDevice *)GetProcAddress(module, "oidnCommitDevice");
    PFun_oidnGetDeviceInt *oidnGetDeviceInt_p =
        (PFun_oidnGetDeviceInt *)GetProcAddress(module, "oidnGetDeviceInt");
    PFun_oidnNewFilter *oidnNewFilter_p =
        (PFun_oidnNewFilter *)GetProcAddress(module, "oidnNewFilter");
    PFun_oidnNewSharedBufferFromWin32Handle *oidnNewSharedBufferFromWin32Handle_p =
        (PFun_oidnNewSharedBufferFromWin32Handle *)GetProcAddress(
            module, "oidnNewSharedBufferFromWin32Handle");
    PFun_oidnSetFilterImage *oidnSetFilterImage_p =
        (PFun_oidnSetFilterImage *)GetProcAddress(module, "oidnSetFilterImage");
    PFun_oidnCommitFilter *oidnCommitFilter_p =
        (PFun_oidnCommitFilter *)GetProcAddress(module, "oidnCommitFilter");
    PFun_oidnExecuteFilter *oidnExecuteFilter_p =
        (PFun_oidnExecuteFilter *)GetProcAddress(module, "oidnExecuteFilter");
    PFun_oidnSetFilterBool *oidnSetFilterBool_p =
        (PFun_oidnSetFilterBool *)GetProcAddress(module, "oidnSetFilterBool");
    PFun_oidnSetFilterInt *oidnSetFilterInt_p =
        (PFun_oidnSetFilterInt *)GetProcAddress(module, "oidnSetFilterInt");
    PFun_oidnReleaseBuffer *oidnReleaseBuffer_p =
        (PFun_oidnReleaseBuffer *)GetProcAddress(module, "oidnReleaseBuffer");
    // PFun_oidnRemoveFilterImage *oidnRemoveFilterImage_p =
    //     (PFun_oidnRemoveFilterImage *)GetProcAddress(module, "oidnRemoveFilterImage");
    PFun_oidnUpdateFilterData *oidnUpdateFilterData_p =
        (PFun_oidnUpdateFilterData *)GetProcAddress(module, "oidnUpdateFilterData");

    OIDNDevice oidnDevice = oidnNewDevice_p(OIDN_DEVICE_TYPE_DEFAULT);
    oidnCommitDevice_p(oidnDevice);

    int externalMemoryTypes = oidnGetDeviceInt_p(oidnDevice, "externalMemoryTypes");
    int deviceType          = oidnGetDeviceInt_p(oidnDevice, "type");

    OIDNFilter filter = oidnNewFilter_p(oidnDevice, "RT");

    HANDLE accumulatedImageOIDNHandle = device->GetWin32Handle(&accumulatedImage);
    HANDLE albedoOIDNHandle           = device->GetWin32Handle(&accumulatedAlbedo);
    HANDLE normalOIDNHandle           = device->GetWin32Handle(&normals);

    OIDNBuffer colorBuf = oidnNewSharedBufferFromWin32Handle_p(
        oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32, accumulatedImageOIDNHandle, 0,
        accumulatedImage.req.size);
    OIDNBuffer albedoBuf = oidnNewSharedBufferFromWin32Handle_p(
        oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32, albedoOIDNHandle, 0,
        accumulatedAlbedo.req.size);
    OIDNBuffer normalBuf = oidnNewSharedBufferFromWin32Handle_p(
        oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32, normalOIDNHandle, 0,
        normals.req.size);
    OIDNBuffer outputBuf = oidnNewSharedBufferFromWin32Handle_p(
        oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32, accumulatedImageOIDNHandle, 0,
        imageOutSize);
    oidnSetFilterImage_p(filter, "color", colorBuf, OIDN_FORMAT_FLOAT3, targetWidth,
                         targetHeight, 0, 0, 0);
    oidnSetFilterImage_p(filter, "albedo", albedoBuf, OIDN_FORMAT_FLOAT3, targetWidth,
                         targetHeight, 0, 0, 0);
    oidnSetFilterImage_p(filter, "normal", normalBuf, OIDN_FORMAT_FLOAT3, targetWidth,
                         targetHeight, 0, 0, 0);
    oidnSetFilterImage_p(filter, "output", outputBuf, OIDN_FORMAT_FLOAT3, targetWidth,
                         targetHeight, 0, 0, 0);

    oidnSetFilterBool_p(filter, "hdr", true);
    oidnSetFilterInt_p(filter, "quality", OIDN_QUALITY_HIGH);
    // oidnCommitFilter_p(filter);

    transferCmd->Barrier(&gpuEnvMap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transferCmd->FlushBarriers();

    ScratchArena sceneScratch;

    float radius    = 1.5f;
    Vec2f minDomain = -Vec2f(radius);
    Vec2f maxDomain = Vec2f(radius);
    int arrayWidth  = 32 * radius;
    StaticArray<float> filterValues(sceneScratch.temp.arena, Sqr(arrayWidth));
    float sigma = 0.5f;
    float expX  = Gaussian(radius, 0, sigma);
    float expY  = Gaussian(radius, 0, sigma);

    for (int y = 0; y < arrayWidth; y++)
    {
        for (int x = 0; x < arrayWidth; x++)
        {
            Vec2f t((x + 0.5f) / (float)arrayWidth, (y + 0.5f) / (float)arrayWidth);
            Vec2f p(Lerp(t.x, minDomain.x, maxDomain.x), Lerp(t.y, minDomain.y, maxDomain.y));
            // mitchellDistribution.Push(MitchellEvaluate(p, radius));
            filterValues.Push(GaussianEvaluate(p, sigma, expX, expY));
        }
    }

    PiecewiseConstant2D filterDistribution(sceneScratch.temp.arena, filterValues.data,
                                           arrayWidth, arrayWidth, minDomain, maxDomain);

    StaticArray<float> cdfs(sceneScratch.temp.arena, (arrayWidth + 1) * (1 + arrayWidth));
    for (int i = 0; i <= arrayWidth; i++)
    {
        cdfs.Push(filterDistribution.marginal.cdf[i]);
    }
    for (int i = 0; i < arrayWidth; i++)
    {
        PiecewiseConstant1D &conditional = filterDistribution.conditional[i];
        for (int j = 0; j <= arrayWidth; j++)
        {
            cdfs.Push(conditional.cdf[j]);
        }
    }

    TransferBuffer filterCDFBuffer = transferCmd->SubmitBuffer(
        cdfs.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(f32) * cdfs.Length());
    TransferBuffer filterValuesBuffer =
        transferCmd->SubmitBuffer(filterValues.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  sizeof(f32) * filterValues.Length());

    // VolumeData volumeData = Volumes(transferCmd, arena);
    // TransferBuffer volumeOctreeBuffer =
    //     transferCmd->SubmitBuffer(volumeData.octree.data,
    //     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    //                               sizeof(GPUOctreeNode) * volumeData.octree.Length());
    // int bindlessVdbIndex = device->BindlessStorageIndex(&volumeData.vdbDataBuffer.buffer);
    // Assert(bindlessVdbIndex == 0);

    submitSemaphore.signalValue = 1;
    transferCmd->SignalOutsideFrame(submitSemaphore);
    device->SubmitCommandBuffer(transferCmd);
    device->Wait(submitSemaphore);
    device->DestroyBuffer(&gpuEnvMapTransfer.stagingBuffer);

    // Create descriptor set layout and pipeline
    VkShaderStageFlags flags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                               VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                               VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    // VkShaderStageFlags flags   = VK_SHADER_STAGE_COMPUTE_BIT;
    DescriptorSetLayout layout = {};
    layout.pipelineLayout      = nullptr;
    layout.AddBinding((u32)RTBindings::Accel, DescriptorType::AccelerationStructure, flags);
    layout.AddBinding((u32)RTBindings::Image, DescriptorType::StorageImage, flags);
    layout.AddBinding((u32)RTBindings::Scene, DescriptorType::UniformBuffer, flags);
    layout.AddBinding((u32)RTBindings::GPUMaterial, DescriptorType::StorageBuffer, flags);
    layout.AddBinding((u32)RTBindings::PageTable, DescriptorType::StorageBuffer, flags);
    layout.AddBinding((u32)RTBindings::ShaderDebugInfo, DescriptorType::UniformBuffer, flags);
    layout.AddBinding((u32)RTBindings::ClusterPageData, DescriptorType::StorageBuffer, flags);
    layout.AddBinding((u32)RTBindings::PtexFaceData, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(13, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(14, DescriptorType::StorageBuffer, flags);
    // layout.AddBinding(15, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(16, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(17, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(18, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(19, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(20, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(21, DescriptorType::StorageImage, flags);
    layout.AddBinding(22, DescriptorType::StorageImage, flags);
    layout.AddBinding(23, DescriptorType::StorageBuffer, flags);
    layout.AddBinding((u32)RTBindings::Feedback, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(26, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(27, DescriptorType::StorageBuffer, flags);
    layout.AddBinding(29, DescriptorType::StorageBuffer, flags);

    layout.AddImmutableSamplers();

    RayTracingState rts = device->CreateRayTracingPipeline(groups, ArrayLength(groups),
                                                           &pushConstant, &layout, 3, true);
    // VkPipeline pipeline = device->CreateComputePipeline(&shader, &layout, &pushConstant);
    // Build clusters

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

        Mesh *meshes = (Mesh *)scene->primitives;
        Assert(scene->geometryType == GeometryType::TriangleMesh);
        blasScenes.Push(scene);
    }

    Scene *rootScene = GetScene();

    struct GPUTextureInfo
    {
        u8 *packedFaceData;
        int packedDataSize;

        u32 textureIndex;
        u32 faceOffset;
        u32 numFaces;

        // int minLog2Dim;
        // int numVirtualOffsetBits;
        // int numFaceDimBits;
        // int numFaceIDBits;
        // u32 virtualAddressIndex;
        // u32 faceDataOffset;
    };
    StaticArray<GPUTextureInfo> gpuTextureInfo(sceneScratch.temp.arena,
                                               rootScene->ptexTextures.size());
    // rootScene->ptexTextures.size());

    ImageDesc testDesc(
        ImageType::Array2D, 1, 1, 1, 1, 1, VK_FORMAT_R8G8B8A8_SRGB, MemoryUsage::GPU_ONLY,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_OPTIMAL);
    GPUImage testImage = device->CreateImage(
        testDesc, -1, QueueFlag_Copy | QueueFlag_Compute | QueueFlag_Graphics);

    VirtualTextureManager virtualTextureManager(sceneScratch.temp.arena, megabytes(512),
                                                kilobytes(512), VK_FORMAT_R8G8B8A8_SRGB);

    CommandBuffer *tileCmd          = device->BeginCommandBuffer(QueueType_Compute);
    Semaphore tileSubmitSemaphore   = device->CreateSemaphore();
    tileSubmitSemaphore.signalValue = 1;
    Semaphore texSemaphore          = device->CreateSemaphore();
    texSemaphore.signalValue        = 0;
    virtualTextureManager.ClearTextures(tileCmd);
    tileCmd->FlushBarriers();
    tileCmd->SignalOutsideFrame(tileSubmitSemaphore);
    device->SubmitCommandBuffer(tileCmd);
    device->Wait(tileSubmitSemaphore);

    struct TextureTempData
    {
        Tokenizer tokenizer;
        TextureMetadata metadata;
        string filename;
    };

    struct RequestHandle
    {
        u16 sortKey;
        int requestIndex;
        int ptexIndex;
    };

    StaticArray<TextureTempData> textureTempData(sceneScratch.temp.arena,
                                                 rootScene->ptexTextures.size(),
                                                 rootScene->ptexTextures.size());

    RequestHandle *handles = PushArrayNoZero(sceneScratch.temp.arena, RequestHandle,
                                             rootScene->ptexTextures.size());
    int numHandles         = 0;
    int faceOffset         = 0;

    for (int i = 0; i < rootScene->ptexTextures.size(); i++)
    {
        PtexTexture &ptexTexture = rootScene->ptexTextures[i];
        Ptex::String error;
        Ptex::PtexTexture *t = cache->get((char *)ptexTexture.filename.str, error);
        Assert(t);

        u32 numFaces = t->numFaces();

        u32 faceDataBitOffset = 0;
        u8 *faceDataStream    = PushArray(sceneScratch.temp.arena, u8, numFaces);
        for (u32 faceIndex = 0; faceIndex < numFaces; faceIndex++)
        {
            const Ptex::FaceInfo &f = t->getFaceInfo(faceIndex);
            Ptex::Res res           = f.res;
            Assert(res.ulog2 < 16 && res.vlog2 < 16);

            WriteBits((u32 *)faceDataStream, faceDataBitOffset, res.ulog2, 4);
            WriteBits((u32 *)faceDataStream, faceDataBitOffset, res.vlog2, 4);
        }
        t->release();

        if (i > 0)
        {
            bool result = device->Wait(texSemaphore, 10e9);
            Assert(result);
        }
        device->ResetDescriptorPool(0);
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Compute, "help");
        u32 allocIndex     = virtualTextureManager.AllocateVirtualPages(
            sceneScratch.temp.arena, ptexTexture.filename, faceDataStream, numFaces, cmd);
        texSemaphore.signalValue++;
        cmd->SignalOutsideFrame(texSemaphore);
        cmd->Barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
                     VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT);
        cmd->FlushBarriers();
        device->SubmitCommandBuffer(cmd);

        GPUTextureInfo info;
        info.textureIndex   = allocIndex;
        info.faceOffset     = faceOffset;
        info.packedFaceData = faceDataStream;
        info.packedDataSize = numFaces;
        info.numFaces       = numFaces;
        faceOffset += numFaces;

        gpuTextureInfo.Push(info);
    }
    bool finished = device->Wait(texSemaphore, 10e9);
    Assert(finished);

    u8 *faceDataByteBuffer = PushArrayNoZero(sceneScratch.temp.arena, u8, faceOffset);
    u32 ptexOffset         = 0;

    for (int i = 0; i < rootScene->ptexTextures.size(); i++)
    {
        GPUTextureInfo &info = gpuTextureInfo[i];
        MemoryCopy(faceDataByteBuffer + ptexOffset, info.packedFaceData, info.packedDataSize);
        ptexOffset += info.packedDataSize;
    }

    // Populate GPU materials
    StaticArray<GPUMaterial> gpuMaterials(sceneScratch.temp.arena,
                                          rootScene->materials.Length());

    for (int i = 0; i < rootScene->materials.Length(); i++)
    {
        GPUMaterial material  = rootScene->materials[i]->ConvertToGPU();
        material.textureIndex = -1;
        int index             = rootScene->materials[i]->ptexReflectanceIndex;
        u32 materialID        = index;
        if (index != -1)
        {
            GPUTextureInfo &textureInfo = gpuTextureInfo[index];
            material.textureIndex       = textureInfo.textureIndex;
            material.faceOffset         = textureInfo.faceOffset;
            // if (textureInfo.numVirtualOffsetBits != 0)
            {
                // material.minLog2Dim           = textureInfo.minLog2Dim;
                // material.numVirtualOffsetBits = textureInfo.numVirtualOffsetBits;
                // material.numFaceDimBits       = textureInfo.numFaceDimBits;
                // material.numFaceIDBits        = textureInfo.numFaceIDBits;
                // material.faceDataOffset       = textureInfo.faceDataOffset;
            }
            materialID |= (1u << 31u);
        }
        gpuMaterials.Push(material);
    }

    // Populate GPU lights
    StaticArray<GPULight> gpuLights(sceneScratch.temp.arena, rootScene->lights.size());

    for (int i = 0; i < rootScene->lights.size(); i++)
    {
        GPULight light = rootScene->lights[i]->ConvertToGPU();
        gpuLights.Push(light);
    }

    // Populate GPU media
    CommandBuffer *newTileCmd = device->BeginCommandBuffer(QueueType_Compute);
    StaticArray<GPUMedium> gpuMedia(sceneScratch.temp.arena, rootScene->media.size());
    StaticArray<int> nanovdbIndices(sceneScratch.temp.arena, rootScene->media.size());

#if 0
    for (int i = 0; i < rootScene->media.size(); i++)
    {
        Medium *medium = rootScene->media[i];
        switch (medium->GetType())
        {
            case GPUMediumType::Nanovdb:
            {
                nanovdbIndices.Push(i);
            }
            break;
            case GPUMediumType::Homogeneous:
            {
                // TODO: not implemented yet
                Assert(0);
                GPUMedium medium;
                medium.mediumType = medium.mediumType;
                gpuMedia.Push(medium);
            }
            break;
            default: Assert(0);
        }
    }

    for (int index : nanovdbIndices)
    {
        NanovdbMedium *medium = (NanovdbMedium *)rootScene->media[index];
        VolumeData volumeData = Volumes(medium->filename, newTileCmd, arena);

        TransferBuffer volumeOctreeBuffer = newTileCmd->SubmitBuffer(
            volumeData.octree.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sizeof(GPUOctreeNode) * volumeData.octree.Length());

        int bindlessOctreeIndex = device->BindlessStorageIndex(&volumeOctreeBuffer.buffer);
        int bindlessVdbIndex = device->BindlessStorageIndex(&volumeData.vdbDataBuffer.buffer);

        GPUMedium gpuMedium                  = {};
        gpuMedium.mediumType                 = GPUMediumType::Nanovdb;
        gpuMedium.bindlessOctreeIndex        = bindlessOctreeIndex;
        gpuMedium.bindlessNanovdbBufferIndex = bindlessVdbIndex;
        gpuMedia.Push(gpuMedium);
    }
#endif

    // Process nanovdb media properly

    Semaphore newTileSubmitSemaphore   = device->CreateSemaphore();
    newTileSubmitSemaphore.signalValue = 1;

    GPUBuffer mediumBuffer;

    if (gpuMedia.Length())
    {
        mediumBuffer = newTileCmd
                           ->SubmitBuffer(gpuMedia.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          sizeof(GPUMedium) * gpuMedia.Length())
                           .buffer;
    }

    GPUBuffer materialBuffer =
        newTileCmd
            ->SubmitBuffer(gpuMaterials.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(GPUMaterial) * gpuMaterials.Length())
            .buffer;

    GPUBuffer lightBuffer =
        newTileCmd
            ->SubmitBuffer(gpuLights.data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           sizeof(GPULight) * gpuLights.Length())
            .buffer;

    GPUBuffer faceDataBuffer =
        newTileCmd
            ->SubmitBuffer(faceDataByteBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, faceOffset)
            .buffer;

    newTileCmd->SignalOutsideFrame(newTileSubmitSemaphore);
    newTileCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    newTileCmd->FlushBarriers();
    device->SubmitCommandBuffer(newTileCmd);
    device->Wait(tileSubmitSemaphore);

    u32 numBlas = blasScenes.Length();

    // Virtual geometry initialization
    VirtualGeometryManager virtualGeometryManager(sceneScratch.temp.arena, targetWidth,
                                                  targetHeight, blasScenes.Length());
    StaticArray<AABB> blasSceneBounds(sceneScratch.temp.arena, numBlas);
    StaticArray<string> virtualGeoFilenames(sceneScratch.temp.arena, numBlas);
    HashIndex filenameHash(sceneScratch.temp.arena, NextPowerOfTwo(numBlas),
                           NextPowerOfTwo(numBlas));

    Semaphore geoSemaphore   = device->CreateSemaphore();
    geoSemaphore.signalValue = 0;

    u32 lastSize           = 0;
    int totalSaved         = 0;
    GPUBuffer sizeReadback = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, virtualGeometryManager.totalAccelSizesBuffer.size,
        MemoryUsage::GPU_TO_CPU);

    GPUBuffer globalsReadback = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, GLOBALS_SIZE * sizeof(u32), MemoryUsage::GPU_TO_CPU);

    PerformanceCounter blasCounter = OS_StartCounter();
    for (int sceneIndex = 0; sceneIndex < numBlas; sceneIndex++)
    {
        ScenePrimitives *scene    = blasScenes[sceneIndex];
        string virtualGeoFilename = scene->virtualGeoFilename;
        u32 hash                  = Hash(virtualGeoFilename);

        int resourceID = -1;
        for (int hashIndex = filenameHash.FirstInHash(hash); hashIndex != -1;
             hashIndex     = filenameHash.NextInHash(hashIndex))
        {
            if (virtualGeoFilenames[hashIndex] == virtualGeoFilename)
            {
                resourceID = hashIndex;
                break;
            }
        }
        if (resourceID != -1)
        {
            scene->sceneIndex = resourceID;
            continue;
        }

        scene->sceneIndex = virtualGeoFilenames.Length();
        virtualGeoFilenames.Push(virtualGeoFilename);
        filenameHash.AddInHash(hash, scene->sceneIndex);

        if (sceneIndex > 0)
        {
            bool result = device->Wait(geoSemaphore, 10e9);
            ErrorExit(result, "failed scene %u\n", sceneIndex);
        }
        device->ResetDescriptorPool(0);
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Compute);

        if (sceneIndex == 0)
        {
            cmd->ClearBuffer(&virtualGeometryManager.totalAccelSizesBuffer, 0);
            cmd->Barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT);
            cmd->FlushBarriers();
        }

        bool debug = false;
        geoSemaphore.signalValue++;
        cmd->SignalOutsideFrame(geoSemaphore);
        cmd->ClearBuffer(&virtualGeometryManager.clasGlobalsBuffer, 0);
        cmd->Barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
                     VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT);
        cmd->FlushBarriers();
        bool cullSubpixel =
            Contains(virtualGeoFilename, "isBeach", MatchFlag_CaseInsensitive) |
            Contains(virtualGeoFilename, "isCoral", MatchFlag_CaseInsensitive) |
            Contains(virtualGeoFilename, "isCoastline", MatchFlag_CaseInsensitive) |
            Contains(virtualGeoFilename, "isBayCedarA1", MatchFlag_CaseInsensitive);

        QueryPool pool;
        GPUAccelerationStructurePayload payload;
        u32 meshInfoIndex = virtualGeometryManager.AddNewMesh(
            sceneScratch.temp.arena, cmd, virtualGeoFilename, cullSubpixel, payload, pool);

        if (virtualGeometryManager.meshInfos[meshInfoIndex].numClusters < 10 && !cullSubpixel)
        {
            Print("? %S\n", virtualGeoFilename);
        }

        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_TRANSFER_READ_BIT);
        cmd->FlushBarriers();

        cmd->CopyBuffer(&sizeReadback, &virtualGeometryManager.totalAccelSizesBuffer);
        cmd->CopyBuffer(&globalsReadback, &virtualGeometryManager.clasGlobalsBuffer);

        device->SubmitCommandBuffer(cmd);

#if 0
        bool result = device->Wait(geoSemaphore, 10e9);

        u32 clasSize               = *(u32 *)sizeReadback.mappedPtr;
        VkDeviceSize compactedSize = 0;
        vkGetQueryPoolResults(device->device, pool.queryPool, 0, 1, sizeof(VkDeviceSize),
                              &compactedSize, 0,
                              VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT);

        u32 actualSize = clasSize - lastSize;
        Print("%S %u %llu num v %u num tri %u\n", virtualGeoFilename, actualSize,
              compactedSize,
              ((u32 *)globalsReadback.mappedPtr)[GLOBALS_VERTEX_BUFFER_OFFSET_INDEX],
              ((u32 *)globalsReadback.mappedPtr)[GLOBALS_INDEX_BUFFER_OFFSET_INDEX]);

        totalSaved += (int)compactedSize - (int)actualSize;

        lastSize = clasSize;
        device->DestroyAccelerationStructure(payload.as.as);
        device->DestroyBuffer(&payload.as.buffer);
        device->DestroyBuffer(&payload.scratch);
#endif

        auto &meshInfo =
            virtualGeometryManager.meshInfos[virtualGeometryManager.meshInfos.Length() - 1];
        Vec3f boundsMin = meshInfo.boundsMin;
        Vec3f boundsMax = meshInfo.boundsMax;

        AABB aabb;
        aabb.minX = boundsMin[0];
        aabb.minY = boundsMin[1];
        aabb.minZ = boundsMin[2];
        aabb.maxX = boundsMax[0];
        aabb.maxY = boundsMax[1];
        aabb.maxZ = boundsMax[2];
        blasSceneBounds.Push(aabb);
    }
    bool result = device->Wait(geoSemaphore, 10e9);
    Assert(result);

    f32 blasTime = OS_GetMilliseconds(blasCounter);

#if 0
    Print("total saved: %i\n", totalSaved);
    printf("blas build time: %fms\n", blasTime);
#endif

    // count++;
    // TODO: destroy temp memory
    device->DestroyBuffer(&virtualGeometryManager.vertexBuffer);
    device->DestroyBuffer(&virtualGeometryManager.indexBuffer);
    device->DestroyBuffer(&virtualGeometryManager.clasScratchBuffer);
    device->DestroyBuffer(&virtualGeometryManager.blasScratchBuffer);
    device->DestroyBuffer(&virtualGeometryManager.buildClusterTriangleInfoBuffer);
    device->DestroyBuffer(&virtualGeometryManager.decodeClusterDataBuffer);
    device->DestroyBuffer(&virtualGeometryManager.pageUploadBuffer);

    CommandBuffer *dgfTransferCmd = device->BeginCommandBuffer(QueueType_Compute);
    virtualGeometryManager.FinalizeResources(dgfTransferCmd);

    // Photon mapping
    PhotonMapper photonMapper(sceneScratch.temp.arena);

    // Path guiding
    // PathGuiding pathGuiding(sceneScratch.temp.arena);

    Semaphore sem   = device->CreateSemaphore();
    sem.signalValue = 1;
    dgfTransferCmd->SignalOutsideFrame(sem);
    dgfTransferCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_NONE);
    dgfTransferCmd->FlushBarriers();

    device->SubmitCommandBuffer(dgfTransferCmd);

    device->Wait(sem);

    CommandBuffer *allCommandBuffer = device->BeginCommandBuffer(QueueType_Graphics);
    allCommandBuffer->Wait(sem);
    Semaphore tlasSemaphore       = device->CreateSemaphore();
    GPUAccelerationStructure tlas = {};

    // Build the TLAS over BLAS
    StaticArray<Instance> instances;
    StaticArray<AffineSpace> instanceTransforms;

    PerformanceCounter tlasCounter = OS_StartCounter();
    if (tlasScenes.Length() >= 1)
    {
        ScratchArena scratch(&sceneScratch.temp.arena, 1);
        ScenePrimitives *scene = tlasScenes[0];
        Array<Instance> flattenedInstances(scratch.temp.arena, scene->numPrimitives);
        Array<AffineSpace> flattenedTransforms(scratch.temp.arena, scene->numPrimitives);
        AffineSpace transform = AffineSpace::Identity();

        FlattenInstances(scene, transform, flattenedInstances, flattenedTransforms);
        instances = StaticArray<Instance>(sceneScratch.temp.arena, flattenedInstances.Length(),
                                          flattenedInstances.Length());
        instanceTransforms =
            StaticArray<AffineSpace>(sceneScratch.temp.arena, flattenedTransforms.Length(),
                                     flattenedTransforms.Length());
        MemoryCopy(instances.data, flattenedInstances.data,
                   sizeof(Instance) * instances.Length());
        MemoryCopy(instanceTransforms.data, flattenedTransforms.data,
                   sizeof(AffineSpace) * flattenedTransforms.Length());
    }
    else
    {
        instances            = StaticArray<Instance>(sceneScratch.temp.arena, 1);
        instanceTransforms   = StaticArray<AffineSpace>(sceneScratch.temp.arena, 1);
        Instance instance    = {};
        AffineSpace identity = AffineSpace::Identity();
        instances.Push(instance);
        instanceTransforms.Push(identity);
        Assert(numScenes == 1);
    }

    bool built =
        virtualGeometryManager.AddInstances(sceneScratch.temp.arena, allCommandBuffer,
                                            instances, instanceTransforms, params->filename);

    f32 tlasTime = OS_GetMilliseconds(tlasCounter);
    printf("tlas time: %fms\n", tlasTime);

    tlasSemaphore.signalValue = 1;
    allCommandBuffer->SignalOutsideFrame(tlasSemaphore);

    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    allCommandBuffer->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    allCommandBuffer->FlushBarriers();
    device->SubmitCommandBuffer(allCommandBuffer);

    device->Wait(tlasSemaphore);
    device->DestroyBuffer(&virtualGeometryManager.instanceTransformsUploadBuffer);
    device->DestroyBuffer(&virtualGeometryManager.resourceIDsUploadBuffer);
    device->DestroyBuffer(&virtualGeometryManager.partitionInfosUploadBuffer);
    if (built)
    {
        device->DestroyBuffer(&virtualGeometryManager.blasProxyScratchBuffer);
        device->DestroyBuffer(&virtualGeometryManager.mergedInstancesAABBBuffer);
    }

    f32 frameDt = 1.f / 60.f;
    int envMapBindlessIndex;

    ViewCamera camera = {};
    // camera.position       = Vec3f(5441.899414, -224.013306, -10116.300781);
    // camera.pitch          = -0.138035f;
    // camera.yaw            = 1.092227f;
    ViewCamera prevCamera = {};

#if 1
    camera.position = Vec3f(0);
    // camera.position = Vec3f(4892.06055f, 767.444824f, -11801.2275f);
    // camera.position = Vec3f(5128.51562f, 1104.60583f, -6173.79395f);
    camera.forward = Normalize(params->look - params->pCamera);
    // camera.forward = Vec3f(-.290819466f, .091174677f, .9524323811f);
    camera.right = Normalize(Cross(camera.forward, params->up));

    camera.pitch = ArcSin(camera.forward.y);
    camera.yaw   = -Atan2(camera.forward.z, camera.forward.x);
#endif

    Vec3f baseForward = camera.forward;
    Vec3f baseRight   = camera.right;

    TransferBuffer sceneTransferBuffers[2] = {
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene)),
        device->GetStagingBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(GPUScene))};

    ResourceHandle sceneTransferBufferHandles[2] = {
        rg->RegisterExternalResource("scene buffer 0", &sceneTransferBuffers[0].buffer),
        rg->RegisterExternalResource("scene buffer 1", &sceneTransferBuffers[1].buffer),
    };

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

    Semaphore frameSemaphore = device->CreateSemaphore();

    GPUBuffer readback =
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, 3 * sizeof(u32) * GLOBALS_SIZE,
                             MemoryUsage::GPU_TO_CPU);

    GPUBuffer debugBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 3 * sizeof(u32) * GLOBALS_SIZE);

    Semaphore transferSem   = device->CreateSemaphore();
    transferSem.signalValue = 1;

    u32 testCount                   = 0;
    u32 maxUnused                   = 0;
    f32 speed                       = 5000.f;
    u32 numFramesAccumulated        = 0;
    f32 lastFrameTime               = OS_NowSeconds();
    const u32 numFramesUntilDenoise = 128;

    f32 time = OS_GetMilliseconds(counter);
    printf("scene initialization time: %fms\n", time);

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
        f32 deltaTime = frameTime - lastFrameTime;
        lastFrameTime = frameTime;

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
                else if (node->values[i].key == OS_Key_L)
                {
                    speed *= 2.f;
                }
                else if (node->values[i].key == OS_Key_K)
                {
                    speed /= 2.f;
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

        if (!(prevCamera.position == camera.position && prevCamera.yaw == camera.yaw &&
              prevCamera.pitch == camera.pitch))
        {
            numFramesAccumulated = 0;
        }

        numFramesAccumulated++;

        events.Clear();

        Mat4 clipToPrevClip;
        Mat4 prevClipToClip;
        u32 numPhases     = 8 * u32(Ceil(Sqr(params->height / (f32)targetHeight)));
        f32 haltonSampleX = Halton((device->frameCount + 1) % numPhases, 2);
        f32 haltonSampleY = Halton((device->frameCount + 1) % numPhases, 3);
        float haltonX     = 2.f * haltonSampleX - 1.f;
        float haltonY     = 2.f * haltonSampleY - 1.f;
        float jitterX     = (.5f * haltonX / targetWidth);
        float jitterY     = (.5f * haltonY / targetHeight);

        float outJitterX = .5f * (haltonSampleX - 0.5f);
        float outJitterY = .5f * (0.5f - haltonSampleY);

        // Input
        {
            prevCamera = camera;

            f32 rotationSpeed = 0.001f * PI;
            camera.RotateCamera(dMouseP, rotationSpeed);

            camera.position += (dir[2] - dir[3]) * camera.forward * speed * deltaTime;
            camera.position += (dir[0] - dir[1]) * camera.right * speed * deltaTime;

            camera.forward.x = Cos(camera.yaw) * Cos(camera.pitch);
            camera.forward.y = Sin(camera.pitch);
            camera.forward.z = -Sin(camera.yaw) * Cos(camera.pitch);
            camera.forward   = Normalize(camera.forward);

            camera.right = Normalize(Cross(camera.forward, params->up));

            if (Length(camera.right) < 1e-8f)
            {
                // camera.right = Normalize(Cross
            }

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
                AffineSpace(axis, Vec3f(0)) * Translate(-camera.position);
            AffineSpace renderFromCamera = Inverse(cameraFromRender);

            Mat4 clipFromCamera = params->NDCFromCamera;
            // clipFromCamera[2][0] -= jitterX;
            // clipFromCamera[2][1] -= jitterY;

            clipToPrevClip = params->NDCFromCamera * Mat4(gpuScene.cameraFromRender) *
                             Mat4(renderFromCamera) * params->cameraFromClip;
            prevClipToClip = Inverse(clipToPrevClip);

            gpuScene.cameraP          = camera.position;
            gpuScene.renderFromCamera = renderFromCamera;
            gpuScene.cameraFromRender = cameraFromRender;
            gpuScene.prevClipFromClip = clipToPrevClip;
            gpuScene.clipFromPrevClip = prevClipToClip;
            gpuScene.clipFromRender   = clipFromCamera * Mat4(cameraFromRender);
            // gpuScene.jitterX          = jitterX;
            // gpuScene.jitterY          = jitterY;
            OS_GetMousePos(params->window, shaderDebug.mousePos.x, shaderDebug.mousePos.y);
        }
        u32 dispatchDimX =
            (targetWidth + PATH_TRACE_NUM_THREADS_X - 1) / PATH_TRACE_NUM_THREADS_X;
        u32 dispatchDimY =
            (targetHeight + PATH_TRACE_NUM_THREADS_Y - 1) / PATH_TRACE_NUM_THREADS_Y;
        gpuScene.dispatchDimX = dispatchDimX;
        gpuScene.dispatchDimY = dispatchDimY;

        u32 *data = (u32 *)readback.mappedPtr;
        if (!device->BeginFrame(false))
        {
            ScratchArena scratch;
            Print("freed partition count: %u visible count: %u, writes: %u updates %u, "
                  "allocate: "
                  "%u freed: %u unused: %u debug: %u, free list count: %u, x: %u, y: %u, z: "
                  "%u, skipped: %u, max unused: %u\n",
                  data[GLOBALS_FREED_PARTITION_COUNT], data[GLOBALS_VISIBLE_PARTITION_COUNT],
                  data[GLOBALS_PTLAS_WRITE_COUNT_INDEX],
                  data[GLOBALS_PTLAS_UPDATE_COUNT_INDEX],
                  data[GLOBALS_ALLOCATED_INSTANCE_COUNT_INDEX],
                  data[GLOBALS_FREED_INSTANCE_COUNT_INDEX],
                  data[GLOBALS_INSTANCE_UNUSED_COUNT], data[GLOBALS_DEBUG],
                  data[GLOBALS_DEBUG2], data[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_X],
                  data[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_Y],
                  data[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_Z], data[GLOBALS_DEBUG3], maxUnused);
            Assert(0);
        }

        Print("frame: %u\n", device->frameCount);

        // maxUnused = Max(maxUnused, data[GLOBALS_DEBUG]);
        Vec3f *vals = (Vec3f *)readback.mappedPtr;
        for (u32 i = 0; i < 22; i++)
        {
            Print("num times sampled %u: %f %f %f", i, vals[i].x, vals[i].y, vals[i].z);
            // Print("num times sampled %u: %u", i, data[i]);
        }
        Print("\n");
        // Print("freed partition count: %u visible count: %u, writes: %u updates %u, "
        //       "allocate: "
        //       "%u freed: %u unused: %u debug: %u, free list count: %u, x: %u, y: %u, z: "
        //       "%u, skipped: %u, max unused: %u\n",
        //       data[GLOBALS_FREED_PARTITION_COUNT], data[GLOBALS_VISIBLE_PARTITION_COUNT],
        //       data[GLOBALS_PTLAS_WRITE_COUNT_INDEX], data[GLOBALS_PTLAS_UPDATE_COUNT_INDEX],
        //       data[GLOBALS_ALLOCATED_INSTANCE_COUNT_INDEX],
        //       data[GLOBALS_FREED_INSTANCE_COUNT_INDEX], data[GLOBALS_INSTANCE_UNUSED_COUNT],
        //       data[GLOBALS_DEBUG], data[GLOBALS_DEBUG2],
        //       data[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_X],
        //       data[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_Y],
        //       data[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_Z], data[GLOBALS_DEBUG3], maxUnused);

        string cmdBufferName =
            PushStr8F(frameScratch.temp.arena, "Graphics Cmd %u", device->frameCount);
        CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Graphics, cmdBufferName);

        rg->BeginFrame();
        debugState.BeginFrame(cmd);

        if (device->frameCount == 0)
        {
            device->Wait(tileSubmitSemaphore);
            device->Wait(tlasSemaphore);

            envMapBindlessIndex = device->BindlessIndex(&gpuEnvMap);
        }

        u32 currentBuffer   = device->frameCount & 1;
        u32 lastFrameBuffer = device->frameCount == 0 ? 0 : !currentBuffer;

        MemoryCopy(sceneTransferBuffers[currentBuffer].mappedPtr, &gpuScene, sizeof(GPUScene));
        cmd->SubmitTransfer(&sceneTransferBuffers[currentBuffer]);
        MemoryCopy(shaderDebugBuffers[currentBuffer].mappedPtr, &shaderDebug,
                   sizeof(ShaderDebugInfo));
        cmd->SubmitTransfer(&shaderDebugBuffers[currentBuffer]);

        virtualTextureManager.Update(cmd);

        // photonMapper.BuildKDTree();
        // pathGuider.PathGuiding();

        rg->StartPass(2,
                      [&clasGlobals      = virtualGeometryManager.clasGlobalsBuffer,
                       instanceBitVector = virtualGeometryManager.instanceBitVectorHandle,
                       &virtualTextureManager, currentBuffer, &readback](CommandBuffer *cmd) {
                          RenderGraph *rg = GetRenderGraph();

                          GPUBuffer *buffer = rg->GetBuffer(instanceBitVector);
                          cmd->ClearBuffer(buffer);
                          // cmd->ClearBuffer(&readback);

                          cmd->ClearBuffer(&clasGlobals, 0);
                          cmd->ClearBuffer(
                              &virtualTextureManager.feedbackBuffers[currentBuffer].buffer, 0,
                              0, sizeof(u32));

                          cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_2_SHADER_WRITE_BIT |
                                           VK_ACCESS_2_SHADER_READ_BIT);
                          cmd->FlushBarriers();
                      })
            .AddHandle(virtualGeometryManager.clasGlobalsBufferHandle,
                       ResourceUsageType::Write)
            .AddHandle(virtualGeometryManager.instanceBitVectorHandle,
                       ResourceUsageType::Write);

#if 0
        if (device->frameCount > 0)
        {
            // TODO: support transient lifetimes that cross frame boundaries
            virtualGeometryManager.ReprojectDepth(targetWidth, targetHeight, depthBufferHandle,
                                                  sceneTransferBufferHandles[currentBuffer]);
            virtualGeometryManager.UpdateHZB();
        }
        else
        {
            rg->StartPass(1, [depthPyramid =
                                  virtualGeometryManager.depthPyramid](CommandBuffer *cmd) {
                  GPUImage *img = GetRenderGraph()->GetImage(depthPyramid);
                  cmd->Barrier(
                      img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->FlushBarriers();
              }).AddHandle(virtualGeometryManager.depthPyramid, ResourceUsageType::Write);
        }
#endif

        // Instance culling
        {
            virtualGeometryManager.PrepareInstances(
                cmd, sceneTransferBufferHandles[currentBuffer], true);
        }

        rg->StartComputePass(prepareIndirectPipeline, prepareIndirectLayout, 1,
                             [](CommandBuffer *cmd) {
                                 device->BeginEvent(cmd, "Prepare indirect");
                                 cmd->Dispatch(1, 1, 1);
                                 cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_ACCESS_2_SHADER_WRITE_BIT,
                                              VK_ACCESS_2_SHADER_READ_BIT);
                                 cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                              VK_ACCESS_2_SHADER_WRITE_BIT,
                                              VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
                                 cmd->FlushBarriers();
                                 device->EndEvent(cmd);
                             })
            .AddHandle(virtualGeometryManager.clasGlobalsBufferHandle, ResourceUsageType::RW);

        virtualGeometryManager.BuildPTLAS(cmd);

        rg->StartPass(0, [](CommandBuffer *cmd) {
            cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                         VK_ACCESS_2_SHADER_READ_BIT);
            cmd->FlushBarriers();
        });

        RayPushConstant pc;
        pc.envMap         = envMapBindlessIndex;
        pc.frameNum       = (u32)device->frameCount;
        pc.width          = envMap->width;
        pc.height         = envMap->height;
        pc.filterIntegral = filterDistribution.marginal.Integral();

        VkPipelineStageFlags2 flags   = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
        // cmd->Barrier(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer, flags,
        //              VK_ACCESS_2_SHADER_WRITE_BIT);
        u64 ptlasAddress =
            device->GetDeviceAddress(virtualGeometryManager.tlasAccelBuffer.buffer);

        // TODO: not adding the handle of all of the barriers. maybe will need to if
        // automatic synchronization is done in the future

        if (0) // numFramesAccumulated > numFramesUntilDenoise)
        {
            rg->StartPass(1, [&](CommandBuffer *cmd) {
                  RenderGraph *rg = GetRenderGraph();
                  u32 offset;
                  GPUBuffer *imageOut       = rg->GetBuffer(imageOutHandle, offset);
                  HANDLE imageOutOIDNHandle = device->GetWin32Handle(imageOut);
                  OIDNBuffer outputBuf      = oidnNewSharedBufferFromWin32Handle_p(
                      oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                      imageOutOIDNHandle, 0, imageOutSize);
                  oidnSetFilterImage_p(filter, "color", colorBuf, OIDN_FORMAT_FLOAT3,
                                       targetWidth, targetHeight, 0, 0, 0);
                  oidnSetFilterImage_p(filter, "albedo", albedoBuf, OIDN_FORMAT_FLOAT3,
                                       targetWidth, targetHeight, 0, 0, 0);
                  oidnSetFilterImage_p(filter, "normal", normalBuf, OIDN_FORMAT_FLOAT3,
                                       targetWidth, targetHeight, 0, 0, 0);
                  oidnSetFilterImage_p(filter, "output", outputBuf, OIDN_FORMAT_FLOAT3,
                                       targetWidth, targetHeight, 0, 0, 0);
                  oidnCommitFilter_p(filter);
                  oidnExecuteFilter_p(filter);

                  CloseHandle(imageOutOIDNHandle);
              }).AddHandle(imageOutHandle, ResourceUsageType::Write);
        }
        else
        {
            rg->StartPass(
                  4,
                  [&layout, &ptlasAddress, &rts = rts, &pushConstant, &pc, bindPoint,
                   targetWidth, targetHeight,
                   &scene = sceneTransferBuffers[currentBuffer].buffer,
                   &debug = shaderDebugBuffers[currentBuffer].buffer, &materialBuffer,
                   &faceDataBuffer, &virtualTextureManager, &virtualGeometryManager,
                   imageHandle = image, &depthBuffer, &readback, &normals, albedoHandle,
                   currentBuffer, &debugBuffer, &filterValuesBuffer,
                   &filterCDFBuffer](CommandBuffer *cmd) {
                      RenderGraph *rg  = GetRenderGraph();
                      GPUImage *image  = rg->GetImage(imageHandle);
                      GPUImage *albedo = rg->GetImage(albedoHandle);

                      cmd->Barrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_NONE,
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                   VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                      cmd->Barrier(albedo, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_NONE,
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                   VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                      cmd->Barrier(&depthBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                   VK_ACCESS_2_SHADER_READ_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
                      cmd->FlushBarriers();

                      cmd->BindPipeline(bindPoint, rts.pipeline);

                      cmd->StartBinding(bindPoint, rts.pipeline, &layout)
                          // .Bind(&tlas.as)
                          .Bind(&ptlasAddress)
                          .Bind(image)
                          .Bind(&scene)
                          .Bind(&materialBuffer)
                          .Bind(&virtualTextureManager.pageHashTableBuffer)
                          .Bind(&debug)
                          .Bind(&virtualGeometryManager.clusterPageDataBuffer)
                          .Bind(&faceDataBuffer)
                          .Bind(&debugBuffer)
                          .Bind(&virtualGeometryManager.instancesBuffer)
                          .Bind(&virtualGeometryManager.resourceTruncatedEllipsoidsBuffer)
                          .Bind(&virtualGeometryManager.instanceTransformsBuffer)
                          .Bind(&virtualGeometryManager.partitionInfosBuffer)
                          .Bind(&virtualGeometryManager.instanceResourceIDsBuffer)
                          .Bind(&virtualGeometryManager.resourceBuffer)
                          .Bind(&depthBuffer)
                          .Bind(albedo)
                          .Bind(&normals)
                          .Bind(&virtualTextureManager.feedbackBuffers[currentBuffer].buffer)
                          .Bind(&filterCDFBuffer.buffer)
                          .Bind(&filterValuesBuffer.buffer)
                          // TODO IMPORTANT BINDLESS
                          // .Bind(&volumeOctreeBuffer.buffer)
                          .PushConstants(&pushConstant, &pc)
                          .End();

                      int beginIndex = TIMED_GPU_RANGE_BEGIN(cmd, "ray trace");
                      // cmd->Dispatch(dispatchDimX, dispatchDimY, 1);
                      cmd->TraceRays(&rts, targetWidth, targetHeight, 1);
                      TIMED_RANGE_END(beginIndex);

                      {
                          // RenderGraph *rg = GetRenderGraph();
                          // GPUBuffer readback0 =
                          //     device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          //                          debugBuffer.size,
                          //                          MemoryUsage::GPU_TO_CPU);
                          cmd->Barrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_ACCESS_2_SHADER_WRITE_BIT,
                                       VK_ACCESS_2_TRANSFER_READ_BIT);
                          // cmd->Barrier(&imageOut, VK_IMAGE_LAYOUT_GENERAL,
                          //              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          //              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                          //              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                          // VK_ACCESS_2_SHADER_WRITE_BIT,
                          //              VK_ACCESS_2_TRANSFER_READ_BIT);

                          // cmd->Barrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                          //     //              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                          //     // VK_ACCESS_2_SHADER_WRITE_BIT,
                          //     //              VK_ACCESS_2_TRANSFER_READ_BIT);
                          cmd->FlushBarriers();

                          // BufferImageCopy copy = {};
                          // copy.extent =
                          //     Vec3u(filterWeightsImage.desc.width,
                          // filterWeightsImage.desc.height,
                          // 1);

                          cmd->CopyBuffer(&readback, &debugBuffer);
                          // cmd->CopyBuffer(&readback2, buffer1);
                          // cmd->CopyImageToBuffer(&readback0, &filterWeightsImage, &copy, 1);
                      }
                  })
                .AddHandle(image, ResourceUsageType::Write)
                .AddHandle(depthBufferHandle, ResourceUsageType::Write)
                .AddHandle(normalsHandle, ResourceUsageType::Write)
                .AddHandle(albedoHandle, ResourceUsageType::Write);

            NumPushConstant accumulatePush;
            accumulatePush.num = numFramesAccumulated;
            rg->StartComputePass(
                  accumulatePipeline, accumulateLayout, 4,
                  [targetWidth, targetHeight, image, albedoHandle](CommandBuffer *cmd) {
                      RenderGraph *rg      = GetRenderGraph();
                      GPUImage *frameImage = rg->GetImage(image);
                      GPUImage *albedo     = rg->GetImage(albedoHandle);
                      device->BeginEvent(cmd, "Accumulate");
                      cmd->Barrier(frameImage, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                      cmd->Barrier(albedo, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                      cmd->Dispatch((targetWidth + 7) / 8, (targetHeight + 7) / 8, 1);
                      device->EndEvent(cmd);

                      rg->SetSubmit();
                  },
                  &accumulatePC, &accumulatePush)
                .AddHandle(image, ResourceUsageType::Read)
                .AddHandle(accumulatedImageHandle, ResourceUsageType::Write)
                .AddHandle(albedoHandle, ResourceUsageType::Read)
                .AddHandle(accumulatedAlbedoHandle, ResourceUsageType::Write);
        }

        Pass &pass = rg->StartComputePass(
            copyDenoisedPipeline, copyDenoisedLayout, 2, [&](CommandBuffer *cmd) {
                GPUImage *image = rg->GetImage(finalImageHandle);
                device->BeginEvent(cmd, "Write Denoised Output");
                cmd->Barrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT);
                cmd->Barrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                cmd->FlushBarriers();
                cmd->Dispatch((targetWidth + 7) / 8, (targetHeight + 7) / 8, 1);
                device->EndEvent(cmd);
            });

        if (0) // numFramesAccumulated > numFramesUntilDenoise)
        {
            pass.AddHandle(imageOutHandle, ResourceUsageType::Read);
        }
        else
        {
            pass.AddHandle(accumulatedImageHandle, ResourceUsageType::Read);
        }
        pass.AddHandle(finalImageHandle, ResourceUsageType::Write);

        // Copy feedback from device to host
        rg->StartPass(0, [&virtualTextureManager, currentBuffer](CommandBuffer *cmd) {
            cmd->Barrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT);
            cmd->CopyBuffer(
                &virtualTextureManager.feedbackBuffers[currentBuffer].stagingBuffer,
                &virtualTextureManager.feedbackBuffers[currentBuffer].buffer);
        });

        debugState.EndFrame(cmd);

        rg->StartPass(1, [&swapchain, finalImageHandle](CommandBuffer *cmd) {
              GPUImage *image = GetRenderGraph()->GetImage(finalImageHandle);
              device->CopyFrameBuffer(&swapchain, cmd, image);
          }).AddHandle(finalImageHandle, ResourceUsageType::Read);

        rg->Compile();
        rg->Execute(cmd);
        rg->EndFrame();

        device->EndFrame(QueueFlag_Graphics);

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
