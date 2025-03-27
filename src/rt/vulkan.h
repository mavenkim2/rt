#ifndef RT_VULKAN_H
#define RT_VULKAN_H

#include <functional>

#include "base.h"
#include "bvh/bvh_types.h"
#include "containers.h"
#include "platform.h"

#define VK_NO_PROTOTYPES

#include "../third_party/vulkan/vulkan/vulkan.h"
#include "../third_party/vulkan/volk.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "../third_party/vulkan/vk_mem_alloc.h"

namespace rt
{
struct AffineSpace;
struct Instance;
struct Mesh;
struct ScenePrimitives;

#define VK_CHECK(check)                                                                       \
    do                                                                                        \
    {                                                                                         \
        VkResult result_ = check;                                                             \
        Assert(result_ == VK_SUCCESS);                                                        \
    } while (0);

static const int numActiveFrames = 2;
using CopyFunction               = std::function<void(void *)>;

inline u32 GetFormatSize(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_R8G8B8_SRGB: return 3;
        case VK_FORMAT_R8G8B8A8_SRGB: return 4;
        default: Assert(0); return 0;
    }
}

enum class ResourceUsage : u32
{
    None     = 0,
    Graphics = 1 << 1,
    Depth    = 1 << 4,
    Stencil  = 1 << 5,

    // Pipeline stages
    Indirect       = 1 << 8,
    Vertex         = 1 << 9,
    Fragment       = 1 << 10,
    Index          = 1 << 11,
    Input          = 1 << 12,
    Shader         = 1 << 13,
    VertexInput    = Vertex | Input,
    IndexInput     = Index | Input,
    VertexShader   = Vertex | Shader,
    FragmentShader = Fragment | Shader,

    // Transfer
    TransferSrc = 1 << 14,
    TransferDst = 1 << 15,

    // Bindless
    Bindless = (1 << 16) | TransferDst,

    // Attachments
    ColorAttachment = 1 << 17,

    ShaderRead  = 1 << 25,
    UniformRead = 1 << 2,

    ComputeRead  = 1 << 3,
    ComputeWrite = 1 << 26,

    VertexBuffer = (1 << 6) | Vertex,
    IndexBuffer  = (1 << 7) | Index,

    DepthStencil = Depth | Stencil,

    UniformBuffer = (1 << 18),
    UniformTexel  = (1 << 19),

    StorageBufferRead = (1 << 20),
    StorageBuffer     = (1 << 21),
    StorageTexel      = (1 << 22),

    SampledImage = (1 << 23),
    StorageImage = (1 << 24),

    ReadOnly  = ComputeRead | ShaderRead | SampledImage,
    WriteOnly = ComputeWrite | StorageImage,

    ShaderGlobals = StorageBufferRead | Bindless,
    Reset         = 0xffffffff,
};

ENUM_CLASS_FLAGS(ResourceUsage)

enum class ValidationMode
{
    Disabled,
    Enabled,
    Verbose,
};

enum class GPUDevicePreference
{
    Discrete,
    Integrated,
};

typedef u32 DeviceCapabilities;
enum
{
    DeviceCapabilities_MeshShader,
    DeviceCapabilities_VariableShading,
};

enum DescriptorType
{
    DescriptorType_SampledImage,
    DescriptorType_StorageBuffer,
    // DescriptorType_UniformTexel,
    // DescriptorType_StorageTexelBuffer,
    DescriptorType_Count,
};

enum QueryType
{
    QueryType_Occlusion,
    QueryType_Timestamp,
    QueryType_PipelineStatistics,
    QueryType_CompactSize,
};

enum class ShaderStage
{
    Vertex,
    Geometry,
    Fragment,
    Compute,
    Raygen,
    Miss,
    Hit,
    Intersect,
    Count,
};
ENUM_CLASS_FLAGS(ShaderStage)

inline VkShaderStageFlags GetVulkanShaderStage(ShaderStage stage)
{
    VkShaderStageFlags flags = 0;
    if (EnumHasAnyFlags(stage, ShaderStage::Vertex)) flags |= VK_SHADER_STAGE_VERTEX_BIT;
    if (EnumHasAnyFlags(stage, ShaderStage::Geometry)) flags |= VK_SHADER_STAGE_GEOMETRY_BIT;
    if (EnumHasAnyFlags(stage, ShaderStage::Fragment)) flags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    if (EnumHasAnyFlags(stage, ShaderStage::Compute)) flags |= VK_SHADER_STAGE_COMPUTE_BIT;
    if (EnumHasAnyFlags(stage, ShaderStage::Raygen)) flags |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    if (EnumHasAnyFlags(stage, ShaderStage::Miss)) flags |= VK_SHADER_STAGE_MISS_BIT_KHR;
    if (EnumHasAnyFlags(stage, ShaderStage::Hit)) flags |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    if (EnumHasAnyFlags(stage, ShaderStage::Intersect))
        flags |= VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

    Assert(flags);
    return flags;
}

struct Shader
{
    VkShaderModule module;
    ShaderStage stage;
};

enum class RayTracingShaderGroupType
{
    Triangle,
    Procedural,
};

static const int MAX_SHADERS_IN_GROUP = 3;
struct RayTracingShaderGroup
{
    Shader shaders[MAX_SHADERS_IN_GROUP];
    ShaderStage stage[MAX_SHADERS_IN_GROUP];
    int numShaders;
    RayTracingShaderGroupType type;
};

struct PushConstant
{
    ShaderStage stage;
    u32 offset;
    u32 size;
};

// struct GPUBufferDesc
// {
//     u64 size;
//     // BindFlag mFlags    = 0;
//     MemoryUsage usage = MemoryUsage::GPU_ONLY;
//     ResourceUsage resourceUsage;
// };

struct GraphicsObject
{
    void *internalState = 0;

    b32 IsValid() { return internalState != 0; }
};

struct GPUBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    size_t size;
    VkPipelineStageFlags2 lastStage;
    VkAccessFlags2 lastAccess;
};

enum class RTBindings
{
    Accel,
    Image,
    Scene,
    RTBindingData,
    GPUMaterial,
    DenseGeometryData,
    PackedDenseGeometryHeaders,
    ShaderDebugInfo,
};

enum RayShaderTypes
{
    RST_Raygen,
    RST_Miss,
    RST_Hit,
    RST_Intersect,
    RST_Max,
};

struct RayTracingState
{
    VkPipeline pipeline;
    VkPipelineLayout layout;
    union
    {
        struct
        {
            VkStridedDeviceAddressRegionKHR raygen;
            VkStridedDeviceAddressRegionKHR miss;
            VkStridedDeviceAddressRegionKHR hit;
            VkStridedDeviceAddressRegionKHR call;
        };
        VkStridedDeviceAddressRegionKHR addresses[RST_Max];
    };
};

struct QueryPool
{
    VkQueryPool queryPool;
    int count;
};

struct Swapchain
{
    VkSwapchainKHR swapchain;
    VkSurfaceKHR surface;
    VkExtent2D extent;
    VkFormat format;

    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

    std::vector<VkSemaphore> acquireSemaphores;
    VkSemaphore releaseSemaphore;

    u32 acquireSemaphoreIndex;
    u32 imageIndex;
};

struct GPUImage
{
    VkImage image;
    int width;
    int height;
    VkImageView imageView;
    VmaAllocation allocation;
    VkPipelineStageFlags2 lastPipeline;
    VkImageLayout lastLayout;
    VkAccessFlags2 lastAccess;
    VkImageAspectFlags aspect;
};

struct GPUMesh
{
    GPUBuffer buffer;

    u64 deviceAddress;
    u64 vertexOffset;
    u64 vertexSize;
    u64 vertexStride;
    u64 indexOffset;
    u64 indexSize;
    u64 indexStride;
    u64 normalOffset;
    u64 normalSize;
    u64 normalStride;

    u32 numIndices;
    u32 numVertices;
    u32 numFaces;
};

struct GPUAccelerationStructure
{
    VkAccelerationStructureKHR as;
    GPUBuffer buffer;
    VkDeviceAddress address;
    VkAccelerationStructureTypeKHR type;
};

struct TransferBuffer
{
    union
    {
        GPUBuffer buffer;
        GPUImage image;
    };
    GPUBuffer stagingBuffer;
    void *mappedPtr;
};

enum QueueType
{
    QueueType_Graphics,
    QueueType_Compute,
    QueueType_Copy,

    QueueType_Count,
};

struct Semaphore
{
    VkSemaphore semaphore;
    u64 signalValue;
};

struct DescriptorSetLayout;
struct DescriptorSet
{
    union DescriptorInfo
    {
        VkWriteDescriptorSetAccelerationStructureKHR accel;
        VkDescriptorImageInfo image;
        VkDescriptorBufferInfo buffer;
    };

    VkDescriptorSet set;
    std::vector<DescriptorInfo> descriptorInfo;
    std::vector<VkWriteDescriptorSet> writeDescriptorSets;
    DescriptorSetLayout *layout;

    DescriptorSet &Bind(int index, GPUImage *image);
    DescriptorSet &Bind(int index, GPUBuffer *buffer);
    DescriptorSet &Bind(int index, VkAccelerationStructureKHR *accel);
};

struct DescriptorSetLayout
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<bool> nullBindings;
    VkDescriptorSetLayout layout;

    VkPipelineLayout pipelineLayout;

    int AddBinding(u32 binding, VkDescriptorType type, VkShaderStageFlags stage,
                   bool null = false);
    VkDescriptorSetLayout *GetVulkanLayout();
    DescriptorSet CreateDescriptorSet();
    void AddImmutableSamplers();
    void CreatePipelineLayout(PushConstant *pc = 0);
};

struct CommandBuffer
{
    QueueType type;
    VkCommandBuffer buffer;

    VkSemaphore semaphore;
    u64 submissionID;

    u32 currentBuffer = 0;
    // PipelineState *currentPipeline              = 0;

    std::vector<VkMemoryBarrier2> memBarriers;
    std::vector<VkImageMemoryBarrier2> imageBarriers;
    std::vector<VkBufferMemoryBarrier2> bufferBarriers;
    std::vector<Semaphore> waitSemaphores;
    std::vector<Semaphore> signalSemaphores;

    // Descriptor bindings
    VkDescriptorSet descriptorSet;

    // BindedResource srvTable[cMaxBindings] = {};
    // BindedResource uavTable[cMaxBindings] = {};
    // Sampler *samTable[cMaxBindings]       = {};

    void Wait(Semaphore s) { waitSemaphores.push_back(s); }
    void Signal(Semaphore s) { signalSemaphores.push_back(s); }
    void WaitOn(CommandBuffer *other);
    void SubmitTransfer(TransferBuffer *buffer);
    GPUBuffer SubmitBuffer(void *ptr, VkBufferUsageFlags2 flags, size_t totalSize);
    GPUImage SubmitImage(void *ptr, VkImageUsageFlags flags, VkFormat format, VkImageType type,
                         u32 width, u32 height);
    void BindPipeline(VkPipelineBindPoint bindPoint, VkPipeline pipeline);
    void BindDescriptorSets(VkPipelineBindPoint bindPoint, DescriptorSet *set,
                            VkPipelineLayout pipeLayout);
    void TraceRays(RayTracingState *state, u32 width, u32 height, u32 depth);
    void Dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ);
    void Barrier(GPUImage *image, VkImageLayout layout, VkPipelineStageFlags2 stage,
                 VkAccessFlags2 access);
    void Barrier(GPUBuffer *buffer, VkPipelineStageFlags2 stage, VkAccessFlags2 access);
    void Barrier(VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage,
                 VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess);

    void FlushBarriers();
    void PushConstants(PushConstant *pc, void *ptr, VkPipelineLayout layout);
    void CompactAS(QueryPool &pool, GPUAccelerationStructure **as, int count);
    GPUAccelerationStructure BuildAS(VkAccelerationStructureTypeKHR type,
                                     VkAccelerationStructureGeometryKHR *geometries, int count,
                                     VkAccelerationStructureBuildRangeInfoKHR *buildRanges,
                                     u32 *maxPrimitiveCounts);

    void BuildCLAS(GPUBuffer *triangleClusterInfo, GPUBuffer *dstAddresses,
                   GPUBuffer *dstSizes, GPUBuffer *srcInfosCount, u32 offset,
                   ScenePrimitives *scene, int numClusters, u32 numTriangles, u32 numVertices);
    void BuildClusterBLAS(GPUBuffer *bottomLevelInfo, GPUBuffer *dstAddresses,
                          GPUBuffer *srcInfosCount, u32 srcInfosOffset, u32 numClusters);

    GPUBuffer CreateTLASInstances(Instance *instances, int numInstances,
                                  AffineSpace *transforms, ScenePrimitives **childScenes);
    GPUAccelerationStructure BuildTLAS(GPUBuffer *instanceData, u32 numInstances);
    GPUAccelerationStructure BuildTLAS(Instance *instances, int numInstances,
                                       AffineSpace *transforms, ScenePrimitives **childScenes);
    GPUAccelerationStructure BuildBLAS(const GPUMesh *meshes, int count);
    GPUAccelerationStructure BuildCustomBLAS(GPUBuffer *aabbsBuffer, u32 numAabbs);
};

typedef ChunkedLinkedList<CommandBuffer> CommandBufferList;
typedef StaticArray<CommandBufferList> CommandBufferPool;
typedef StaticArray<ChunkedLinkedList<CommandBuffer *>> CommandBufferFreeList;

struct alignas(CACHE_LINE_SIZE) ThreadPool
{
    static const int commandBufferPoolSize = 16;
    Arena *arena;
    u64 currentFrame;

    StaticArray<VkCommandPool> pool;
    CommandBufferPool buffers;
    CommandBufferFreeList freeList;

    VkDescriptorPool descriptorPool[numActiveFrames];
    std::vector<VkDescriptorSet> sets;
};

struct CommandQueue
{
    VkQueue queue;
    Mutex lock = {};

    VkSemaphore submitSemaphore[numActiveFrames];
    u64 submissionID;
};

struct Vulkan
{
    Arena *arena;
    Mutex arenaMutex = {};
    f64 cTimestampPeriod;
    u64 frameCount;
    DeviceCapabilities capabilities;

    //////////////////////////////
    // API State
    //
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkDebugUtilsMessengerEXT debugMessenger;
    StaticArray<VkQueueFamilyProperties2> queueFamilyProperties;
    StaticArray<u32> families;
    u32 graphicsFamily = VK_QUEUE_FAMILY_IGNORED;
    u32 computeFamily  = VK_QUEUE_FAMILY_IGNORED;
    u32 copyFamily     = VK_QUEUE_FAMILY_IGNORED;

    VkPhysicalDeviceMemoryProperties2 memoryProperties;
    VkPhysicalDeviceProperties2 deviceProperties;
    VkPhysicalDeviceVulkan11Properties properties11;
    VkPhysicalDeviceVulkan12Properties properties12;
    VkPhysicalDeviceVulkan13Properties properties13;
    VkPhysicalDeviceVulkan14Properties properties14;

    VkPhysicalDeviceMeshShaderPropertiesEXT meshShaderProperties;
    VkPhysicalDeviceFragmentShadingRatePropertiesKHR variableShadingRateProperties;

    VkPhysicalDeviceFeatures2 deviceFeatures;
    VkPhysicalDeviceVulkan11Features features11;
    VkPhysicalDeviceVulkan12Features features12;
    VkPhysicalDeviceVulkan13Features features13;
    VkPhysicalDeviceVulkan14Features features14;
    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures;
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR variableShadingRateFeatures;

    // RT + CLAS
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelStructProps;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelStructFeats;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipeProperties;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipeFeatures;

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures;

    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clasPropertiesNV;
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clasFeaturesNV;

    VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV reorderPropertiesNV;
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV reorderFeaturesNV;

    VkPhysicalDeviceMemoryProperties2 memProperties;

    //////////////////////////////
    // Queues, command pools & buffers, fences
    //
    CommandQueue queues[QueueType_Count];

    b32 debugUtils = false;

    //////////////////////////////
    // Descriptors
    //
    VkDescriptorPool bindlessPool;

    //////////////////////////////
    // Pipelines
    //
    std::vector<VkDynamicState> dynamicStates;
    VkPipelineDynamicStateCreateInfo dynamicStateInfo;

    struct PipelineStateVulkan
    {
        VkPipeline pipeline;
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        std::vector<VkDescriptorSet> descriptorSets;
        // u32 mCurrentSet = 0;
        VkPipelineLayout pipelineLayout;

        VkPushConstantRange pushConstantRange = {};
        PipelineStateVulkan *next;
    };

    struct ShaderVulkan
    {
        VkShaderModule module;
        VkPipelineShaderStageCreateInfo pipelineStageInfo;
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        VkPushConstantRange pushConstantRange;

        ShaderVulkan *next;
    };

    //////////////////////////////
    // Buffers
    //

    //////////////////////////////
    // Textures/Samplers
    //

    struct TextureVulkan
    {
        VkImage image            = VK_NULL_HANDLE;
        VkBuffer stagingBuffer   = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;

        struct Subresource
        {
            VkImageView imageView = VK_NULL_HANDLE;
            u32 baseLayer;
            u32 numLayers;
            u32 baseMip;
            u32 numMips;
            i32 descriptorIndex;

            b32 IsValid() { return imageView != VK_NULL_HANDLE; }
        };
        Subresource subresource;               // whole view
        std::vector<Subresource> subresources; // sub views
        TextureVulkan *next;
    };

    struct SamplerVulkan
    {
        VkSampler sampler;
    };

    std::vector<VkSampler> immutableSamplers;

    //////////////////////////////
    // Allocation/Deferred cleanup
    //

    VmaAllocator allocator;
    Mutex cleanupMutex = {};
    std::vector<VkSemaphore> cleanupSemaphores[numActiveFrames];
    std::vector<VkSwapchainKHR> cleanupSwapchains[numActiveFrames];
    std::vector<VkImageView> cleanupImageViews[numActiveFrames];
    std::vector<VkBufferView> cleanupBufferViews[numActiveFrames];

    struct CleanupBuffer
    {
        VkBuffer buffer;
        VmaAllocation allocation;
    };
    std::vector<CleanupBuffer> cleanupBuffers[numActiveFrames];
    struct CleanupTexture
    {
        VkImage image;
        VmaAllocation allocation;
    };
    std::vector<CleanupTexture> cleanupTextures[numActiveFrames];
    // std::vector<VmaAllocation> mCleanupAllocations[cNumBuffers];

    void Cleanup();

    // Frame allocators
    // struct FrameData
    // {
    //     GPUBuffer buffer;
    //     std::atomic<u64> offset = 0;
    //     // u64 mTotalSize           = 0;
    //     u32 alignment = 0;
    // } frameAllocator[cNumBuffers];
    //
    // // The gpu buffer should already be created
    // FrameAllocation FrameAllocate(u64 size);
    // void FrameAllocate(GPUBuffer *inBuf, void *inData, Commandstd::vector cmd, u64 inSize =
    // ~0,
    //                    u64 srcOffset = 0);
    // void CommitFrameAllocation(Commandstd::vector cmd, FrameAllocation &alloc,
    //                            GPUBuffer *dstBuffer, u64 dstOffset = 0);

    //////////////////////////////
    // Functions
    //

    Vulkan(ValidationMode validationMode,
           GPUDevicePreference preference = GPUDevicePreference::Discrete);
    Swapchain CreateSwapchain(OS_Handle window, VkFormat format, u32 width, u32 height);
    Semaphore CreateGraphicsSemaphore();
    void AllocateCommandBuffers(ThreadPool &pool, QueueType type);
    void CheckInitializedThreadPool(int threadIndex);
    CommandBuffer *BeginCommandBuffer(QueueType queue);
    void SubmitCommandBuffer(CommandBuffer *cmd);
    VkImageMemoryBarrier2
    ImageMemoryBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                       VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask,
                       VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask,
                       VkImageAspectFlags aspectFlags);
    void CopyFrameBuffer(Swapchain *swapchain, CommandBuffer *cmd, GPUImage *image);
    ThreadPool &GetThreadPool(int threadIndex);
    DescriptorSetLayout CreateDescriptorSetLayout(u32 binding, VkDescriptorType type,
                                                  VkShaderStageFlags stage);
    GPUBuffer CreateBuffer(VkBufferUsageFlags flags, size_t totalSize,
                           VmaAllocationCreateFlags vmaFlags = 0);
    GPUImage CreateImage(VkImageUsageFlags flags, VkFormat format, VkImageType type,
                         int width = 1, int height = 1, int depth = 1, int numMips = 1,
                         int numLayers = 1);
    int BindlessIndex(GPUImage *image);
    int BindlessStorageIndex(GPUBuffer *buffer, size_t offset = 0,
                             size_t range = VK_WHOLE_SIZE);
    u64 GetMinAlignment(VkBufferUsageFlags flags);
    TransferBuffer GetStagingBuffer(VkBufferUsageFlags flags, size_t totalSize,
                                    int numRanges = 0);
    TransferBuffer GetStagingImage(VkImageUsageFlags flags, VkFormat format, VkImageType type,
                                   u32 width, u32 height);
    u64 GetDeviceAddress(VkBuffer buffer);
    void BeginEvent(CommandBuffer *cmd, string name);
    void EndEvent(CommandBuffer *cmd);
    Shader CreateShader(ShaderStage stage, string name, string shaderData);
    void BindAccelerationStructure(VkDescriptorSet descriptorSet,
                                   VkAccelerationStructureKHR accel);
    VkPipeline CreateComputePipeline(Shader *shader, DescriptorSetLayout *layout,
                                     PushConstant *pc = 0);
    RayTracingState CreateRayTracingPipeline(RayTracingShaderGroup *shaderGroups,
                                             int numGroups, PushConstant *pc,
                                             DescriptorSetLayout *layout, u32 maxDepth,
                                             bool useClusters = false);
    QueryPool CreateQuery(QueryType type, int count);

    VkAccelerationStructureInstanceKHR GetVkInstance(const AffineSpace &transform,
                                                     GPUAccelerationStructure &as);
    QueryPool GetCompactionSizes(CommandBuffer *cmd, GPUAccelerationStructure **as, int count);
    void BeginFrame();
    void EndFrame();

    void Wait(Semaphore s);

    // void CopyBuffer(CommandBuffer cmd, GPUBuffer *dest, GPUBuffer *src, u32 size);
    // void ClearBuffer(CommandBuffer cmd, GPUBuffer *dst);
    // void CopyTexture(CommandBuffer cmd, Texture *dst, Texture *src, Rect3U32 *rect = 0);
    // void CopyImage(CommandBuffer cmd, Swapchain *dst, Texture *src);
    // void DeleteBuffer(GPUBuffer *buffer);
    // void DeleteTexture(Texture *texture);
    // void CreateSampler(Sampler *sampler, SamplerDesc desc);
    // void BindSampler(CommandBuffer cmd, Sampler *sampler, u32 slot);
    // void BindResource(GPUResource *resource, ResourceViewType type, u32 slot,
    //                   CommandBuffer cmd, i32 subresource = -1);
    // i32 GetDescriptorIndex(GPUResource *resource, ResourceViewType type,
    //                        i32 subresourceIndex = -1);
    // i32 CreateSubresource(GPUBuffer *buffer, ResourceViewType type, u64 offset = 0ull,
    //                       u64 size = ~0ull, Format format = Format::Null,
    //                       const char *name = 0);
    // i32 CreateSubresource(Texture *texture, u32 baseLayer = 0, u32 numLayers = ~0u,
    //                       u32 baseMip = 0, u32 numMips = ~0u);
    // void UpdateDescriptorSet(CommandBuffer cmd, b8 isCompute = 0);
    // CommandBuffer BeginCommandBuffer(QueueType queue);
    // void BeginRenderPass(Swapchain *inSwapchain, CommandBuffer commandBuffer);
    // void BeginRenderPass(RenderPassImage *images, u32 count, CommandBuffer cmd);
    // void Draw(CommandBuffer cmd, u32 vertexCount, u32 firstVertex);
    // void DrawIndexed(CommandBuffer cmd, u32 indexCount, u32 firstVertex, u32 baseVertex);
    // void DrawIndexedIndirect(CommandBuffer cmd, GPUBuffer *indirectBuffer, u32 drawCount,
    //                          u32 offset = 0, u32 stride = 20);
    // void DrawIndexedIndirectCount(CommandBuffer cmd, GPUBuffer *indirectBuffer,
    //                               GPUBuffer *countBuffer, u32 maxDrawCount,
    //                               u32 indirectOffset = 0, u32 countOffset = 0,
    //                               u32 stride = 20);
    // void BindVertexBuffer(CommandBuffer cmd, GPUBuffer **buffers, u32 count = 1,
    //                       u32 *offsets = 0);
    // void BindIndexBuffer(CommandBuffer cmd, GPUBuffer *buffer, u64 offset = 0);
    // void Dispatch(CommandBuffer cmd, u32 groupCountX, u32 groupCountY, u32 groupCountZ);
    // void DispatchIndirect(CommandBuffer cmd, GPUBuffer *buffer, u32 offset = 0);
    // void SetViewport(CommandBuffer cmd, Viewport *viewport);
    // void SetScissor(CommandBuffer cmd, Rect2 scissor);
    // void EndRenderPass(CommandBuffer cmd);
    // void EndRenderPass(Swapchain *swapchain, CommandBuffer cmd);
    // void SubmitCommandBuffers();
    // void BindCompute(PipelineState *ps, CommandBuffer cmd);
    // void WaitForGPU();
    // void Wait(CommandBuffer waitFor, CommandBuffer cmd);
    // void Wait(CommandBuffer wait);
    // void Barrier(CommandBuffer cmd, GPUBarrier *barriers, u32 count);
    // b32 IsSignaled(FenceTicket ticket);
    // b32 IsLoaded(GPUResource *resource);

    // Query pool
    // void CreateQueryPool(QueryPool *queryPool, QueryType type, u32 queryCount);
    // void BeginQuery(QueryPool *queryPool, CommandBuffer cmd, u32 queryIndex);
    // void EndQuery(QueryPool *queryPool, CommandBuffer cmd, u32 queryIndex);
    // void ResolveQuery(QueryPool *queryPool, CommandBuffer cmd, GPUBuffer *buffer,
    //                   u32 queryIndex, u32 count, u32 destOffset);
    // void ResetQuery(QueryPool *queryPool, CommandBuffer cmd, u32 index, u32 count);
    // u32 GetCount(Fence f);

    // void SetName(GPUResource *resource, const char *name);
    // void SetName(GPUResource *resource, string name);

    u32 GetCurrentBuffer() { return frameCount % numActiveFrames; }
    u32 GetNextBuffer() { return (frameCount + 1) % numActiveFrames; }

    const i32 cPoolSize = 128;
    b32 CreateSwapchain(Swapchain *inSwapchain);

    //////////////////////////////
    // Dedicated transfer queue
    //
    struct RingAllocation
    {
        void *mappedData;
        u64 size;
        u32 offset;
        u32 ringId;
        b8 freed;
    };

    StaticArray<ThreadPool> commandPools;

    //////////////////////////////
    // Bindless resources
    //
    struct BindlessDescriptorPool
    {
        VkDescriptorPool pool        = VK_NULL_HANDLE;
        VkDescriptorSet set          = VK_NULL_HANDLE;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;

        u32 descriptorCount;
        std::vector<i32> freeList;

        Mutex mutex = {};

        i32 Allocate()
        {
            i32 result = -1;
            MutexScope(&mutex)
            {
                if (freeList.size() != 0)
                {
                    result = freeList.back();
                    freeList.pop_back();
                }
            }
            return result;
        }

        void Free(i32 i)
        {
            if (i >= 0)
            {
                MutexScope(&mutex) { freeList.push_back(i); }
            }
        }
    };

    BindlessDescriptorPool bindlessDescriptorPools[DescriptorType_Count];

    std::vector<VkDescriptorSet> bindlessDescriptorSets;
    std::vector<VkDescriptorSetLayout> bindlessDescriptorSetLayouts;

    //////////////////////////////
    // Default samplers
    //
    VkSampler nullSampler;

    VkImage nullImage2D;
    VmaAllocation nullImage2DAllocation;
    VkImageView nullImageView2D;
    VkImageView nullImageView2DArray;

    VkBuffer nullBuffer;
    VmaAllocation nullBufferAllocation;

    //////////////////////////////
    // Debug
    //
    void SetName(u64 handle, VkObjectType type, const char *name);
    void SetName(VkDescriptorSetLayout handle, const char *name);
    void SetName(VkDescriptorSet handle, const char *name);
    void SetName(VkShaderModule handle, const char *name);
    void SetName(VkPipeline handle, const char *name);
    void SetName(VkQueue handle, const char *name);
    void SetName(GPUBuffer *buffer, const char *name);

    //////////////////////////////
    // Memory
    //
    i32 GetMemoryTypeIndex(u32 typeBits, VkMemoryPropertyFlags flags);
};

inline VkTransformMatrixKHR ConvertMatrix(const AffineSpace &space)
{
    VkTransformMatrixKHR matrix;
    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 4; c++)
        {
            matrix.matrix[r][c] = space[c][r];
        }
    }
    return matrix;
}

extern Vulkan *device;

} // namespace rt
#endif
