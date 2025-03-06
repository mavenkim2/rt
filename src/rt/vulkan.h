#ifndef RT_VULKAN_H
#define RT_VULKAN_H

#include "bvh/bvh_types.h"
#define VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#include "../third_party/vulkan/volk.h"

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include "../third_party/vulkan/vk_mem_alloc.h"

// std::vector of things to do:
// 1. create shaders in HLSL and compile
// 2. shader binding table
// 3. create acceleration structures (TLAS and BLAS, how do I handle overlaps? multi
// level instancing?)
// 4. subdivision surfaces using opensubdiv on gpu?
//
// START: render JUST the ocean with real time path tracing. No clusters, no
// subdivision, just triangles.

namespace rt
{

#ifdef _WIN32
typedef HWND Window;
#endif

#define VK_CHECK(check)                                                                       \
    do                                                                                        \
    {                                                                                         \
        VkResult result_ = check;                                                             \
        Assert(result_ == VK_SUCCESS);                                                        \
    } while (0);

static const int cNumBuffers = 2;
using CopyFunction           = std::function<void(void *)>;

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
    DescriptorType_UniformTexel,
    DescriptorType_StorageBuffer,
    DescriptorType_StorageTexelBuffer,
    DescriptorType_Count,
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
};

// struct GPUBuffer
// {
//     GPUBufferDesc desc;
//     void *mappedData;
// };
//

struct Swapchain
{
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkSurfaceKHR surface;
    VkExtent2D extent;
    VkFormat format;

    u32 width;
    u32 height;

    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

    std::vector<VkSemaphore> acquireSemaphores;
    VkSemaphore releaseSemaphore = VK_NULL_HANDLE;

    u32 acquireSemaphoreIndex = 0;
    u32 imageIndex;
};

struct GPUBVH
{
};

struct TransferBuffer
{
    GPUBuffer buffer;
    GPUBuffer stagingBuffer;
    void *mappedPtr;
};

struct TransferCommandBuffer
{
    VkCommandBuffer buffer;
    VkSemaphore semaphore;
    u64 submissionID;
    void SubmitTransfer(TransferBuffer *buffer);
    void SubmitToQueue();
};

enum QueueType
{
    QueueType_Graphics,
    QueueType_Compute,
    QueueType_Copy,

    QueueType_Count,
};

struct CommandBuffer
{
    QueueType type;
    VkCommandBuffer buffer;
    u32 currentBuffer = 0;
    // PipelineState *currentPipeline              = 0;
    VkSemaphore semaphore;
    std::atomic_bool waitedOn{false};

    std::vector<VkImageMemoryBarrier2> endPassImageMemoryBarriers;
    std::vector<Swapchain> updateSwapchains;

    // Descriptor bindings

    // BindedResource srvTable[cMaxBindings] = {};
    // BindedResource uavTable[cMaxBindings] = {};
    // Sampler *samTable[cMaxBindings]       = {};

    // Descriptor set
    // VkDescriptorSet mDescriptorSets[cNumBuffers][QueueType_Count];
    std::vector<VkDescriptorSet> descriptorSets[cNumBuffers];
    u32 currentSet = 0;
    // b32 mIsDirty[cNumBuffers][QueueType_Count] = {};
};

typedef StaticArray<ChunkedLinkedList<VkCommandBuffer>> CommandBufferPool;
typedef ChunkedLinkedList<TransferCommandBuffer> TransferCommandBufferPool;

struct alignas(CACHE_LINE_SIZE) ThreadCommandPool
{
    Arena *arena;
    static const int commandBufferPoolSize = 16;

    StaticArray<VkCommandPool> pool;
    CommandBufferPool buffers;

    TransferCommandBufferPool freeTransferBuffers;
};

struct CommandQueue
{
    VkQueue queue;
    Mutex lock = {};
    u64 submissionID;

    std::vector<VkSemaphore> waitSemaphores;
    std::vector<u64> waitSemaphoreValues;
};

struct GPUAccelerationStructure
{
    VkAccelerationStructureKHR as;
    GPUBuffer buffer;
};

struct GPUMesh
{
    u64 vertexAddress;
    u64 indexAddress;

    u32 numIndices;
    u32 numVertices;
    u32 numFaces;
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
    VkPhysicalDeviceBufferDeviceAddressFeatures deviceAddressFeatures;

    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelStructProps;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelStructFeats;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipeProperties;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipeFeatures;

    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clasPropertiesNV;
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clasFeaturesNV;

    VkPhysicalDeviceMemoryProperties2 memProperties;

    //////////////////////////////
    // Queues, command pools & buffers, fences
    //
    CommandQueue queues[QueueType_Count];
    VkFence frameFences[cNumBuffers][QueueType_Count] = {};

    u32 numCommandLists;
    std::vector<CommandBuffer> commandLists;
    TicketMutex commandMutex = {};

    b32 debugUtils = false;

    //////////////////////////////
    // Descriptors
    //
    VkDescriptorPool pool;

    //////////////////////////////
    // Pipelines
    //
    std::vector<VkDynamicState> dynamicStates;
    VkPipelineDynamicStateCreateInfo dynamicStateInfo;

    // TODO: the descriptor sets shouldn't be here.
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
    // struct GPUBufferVulkan
    // {
    //     VkBuffer buffer;
    //     VmaAllocation allocation;
    //
    //     struct Subresource
    //     {
    //         DescriptorType type;
    //         VkDescriptorBufferInfo info;
    //
    //         // Texel buffer views infromation
    //         Format format;
    //         VkBufferView view = VK_NULL_HANDLE;
    //
    //         // Bindless descriptor index
    //         i32 descriptorIndex = -1;
    //
    //         b32 IsBindless() { return descriptorIndex != -1; }
    //     };
    //     i32 subresourceSrv;
    //     i32 subresourceUav;
    //     std::vector<Subresource> subresources;
    //     GPUBufferVulkan *next;
    // };

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
        SamplerVulkan *next;
    };

    //////////////////////////////
    // Allocation/Deferred cleanup
    //

    VmaAllocator allocator;
    Mutex cleanupMutex = {};
    std::vector<VkSemaphore> cleanupSemaphores[cNumBuffers];
    std::vector<VkSwapchainKHR> cleanupSwapchains[cNumBuffers];
    std::vector<VkImageView> cleanupImageViews[cNumBuffers];
    std::vector<VkBufferView> cleanupBufferViews[cNumBuffers];

    struct CleanupBuffer
    {
        VkBuffer buffer;
        VmaAllocation allocation;
    };
    std::vector<CleanupBuffer> cleanupBuffers[cNumBuffers];
    struct CleanupTexture
    {
        VkImage image;
        VmaAllocation allocation;
    };
    std::vector<CleanupTexture> cleanupTextures[cNumBuffers];
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

    Vulkan(ValidationMode validationMode, GPUDevicePreference preference);
    // u64 GetMinAlignment(GPUBufferDesc *inDesc);
    Swapchain CreateSwapchain(Window window, u32 width, u32 height);
    // void CreatePipeline(PipelineStateDesc *inDesc, PipelineState *outPS, string name);
    // void CreateComputePipeline(PipelineStateDesc *inDesc, PipelineState *outPS, string
    // name); void CreateShader(Shader *shader, string shaderData); void AddPCTemp(Shader
    // *shader, u32 offset, u32 size); void CreateBufferCopy(GPUBuffer *inBuffer, GPUBufferDesc
    // inDesc, CopyFunction initCallback);
    void AllocateCommandBuffers(ThreadCommandPool &pool, QueueType type);
    void AllocateTransferCommandBuffers(ThreadCommandPool &pool);
    void CheckInitializedThreadCommandPool(int threadIndex);
    CommandBuffer BeginCommandBuffer(QueueType queue);
    ThreadCommandPool &GetThreadCommandPool(int threadIndex);
    TransferCommandBuffer BeginTransfers();
    GPUBuffer CreateBuffer(VkBufferUsageFlags flags, size_t totalSize,
                           VmaAllocationCreateFlags vmaFlags = 0);
    TransferBuffer GetStagingBuffer(VkBufferUsageFlags flags, size_t totalSize);
    void SubmitTransfer(TransferBuffer *buffer);
    u64 GetDeviceAddress(VkBuffer buffer);
    void SubmitToQueue();
    GPUAccelerationStructure CreateBLAS(CommandBuffer *cmd, const GPUMesh *meshes, int count);
    void BeginEvent(CommandBuffer *cmd, string name);
    void EndEvent(CommandBuffer *cmd);

    // void CreateBuffer(GPUBuffer *inBuffer, GPUBufferDesc inDesc, void *inData = 0)
    // {
    //     if (inData == 0)
    //     {
    //         CreateBufferCopy(inBuffer, inDesc, 0);
    //     }
    //     else
    //     {
    //         CopyFunction func = [&](void *dest) { MemoryCopy(dest, inData, inDesc.size); };
    //         CreateBufferCopy(inBuffer, inDesc, func);
    //     }
    // }

    // void CopyBuffer(CommandBuffer cmd, GPUBuffer *dest, GPUBuffer *src, u32 size);
    // void ClearBuffer(CommandBuffer cmd, GPUBuffer *dst);
    // void CopyTexture(CommandBuffer cmd, Texture *dst, Texture *src, Rect3U32 *rect = 0);
    // void CopyImage(CommandBuffer cmd, Swapchain *dst, Texture *src);
    // void DeleteBuffer(GPUBuffer *buffer);
    // void CreateTexture(Texture *outTexture, TextureDesc desc, void *inData);
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
    // TransferCommandBuffer BeginTransfers();
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
    // void BindPipeline(PipelineState *ps, CommandBuffer cmd);
    // void BindCompute(PipelineState *ps, CommandBuffer cmd);
    // void PushConstants(CommandBuffer cmd, u32 size, void *data, u32 offset = 0);
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

    u32 GetCurrentBuffer() { return frameCount % cNumBuffers; }
    u32 GetNextBuffer() { return (frameCount + 1) % cNumBuffers; }

private:
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

    StaticArray<ThreadCommandPool> commandPools;

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

    // Linear wrap, nearest wrap, cmp > clamp to edge
    std::vector<VkSampler> immutableSamplers;
    // VkSampler mLinearSampler;
    // VkSampler mNearestSampler;

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

    //////////////////////////////
    // Memory
    //
    i32 GetMemoryTypeIndex(u32 typeBits, VkMemoryPropertyFlags flags);
};

static Vulkan *device;

} // namespace rt
#endif
