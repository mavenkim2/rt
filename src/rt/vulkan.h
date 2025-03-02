#ifndef RT_VULKAN_H
#define RT_VULKAN_H
#include "base.h"
#include "scene_load.h"
#include <vulkan.h>

#define VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#include "../third_party/vulkan/vulkan.h"
#include "../third_party/vulkan/volk.h"
#include "../third_party/vulkan/vk_mem_alloc.h"

// List of things to do:
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

using CopyFunction = std::function<void(void *)>;

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

struct GPUBufferDesc
{
    u64 size;
    // BindFlag mFlags    = 0;
    MemoryUsage usage = MemoryUsage::GPU_ONLY;
    ResourceUsage resourceUsage;
};

struct GraphicsObject
{
    void *internalState = 0;

    b32 IsValid() { return internalState != 0; }
};

struct GPUBuffer : GraphicsObject
{
    GPUBufferDesc desc;
    void *mappedData;
};

struct GPUBVH : GraphicsObject
{
};

struct CommandList : GraphicsObject
{
};

struct Vulkan
{
    Vulkan(ValidationMode validationMode, GPUDevicePreference preference);

    Arena *arena;
    Mutex arenaMutex = {};
    DeviceCapabilities capabilities;

    //////////////////////////////
    // API State
    //
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkDebugUtilsMessengerEXT debugMessenger;
    list<VkQueueFamilyProperties2> queueFamilyProperties;
    list<u32> families;
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

    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clasPropertiesNV;
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clasFeaturesNV;

    VkPhysicalDeviceMemoryProperties2 memProperties;

    //////////////////////////////
    // Queues, command pools & buffers, fences
    //
    struct CommandQueue
    {
        VkQueue queue;
        Mutex lock = {};
    } queues[QueueType_Count];

    VkFence frameFences[cNumBuffers][QueueType_Count] = {};

    struct CommandListVulkan
    {
        QueueType type;
        VkCommandPool commandPools[cNumBuffers]     = {};
        VkCommandBuffer commandBuffers[cNumBuffers] = {};
        u32 currentBuffer                           = 0;
        PipelineState *currentPipeline              = 0;
        VkSemaphore semaphore;
        list<CommandList> waitForCmds;
        std::atomic_bool waitedOn{false};

        list<VkImageMemoryBarrier2> endPassImageMemoryBarriers;
        list<Swapchain> updateSwapchains;

        // Descriptor bindings

        BindedResource srvTable[cMaxBindings] = {};
        BindedResource uavTable[cMaxBindings] = {};
        Sampler *samTable[cMaxBindings]       = {};

        // Descriptor set
        // VkDescriptorSet mDescriptorSets[cNumBuffers][QueueType_Count];
        list<VkDescriptorSet> descriptorSets[cNumBuffers];
        u32 currentSet = 0;
        // b32 mIsDirty[cNumBuffers][QueueType_Count] = {};

        CommandListVulkan *next;

        const VkCommandBuffer GetCommandBuffer() const
        {
            return commandBuffers[currentBuffer];
        }

        const VkCommandPool GetCommandPool() const { return commandPools[currentBuffer]; }
    };

    // TODO: consider using buckets?
    u32 numCommandLists = 0;
    list<CommandListVulkan *> commandLists;
    // list<CommandListVulkan *> computeCommandLists;
    TicketMutex commandMutex = {};

    CommandListVulkan &GetCommandList(CommandList cmd)
    {
        Assert(cmd.IsValid());
        return *(CommandListVulkan *)(cmd.internalState);
    }

    u32 GetCurrentBuffer() { return frameCount % cNumBuffers; }

    u32 GetNextBuffer() { return (frameCount + 1) % cNumBuffers; }

    b32 debugUtils = false;

    // Shaders
    // VkPipelineShaderStage
    struct SwapchainVulkan
    {
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VkSurfaceKHR surface;
        VkExtent2D extent;

        list<VkImage> images;
        list<VkImageView> imageViews;

        list<VkSemaphore> acquireSemaphores;
        VkSemaphore releaseSemaphore = VK_NULL_HANDLE;

        u32 acquireSemaphoreIndex = 0;
        u32 imageIndex;

        SwapchainVulkan *next;
    };

    //////////////////////////////
    // Descriptors
    //
    VkDescriptorPool pool;

    //////////////////////////////
    // Pipelines
    //
    list<VkDynamicState> dynamicStates;
    VkPipelineDynamicStateCreateInfo dynamicStateInfo;

    // TODO: the descriptor sets shouldn't be here.
    struct PipelineStateVulkan
    {
        VkPipeline pipeline;
        list<VkDescriptorSetLayoutBinding> layoutBindings;
        list<VkDescriptorSetLayout> descriptorSetLayouts;
        list<VkDescriptorSet> descriptorSets;
        // u32 mCurrentSet = 0;
        VkPipelineLayout pipelineLayout;

        VkPushConstantRange pushConstantRange = {};
        PipelineStateVulkan *next;
    };

    struct ShaderVulkan
    {
        VkShaderModule module;
        VkPipelineShaderStageCreateInfo pipelineStageInfo;
        list<VkDescriptorSetLayoutBinding> layoutBindings;
        VkPushConstantRange pushConstantRange;

        ShaderVulkan *next;
    };

    //////////////////////////////
    // Buffers
    //
    struct GPUBufferVulkan
    {
        VkBuffer buffer;
        VmaAllocation allocation;

        struct Subresource
        {
            DescriptorType type;
            VkDescriptorBufferInfo info;

            // Texel buffer views infromation
            Format format;
            VkBufferView view = VK_NULL_HANDLE;

            // Bindless descriptor index
            i32 descriptorIndex = -1;

            b32 IsBindless() { return descriptorIndex != -1; }
        };
        i32 subresourceSrv;
        i32 subresourceUav;
        list<Subresource> subresources;
        GPUBufferVulkan *next;
    };

    //////////////////////////////
    // Fences
    //
    struct FenceVulkan
    {
        u32 count;
        VkFence fence;
        FenceVulkan *next;
    };

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
        Subresource subresource;        // whole view
        list<Subresource> subresources; // sub views
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
    list<VkSemaphore> cleanupSemaphores[cNumBuffers];
    list<VkSwapchainKHR> cleanupSwapchains[cNumBuffers];
    list<VkImageView> cleanupImageViews[cNumBuffers];
    list<VkBufferView> cleanupBufferViews[cNumBuffers];

    struct CleanupBuffer
    {
        VkBuffer buffer;
        VmaAllocation allocation;
    };
    list<CleanupBuffer> cleanupBuffers[cNumBuffers];
    struct CleanupTexture
    {
        VkImage image;
        VmaAllocation allocation;
    };
    list<CleanupTexture> cleanupTextures[cNumBuffers];
    // list<VmaAllocation> mCleanupAllocations[cNumBuffers];

    void Cleanup();

    // Frame allocators
    struct FrameData
    {
        GPUBuffer buffer;
        std::atomic<u64> offset = 0;
        // u64 mTotalSize           = 0;
        u32 alignment = 0;
    } frameAllocator[cNumBuffers];

    // The gpu buffer should already be created
    FrameAllocation FrameAllocate(u64 size);
    void FrameAllocate(GPUBuffer *inBuf, void *inData, CommandList cmd, u64 inSize = ~0,
                       u64 srcOffset = 0);
    void CommitFrameAllocation(CommandList cmd, FrameAllocation &alloc, GPUBuffer *dstBuffer,
                               u64 dstOffset = 0);

    //////////////////////////////
    // Functions
    //
    SwapchainVulkan *freeSwapchain     = 0;
    CommandListVulkan *freeCommandList = 0;
    PipelineStateVulkan *freePipeline  = 0;
    GPUBufferVulkan *freeBuffer        = 0;
    TextureVulkan *freeTexture         = 0;
    SamplerVulkan *freeSampler         = 0;
    ShaderVulkan *freeShader           = 0;
    FenceVulkan *freeFence             = 0;

    SwapchainVulkan *ToInternal(Swapchain *swapchain)
    {
        Assert(swapchain->IsValid());
        return (SwapchainVulkan *)(swapchain->internalState);
    }

    CommandListVulkan *ToInternal(CommandList commandlist)
    {
        Assert(commandlist.IsValid());
        return (CommandListVulkan *)(commandlist.internalState);
    }

    PipelineStateVulkan *ToInternal(PipelineState *ps)
    {
        Assert(ps->IsValid());
        return (PipelineStateVulkan *)(ps->internalState);
    }

    GPUBufferVulkan *ToInternal(GPUBuffer *gb)
    {
        Assert(gb->IsValid());
        return (GPUBufferVulkan *)(gb->internalState);
    }

    TextureVulkan *ToInternal(Texture *texture)
    {
        Assert(texture->IsValid());
        return (TextureVulkan *)(texture->internalState);
    }

    ShaderVulkan *ToInternal(Shader *shader)
    {
        Assert(shader->IsValid());
        return (ShaderVulkan *)(shader->internalState);
    }

    FenceVulkan *ToInternal(Fence *fence)
    {
        Assert(fence->IsValid());
        return (FenceVulkan *)(fence->internalState);
    }

    VkQueryPool ToInternal(QueryPool *queryPool)
    {
        Assert(queryPool->IsValid());
        return (VkQueryPool)(queryPool->internalState);
    }

    SamplerVulkan *ToInternal(Sampler *sampler)
    {
        Assert(sampler->IsValid());
        return (SamplerVulkan *)(sampler->internalState);
    }

    mkGraphicsVulkan(ValidationMode validationMode, GPUDevicePreference preference);
    u64 GetMinAlignment(GPUBufferDesc *inDesc);
    b32 CreateSwapchain(Window window, SwapchainDesc *desc, Swapchain *swapchain);
    void CreatePipeline(PipelineStateDesc *inDesc, PipelineState *outPS, string name);
    void CreateComputePipeline(PipelineStateDesc *inDesc, PipelineState *outPS, string name);
    void CreateShader(Shader *shader, string shaderData);
    void AddPCTemp(Shader *shader, u32 offset, u32 size);
    void CreateBufferCopy(GPUBuffer *inBuffer, GPUBufferDesc inDesc,
                          CopyFunction initCallback);

    void CreateBuffer(GPUBuffer *inBuffer, GPUBufferDesc inDesc, void *inData = 0)
    {
        if (inData == 0)
        {
            CreateBufferCopy(inBuffer, inDesc, 0);
        }
        else
        {
            CopyFunction func = [&](void *dest) { MemoryCopy(dest, inData, inDesc.size); };
            CreateBufferCopy(inBuffer, inDesc, func);
        }
    }

    void CopyBuffer(CommandList cmd, GPUBuffer *dest, GPUBuffer *src, u32 size);
    void ClearBuffer(CommandList cmd, GPUBuffer *dst);
    void CopyTexture(CommandList cmd, Texture *dst, Texture *src, Rect3U32 *rect = 0);
    void CopyImage(CommandList cmd, Swapchain *dst, Texture *src);
    void DeleteBuffer(GPUBuffer *buffer);
    void CreateTexture(Texture *outTexture, TextureDesc desc, void *inData);
    void DeleteTexture(Texture *texture);
    void CreateSampler(Sampler *sampler, SamplerDesc desc);
    void BindSampler(CommandList cmd, Sampler *sampler, u32 slot);
    void BindResource(GPUResource *resource, ResourceViewType type, u32 slot, CommandList cmd,
                      i32 subresource = -1);
    i32 GetDescriptorIndex(GPUResource *resource, ResourceViewType type,
                           i32 subresourceIndex = -1);
    i32 CreateSubresource(GPUBuffer *buffer, ResourceViewType type, u64 offset = 0ull,
                          u64 size = ~0ull, Format format = Format::Null,
                          const char *name = 0);
    i32 CreateSubresource(Texture *texture, u32 baseLayer = 0, u32 numLayers = ~0u,
                          u32 baseMip = 0, u32 numMips = ~0u);
    void UpdateDescriptorSet(CommandList cmd, b8 isCompute = 0);
    CommandList BeginCommandList(QueueType queue);
    void BeginRenderPass(Swapchain *inSwapchain, CommandList inCommandList);
    void BeginRenderPass(RenderPassImage *images, u32 count, CommandList cmd);
    void Draw(CommandList cmd, u32 vertexCount, u32 firstVertex);
    void DrawIndexed(CommandList cmd, u32 indexCount, u32 firstVertex, u32 baseVertex);
    void DrawIndexedIndirect(CommandList cmd, GPUBuffer *indirectBuffer, u32 drawCount,
                             u32 offset = 0, u32 stride = 20);
    void DrawIndexedIndirectCount(CommandList cmd, GPUBuffer *indirectBuffer,
                                  GPUBuffer *countBuffer, u32 maxDrawCount,
                                  u32 indirectOffset = 0, u32 countOffset = 0,
                                  u32 stride = 20);
    void BindVertexBuffer(CommandList cmd, GPUBuffer **buffers, u32 count = 1,
                          u32 *offsets = 0);
    void BindIndexBuffer(CommandList cmd, GPUBuffer *buffer, u64 offset = 0);
    void Dispatch(CommandList cmd, u32 groupCountX, u32 groupCountY, u32 groupCountZ);
    void DispatchIndirect(CommandList cmd, GPUBuffer *buffer, u32 offset = 0);
    void SetViewport(CommandList cmd, Viewport *viewport);
    void SetScissor(CommandList cmd, Rect2 scissor);
    void EndRenderPass(CommandList cmd);
    void EndRenderPass(Swapchain *swapchain, CommandList cmd);
    void SubmitCommandLists();
    void BindPipeline(PipelineState *ps, CommandList cmd);
    void BindCompute(PipelineState *ps, CommandList cmd);
    void PushConstants(CommandList cmd, u32 size, void *data, u32 offset = 0);
    void WaitForGPU();
    void Wait(CommandList waitFor, CommandList cmd);
    void Wait(CommandList wait);
    void Barrier(CommandList cmd, GPUBarrier *barriers, u32 count);
    b32 IsSignaled(FenceTicket ticket);
    b32 IsLoaded(GPUResource *resource);

    // Query pool
    void CreateQueryPool(QueryPool *queryPool, QueryType type, u32 queryCount);
    void BeginQuery(QueryPool *queryPool, CommandList cmd, u32 queryIndex);
    void EndQuery(QueryPool *queryPool, CommandList cmd, u32 queryIndex);
    void ResolveQuery(QueryPool *queryPool, CommandList cmd, GPUBuffer *buffer, u32 queryIndex,
                      u32 count, u32 destOffset);
    void ResetQuery(QueryPool *queryPool, CommandList cmd, u32 index, u32 count);
    u32 GetCount(Fence f);

    // Debug
    void BeginEvent(CommandList cmd, string name);
    void EndEvent(CommandList cmd);
    void SetName(GPUResource *resource, const char *name);
    void SetName(GPUResource *resource, string name);

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
    Mutex mTransferMutex = {};
    struct TransferCommand
    {
        VkCommandPool cmdPool     = VK_NULL_HANDLE; // command pool to issue transfer request
        VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
        VkCommandPool transitionPool =
            VK_NULL_HANDLE; // command pool to issue transfer request
        VkCommandBuffer transitionBuffer = VK_NULL_HANDLE;
        // VkFence mFence                               = VK_NULL_HANDLE; // signals cpu that
        // transfer is complete
        Fence fence;
        VkSemaphore semaphores[QueueType_Count - 1] = {}; // graphics, compute
        RingAllocation *ringAllocation;

        const b32 IsValid() { return cmdPool != VK_NULL_HANDLE; }
    };
    list<TransferCommand> transferFreeList;

    TransferCommand Stage(u64 size);

    void Submit(TransferCommand cmd);

    struct RingAllocator
    {
        TicketMutex lock;
        GPUBuffer transferRingBuffer;
        u64 ringBufferSize;
        u32 writePos;
        u32 readPos;
        u32 alignment;

        RingAllocation allocations[256];
        u16 allocationReadPos;
        u16 allocationWritePos;

    } stagingRingAllocators[4];

    // NOTE: there is a potential case where the allocation has transferred, but the fence
    // isn't signaled (when command buffer is being reused). current solution is to just not
    // care, since it doesn't impact anything yet.
    RingAllocation *RingAlloc(u64 size);
    RingAllocation *RingAllocInternal(u32 ringId, u64 size);
    void RingFree(RingAllocation *allocation);

    //////////////////////////////
    // Bindless resources
    //
    struct BindlessDescriptorPool
    {
        VkDescriptorPool pool        = VK_NULL_HANDLE;
        VkDescriptorSet set          = VK_NULL_HANDLE;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;

        u32 descriptorCount;
        list<i32> freeList;

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

    list<VkDescriptorSet> bindlessDescriptorSets;
    list<VkDescriptorSetLayout> bindlessDescriptorSetLayouts;

    //////////////////////////////
    // Default samplers
    //
    VkSampler nullSampler;

    // Linear wrap, nearest wrap, cmp > clamp to edge
    list<VkSampler> immutableSamplers;
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

    //////////////////////////////
    // Ray tracing
    //
    struct GPUBVHVulkan
    {
        VkAccelerationStructureKHR as;
    };

    void CreateRayTracingPipeline();
    GPUBVH *CreateBLAS(const GPUMesh *meshes, int count);
};

static Vulkan *device;

} // namespace rt
#endif
