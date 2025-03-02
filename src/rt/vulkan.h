#ifndef RT_VULKAN_H
#define RT_VULKAN_H
#include <vulkan.h>

#define VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#include "../third_party/vulkan/vulkan.h"
#include "../third_party/vulkan/volk.h"
#include "../third_party/vulkan/vk_mem_alloc.h"

namespace rt
{

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

    u32 GetCurrentBuffer() override { return frameCount % cNumBuffers; }

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
    FrameAllocation FrameAllocate(u64 size) override;
    void FrameAllocate(GPUBuffer *inBuf, void *inData, CommandList cmd, u64 inSize = ~0,
                       u64 srcOffset = 0) override;
    void CommitFrameAllocation(CommandList cmd, FrameAllocation &alloc, GPUBuffer *dstBuffer,
                               u64 dstOffset = 0) override;

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
    u64 GetMinAlignment(GPUBufferDesc *inDesc) override;
    b32 CreateSwapchain(Window window, SwapchainDesc *desc, Swapchain *swapchain) override;
    void CreatePipeline(PipelineStateDesc *inDesc, PipelineState *outPS, string name) override;
    void CreateComputePipeline(PipelineStateDesc *inDesc, PipelineState *outPS,
                               string name) override;
    void CreateShader(Shader *shader, string shaderData) override;
    void AddPCTemp(Shader *shader, u32 offset, u32 size) override;
    void CreateBufferCopy(GPUBuffer *inBuffer, GPUBufferDesc inDesc,
                          CopyFunction initCallback) override;
    void CopyBuffer(CommandList cmd, GPUBuffer *dest, GPUBuffer *src, u32 size) override;
    void ClearBuffer(CommandList cmd, GPUBuffer *dst) override;
    void CopyTexture(CommandList cmd, Texture *dst, Texture *src, Rect3U32 *rect = 0) override;
    void CopyImage(CommandList cmd, Swapchain *dst, Texture *src) override;
    void DeleteBuffer(GPUBuffer *buffer) override;
    void CreateTexture(Texture *outTexture, TextureDesc desc, void *inData) override;
    void DeleteTexture(Texture *texture) override;
    void CreateSampler(Sampler *sampler, SamplerDesc desc) override;
    void BindSampler(CommandList cmd, Sampler *sampler, u32 slot) override;
    void BindResource(GPUResource *resource, ResourceViewType type, u32 slot, CommandList cmd,
                      i32 subresource = -1) override;
    i32 GetDescriptorIndex(GPUResource *resource, ResourceViewType type,
                           i32 subresourceIndex = -1) override;
    i32 CreateSubresource(GPUBuffer *buffer, ResourceViewType type, u64 offset = 0ull,
                          u64 size = ~0ull, Format format = Format::Null,
                          const char *name = 0) override;
    i32 CreateSubresource(Texture *texture, u32 baseLayer = 0, u32 numLayers = ~0u,
                          u32 baseMip = 0, u32 numMips = ~0u) override;
    void UpdateDescriptorSet(CommandList cmd, b8 isCompute = 0);
    CommandList BeginCommandList(QueueType queue) override;
    void BeginRenderPass(Swapchain *inSwapchain, CommandList inCommandList) override;
    void BeginRenderPass(RenderPassImage *images, u32 count, CommandList cmd) override;
    void Draw(CommandList cmd, u32 vertexCount, u32 firstVertex) override;
    void DrawIndexed(CommandList cmd, u32 indexCount, u32 firstVertex,
                     u32 baseVertex) override;
    void DrawIndexedIndirect(CommandList cmd, GPUBuffer *indirectBuffer, u32 drawCount,
                             u32 offset = 0, u32 stride = 20) override;
    void DrawIndexedIndirectCount(CommandList cmd, GPUBuffer *indirectBuffer,
                                  GPUBuffer *countBuffer, u32 maxDrawCount,
                                  u32 indirectOffset = 0, u32 countOffset = 0,
                                  u32 stride = 20) override;
    void BindVertexBuffer(CommandList cmd, GPUBuffer **buffers, u32 count = 1,
                          u32 *offsets = 0) override;
    void BindIndexBuffer(CommandList cmd, GPUBuffer *buffer, u64 offset = 0) override;
    void Dispatch(CommandList cmd, u32 groupCountX, u32 groupCountY, u32 groupCountZ) override;
    void DispatchIndirect(CommandList cmd, GPUBuffer *buffer, u32 offset = 0) override;
    void SetViewport(CommandList cmd, Viewport *viewport) override;
    void SetScissor(CommandList cmd, Rect2 scissor) override;
    void EndRenderPass(CommandList cmd) override;
    void EndRenderPass(Swapchain *swapchain, CommandList cmd) override;
    void SubmitCommandLists() override;
    void BindPipeline(PipelineState *ps, CommandList cmd) override;
    void BindCompute(PipelineState *ps, CommandList cmd) override;
    void PushConstants(CommandList cmd, u32 size, void *data, u32 offset = 0) override;
    void WaitForGPU() override;
    void Wait(CommandList waitFor, CommandList cmd) override;
    void Wait(CommandList wait) override;
    void Barrier(CommandList cmd, GPUBarrier *barriers, u32 count) override;
    b32 IsSignaled(FenceTicket ticket) override;
    b32 IsLoaded(GPUResource *resource) override;

    // Query pool
    void CreateQueryPool(QueryPool *queryPool, QueryType type, u32 queryCount) override;
    void BeginQuery(QueryPool *queryPool, CommandList cmd, u32 queryIndex) override;
    void EndQuery(QueryPool *queryPool, CommandList cmd, u32 queryIndex) override;
    void ResolveQuery(QueryPool *queryPool, CommandList cmd, GPUBuffer *buffer, u32 queryIndex,
                      u32 count, u32 destOffset) override;
    void ResetQuery(QueryPool *queryPool, CommandList cmd, u32 index, u32 count) override;
    u32 GetCount(Fence f) override;

    // Debug
    void BeginEvent(CommandList cmd, string name) override;
    void EndEvent(CommandList cmd) override;
    void SetName(GPUResource *resource, const char *name) override;
    void SetName(GPUResource *resource, string name) override;

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
};

Vulkan::Vulkan(ValidationMode validationMode, GPUDevicePreference preference)
{
    arena           = ArenaAlloc();
    const i32 major = 0;
    const i32 minor = 0;
    const i32 patch = 1;

    VK_CHECK(volkInitialize());

    // Create the application
    VkApplicationInfo appInfo  = {};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "RT";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.pEngineName        = "RT";
    appInfo.engineVersion      = VK_MAKE_API_VERSION(0, major, minor, patch);
    appInfo.apiVersion         = VK_API_VERSION_1_4;

    // Load available layers
    u32 layerCount = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, 0));
    list<VkLayerProperties> availableLayers(layerCount);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()));

    // Load extension info
    u32 extensionCount = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(0, &extensionCount, 0));
    list<VkExtensionProperties> extensionProperties(extensionCount);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(0, &extensionCount,
                                                    extensionProperties.data()));

    list<const char *> instanceExtensions;
    list<const char *> instanceLayers;
    // Add extensions
    for (auto &availableExtension : extensionProperties)
    {
        if (strcmp(availableExtension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
        {
            debugUtils = true;
            instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }
    instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef WINDOWS
    instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
#error not supported
#endif

    // Add layers
    if (validationMode != ValidationMode::Disabled)
    {
        static const list<const char *> validationPriorityList[] = {
            // Preferred
            {"VK_LAYER_KHRONOS_validation"},
            // Fallback
            {"VK_LAYER_LUNARG_standard_validation"},
            // Individual
            {
                "VK_LAYER_GOOGLE_threading",
                "VK_LAYER_LUNARG_parameter_validation",
                "VK_LAYER_LUNARG_object_tracker",
                "VK_LAYER_LUNARG_core_validation",
                "VK_LAYER_GOOGLE_unique_objects",
            },
            // Last resort
            {
                "VK_LAYER_LUNARG_core_validation",
            },
        };
        for (auto &validationLayers : validationPriorityList)
        {
            bool validated = true;
            for (auto &layer : validationLayers)
            {
                bool found = false;
                for (auto &availableLayer : availableLayers)
                {
                    if (strcmp(availableLayer.layerName, layer) == 0)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    validated = false;
                    break;
                }
            }

            if (validated)
            {
                for (auto &c : validationLayers)
                {
                    instanceLayers.push_back(c);
                }
                break;
            }
        }
    }

    // Create instance
    {
        Assert(volkGetInstanceVersion() >= VK_API_VERSION_1_4);
        VkInstanceCreateInfo instInfo    = {};
        instInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instInfo.pApplicationInfo        = &appInfo;
        instInfo.enabledLayerCount       = (u32)instanceLayers.size();
        instInfo.ppEnabledLayerNames     = instanceLayers.data();
        instInfo.enabledExtensionCount   = (u32)instanceExtensions.size();
        instInfo.ppEnabledExtensionNames = instanceExtensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo = {};

        debugUtilsCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        if (validationMode != ValidationMode::Disabled && debugUtils)
        {
            debugUtilsCreateInfo.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
            debugUtilsCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            if (validationMode == ValidationMode::Verbose)
            {
                debugUtilsCreateInfo.messageSeverity |=
                    (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT);
            }

            debugUtilsCreateInfo.pfnUserCallback = DebugUtilsMessengerCallback;
            instInfo.pNext                       = &debugUtilsCreateInfo;
        }

        VK_CHECK(vkCreateInstance(&instInfo, 0, &instance));
        volkLoadInstanceOnly(instance);

        if (validationMode != ValidationMode::Disabled && debugUtils)
        {
            VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsCreateInfo, 0,
                                                    &debugMessenger));
        }
    }

    // Enumerate physical devices
    {
        u32 deviceCount = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, 0));
        Assert(deviceCount != 0);

        list<VkPhysicalDevice> devices(deviceCount);
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()));

        list<const char *> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        VkPhysicalDevice preferred = VK_NULL_HANDLE;
        VkPhysicalDevice fallback  = VK_NULL_HANDLE;

        for (auto &testDevice : devices)
        {
            VkPhysicalDeviceProperties2 props = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            vkGetPhysicalDeviceProperties2(testDevice, &props);
            if (props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;

            u32 queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties2(testDevice, &queueFamilyCount, 0);

            list<VkQueueFamilyProperties2> queueFamilyProps;
            queueFamilyProps.resize(queueFamilyCount);
            for (u32 i = 0; i < queueFamilyCount; i++)
            {
                queueFamilyProps[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
            }

            vkGetPhysicalDeviceQueueFamilyProperties2(testDevice, &queueFamilyCount,
                                                      queueFamilyProps.data());

            u32 graphicsIndex = VK_QUEUE_FAMILY_IGNORED;
            for (u32 i = 0; i < queueFamilyCount; i++)
            {
                if (queueFamilyProps[i].queueFamilyProperties.queueFlags &
                    VK_QUEUE_GRAPHICS_BIT)
                {
                    graphicsIndex = i;
                    break;
                }
            }
            if (graphicsIndex == VK_QUEUE_FAMILY_IGNORED) continue;

#ifdef _WIN32
            if (!vkGetPhysicalDeviceWin32PresentationSupportKHR(testDevice, graphicsIndex))
                continue;
#endif
            if (props.properties.apiVersion < VK_API_VERSION_1_4) continue;

            b32 suitable = props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            if (preference == GPUDevicePreference::Integrated)
            {
                suitable =
                    props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
            }
            if (!preferred && suitable)
            {
                preferred = testDevice;
            }
            if (!fallback)
            {
                fallback = testDevice;
            }
        }
        physicalDevice = preferred ? preferred : fallback;
        if (!physicalDevice)
        {
            Print("Error: No GPU selected\n");
            Assert(0);
        }
        // Printf("Selected GPU: %s\n", deviceProperties.properties.deviceName);

        deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features11.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        features12.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features13.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features14.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        deviceFeatures.pNext = &features11;
        features11.pNext     = &features12;
        features12.pNext     = &features13;
        features13.pNext     = &features14;

        void **featuresChain = &features14.pNext;
        *featuresChain       = 0;

        deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties11.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        properties12.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
        properties13.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        properties14.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
        deviceProperties.pNext = &properties11;
        properties11.pNext     = &properties12;
        properties12.pNext     = &properties13;
        properties13.pNext     = &properties14;
        void **propertiesChain = &properties14.pNext;

        u32 deviceExtCount = 0;
        VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &deviceExtCount, 0));
        list<VkExtensionProperties> availableDevExt(deviceExtCount);
        VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &deviceExtCount,
                                                      availableDevExt.data()));

        auto checkAndAddExtension = [&](const char *extName, auto *prop = 0, auto *feat = 0) {
            for (auto &extension : availableDevExt)
            {
                if (strcmp(extension.extensionName, extName) == 0)
                {
                    if (prop)
                    {
                        *propertiesChain = prop;
                        propertiesChain  = &prop->pNext;
                    }
                    if (feat)
                    {
                        *featuresChain = &feat;
                        featuresChain  = &feat->pNext;
                    }
                    deviceExtensions.push_back(extName);
                    return true;
                }
            }
            return false;
        };

        if (checkExtension(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME))
        {
            deviceExtensions.push_back(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME);
        }

        meshShaderProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
        meshShaderFeatures   = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        if (checkAndAddExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshShaderProperties,
                                 &meshShaderFeatures))
        {
            capabilities |= DeviceCapabilities_MeshShader;
        }
        variableShadingRateProperties = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR};
        variableShadingRateFeatures = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};
        if (checkAndAddExtension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME,
                                 &variableShadingRateProperties, &variableShadingRateFeatures))
        {
            capabilities |= DeviceCapabilities_VariableShading;
        }

        // Ray tracing extensions
        {
            accelStructProps = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
            accelStructFeats = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
            bool result = checkAndAddExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                               &accelStructProps, &accelStructFeats);
            ErrorExit(result,
                      "Machine doesn't support VK_KHR_acceleration_structure. Exiting\n");

            rtPipeProperties = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
            rtPipeFeatures = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
            result = checkAndAddExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                          &rtPipeProperties, &rtPipeFeatures);
            ErrorExit(result,
                      "Machine doesn't support VK_KHR_acceleration_structure. Exiting\n");

            checkAndAddExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

            clasPropertiesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
            clasFeaturesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};
            bool result =
                checkAndAddExtension(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                     &clasPropertiesNV, &clasFeaturesNV);
            ErrorExit(
                result,
                "Machine doesn't support VK_NV_cluster_acceleration_structure. Exiting\n");
        }

        vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures);

        // Ensure core functionlity is supported
        Assert(deviceFeatures.features.multiDrawIndirect == VK_TRUE);
        Assert(deviceFeatures.features.pipelineStatisticsQuery == VK_TRUE);
        Assert(features13.dynamicRendering == VK_TRUE);
        Assert(features12.descriptorIndexing == VK_TRUE);
        Assert(features12.bufferDeviceAddress == VK_TRUE);

        if (capabilities & DeviceCapabilities_MeshShader)
        {
            Assert(meshShaderFeatures.meshShader == VK_TRUE);
            Assert(meshShaderFeatures.taskShader == VK_TRUE);
        }

        vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties);
        cTimestampPeriod = (f64)deviceProperties.properties.limits.timestampPeriod * 1e-9;

        u32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &queueFamilyCount, 0);
        queueFamilyProperties.resize(queueFamilyCount);
        for (u32 i = 0; i < queueFamilyCount; i++)
        {
            queueFamilyProperties[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        }
        vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &queueFamilyCount,
                                                  queueFamilyProperties.data());

        // Device exposes 1+ queue families, queue families have 1+ queues. Each family
        // supports a combination of the below:
        // 1. Graphics
        // 2. Compute
        // 3. Transfer
        // 4. Sparse Memory Management

        // Find queues in queue family
        for (u32 i = 0; i < queueFamilyProperties.size(); i++)
        {
            auto &queueFamily = queueFamilyProperties[i];
            if (queueFamily.queueFamilyProperties.queueCount > 0)
            {
                if (graphicsFamily == VK_QUEUE_FAMILY_IGNORED &&
                    queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    graphicsFamily = i;
                }
                if ((queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                    (copyFamily == VK_QUEUE_FAMILY_IGNORED ||
                     (!(queueFamily.queueFamilyProperties.queueFlags &
                        VK_QUEUE_GRAPHICS_BIT) &&
                      !(queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT))))

                {
                    copyFamily = i;
                }
                if ((queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    (computeFamily == VK_QUEUE_FAMILY_IGNORED ||
                     !(queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT)))

                {
                    computeFamily = i;
                }
            }
        }

        // Create the device queues
        list<VkDeviceQueueCreateInfo> queueCreateInfos;
        f32 queuePriority = 1.f;
        for (u32 i = 0; i < 3; i++)
        {
            u32 queueFamily = 0;
            if (i == 0)
            {
                queueFamily = graphicsFamily;
            }
            else if (i == 1)
            {
                if (graphicsFamily == computeFamily)
                {
                    continue;
                }
                queueFamily = computeFamily;
            }
            else if (i == 2)
            {
                if (graphicsFamily == copyFamily || computeFamily == copyFamily)
                {
                    continue;
                }
                queueFamily = copyFamily;
            }
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount       = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);

            families.push_back(queueFamily);
        }

        VkDeviceCreateInfo createInfo      = {};
        createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount    = (u32)queueCreateInfos.size();
        createInfo.pQueueCreateInfos       = queueCreateInfos.data();
        createInfo.pEnabledFeatures        = 0;
        createInfo.pNext                   = &deviceFeatures;
        createInfo.enabledExtensionCount   = (u32)deviceExtensions.size();
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, 0, &device));

        volkLoadDevice(device);
    }

    memoryProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memoryProperties);

    // Get the device queues
    vkGetDeviceQueue(device, graphicsFamily, 0, &queues[QueueType_Graphics].queue);
    vkGetDeviceQueue(device, computeFamily, 0, &queues[QueueType_Compute].queue);
    vkGetDeviceQueue(device, copyFamily, 0, &queues[QueueType_Copy].queue);

    SetName(queues[QueueType_Graphics].queue, "Graphics Queue");
    SetName(queues[QueueType_Copy].queue, "Transfer Queue");

    // TODO: unified memory access architectures
    memProperties       = {};
    memProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProperties);

    VmaAllocatorCreateInfo allocCreateInfo = {};
    allocCreateInfo.physicalDevice         = physicalDevice;
    allocCreateInfo.device                 = device;
    allocCreateInfo.instance               = instance;
    allocCreateInfo.vulkanApiVersion       = VK_API_VERSION_1_4;
    // these are promoted to core, so this doesn't do anything
    allocCreateInfo.flags = VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;

#if VMA_DYNAMIC_VULKAN_FUNCTIONS
    VmaVulkanFunctions vulkanFunctions    = {};
    vulkanFunctions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    allocCreateInfo.pVulkanFunctions      = &vulkanFunctions;
#else
#error
#endif

    VK_CHECK(vmaCreateAllocator(&allocCreateInfo, &allocator));

    // Set up dynamic pso
    dynamicStates = {
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_VIEWPORT,
    };

    // Set up frame fences
    for (u32 buffer = 0; buffer < cNumBuffers; buffer++)
    {
        for (u32 queue = 0; queue < QueueType_Count; queue++)
        {
            if (queues[queue].queue == VK_NULL_HANDLE)
            {
                continue;
            }
            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VK_CHECK(vkCreateFence(device, &fenceInfo, 0, &frameFences[buffer][queue]));
        }
    }

    dynamicStateInfo                   = {};
    dynamicStateInfo.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = (u32)dynamicStates.size();
    dynamicStateInfo.pDynamicStates    = dynamicStates.data();

    // Init descriptor pool
    {
        VkDescriptorPoolSize poolSizes[2];

        u32 count = 0;
        // Uniform buffers
        poolSizes[count].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[count].descriptorCount = cPoolSize;
        count++;

        // Combined samplers
        poolSizes[count].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[count].descriptorCount = cPoolSize;

        VkDescriptorPoolCreateInfo createInfo = {};
        createInfo.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.poolSizeCount              = count;
        createInfo.pPoolSizes                 = poolSizes;
        createInfo.maxSets                    = cPoolSize;

        VK_CHECK(vkCreateDescriptorPool(device, &createInfo, 0, &pool));
    }

    // Bindless descriptor pools
    {
        for (DescriptorType type = (DescriptorType)0; type < DescriptorType_Count;
             type                = (DescriptorType)(type + 1))
        {
            VkDescriptorType descriptorType;
            switch (type)
            {
                case DescriptorType_SampledImage:
                    descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    break;
                case DescriptorType_UniformTexel:
                    descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                    break;
                case DescriptorType_StorageBuffer:
                    descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    break;
                case DescriptorType_StorageTexelBuffer:
                    descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                    break;
                default: Assert(0);
            }

            BindlessDescriptorPool &bindlessDescriptorPool = bindlessDescriptorPools[type];
            VkDescriptorPoolSize poolSize                  = {};
            poolSize.type                                  = descriptorType;
            if (type == DescriptorType_StorageBuffer ||
                type == DescriptorType_StorageTexelBuffer)
            {
                poolSize.descriptorCount =
                    Min(10000,
                        deviceProperties.properties.limits.maxDescriptorSetStorageBuffers / 4);
            }
            else if (type == DescriptorType_SampledImage)
            {
                poolSize.descriptorCount =
                    Min(10000,
                        deviceProperties.properties.limits.maxDescriptorSetSampledImages / 4);
            }
            else if (type == DescriptorType_UniformTexel)
            {
                poolSize.descriptorCount =
                    Min(10000,
                        deviceProperties.properties.limits.maxDescriptorSetUniformBuffers / 4);
            }
            bindlessDescriptorPool.descriptorCount = poolSize.descriptorCount;

            VkDescriptorPoolCreateInfo createInfo = {};
            createInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            createInfo.poolSizeCount = 1;
            createInfo.pPoolSizes    = &poolSize;
            createInfo.maxSets       = 1;
            createInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
            VK_CHECK(
                vkCreateDescriptorPool(device, &createInfo, 0, &bindlessDescriptorPool.pool));

            VkDescriptorSetLayoutBinding binding = {};
            binding.binding                      = 0;
            binding.pImmutableSamplers           = 0;
            binding.stageFlags                   = VK_SHADER_STAGE_ALL;
            binding.descriptorType               = descriptorType;
            binding.descriptorCount              = bindlessDescriptorPool.descriptorCount;

            // These flags enable bindless:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorBindingFlagBits.html
            VkDescriptorBindingFlags bindingFlags =
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCreate = {};
            bindingFlagsCreate.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            bindingFlagsCreate.bindingCount  = 1;
            bindingFlagsCreate.pBindingFlags = &bindingFlags;

            VkDescriptorSetLayoutCreateInfo createSetLayout = {};
            createSetLayout.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            createSetLayout.bindingCount = 1;
            createSetLayout.pBindings    = &binding;
            createSetLayout.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
            createSetLayout.pNext = &bindingFlagsCreate;

            VK_CHECK(vkCreateDescriptorSetLayout(device, &createSetLayout, 0,
                                                 &bindlessDescriptorPool.layout));

            VkDescriptorSetAllocateInfo allocInfo = {};
            allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool     = bindlessDescriptorPool.pool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts        = &bindlessDescriptorPool.layout;
            VK_CHECK(
                vkAllocateDescriptorSets(device, &allocInfo, &bindlessDescriptorPool.set));

            for (u32 i = 0; i < poolSize.descriptorCount; i++)
            {
                bindlessDescriptorPool.freeList.push_back(poolSize.descriptorCount - i - 1);
            }
            bindlessDescriptorSets.push_back(bindlessDescriptorPool.set);
            bindlessDescriptorSetLayouts.push_back(bindlessDescriptorPool.layout);

            // Set debug names
            TempArena temp = ScratchStart(0, 0);
            string typeName;
            switch (type)
            {
                case DescriptorType_SampledImage: typeName = "Sampled Image"; break;
                case DescriptorType_StorageBuffer: typeName = "Storage Buffer"; break;
                case DescriptorType_UniformTexel: typeName = "Uniform Texel Buffer"; break;
                case DescriptorType_StorageTexelBuffer:
                    typeName = "Storage Texel Buffer";
                    break;
            }
            string name =
                PushStr8F(temp.arena, "Bindless Descriptor Set Layout: %S", typeName);
            SetName(bindlessDescriptorPool.layout, (const char *)name.str);

            name = PushStr8F(temp.arena, "Bindless Descriptor Set: %S", typeName);
            SetName(bindlessDescriptorPool.set, (const char *)name.str);
            ScratchEnd(temp);
        }
    }

    // Init frame allocators
    {
        GPUBufferDesc desc;
        desc.usage         = MemoryUsage::CPU_TO_GPU;
        desc.size          = megabytes(32);
        desc.resourceUsage = ResourceUsage_TransferSrc;
        for (u32 i = 0; i < cNumBuffers; i++)
        {
            CreateBuffer(&frameAllocator[i].buffer, desc, 0);
            frameAllocator[i].alignment = 16;
        }
    }

    // Initialize ring buffer
    {
        u32 ringBufferSize = megabytes(128);
        GPUBufferDesc desc;
        desc.usage         = MemoryUsage::CPU_TO_GPU;
        desc.size          = ringBufferSize;
        desc.resourceUsage = ResourceUsage_TransferSrc;

        for (u32 i = 0; i < ArrayLength(stagingRingAllocators); i++)
        {
            RingAllocator &stagingRingAllocator = stagingRingAllocators[i];
            CreateBuffer(&stagingRingAllocator.transferRingBuffer, desc, 0);
            SetName(&stagingRingAllocator.transferRingBuffer, "Transfer Staging Buffer");

            stagingRingAllocator.ringBufferSize = ringBufferSize;
            stagingRingAllocator.writePos = stagingRingAllocator.readPos = 0;
            stagingRingAllocator.allocationReadPos                       = 0;
            stagingRingAllocator.allocationWritePos                      = 0;
            stagingRingAllocator.alignment                               = 16;
            stagingRingAllocator.lock.Init();
        }
    }

    // Default samplers
    {
        // Null sampler
        VkSamplerCreateInfo samplerCreate = {};
        samplerCreate.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &nullSampler));

        samplerCreate.anisotropyEnable        = VK_FALSE;
        samplerCreate.maxAnisotropy           = 0;
        samplerCreate.minLod                  = 0;
        samplerCreate.maxLod                  = FLT_MAX;
        samplerCreate.mipLodBias              = 0;
        samplerCreate.unnormalizedCoordinates = VK_FALSE;
        samplerCreate.compareEnable           = VK_FALSE;
        samplerCreate.compareOp               = VK_COMPARE_OP_NEVER;

        samplerCreate.minFilter    = VK_FILTER_LINEAR;
        samplerCreate.magFilter    = VK_FILTER_LINEAR;
        samplerCreate.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCreate.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreate.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreate.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        // sampler linear wrap
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));

        // samler nearest wrap
        samplerCreate.minFilter  = VK_FILTER_NEAREST;
        samplerCreate.magFilter  = VK_FILTER_NEAREST;
        samplerCreate.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));

        // sampler linear clamp
        samplerCreate.minFilter    = VK_FILTER_LINEAR;
        samplerCreate.magFilter    = VK_FILTER_LINEAR;
        samplerCreate.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCreate.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreate.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreate.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));

        // sampler nearest clamp
        samplerCreate.minFilter  = VK_FILTER_NEAREST;
        samplerCreate.magFilter  = VK_FILTER_NEAREST;
        samplerCreate.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));

        // sampler nearest compare
        samplerCreate.compareEnable = VK_TRUE;
        samplerCreate.compareOp     = VK_COMPARE_OP_GREATER_OR_EQUAL;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));
    }

    // Default views
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType         = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width      = 1;
        imageInfo.extent.height     = 1;
        imageInfo.extent.depth      = 1;
        imageInfo.mipLevels         = 1;
        imageInfo.arrayLayers       = 1;
        imageInfo.format            = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &nullImage2D,
                                &nullImage2DAllocation, 0));

        VkImageViewCreateInfo createInfo           = {};
        createInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount     = 1;
        createInfo.subresourceRange.baseMipLevel   = 0;
        createInfo.subresourceRange.levelCount     = 1;
        createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.format                          = VK_FORMAT_R8G8B8A8_UNORM;

        createInfo.image    = nullImage2D;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

        VK_CHECK(vkCreateImageView(device, &createInfo, 0, &nullImageView2D));

        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        VK_CHECK(vkCreateImageView(device, &createInfo, 0, &nullImageView2DArray));

        // Transitions
        TransferCommand cmd = Stage(0);

        VkImageMemoryBarrier2 imageBarrier       = {};
        imageBarrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        imageBarrier.image                       = nullImage2D;
        imageBarrier.oldLayout                   = imageInfo.initialLayout;
        imageBarrier.newLayout                   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageBarrier.srcAccessMask               = VK_ACCESS_2_NONE;
        imageBarrier.dstAccessMask               = VK_ACCESS_2_SHADER_READ_BIT;
        imageBarrier.srcStageMask                = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        imageBarrier.dstStageMask                = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.baseMipLevel   = 0;
        imageBarrier.subresourceRange.layerCount     = 1;
        imageBarrier.subresourceRange.levelCount     = 1;
        imageBarrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;

        VkDependencyInfo dependencyInfo        = {};
        dependencyInfo.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers    = &imageBarrier;

        vkCmdPipelineBarrier2(cmd.transitionBuffer, &dependencyInfo);

        Submit(cmd);
    }

    // Null buffer
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size               = 4;
        bufferInfo.usage =
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.preferredFlags          = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        VK_CHECK(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &nullBuffer,
                                 &nullBufferAllocation, 0));
    }
}

void RT()
{
    VkAccelerationStructureGeometryKHR geometry = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = ;

    vkCmdBuildkj
}
} // namespace rt
#endif
