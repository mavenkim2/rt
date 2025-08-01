#ifndef RT_VULKAN_H
#define RT_VULKAN_H

#include "../base.h"
#include "../bvh/bvh_types.h"
#include "../containers.h"
#include "../math/math_include.h"
#include "../platform.h"

#define VK_NO_PROTOTYPES

#include "../../third_party/vulkan/vulkan/vulkan.h"
#include "../../third_party/vulkan/volk.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "../../third_party/vulkan/vk_mem_alloc.h"

namespace rt
{

struct AffineSpace;
struct Instance;
struct Mesh;

#define VK_CHECK(check)                                                                       \
    do                                                                                        \
    {                                                                                         \
        VkResult result_ = check;                                                             \
        Assert(result_ == VK_SUCCESS);                                                        \
    } while (0);

static const int numActiveFrames = 1;

inline u32 GetFormatSize(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_B8G8R8_UNORM:
        case VK_FORMAT_R8G8B8_SRGB: return 3;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB: return 4;
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return 8;
        default: Assert(0); return 0;
    }
}

inline u32 GetBlockShift(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return 2;
        default: return 0;
    }
}

inline u32 GetBlockSize(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        {
            return 4;
        }
        break;
        default: return 1;
    }
}

inline u32 GetNumLevels(u32 width, u32 height)
{
    return Max(1, Max(Log2Int(NextPowerOfTwo(width)), Log2Int(NextPowerOfTwo(height))));
}

enum class CLASOpType
{
    Move = (1 << 0),
    CLAS = (1 << 1),
    BLAS = (1 << 2),
};
ENUM_CLASS_FLAGS(CLASOpType)

enum class CLASOpMode
{
    ImplicitDestinations,
    ExplicitDestinations,
    ComputeSizes,
};

struct CLASOpInput
{
    union
    {
        struct MoveObjects
        {
            u64 maxMovedBytes;
            int maxNumClusters;
            bool noMoveOverlap;
        } moveObjects;

        struct TriangleClusters
        {
            // TODO: add the rest
            u32 maxNumTriangles;
            u32 maxNumVertices;
        } triangleClusters;

        struct ClusterBottomLevel
        {
            u32 maxTotalClusterCount;
            u32 maxClusterCountPerAccelerationStructure;
        } clusterBottomLevel;
    };
    u32 maxAccelerationStructureCount;
};

union VkOpInput
{
    VkClusterAccelerationStructureClustersBottomLevelInputNV clustersBottomLevel;
    VkClusterAccelerationStructureTriangleClusterInputNV triangleClusters;
    VkClusterAccelerationStructureMoveObjectsInputNV moveObjects;
};

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

enum class MemoryUsage
{
    GPU_ONLY,
    CPU_ONLY,
    CPU_TO_GPU,
    GPU_TO_CPU,
};
ENUM_CLASS_FLAGS(MemoryUsage)

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

    PushConstant() {}
    PushConstant(ShaderStage stage, u32 offset, u32 size)
        : stage(stage), offset(offset), size(size)
    {
    }
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
    void *mappedPtr;
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
    GPUMaterial     = 4,
    PageTable       = 5,
    PhysicalPages   = 6,
    ShaderDebugInfo = 7,

    ClusterPageData = 8,
    PtexFaceData    = 11,
    Feedback        = 12,
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
    QueryType type;
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
    std::vector<VkSemaphore> releaseSemaphores;

    u32 acquireSemaphoreIndex;
    u32 imageIndex;
};

enum class ImageType
{
    Type1D,
    Array1D,
    Type2D,
    Array2D,
    Cubemap,
};

struct ImageLimits
{
    u32 max1DImageDim;
    u32 max2DImageDim;
    u32 maxNumLayers;
};

struct ImageDesc
{
    ImageType imageType = ImageType::Type2D;
    u32 width           = 1;
    u32 height          = 1;
    u32 depth           = 1;
    u32 numMips         = 1;
    u32 numLayers       = 1;
    VkFormat format;
    MemoryUsage memUsage = MemoryUsage::GPU_ONLY;
    VkImageUsageFlags imageUsage;
    VkImageTiling tiling;

    ImageDesc() {}
    ImageDesc(ImageType imageType, u32 width, u32 height, u32 depth, u32 numMips,
              u32 numLayers, VkFormat format, MemoryUsage memUsage,
              VkImageUsageFlags imageUsage, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL)
        : imageType(imageType), width(width), height(height), depth(depth), numMips(numMips),
          numLayers(numLayers), format(format), memUsage(memUsage), imageUsage(imageUsage),
          tiling(tiling)
    {
    }
};

struct GPUImage
{
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    ImageDesc desc;

    VkPipelineStageFlags2 lastPipeline;
    VkImageLayout lastLayout;
    VkAccessFlags2 lastAccess;
    VkImageAspectFlags aspect;

    struct Subresource
    {
        VkImageView imageView;
        u32 baseLayer;
        u32 numLayers;
        u32 baseMip;
        u32 numMips;
        // i32 descriptorIndex;
    };
    StaticArray<Subresource> subresources;

    GPUImage() {}
};

struct BufferToBufferCopy
{
    u32 srcOffset;
    u32 dstOffset;
    u32 size;
};

struct BufferImageCopy
{
    u64 bufferOffset;
    u32 rowLength;
    u32 imageHeight;

    u32 mipLevel;
    u32 baseLayer;
    u32 layerCount = 1;

    Vec3i offset;
    Vec3u extent;
};

struct ImageToImageCopy
{
    u32 srcMipLevel;
    u32 srcBaseLayer;
    u32 srcLayerCount = 1;
    Vec3i srcOffset;

    u32 dstMipLevel;
    u32 dstBaseLayer;
    u32 dstLayerCount = 1;
    Vec3i dstOffset;

    Vec3u extent;
};

struct GPUAccelerationStructure
{
    VkAccelerationStructureKHR as;
    GPUBuffer buffer;
    VkDeviceAddress address;
    VkAccelerationStructureTypeKHR type;
};

struct GPUAccelerationStructurePayload
{
    GPUAccelerationStructure as;
    GPUBuffer scratch;
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

    TransferBuffer() {}
};

enum QueueType
{
    QueueType_Graphics,
    QueueType_Compute,
    QueueType_Copy,

    QueueType_Count,
};

enum QueueFlag
{
    QueueFlag_Graphics = 1 << 0,
    QueueFlag_Compute  = 1 << 1,
    QueueFlag_Copy     = 1 << 2,
};

struct Semaphore
{
    VkSemaphore semaphore;
    u64 signalValue;
};

enum class DescriptorType
{
    Sampler,
    CombinedImageSampler,
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    UniformBufferDynamic,
    StorageBufferDynamic,
    InputAttachment,
    AccelerationStructure,
    PTLAS,

    Count,
};

struct DescriptorSetLayout;
struct CommandBuffer;

struct DescriptorSet
{
    union DescriptorInfo
    {
        VkWriteDescriptorSetAccelerationStructureKHR accel;
        VkDescriptorImageInfo image;
        VkDescriptorBufferInfo buffer;
        VkWriteDescriptorSetPartitionedAccelerationStructureNV ptlas;
    };

    VkDescriptorPool pool;
    VkDescriptorSet set;
    std::vector<DescriptorInfo> descriptorInfo;
    std::vector<VkWriteDescriptorSet> writeDescriptorSets;
    DescriptorSetLayout *layout;

    u32 numBinds = 0;

    DescriptorSet &Bind(int index, GPUImage *image, int subresource = -1);
    DescriptorSet &Bind(int index, GPUBuffer *buffer, u64 offset = 0,
                        u64 size = VK_WHOLE_SIZE);
    DescriptorSet &Bind(int index, VkAccelerationStructureKHR *accel);
    DescriptorSet &Bind(int index, u64 *ptlasAddress);

    DescriptorSet &Bind(GPUBuffer *buffer, u64 offset = 0, u64 size = VK_WHOLE_SIZE);
    DescriptorSet &Bind(GPUImage *img);
    DescriptorSet &Bind(VkAccelerationStructureKHR *accel);
    DescriptorSet &Bind(u64 *ptlasAddress);
    void Reset();
};

struct DescriptorSetLayout
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayout layout;

    VkPipelineLayout pipelineLayout;

    int AddBinding(u32 binding, DescriptorType type, VkShaderStageFlags stage);
    VkDescriptorSetLayout *GetVulkanLayout();
    DescriptorSet CreateDescriptorSet();
    DescriptorSet CreateNewDescriptorSet();
    void AddImmutableSamplers();
    void CreatePipelineLayout(PushConstant *pc = 0);
};

struct ResourceBinding
{
    VkPipelineBindPoint bindPoint;
    CommandBuffer *cmd;
    DescriptorSet ds;

    ResourceBinding &Bind(int index, GPUImage *image, int subresource = -1)
    {
        ds.Bind(index, image, subresource);
        return *this;
    }
    ResourceBinding &Bind(int index, GPUBuffer *buffer, u64 offset = 0,
                          u64 size = VK_WHOLE_SIZE)
    {
        ds.Bind(index, buffer, offset, size);
        return *this;
    }
    ResourceBinding &Bind(int index, VkAccelerationStructureKHR *accel)
    {
        ds.Bind(index, accel);
        return *this;
    }

    ResourceBinding &Bind(GPUBuffer *buffer, u64 offset = 0, u64 size = VK_WHOLE_SIZE)
    {
        ds.Bind(buffer, offset, size);
        return *this;
    }
    ResourceBinding &Bind(GPUImage *img)
    {
        ds.Bind(img);
        return *this;
    }
    ResourceBinding &Bind(VkAccelerationStructureKHR *accel)
    {
        ds.Bind(accel);
        return *this;
    }
    ResourceBinding &PushConstants(PushConstant *push, void *ptr);
    void End();
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
    void SignalOutsideFrame(Semaphore s)
    {
        signalSemaphores.push_back(s);
        submissionID = s.signalValue;
        semaphore    = s.semaphore;
    }
    void WaitOn(CommandBuffer *other);
    void CopyBuffer(GPUBuffer *dst, GPUBuffer *src);
    void SubmitTransfer(TransferBuffer *buffer, u32 dstOffset = 0);
    TransferBuffer SubmitBuffer(void *ptr, VkBufferUsageFlags2 flags, size_t totalSize,
                                u32 dstOffset = 0);
    void SubmitBuffer(GPUBuffer *dst, void *ptr, size_t totalSize, u32 dstOffset = 0);
    TransferBuffer SubmitImage(void *ptr, ImageDesc desc);
    void CopyBuffer(GPUBuffer *dst, GPUBuffer *src, BufferToBufferCopy *copies, u32 num);
    void CopyImage(GPUBuffer *transfer, GPUImage *image, BufferImageCopy *copies, u32 num);
    void CopyImage(GPUImage *dst, GPUImage *src, const ImageToImageCopy &copy);
    void CopyImageToBuffer(GPUBuffer *dst, GPUImage *src, const BufferImageCopy *copies,
                           u32 num);

    void BindPipeline(VkPipelineBindPoint bindPoint, VkPipeline pipeline);
    void BindDescriptorSets(VkPipelineBindPoint bindPoint, DescriptorSet *set,
                            VkPipelineLayout pipeLayout);
    ResourceBinding StartBinding(VkPipelineBindPoint bindPoint, VkPipeline pipeline,
                                 DescriptorSetLayout *layout);
    ResourceBinding StartBindingCompute(VkPipeline pipeline, DescriptorSetLayout *layout);

    void TraceRays(RayTracingState *state, u32 width, u32 height, u32 depth);
    void Dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ);
    void DispatchIndirect(GPUBuffer *buffer, u32 offset = 0);
    void Barrier(GPUImage *image, VkImageLayout oldLayout, VkImageLayout newLayout,
                 VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask,
                 VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask,
                 QueueType fromQueue, QueueType toQueue);
    void Barrier(GPUImage *image, VkImageLayout layout, VkPipelineStageFlags2 stage,
                 VkAccessFlags2 access);
    void Barrier(GPUBuffer *buffer, VkPipelineStageFlags2 stage, VkAccessFlags2 access);
    void Barrier(VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage,
                 VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess);
    void TransferWriteBarrier(GPUImage *image);
    void UAVBarrier(GPUImage *image);
    void FlushBarriers();
    void PushConstants(PushConstant *pc, void *ptr, VkPipelineLayout layout);
    QueryPool GetCompactionSizes(const GPUAccelerationStructurePayload *as);
    GPUAccelerationStructure CompactAS(QueryPool &pool,
                                       const GPUAccelerationStructurePayload *as);

    GPUAccelerationStructurePayload
    BuildAS(VkAccelerationStructureTypeKHR type,
            VkAccelerationStructureGeometryKHR *geometries, int count,
            VkAccelerationStructureBuildRangeInfoKHR *buildRanges, u32 *maxPrimitiveCounts);

    VkAccelerationStructureKHR
    BuildAS(GPUBuffer *accel, GPUBuffer *scratch, VkAccelerationStructureTypeKHR accelType,
            VkAccelerationStructureGeometryKHR *geometries, int count,
            VkAccelerationStructureBuildRangeInfoKHR *buildRanges, u32 *maxPrimitiveCounts);

    void CLASIndirect(CLASOpInput opInput, CLASOpMode opMode, CLASOpType opType,
                      GPUBuffer *dstImplicitData, GPUBuffer *scratchBuffer,
                      GPUBuffer *dstAddresses, GPUBuffer *dstSizes, GPUBuffer *srcInfosArray,
                      GPUBuffer *srcInfosCount, u32 srcInfosOffset, u32 dstClasOffset = 0);
    void ComputeCLASSizes(GPUBuffer *srcInfosArray, GPUBuffer *scratchBuffer,
                          GPUBuffer *dstSizes, GPUBuffer *srcInfosCount, u32 srcInfosOffset,
                          u32 dstClasOffset, u32 maxNumTriangles, u32 maxNumVertices,
                          u32 maxNumClusters);
    void ComputeBLASSizes(GPUBuffer *srcInfosArray, GPUBuffer *scratchBuffer,
                          GPUBuffer *dstSizes, GPUBuffer *srcInfosCount, u32 srcInfosOffset,
                          u32 maxTotalClusterCount,
                          u32 maxClusterCountPerAccelerationStructure,
                          u32 maxAccelerationStructureCount);
    void BuildCLAS(CLASOpMode opMode, GPUBuffer *dstImplicitData, GPUBuffer *scratchBuffer,
                   GPUBuffer *triangleClusterInfo, GPUBuffer *dstAddresses,
                   GPUBuffer *dstSizes, GPUBuffer *srcInfosCount, u32 srcInfosOffset,
                   int maxNumClusters, u32 maxNumTriangles, u32 maxNumVertices,
                   u32 dstClasOffset);
    void MoveCLAS(CLASOpMode opMode, GPUBuffer *dstImplicitData, GPUBuffer *scratchBuffer,
                  GPUBuffer *dstAddresses, GPUBuffer *dstSizes, GPUBuffer *srcInfosArray,
                  GPUBuffer *srcInfosCount, u32 srcInfosOffset, int maxNumClusters,
                  u64 maxMovedBytes, bool noMoveOverlap, u32 dstClasOffset = 0);
    void BuildPTLAS(GPUBuffer *ptlasBuffer, GPUBuffer *scratchBuffer, GPUBuffer *srcInfos,
                    GPUBuffer *srcInfosCount, u32 srcInfosOffset, u32 instanceCount,
                    u32 maxInstancesPerPartition, u32 partitionCount,
                    u32 maxInstanceInGlobalPartitionCount);
    void BuildClusterBLAS(CLASOpMode opMode, GPUBuffer *implicitBuffer,
                          GPUBuffer *scratchBuffer, GPUBuffer *bottomLevelInfo,
                          GPUBuffer *dstAddresses, GPUBuffer *dstSizes,
                          GPUBuffer *srcInfosCount, u32 srcInfosOffset,
                          u32 maxClusterCountPerAccelerationStructure,
                          u32 maxTotalClusterCount, u32 maxAccelerationStructureCount);

    GPUAccelerationStructurePayload BuildTLAS(GPUBuffer *instanceData, u32 numInstances);
    VkAccelerationStructureKHR BuildTLAS(GPUBuffer *accelBuffer, GPUBuffer *scratchBuffer,
                                         GPUBuffer *instanceData, u32 numInstances);
    VkAccelerationStructureKHR BuildTLAS(GPUBuffer *accelBuffer, GPUBuffer *scratchBuffer,
                                         GPUBuffer *instanceData, GPUBuffer *buildRangeBuffer,
                                         u32 maxInstances);
    GPUAccelerationStructurePayload BuildCustomBLAS(GPUBuffer *aabbsBuffer, u32 numAabbs);
    void ClearBuffer(GPUBuffer *b, u32 val = 0);
    void ClearImage(GPUImage *image, u32 value, u32 baseMip = 0,
                    u32 numMips = VK_REMAINING_MIP_LEVELS, u32 baseLayer = 0,
                    u32 numLayers = VK_REMAINING_ARRAY_LAYERS);
    void ClearImage(GPUImage *image, const Vec4f &value, u32 baseMip = 0,
                    u32 numMips = VK_REMAINING_MIP_LEVELS, u32 baseLayer = 0,
                    u32 numLayers = VK_REMAINING_ARRAY_LAYERS);

    void BeginQuery(QueryPool *queryPool, u32 queryIndex);
    void EndQuery(QueryPool *queryPool, u32 queryIndex);
    void ResolveQuery(QueryPool *queryPool, GPUBuffer *gpuBuffer, u32 queryIndex, u32 count,
                      u32 destOffset);
    void ResetQuery(QueryPool *queryPool, u32 index, u32 count);
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
    static const u32 numBuffers = 2;
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

    VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV ptlasPropertiesNV;
    VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV ptlasFeaturesNV;

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
    inline f64 GetTimestampPeriod() { return cTimestampPeriod; }
    Swapchain CreateSwapchain(OS_Handle window, VkFormat format, u32 width, u32 height);
    ImageLimits GetImageLimits();
    Semaphore CreateSemaphore();
    void AllocateCommandBuffers(ThreadPool &pool, QueueType type);
    void CheckInitializedThreadPool(int threadIndex);
    CommandBuffer *BeginCommandBuffer(QueueType queue, string name = "Command Buffer");
    void SubmitCommandBuffer(CommandBuffer *cmd, bool frame = false, bool parallel = false);
    VkImageMemoryBarrier2
    ImageMemoryBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                       VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask,
                       VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask,
                       VkImageAspectFlags aspectFlags, QueueType fromQueue = QueueType_Count,
                       QueueType toQueue = QueueType_Count);
    void CopyFrameBuffer(Swapchain *swapchain, CommandBuffer *cmd, GPUImage *image);
    ThreadPool &GetThreadPool(int threadIndex);
    DescriptorSetLayout CreateDescriptorSetLayout(u32 binding, VkDescriptorType type,
                                                  VkShaderStageFlags stage);
    GPUBuffer CreateBuffer(VkBufferUsageFlags flags, size_t totalSize,
                           MemoryUsage usage = MemoryUsage::GPU_ONLY);
    GPUImage CreateImage(ImageDesc desc, int numSubresources = -1, int ownedQueue = -1);
    int CreateSubresource(GPUImage *image, u32 baseMip = 0,
                          u32 numMips = VK_REMAINING_MIP_LEVELS, u32 baseLayer = 0,
                          u32 numLayers = VK_REMAINING_ARRAY_LAYERS);
    u32 GetImageSize(GPUImage *image);
    void DestroyBuffer(GPUBuffer *buffer);
    void DestroyImage(GPUImage *image);
    void DestroyAccelerationStructure(GPUAccelerationStructure *as);
    void DestroyPool(VkDescriptorPool pool);
    int BindlessIndex(GPUImage *image);
    int BindlessStorageIndex(GPUImage *image, int subresourceIndex = -1);
    int BindlessStorageIndex(GPUBuffer *buffer, size_t offset = 0,
                             size_t range = VK_WHOLE_SIZE);
    u64 GetMinAlignment(VkBufferUsageFlags flags);
    TransferBuffer GetStagingBuffer(VkBufferUsageFlags flags, size_t totalSize,
                                    int numRanges = 0);
    TransferBuffer GetStagingImage(ImageDesc desc);
    TransferBuffer GetReadbackBuffer(VkBufferUsageFlags flags, size_t totalSize);
    u64 GetDeviceAddress(VkBuffer buffer);
    void BeginEvent(CommandBuffer *cmd, string name);
    void EndEvent(CommandBuffer *cmd);
    Shader CreateShader(ShaderStage stage, string name, string shaderData);
    void BindAccelerationStructure(VkDescriptorSet descriptorSet,
                                   VkAccelerationStructureKHR accel);
    VkPipeline CreateComputePipeline(Shader *shader, DescriptorSetLayout *layout,
                                     PushConstant *pc = 0, string pipelineName = {});
    RayTracingState CreateRayTracingPipeline(RayTracingShaderGroup *shaderGroups,
                                             int numGroups, PushConstant *pc,
                                             DescriptorSetLayout *layout, u32 maxDepth,
                                             bool useClusters = false);
    void GetClusterBuildSizes(CLASOpInput opInput, CLASOpMode opMode, CLASOpType opType,
                              u32 &scratchSize, u32 &updateScratchSize,
                              u32 &accelerationStructureSize);
    void GetCLASBuildSizes(CLASOpMode opMode, int maxNumClusters, u32 maxNumTriangles,
                           u32 maxNumVertices, u32 &scratchSize,
                           u32 &accelerationStructureSize);
    void GetClusterBLASBuildSizes(CLASOpMode opMode, u32 maxTotalClusterCount,
                                  u32 maxClusterCountPerAccelerationStructure,
                                  u32 maxAccelerationStructureCount, u32 &scratchSize,
                                  u32 &accelerationStructureSize);
    void GetMoveBuildSizes(CLASOpMode opMode, int maxNumClusters, u64 maxMovedBytes,
                           bool noMoveOverlap, u32 &scratchSize,
                           u32 &accelerationStructureSize);
    void GetPTLASBuildSizes(u32 instanceCount, u32 maxInstancesPerPartition,
                            u32 partitionCount, u32 maxInstanceInGlobalPartitionCount,
                            u32 &scratchSize, u32 &accelSize);
    VkAccelerationStructureKHR CreatePTLAS(GPUBuffer *tlasData);
    void GetBuildSizes(VkAccelerationStructureTypeKHR accelType,
                       VkAccelerationStructureGeometryKHR *geometries, int count,
                       VkAccelerationStructureBuildRangeInfoKHR *buildRanges,
                       u32 *maxPrimitiveCounts, u32 &buildScratchSize, u32 &accelSize);
    void GetTLASBuildSizes(u32 numInstances, u32 &buildScratchSize, u32 &accelSize);
    QueryPool CreateQuery(QueryType type, int count);

    VkAccelerationStructureInstanceKHR GetVkInstance(const AffineSpace &transform,
                                                     GPUAccelerationStructure &as);
    void ConvertCLASIndirectInfo(CLASOpInput opInput, CLASOpType opType,
                                 VkClusterAccelerationStructureInputInfoNV &inputInfo,
                                 VkOpInput &vkOpInput, VkDeviceSize &srcInfosArrayStride);
    u32 GetQueueFamily(QueueType type);
    bool BeginFrame(bool doubleBuffer = true);
    void EndFrame(int queueType);

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
    void CreateQueryPool(QueryPool *queryPool, QueryType type, u32 queryCount);
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

    BindlessDescriptorPool bindlessDescriptorPools[3];

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
    void SetName(VkPipelineLayout handle, const char *name);
    void SetName(VkQueue handle, const char *name);
    void SetName(VkCommandBuffer handle, const char *name);
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

inline VkDescriptorType ConvertDescriptorType(DescriptorType type)
{
    switch (type)
    {
        case DescriptorType::Sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;

        case DescriptorType::CombinedImageSampler:
            return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

        case DescriptorType::SampledImage: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;

        case DescriptorType::StorageImage: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case DescriptorType::UniformTexelBuffer:
            return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        case DescriptorType::StorageTexelBuffer:
            return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        case DescriptorType::UniformBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case DescriptorType::StorageBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case DescriptorType::UniformBufferDynamic:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        case DescriptorType::StorageBufferDynamic:
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        case DescriptorType::InputAttachment: return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        case DescriptorType::AccelerationStructure:
            return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        case DescriptorType::PTLAS:
            return VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV;
        default: Assert(0); return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
}

extern Vulkan *device;

} // namespace rt
#endif
