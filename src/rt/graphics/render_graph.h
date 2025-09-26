#ifndef RENDER_GRAPH_H_
#define RENDER_GRAPH_H_

#include "vulkan.h"
#include "../containers.h"
#include <functional>

namespace rt
{

using PassFunction           = std::function<void(CommandBuffer *cmd)>;
using CrossQueueDependencies = FixedArray<StaticArray<u32>, QueueType_Count>;

enum class ResourceUsageType
{
    Read,
    Write,
    RW,
};

struct ResourceHandle
{
    u32 index;
};

enum class ResourceFlags
{
    Transient = (1u << 0u),
    External  = (1u << 1u),
    Buffer    = (1u << 2u),
    Image     = (1u << 3u),
};
ENUM_CLASS_FLAGS(ResourceFlags)

struct ResourceLifeTimeRange
{
    u32 passStart = ~0u;
    u32 passEnd[QueueType_Count];

    void Extend(u32 pass, QueueType queue);
};

struct RenderGraphResource
{
    string name;
    union
    {
        struct
        {
            ImageDesc imageDesc;
            GPUImage image;
            StaticArray<Subresource> subresources;
        };
        struct
        {
            VkBufferUsageFlags2 bufferUsageFlags;
            GPUBuffer buffer;
        };
    };
    u32 alignment;
    u32 size;

    // Frame temp
    ResourceLifeTimeRange lifeTime;
    u32 latestWritePass;

    ResourceFlags flags;

    int residentResourceIndex;
    int offset;
    int aliasOffset;
    int aliasNext;
};

struct ResidentResource
{
    u32 lastMemReqBits;
    u32 memReqBits;
    VmaAllocation alloc;
    bool isAlloc;

    // VkBuffer buffer;
    // VkBufferUsageFlags2 bufferUsage;
    u32 bufferSize;
    u32 maxAlignment;
    int start;
    bool dirty;
};

struct Pass
{
    PassFunction func;
    StaticArray<ResourceHandle> resourceHandles;
    StaticArray<ResourceUsageType> resourceUsageTypes;
    StaticArray<int> subresources;

    // Compute
    VkPipeline pipeline;
    DescriptorSetLayout *layout;

    // Image aliasing

    Pass &AddHandle(ResourceHandle handle, ResourceUsageType type, int subresource = -1);
};

struct RenderGraph
{
    Arena *arena;
    StaticArray<Pass> passes;
    StaticArray<RenderGraphResource> resources;

    u32 watermark;
    // HashIndex residentResourceHash;
    StaticArray<ResidentResource> residentResources;
    // CrossQueueDependencies dependencies;

    RenderGraph();
    void BeginFrame();
    void EndFrame();
    inline bool IsBuffer(const RenderGraphResource &resource)
    {
        return EnumHasAnyFlags(resource.flags, ResourceFlags::Buffer);
    };
    inline bool IsImage(const RenderGraphResource &resource)
    {
        return EnumHasAnyFlags(resource.flags, ResourceFlags::Image);
    };
    ResourceHandle CreateBufferResource(string name, VkBufferUsageFlags2 usageFlags, u32 size,
                                        ResourceFlags flags = ResourceFlags::Transient |
                                                              ResourceFlags::Buffer);
    ResourceHandle CreateImageResource(string name, ImageDesc desc,
                                       ResourceFlags flags = ResourceFlags::Transient |
                                                             ResourceFlags::Image);
    void CreateImageSubresources(ResourceHandle handle,
                                 StaticArray<Subresource> &subresources);

    void UpdateBufferResource(ResourceHandle handle, VkBufferUsageFlags2 usageFlags, u32 size);
    ResourceHandle RegisterExternalResource(string name, GPUBuffer *buffer);
    ResourceHandle RegisterExternalResource(string name, GPUImage *image);
    int Overlap(const ResourceLifeTimeRange &lhs, const ResourceLifeTimeRange &rhs) const;
    Pass &StartPass(u32 numResources, PassFunction &&func);
    Pass &StartComputePass(VkPipeline pipeline, DescriptorSetLayout &layout, u32 numResources,
                           PassFunction &&func);
    Pass &StartIndirectComputePass(string name, VkPipeline pipeline,
                                   DescriptorSetLayout &layout, u32 numResources,
                                   ResourceHandle indirectBuffer, u32 indirectBufferOffset,
                                   PassFunction &&func);
    void BindResources(Pass &pass, DescriptorSet &ds);

    template <typename T>
    T *Allocate(T *ptr, u32 count)
    {
        T *out = (T *)PushArrayNoZero(arena, u8, sizeof(T) * count);
        MemoryCopy(out, ptr, sizeof(T) * count);
        return out;
    }

    GPUBuffer *GetBuffer(ResourceHandle handle, u32 &offset, u32 &size);
    GPUBuffer *GetBuffer(ResourceHandle handle, u32 &offset);
    GPUBuffer *GetBuffer(ResourceHandle handle);
    GPUImage *GetImage(ResourceHandle handle);

    template <typename T>
    Pass &StartComputePass(VkPipeline pipeline, DescriptorSetLayout &layout, u32 numResources,
                           PassFunction &&func, PushConstant *pc, T *push)
    {
        u32 passIndex = passes.Length();
        Assert(push);
        T *p = (T *)PushArrayNoZero(arena, u8, sizeof(T));
        MemoryCopy(p, push, sizeof(T));
        auto AddComputePass = [pc, p, passIndex, this, func, pipeline](CommandBuffer *cmd) {
            Pass &pass = passes[passIndex];
            cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            DescriptorSet ds = pass.layout->CreateDescriptorSet();
            BindResources(pass, ds);
            cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                    pass.layout->pipelineLayout);
            cmd->PushConstants(pc, p, pass.layout->pipelineLayout);

            func(cmd);
        };
        Pass pass;
        pass.pipeline           = pipeline;
        pass.layout             = &layout;
        pass.resourceHandles    = StaticArray<ResourceHandle>(arena, numResources);
        pass.resourceUsageTypes = StaticArray<ResourceUsageType>(arena, numResources);
        pass.subresources       = StaticArray<int>(arena, numResources);
        pass.func               = AddComputePass;
        passes.Push(pass);
        return passes[passes.Length() - 1];
    }

    template <typename T>
    Pass &StartIndirectComputePass(VkPipeline pipeline, DescriptorSetLayout &layout,
                                   u32 numResources, ResourceHandle indirectBuffer,
                                   u32 indirectBufferOffset, PassFunction &&func,
                                   PushConstant *pc, T *push)
    {
        u32 passIndex = passes.Length();
        Assert(push);
        T *p = (T *)PushArrayNoZero(arena, u8, sizeof(T));
        MemoryCopy(p, push, sizeof(T));
        auto AddComputePass = [pc, p, passIndex, this, func, indirectBuffer,
                               indirectBufferOffset, pipeline](CommandBuffer *cmd) {
            Pass &pass = passes[passIndex];
            cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            DescriptorSet ds = pass.layout->CreateDescriptorSet();
            BindResources(pass, ds);
            cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                    pass.layout->pipelineLayout);
            cmd->PushConstants(pc, p, pass.layout->pipelineLayout);

            RenderGraphResource &resource = resources[indirectBuffer.index];
            cmd->DispatchIndirect(&resource.buffer, indirectBufferOffset);

            func(cmd);
        };
        Pass pass;
        pass.pipeline           = pipeline;
        pass.layout             = &layout;
        pass.resourceHandles    = StaticArray<ResourceHandle>(arena, numResources);
        pass.resourceUsageTypes = StaticArray<ResourceUsageType>(arena, numResources);
        pass.subresources       = StaticArray<int>(arena, numResources);
        pass.func               = AddComputePass;
        passes.Push(pass);
        return passes[passes.Length() - 1];
    }

    void Compile();
    void Execute(CommandBuffer *cmd);
};

extern RenderGraph *renderGraph_;
inline RenderGraph *GetRenderGraph() { return renderGraph_; }
inline void SetRenderGraph(RenderGraph *g) { renderGraph_ = g; };

} // namespace rt

#endif
