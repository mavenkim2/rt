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
    Transient,
    External,
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
    union
    {
        ImageDesc imageDesc;
        struct
        {
            u32 bufferSize;
            VkBufferUsageFlags2 bufferUsageFlags;

            // External handle
            VkBuffer bufferHandle;
        };
    };

    // Frame temp
    ResourceLifeTimeRange lifeTime;
    u32 latestWritePass;

    ResourceFlags flags;
    u32 alignment;

    int residentResourceIndex;
    int aliasOffset;
    int aliasNext;
};

struct ResidentResource
{
    VkBufferUsageFlags2 bufferUsage;
    u32 bufferSize;
    GPUBuffer gpuBuffer;
    int start;
};

struct Pass
{
    PassFunction func;
    StaticArray<ResourceHandle> resourceHandles;
    StaticArray<ResourceUsageType> resourceUsageTypes;

    // Compute
    VkPipeline pipeline;
    DescriptorSetLayout *layout;

    Pass &AddHandle(ResourceHandle handle, ResourceUsageType type);
};

struct RenderGraph
{
    Arena *arena;
    StaticArray<Pass> passes;
    Array<RenderGraphResource> resources;

    u32 watermark;
    // HashIndex residentResourceHash;
    StaticArray<ResidentResource> residentResources;
    CrossQueueDependencies dependencies;

    void BeginFrame();
    void EndFrame();
    ResourceHandle CreateBufferResource(VkBufferUsageFlags2 usageFlags, u32 size,
                                        ResourceFlags flags = ResourceFlags::Transient);
    void UpdateBufferResource(ResourceHandle handle, VkBufferUsageFlags2 usageFlags, u32 size);
    ResourceHandle RegisterExternalResource(GPUBuffer *buffer);
    ResourceHandle RegisterExternalResource(GPUImage *image);
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

    template <typename T>
    Pass &StartComputePass(VkPipeline pipeline, DescriptorSetLayout &layout, u32 numResources,
                           PassFunction &&func, PushConstant *pc, T *push)
    {
        u32 passIndex = passes.Length();
        Assert(push);
        T *p                = (T *)PushArrayNoZero(arena, u8, sizeof(T));
        auto AddComputePass = [pc, p, passIndex, this, func](CommandBuffer *cmd) {
            Pass &pass       = passes[passIndex];
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
        T *p                = (T *)PushArrayNoZero(arena, u8, sizeof(T));
        auto AddComputePass = [pc, p, passIndex, this, func, indirectBuffer,
                               indirectBufferOffset](CommandBuffer *cmd) {
            Pass &pass       = passes[passIndex];
            DescriptorSet ds = pass.layout->CreateDescriptorSet();
            BindResources(pass, ds);
            cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                    pass.layout->pipelineLayout);
            cmd->PushConstants(pc, p, pass.layout->pipelineLayout);

            RenderGraphResource &resource = resources[indirectBuffer.index];
            ResidentResource &residentResource =
                residentResources[resource.residentResourceIndex];
            cmd->DispatchIndirect(&residentResource.gpuBuffer,
                                  resource.aliasOffset + indirectBufferOffset);

            func(cmd);
        };
        Pass pass;
        pass.pipeline           = pipeline;
        pass.layout             = &layout;
        pass.resourceHandles    = StaticArray<ResourceHandle>(arena, numResources);
        pass.resourceUsageTypes = StaticArray<ResourceUsageType>(arena, numResources);
        pass.func               = AddComputePass;
        passes.Push(pass);
        return passes[passes.Length() - 1];
    }

    void Compile();
    void Execute(CommandBuffer *cmd);
};

extern RenderGraph *renderGraph_;
inline RenderGraph *GetRenderGraph() { return renderGraph_; }

} // namespace rt

#endif
