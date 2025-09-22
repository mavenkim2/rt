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

    HashIndex residentResourceHash;
    StaticArray<ResidentResource> residentResources;
    CrossQueueDependencies dependencies;

    ResourceHandle CreateBufferResource(VkBufferUsageFlags2 usageFlags, u32 size,
                                        ResourceFlags flags = ResourceFlags::Transient);
    void UpdateBufferResource(ResourceHandle handle, VkBufferUsageFlags2 usageFlags, u32 size);
    ResourceHandle RegisterExternalResource(GPUBuffer *buffer);
    int Overlap(const ResourceLifeTimeRange &lhs, const ResourceLifeTimeRange &rhs) const;
    Pass &StartPass(u32 numResources, PassFunction &&func);
    void BindResources(Pass &pass, DescriptorSet &ds);

    GPUBuffer *GetBuffer(ResourceHandle handle, u32 &offset, u32 &size);
    GPUBuffer *GetBuffer(ResourceHandle handle, u32 &offset);
    GPUBuffer *GetBuffer(ResourceHandle handle);

    template <typename T>
    Pass &StartComputePass(VkPipeline pipeline, DescriptorSetLayout &layout, u32 numResources,
                           PassFunction &&func, PushConstant *pc, T *push = 0)
    {
        u32 passIndex = passes.Length();
        T *p          = 0;
        if (push)
        {
            p = (T *)PushArrayNoZero(arena, u8, sizeof(T));
        }
        auto AddComputePass = [pc, p, passIndex, &passes](CommandBuffer *cmd) {
            Pass &pass       = passes[passIndex];
            DescriptorSet ds = pass.layout->CreateDescriptorSet();
            BindResources(pass, ds);
            cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                    pass.layout->pipelineLayout);
            if (p)
            {
                cmd->PushConstants(pc, p, pass.layout->pipelineLayout);
            }

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
                                   PushConstant *pc, T *push = 0)
    {
        u32 passIndex = passes.Length();
        T *p          = 0;
        if (push)
        {
            p = (T *)PushArrayNoZero(arena, u8, sizeof(T));
        }
        auto AddComputePass = [pc, p, passIndex, &passes](CommandBuffer *cmd) {
            Pass &pass       = passes[passIndex];
            DescriptorSet ds = pass.layout->CreateDescriptorSet();
            BindResources(pass, ds);
            cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                    pass.layout->pipelineLayout);
            if (p)
            {
                cmd->PushConstants(pc, p, pass.layout->pipelineLayout);
            }

            RenderGraphResource &resource      = resources[indirectBuffer.index];
            ResidentResource &residentResource = resource.residentResourceIndex;
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

    // template <typename T>
    // Pass &StartComputePass(string eventName, VkPipeline pipeline, DescriptorSetLayout
    // &layout,
    //                        u32 numResources, u32 dispatchX, u32 dispatchY, u32 dispatchZ,
    //                        PushConstant *pc, T *push = 0)
    // {
    //     u32 passIndex = passes.Length();
    //     T *p          = 0;
    //     if (push)
    //     {
    //         p = (T *)PushArrayNoZero(arena, u8, sizeof(T));
    //     }
    //     string name         = PushStr8Copy(arena, eventName);
    //     auto AddComputePass = [name, pc, p, passIndex, &passes](CommandBuffer *cmd) {
    //         Pass &pass       = passes[passIndex];
    //         DescriptorSet ds = pass.layout->CreateDescriptorSet();
    //         for (ResourceHandle &handle : pass.resourceHandles)
    //         {
    //             RenderGraphResource &resource = resources[handle.index];
    //             ResidentResource &residentResource =
    //                 residentResources[resource.residentResourceIndex];
    //             ds.Bind(&residentResource.gpuBuffer, resource.aliasOffset,
    //                     resource.bufferSize);
    //         }
    //         cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
    //                                 pass.layout->pipelineLayout);
    //         if (p)
    //         {
    //             cmd->PushConstants(pc, p, pass.layout->pipelineLayout);
    //         }
    //
    //         device->BeginEvent(cmd, name);
    //         cmd->Dispatch(dispatchX, dispatchY, dispatchZ);
    //         device->EndEvent(cmd);
    //     };
    //     Pass pass;
    //     pass.pipeline           = pipeline;
    //     pass.layout             = &layout;
    //     pass.resourceHandles    = StaticArray<ResourceHandle>(arena, numResources);
    //     pass.resourceUsageTypes = StaticArray<ResourceUsageType>(arena, numResources);
    //     pass.func               = AddComputePass;
    //     passes.Push(pass);
    //     return passes[passes.Length() - 1];
    // }

    void Compile();
    void Execute();
};

extern RenderGraph *renderGraph_;
inline RenderGraph *GetRenderGraph() { return renderGraph_; }

} // namespace rt

#endif
