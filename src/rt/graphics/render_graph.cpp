#include "../radix_sort.h"
#include "vulkan.h"
#include "render_graph.h"
namespace rt
{
RenderGraph *renderGraph_;

// GPU Zen 3: Advance Rendering Technique Chapter 5 Resource Management with Frame Graph in
// Messiah
void ResourceLifeTimeRange::Extend(u32 pass, QueueType queue)
{
    passStart      = Min(pass, passStart);
    passEnd[queue] = Max(pass, passEnd[queue]);
}

ResourceHandle RenderGraph::CreateBufferResource(VkBufferUsageFlags2 usageFlags, u32 size,
                                                 ResourceFlags flags)
{
    RenderGraphResource resource = {};
    resource.bufferSize          = size;
    resource.bufferUsageFlags    = usageFlags;
    resource.alignment           = device->GetMinAlignment(usageFlags);
    resource.flags               = flags;

    ResourceHandle handle;
    handle.index = resources.Length();

    resources.Push(resource);
    return handle;
}

void RenderGraph::UpdateBufferResource(ResourceHandle handle, VkBufferUsageFlags2 usageFlags,
                                       u32 size)
{
    resources[handle.index].bufferSize       = size;
    resources[handle.index].bufferUsageFlags = usageFlags;
}

ResourceHandle RenderGraph::RegisterExternalResource(GPUBuffer *buffer)
{
    RenderGraphResource resource = {};
    resource.bufferSize          = buffer->size;
    resource.bufferUsageFlags    = 0;
    resource.alignment           = 0;
    resource.flags               = ResourceFlags::External;
    resource.bufferHandle        = buffer->buffer;

    ResourceHandle handle;
    handle.index = resources.Length();

    resources.Push(resource);
    return handle;
}

ResourceHandle RenderGraph::RegisterExternalResource(GPUImage *image)
{
    RenderGraphResource resource = {};
    resource.alignment           = 0;
    resource.flags               = ResourceFlags::External;

    ResourceHandle handle;
    handle.index = resources.Length();

    resources.Push(resource);
    return handle;
}

int RenderGraph::Overlap(const ResourceLifeTimeRange &lhs,
                         const ResourceLifeTimeRange &rhs) const
{
    bool before = true;
    bool after  = true;
    for (u32 i = 0; i <= QueueType_Graphics; i++)
    {
        before &= dependencies[i][rhs.passStart] >= dependencies[i][lhs.passEnd[i]];
        after &= dependencies[i][lhs.passStart] >= dependencies[i][rhs.passEnd[i]];
    }
    return before ? -1 : (after ? 1 : 0);
}

Pass &RenderGraph::StartPass(u32 numResources, PassFunction &&func)
{
    Pass pass;
    pass.resourceHandles    = StaticArray<ResourceHandle>(arena, numResources);
    pass.resourceUsageTypes = StaticArray<ResourceUsageType>(arena, numResources);
    pass.func               = func;
    passes.Push(pass);
    return passes[passes.Length() - 1];
}

Pass &RenderGraph::StartComputePass(VkPipeline pipeline, DescriptorSetLayout &layout,
                                    u32 numResources, PassFunction &&func)
{
    u32 passIndex       = passes.Length();
    auto AddComputePass = [passIndex, this, func](CommandBuffer *cmd) {
        Pass &pass       = passes[passIndex];
        DescriptorSet ds = pass.layout->CreateDescriptorSet();
        BindResources(pass, ds);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                pass.layout->pipelineLayout);

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

Pass &RenderGraph::StartIndirectComputePass(string name, VkPipeline pipeline,
                                            DescriptorSetLayout &layout, u32 numResources,
                                            ResourceHandle indirectBuffer,
                                            u32 indirectBufferOffset, PassFunction &&func)
{
    string str          = PushStr8Copy(arena, name);
    u32 passIndex       = passes.Length();
    auto AddComputePass = [str, passIndex, this, func, indirectBuffer,
                           indirectBufferOffset](CommandBuffer *cmd) {
        Pass &pass       = passes[passIndex];
        DescriptorSet ds = pass.layout->CreateDescriptorSet();
        BindResources(pass, ds);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                pass.layout->pipelineLayout);
        device->BeginEvent(cmd, str);
        RenderGraphResource &resource      = resources[indirectBuffer.index];
        ResidentResource &residentResource = residentResources[resource.residentResourceIndex];
        cmd->DispatchIndirect(&residentResource.gpuBuffer,
                              resource.aliasOffset + indirectBufferOffset);

        func(cmd);
        device->EndEvent(cmd);
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

Pass &Pass::AddHandle(ResourceHandle handle, ResourceUsageType type)
{
    resourceHandles.Push(handle);
    resourceUsageTypes.Push(type);
    return *this;
}

void RenderGraph::BindResources(Pass &pass, DescriptorSet &ds)
{
    for (ResourceHandle &handle : pass.resourceHandles)
    {
        RenderGraphResource &resource = resources[handle.index];
        if (resource.flags == ResourceFlags::Transient)
        {
            ResidentResource &residentResource =
                residentResources[resource.residentResourceIndex];
            ds.Bind(&residentResource.gpuBuffer, resource.aliasOffset, resource.bufferSize);
        }
        else if (resource.flags == ResourceFlags::External)
        {
            GPUBuffer temp = {};
            temp.buffer    = resource.bufferHandle;
            ds.Bind(&temp);
        }
    }
}

GPUBuffer *RenderGraph::GetBuffer(ResourceHandle handle, u32 &offset, u32 &size)
{
    RenderGraphResource &resource = resources[handle.index];
    offset                        = resource.aliasOffset;
    size                          = resource.bufferSize;
    Assert(EnumHasAnyFlags(resource.flags, ResourceFlags::Transient));
    return &residentResources[resource.residentResourceIndex].gpuBuffer;
}

GPUBuffer *RenderGraph::GetBuffer(ResourceHandle handle, u32 &offset)
{
    RenderGraphResource &resource = resources[handle.index];
    offset                        = resource.aliasOffset;
    Assert(EnumHasAnyFlags(resource.flags, ResourceFlags::Transient));
    return &residentResources[resource.residentResourceIndex].gpuBuffer;
}

GPUBuffer *RenderGraph::GetBuffer(ResourceHandle handle)
{
    RenderGraphResource &resource = resources[handle.index];
    Assert(EnumHasAnyFlags(resource.flags, ResourceFlags::Transient));
    return &residentResources[resource.residentResourceIndex].gpuBuffer;
}

void RenderGraph::Compile()
{
    ScratchArena scratch;
    struct Handle
    {
        u32 sortKey;
        u32 resourceIndex;
    };

    StaticArray<Handle> handles(scratch.temp.arena, resources.Length());
    for (u32 resourceIndex = 0; resourceIndex < resources.Length(); resourceIndex++)
    {
        RenderGraphResource &resource = resources[resourceIndex];
        if (resource.lifeTime.passStart == ~0u) continue;

        Handle handle;
        handle.sortKey       = resource.bufferSize;
        handle.resourceIndex = resourceIndex;
        handles.Push(handle);
    }
    SortHandles<Handle, false>(handles.data, handles.Length());

    // TODO: async queues
    for (u32 passIndex = 0; passIndex < passes.Length(); passIndex++)
    {
        Pass &pass         = passes[passIndex];
        u32 passDependency = 0;
        for (u32 handleIndex = 0; handleIndex < pass.resourceHandles.Length(); handleIndex++)
        {
            RenderGraphResource &resource = resources[pass.resourceHandles[handleIndex].index];
            ResourceUsageType type        = pass.resourceUsageTypes[handleIndex];
            if (type == ResourceUsageType::Read || type == ResourceUsageType::RW)
            {
                passDependency = Max(passDependency, resource.latestWritePass);
            }
            else if (type == ResourceUsageType::RW || type == ResourceUsageType::Write)
            {
                resource.latestWritePass = Max(resource.latestWritePass, passIndex);
            }
        }
    }

    // Graph<u32> passGraph;
    // passGraph.InitializeStatic(scratch.temp.arena, passes.Length(),
    //                            [&](u32 index, u32 *offsets, u32 *data) {
    //                                Pass &pass = passes[index];
    //                                for (ResourceHandle &handle : pass.resourceHandles)
    //                                {
    //                                    RenderGraphResource &resource =
    //                                    resources[handle.index];
    //                                }
    //                            });

    for (int handleIndex = 0; handleIndex < handles.Length(); handleIndex++)
    {
        Handle &handle                = handles[handleIndex];
        RenderGraphResource &resource = resources[handle.resourceIndex];

        if (!EnumHasAnyFlags(resource.flags, ResourceFlags::Transient)) continue;

        bool aliased = false;
        for (int residentResourceIndex = 0; residentResourceIndex < residentResources.Length();
             residentResourceIndex++)
        {
            ResidentResource &residentResource = residentResources[residentResourceIndex];
            Array<Vec2u> ranges(scratch.temp.arena, 4);
            for (int resourceIndex = residentResource.start; resourceIndex != -1;)
            {
                RenderGraphResource &otherResource = resources[resourceIndex];
                if (Overlap(resource.lifeTime, otherResource.lifeTime) == 0)
                {
                    ranges.Push(Vec2u(otherResource.aliasOffset,
                                      otherResource.aliasOffset + otherResource.bufferSize));
                }

                resourceIndex = otherResource.aliasNext;
            }

            // TODO: need to sort the ranges?
            if (ranges.Length())
            {
                // Find best fit
                StaticArray<Vec2u> freeRanges(scratch.temp.arena, ranges.Length());
                freeRanges.Push(Vec2u(0));
                for (Vec2u &range : ranges)
                {
                    Vec2u &freeRange = freeRanges.Last();
                    if (freeRange.y >= range.x)
                    {
                        freeRange.y = Max(freeRange.y, range.y);
                    }
                    else
                    {
                        freeRanges.Push(range);
                    }
                }
                u32 bestFit     = ~0u;
                u32 aliasOffset = 0;
                for (u32 freeRangeIndex = 0; freeRangeIndex < freeRanges.Length();
                     freeRangeIndex++)
                {
                    Vec2u &freeRange  = freeRanges[freeRangeIndex];
                    u32 alignedOffset = AlignPow2(freeRange.x, resource.alignment);
                    u32 end           = freeRange.y;
                    if (alignedOffset + resource.bufferSize <= end)
                    {
                        u32 fit = freeRange.y - freeRange.x - resource.bufferSize;
                        if (fit < bestFit)
                        {
                            aliasOffset = alignedOffset;
                            bestFit     = fit;
                        }
                    }
                }
                if (bestFit != ~0u)
                {
                    residentResource.bufferUsage |= resource.bufferUsageFlags;
                    resource.aliasOffset           = aliasOffset;
                    resource.aliasNext             = residentResource.start;
                    resource.residentResourceIndex = residentResourceIndex;
                    residentResource.start         = handle.resourceIndex;
                    aliased                        = true;
                    break;
                }
            }
        }
        if (!aliased)
        {
            ResidentResource residentResource;
            residentResource.bufferUsage = resource.bufferUsageFlags;
            residentResource.bufferSize  = resource.bufferSize;
            residentResource.start       = handle.resourceIndex;

            residentResources.Push(residentResource);

            resource.residentResourceIndex = residentResources.Length() - 1;
            resource.aliasOffset           = 0;
            resource.aliasNext             = -1;
        }
    }
    u32 totalSize = 0;
    for (ResidentResource &resource : residentResources)
    {
        if (resource.gpuBuffer.size == 0)
        {
            resource.gpuBuffer =
                device->CreateBuffer(resource.bufferUsage, resource.bufferSize);
        }
        totalSize += resource.gpuBuffer.size;
    }
    Print("Transient Resource Size: %u\n", totalSize);
}

void RenderGraph::Execute(CommandBuffer *cmd)
{
    for (Pass &pass : passes)
    {
        pass.func(cmd);
    }
}

void RenderGraph::BeginFrame() { watermark = ArenaPos(arena); }
void RenderGraph::EndFrame()
{
    ArenaPopTo(arena, watermark);
    passes.Clear();
}

} // namespace rt
