#include "../radix_sort.h"
#include "vulkan.h"
#include "render_graph.h"
#include <algorithm>
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

RenderGraph::RenderGraph()
{
    arena             = ArenaAlloc();
    resources         = StaticArray<RenderGraphResource>(arena, 1000);
    residentResources = StaticArray<ResidentResource>(arena, 1000);
    passes            = StaticArray<Pass>(arena, 1000);
}
ResourceHandle RenderGraph::CreateBufferResource(string name, VkBufferUsageFlags2 usageFlags,
                                                 u32 size, ResourceFlags flags)
{
    RenderGraphResource resource = {};
    resource.name                = PushStr8Copy(arena, name);
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

ResourceHandle RenderGraph::RegisterExternalResource(string name, GPUBuffer *buffer)
{
    RenderGraphResource resource = {};
    resource.name                = PushStr8Copy(arena, name);
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
        // before &= dependencies[i][rhs.passStart] >= dependencies[i][lhs.passEnd[i]];
        // after &= dependencies[i][lhs.passStart] >= dependencies[i][rhs.passEnd[i]];
        before &= rhs.passStart > lhs.passEnd[i];
        after &= lhs.passStart > rhs.passEnd[i];
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
    auto AddComputePass = [passIndex, this, func, pipeline](CommandBuffer *cmd) {
        Pass &pass = passes[passIndex];
        cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
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
    auto AddComputePass = [str, passIndex, this, func, indirectBuffer, indirectBufferOffset,
                           pipeline](CommandBuffer *cmd) {
        Pass &pass = passes[passIndex];
        cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
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

    RenderGraph *rg = GetRenderGraph();
    u32 passIndex   = this - rg->passes.data;

    rg->resources[handle.index].lifeTime.Extend(passIndex, QueueType_Graphics);

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

    u32 unAliasedSize = 0;
    u32 externalSize  = 0;
    for (u32 resourceIndex = 0; resourceIndex < resources.Length(); resourceIndex++)
    {
        RenderGraphResource &resource = resources[resourceIndex];
        if (EnumHasAnyFlags(resource.flags, ResourceFlags::External))
        {
            externalSize += resource.bufferSize;
        }
        if (resource.lifeTime.passStart == ~0u ||
            !EnumHasAnyFlags(resource.flags, ResourceFlags::Transient))
            continue;

        unAliasedSize += resource.bufferSize;
        Handle handle;
        handle.sortKey       = resource.bufferSize;
        handle.resourceIndex = resourceIndex;
        handles.Push(handle);
    }
    SortHandles<Handle, false>(handles.data, handles.Length());

    StaticArray<Handle> residentHandles(scratch.temp.arena, residentResources.Length());
    for (u32 residentResourceIndex = 0; residentResourceIndex < residentResources.Length();
         residentResourceIndex++)
    {
        ResidentResource &resource = residentResources[residentResourceIndex];
        Handle handle;
        handle.sortKey       = resource.bufferSize;
        handle.resourceIndex = residentResourceIndex;
        residentHandles.Push(handle);
    }
    SortHandles<Handle, false>(residentHandles.data, residentHandles.Length());

    // TODO: async queues
    for (u32 passIndex = 0; passIndex < passes.Length(); passIndex++)
    {
        Pass &pass = passes[passIndex];
        Assert(pass.resourceHandles.Length() == pass.resourceHandles.capacity);
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

    for (int handleIndex = 0; handleIndex < handles.Length(); handleIndex++)
    {
        Handle &handle                = handles[handleIndex];
        RenderGraphResource &resource = resources[handle.resourceIndex];

        bool aliased = false;
        for (int residentHandleIndex = 0; residentHandleIndex < residentResources.Length();
             residentHandleIndex++)
        {
            int residentResourceIndex =
                residentHandleIndex >= residentHandles.Length()
                    ? residentHandleIndex
                    : residentHandles[residentHandleIndex].resourceIndex;
            ResidentResource &residentResource = residentResources[residentResourceIndex];
            if (residentResource.bufferSize < resource.bufferSize) continue;
            std::vector<std::pair<u32, u32>> ranges;
            for (int resourceIndex = residentResource.start; resourceIndex != -1;)
            {
                RenderGraphResource &otherResource = resources[resourceIndex];
                if (Overlap(resource.lifeTime, otherResource.lifeTime) == 0)
                {
                    ranges.push_back(
                        std::pair(otherResource.offset,
                                  otherResource.aliasOffset + otherResource.bufferSize));
                }

                resourceIndex = otherResource.aliasNext;
            }

            std::sort(ranges.begin(), ranges.end());
            ranges.push_back(
                std::pair(residentResource.bufferSize, residentResource.bufferSize));
            if (ranges.size())
            {
                // Find best fit
                StaticArray<Vec2u> freeRanges(scratch.temp.arena, ranges.size() + 1);
                freeRanges.Push(Vec2u(0));
                for (auto &range : ranges)
                {
                    Vec2u &freeRange = freeRanges.Last();
                    if (freeRange[1] >= range.first)
                    {
                        freeRange[1] = Max(freeRange[1], range.second);
                    }
                    else
                    {
                        freeRanges.Push(Vec2u(range.first, range.second));
                    }
                }
                u32 bestFit     = ~0u;
                u32 aliasOffset = 0;
                u32 offset      = 0;
                for (u32 freeRangeIndex = 1; freeRangeIndex < freeRanges.Length();
                     freeRangeIndex++)
                {
                    u32 begin = freeRanges[freeRangeIndex - 1].y;
                    u32 end   = freeRanges[freeRangeIndex].x;

                    u32 alignedOffset = AlignPow2(begin, resource.alignment);
                    if (alignedOffset + resource.bufferSize <= end)
                    {
                        u32 fit = end - begin - resource.bufferSize;
                        if (fit < bestFit)
                        {
                            offset      = begin;
                            aliasOffset = alignedOffset;
                            bestFit     = fit;
                        }
                    }
                }
                if (bestFit != ~0u)
                {
                    if ((residentResource.bufferUsage & resource.bufferUsageFlags) !=
                        resource.bufferUsageFlags)
                    {
                        residentResource.dirty = true;
                        residentResource.bufferUsage |= resource.bufferUsageFlags;
                    }
                    resource.offset                = offset;
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
            ResidentResource residentResource = {};
            residentResource.bufferUsage      = resource.bufferUsageFlags;
            residentResource.bufferSize       = resource.bufferSize;
            residentResource.start            = handle.resourceIndex;
            residentResource.dirty            = true;

            residentResources.Push(residentResource);

            resource.residentResourceIndex = residentResources.Length() - 1;
            resource.aliasOffset           = 0;
            resource.aliasNext             = -1;
        }
    }
    u32 totalSize = 0;
    for (ResidentResource &resource : residentResources)
    {
        if (resource.dirty)
        {
            if (resource.gpuBuffer.size)
            {
                device->DestroyBuffer(&resource.gpuBuffer);
            }
            resource.gpuBuffer =
                device->CreateBuffer(resource.bufferUsage, resource.bufferSize);
            resource.dirty = false;
        }
        totalSize += resource.gpuBuffer.size;
    }
    Print("Transient Resource Size: %u\n", totalSize);
    Print("Unaliased Size: %u\n", unAliasedSize);
    Print("External Resource Size: %u\n", externalSize);
}

void RenderGraph::Execute(CommandBuffer *cmd)
{
    for (Pass &pass : passes)
    {
        pass.func(cmd);
    }
}

void RenderGraph::BeginFrame()
{
    watermark = ArenaPos(arena);

    for (RenderGraphResource &resource : resources)
    {
        resource.aliasNext          = -1;
        resource.lifeTime.passStart = ~0u;

        for (u32 i = 0; i < QueueType_Count; i++)
        {
            resource.lifeTime.passEnd[i] = 0;
        }
        resource.latestWritePass = 0;
    }
    for (ResidentResource &resource : residentResources)
    {
        resource.start = -1;
    }
}
void RenderGraph::EndFrame()
{
    ArenaPopTo(arena, watermark);
    passes.Clear();
}

} // namespace rt
