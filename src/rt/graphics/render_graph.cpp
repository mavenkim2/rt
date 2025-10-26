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
    submit            = false;
    semaphore         = device->CreateSemaphore();
}
ResourceHandle RenderGraph::CreateBufferResource(string name, VkBufferUsageFlags2 usageFlags,
                                                 u32 size, MemoryUsage memUsage,
                                                 ResourceFlags flags)
{
    RenderGraphResource resource = {};
    resource.name                = PushStr8Copy(arena, name);
    resource.alignment           = device->GetMinAlignment(usageFlags);
    resource.size                = size;
    resource.bufferUsageFlags    = usageFlags;
    resource.buffer.buffer       = VK_NULL_HANDLE;
    resource.flags               = flags;
    resource.memUsage            = memUsage;

    ResourceHandle handle;
    handle.index = resources.Length();

    resources.Push(resource);
    return handle;
}

void RenderGraph::UpdateBufferResource(ResourceHandle handle, VkBufferUsageFlags2 usageFlags,
                                       u32 size)
{
    resources[handle.index].size             = size;
    resources[handle.index].bufferUsageFlags = usageFlags;
}

ResourceHandle RenderGraph::CreateImageResource(string name, ImageDesc desc,
                                                ResourceFlags flags)
{
    RenderGraphResource resource = {};
    resource.name                = PushStr8Copy(arena, name);
    resource.imageDesc           = desc;
    resource.image.image         = VK_NULL_HANDLE;
    resource.flags               = flags;
    resource.memUsage            = desc.memUsage;

    ResourceHandle handle;
    handle.index = resources.Length();

    resources.Push(resource);
    return handle;
}

void RenderGraph::CreateImageSubresources(ResourceHandle handle,
                                          StaticArray<Subresource> &subresources)
{
    RenderGraphResource &resource = resources[handle.index];
    resource.subresources         = StaticArray<Subresource>(arena, subresources.Length());
    resource.deviceSubresources =
        StaticArray<GPUImage::Subresource>(arena, subresources.Length());
    Copy(resource.subresources, subresources);
}

ResourceHandle RenderGraph::RegisterExternalResource(string name, GPUBuffer *buffer)
{
    RenderGraphResource resource = {};
    resource.name                = PushStr8Copy(arena, name);
    resource.size                = buffer->size;
    resource.bufferUsageFlags    = 0;
    resource.flags               = ResourceFlags::External | ResourceFlags::Buffer;
    resource.buffer              = *buffer;

    ResourceHandle handle;
    handle.index = resources.Length();

    resources.Push(resource);
    return handle;
}

ResourceHandle RenderGraph::RegisterExternalResource(string name, GPUImage *image)
{
    RenderGraphResource resource = {};
    resource.name                = PushStr8Copy(arena, name);
    resource.flags               = ResourceFlags::External | ResourceFlags::Image;
    resource.imageDesc           = image->desc;
    resource.image               = *image;

    ResourceHandle handle;
    handle.index = resources.Length();

    resources.Push(resource);
    return handle;
}

ResourceHandle RenderGraph::RegisterExternalResource(string name, u64 ptlasAddress)
{
    RenderGraphResource resource = {};
    resource.name                = PushStr8Copy(arena, name);
    resource.flags               = ResourceFlags::External | ResourceFlags::PTLAS;
    resource.ptlasAddress        = ptlasAddress;

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
    pass.subresources       = StaticArray<int>(arena, numResources);
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
    pass.subresources       = StaticArray<int>(arena, numResources);
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
        RenderGraphResource &resource = resources[indirectBuffer.index];
        cmd->DispatchIndirect(&resource.buffer, indirectBufferOffset);

        func(cmd);
        device->EndEvent(cmd);
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

Pass &Pass::AddHandle(ResourceHandle handle, ResourceUsageType type, int subresource)
{
    resourceHandles.Push(handle);
    resourceUsageTypes.Push(type);
    subresources.Push(subresource);

    RenderGraph *rg = GetRenderGraph();
    u32 passIndex   = this - rg->passes.data;

    rg->resources[handle.index].lifeTime.Extend(passIndex, QueueType_Graphics);

    return *this;
}

void RenderGraph::BindResources(Pass &pass, DescriptorSet &ds)
{
    for (int handleIndex = 0; handleIndex < pass.resourceHandles.Length(); handleIndex++)
    {
        ResourceHandle handle         = pass.resourceHandles[handleIndex];
        RenderGraphResource &resource = resources[handle.index];
        if (IsBuffer(resource))
        {
            ds.Bind(&resource.buffer);
        }
        else if (IsImage(resource))
        {
            ds.Bind(&resource.image, pass.subresources[handleIndex]);
        }
        else if (IsPTLAS(resource))
        {
            ds.Bind(&resource.ptlasAddress);
        }
    }
}

GPUBuffer *RenderGraph::GetBuffer(ResourceHandle handle, u32 &offset, u32 &size)
{
    RenderGraphResource &resource = resources[handle.index];

    offset = 0;
    size   = resource.size;
    Assert(EnumHasAllFlags(resource.flags, ResourceFlags::Transient | ResourceFlags::Buffer));
    return &resource.buffer;
}

GPUBuffer *RenderGraph::GetBuffer(ResourceHandle handle, u32 &offset)
{
    RenderGraphResource &resource = resources[handle.index];
    offset                        = 0;
    // TODO: kinda hacky, used for OIDN
    resource.buffer.allocation = residentResources[resource.residentResourceIndex].alloc;
    return &resource.buffer;
}

GPUBuffer *RenderGraph::GetBuffer(ResourceHandle handle)
{
    RenderGraphResource &resource = resources[handle.index];
    // Assert(EnumHasAllFlags(resource.flags, ResourceFlags::Transient |
    // ResourceFlags::Buffer));
    return &resource.buffer;
}

int RenderGraph::GetBufferBindlessIndex(ResourceHandle handle)
{
    RenderGraphResource &resource = resources[handle.index];
    Assert(EnumHasAllFlags(resource.flags, ResourceFlags::Buffer | ResourceFlags::Bindless));
    return resource.bindlessBufferIndex;
}

GPUImage *RenderGraph::GetImage(ResourceHandle handle)
{
    RenderGraphResource &resource = resources[handle.index];
    Assert(EnumHasAllFlags(resource.flags, ResourceFlags::Transient | ResourceFlags::Image));

    // TODO: kinda hacky, used for DLSS
    resource.image.allocation = residentResources[resource.residentResourceIndex].alloc;
    return &resource.image;
}

void RenderGraph::Compile()
{
    ScratchArena scratch;

    for (RenderGraphResource &resource : resources)
    {
        if (EnumHasAllFlags(resource.flags, ResourceFlags::Image | ResourceFlags::Transient))
        {
            // TODO: don't do this every frame
            if (resource.image.image)
            {
                device->DestroyImageHandles(&resource.image);
            }
            resource.image              = device->CreateAliasedImage(resource.imageDesc);
            resource.image.subresources = resource.deviceSubresources;
            resource.deviceSubresources.Clear();

            resource.size = resource.image.req.size;
        }
        else if (EnumHasAllFlags(resource.flags,
                                 ResourceFlags::Buffer | ResourceFlags::Transient))
        {
            // TODO: don't do this every frame
            if (resource.buffer.buffer)
            {
                device->DestroyBufferHandle(&resource.buffer);
                if (resource.bindlessBufferIndex != -1)
                {
                    device->FreeBindlessStorageIndex(resource.bindlessBufferIndex);
                }
            }
            resource.buffer = device->CreateAliasedBuffer(resource.bufferUsageFlags,
                                                          resource.size, resource.memUsage);
            resource.size   = resource.buffer.req.size;
        }
    }

    struct Handle
    {
        u32 sortKey;
        u32 resourceIndex;
    };

    StaticArray<Handle> handles(scratch.temp.arena, resources.Length());

    u32 unAliasedSize = 0;
    u64 externalSize  = 0;
    for (u32 resourceIndex = 0; resourceIndex < resources.Length(); resourceIndex++)
    {
        RenderGraphResource &resource = resources[resourceIndex];
        if (EnumHasAnyFlags(resource.flags, ResourceFlags::External))
        {
            externalSize += resource.size;
        }
        if (resource.lifeTime.passStart == ~0u ||
            !EnumHasAnyFlags(resource.flags, ResourceFlags::Transient))
            continue;

        unAliasedSize += resource.size;
        Handle handle;
        handle.sortKey       = resource.size;
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

        u32 bits      = 0;
        u32 alignment = 0;
        if (IsBuffer(resource))
        {
            bits      = resource.buffer.req.bits;
            alignment = Max((u32)resource.buffer.req.alignment, resource.alignment);
        }
        else if (IsImage(resource))
        {
            bits      = resource.image.req.bits;
            alignment = resource.image.req.alignment;
        }
        else
        {
            Assert(0);
        }

        bool aliased = false;
        for (int residentHandleIndex = 0; residentHandleIndex < residentResources.Length();
             residentHandleIndex++)
        {
            int residentResourceIndex =
                residentHandleIndex >= residentHandles.Length()
                    ? residentHandleIndex
                    : residentHandles[residentHandleIndex].resourceIndex;
            ResidentResource &residentResource = residentResources[residentResourceIndex];

            if (residentResource.bufferSize < resource.size ||
                ((residentResource.memReqBits & bits) == 0))
                continue;

            std::vector<std::pair<u32, u32>> ranges;
            for (int resourceIndex = residentResource.start; resourceIndex != -1;)
            {
                RenderGraphResource &otherResource = resources[resourceIndex];
                if (Overlap(resource.lifeTime, otherResource.lifeTime) == 0)
                {
                    ranges.push_back(std::pair(
                        otherResource.offset, otherResource.aliasOffset + otherResource.size));
                }

                resourceIndex = otherResource.aliasNext;
            }

            std::sort(ranges.begin(), ranges.end());
            ranges.push_back(
                std::pair(residentResource.bufferSize, residentResource.bufferSize));
            if (ranges.size())
            {
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

                    u32 alignedOffset = AlignPow2(begin, alignment);
                    if (alignedOffset + resource.size <= end)
                    {
                        u32 fit = end - begin - resource.size;
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
                    resource.offset                = offset;
                    resource.aliasOffset           = aliasOffset;
                    resource.aliasNext             = residentResource.start;
                    resource.residentResourceIndex = residentResourceIndex;
                    residentResource.start         = handle.resourceIndex;
                    residentResource.memReqBits &= bits;
                    residentResource.maxAlignment =
                        Max(residentResource.maxAlignment, alignment);
                    residentResource.memUsage |= resource.memUsage;

                    aliased = true;
                    break;
                }
            }
        }
        if (!aliased)
        {
            ResidentResource residentResource = {};
            if (IsBuffer(resource))
            {
                residentResource.memReqBits = resource.buffer.req.bits;
                residentResource.bufferSize = resource.buffer.req.size;
            }
            else if (IsImage(resource))
            {
                residentResource.memReqBits = resource.image.req.bits;
                residentResource.bufferSize = resource.image.req.size;
            }
            residentResource.maxAlignment = alignment;
            residentResource.start        = handle.resourceIndex;
            residentResource.dirty        = true;
            residentResource.isAlloc      = false;

            residentResources.Push(residentResource);

            resource.residentResourceIndex = residentResources.Length() - 1;
            resource.aliasOffset           = 0;
            resource.aliasNext             = -1;
        }
    }

    u32 totalSize = 0;
    for (ResidentResource &resource : residentResources)
    {
        if (resource.memReqBits != resource.lastMemReqBits)
        {
            resource.dirty = true;
        }

        if (resource.dirty)
        {
            if (resource.isAlloc)
            {
                device->FreeMemory(resource.alloc);
                resource.isAlloc = false;
            }
            MemoryRequirements req;
            req.bits      = resource.memReqBits;
            req.alignment = resource.maxAlignment;
            req.size      = resource.bufferSize;
            req.usage     = resource.memUsage;

            // if (EnumHasAnyFlags(resource.memUsage, MemoryUsage::EXTERNAL))
            // {
            //     DebugBreak();
            //     int stop = 5;
            // }

            resource.alloc   = device->AllocateMemory(req);
            resource.isAlloc = true;
            resource.dirty   = false;
        }

        for (int index = resource.start; index != -1;)
        {
            RenderGraphResource &r = resources[index];
            if (IsImage(r))
            {
                device->BindImageMemory(resource.alloc, r.image.image, r.aliasOffset);
                device->CreateSubresource(&r.image);

                if (r.subresources.Length())
                {
                    for (Subresource &subresource : r.subresources)
                    {
                        device->CreateSubresource(&r.image, subresource.baseMip,
                                                  subresource.numMips, subresource.baseLayer,
                                                  subresource.numLayers);
                    }
                }
            }
            else if (IsBuffer(r))
            {
                device->BindBufferMemory(resource.alloc, r.buffer.buffer, r.aliasOffset);
                if (EnumHasAnyFlags(r.flags, ResourceFlags::Bindless))
                {
                    int bindlessIndex     = device->BindlessStorageIndex(&r.buffer);
                    r.bindlessBufferIndex = bindlessIndex;
                }
            }
            index = r.aliasNext;
        }
        totalSize += resource.bufferSize;
    }
    Print("Transient Resource Size: %u\n", totalSize);
    Print("Unaliased Size: %u\n", unAliasedSize);
    Print("External Resource Size: %llu\n", externalSize);
}

void RenderGraph::Execute(CommandBuffer *cmd)
{
    for (Pass &pass : passes)
    {
        if (cmd == 0)
        {
            cmd = device->BeginCommandBuffer(QueueType_Graphics);
        }
        pass.func(cmd);
        if (submit == true)
        {
            semaphore.signalValue++;
            cmd->SignalOutsideFrame(semaphore);
            submit = false;
            wait   = false;
            device->SubmitCommandBuffer(cmd);
            device->Wait(semaphore);
            cmd = 0;
        }
    }
}

void RenderGraph::SetSubmit() { submit = true; }

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
        resource.lastMemReqBits = resource.memReqBits;
        resource.memReqBits     = ~0u;
        resource.start          = -1;
    }
}
void RenderGraph::EndFrame()
{
    ArenaPopTo(arena, watermark);
    passes.Clear();
}

} // namespace rt
