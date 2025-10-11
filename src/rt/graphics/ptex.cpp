#include "../base.h"
#include "../containers.h"
#include "../debug.h"
#include "../memory.h"
#include "../bit_packing.h"
#include "../string.h"
#include "../radix_sort.h"
#include "../parallel.h"
// #include "../scene.h"
// #include "../integrate.h"
#include "vulkan.h"
#include "ptex.h"
#include <Ptexture.h>
#include <PtexReader.h>
#include <atomic>
#include <cstring>
#include <thread>

namespace Utils
{
void Copy(const void *src, int sstride, void *dst, int dstride, int vres, int rowlen)
{
    // regular non-tiled case
    if (sstride == rowlen && dstride == rowlen)
    {
        // packed case - copy in single block
        MemoryCopy(dst, src, vres * rowlen);
    }
    else
    {
        // copy a row at a time
        const char *sptr = (const char *)src;
        char *dptr       = (char *)dst;
        for (const char *end = sptr + vres * sstride; sptr != end;)
        {
            MemoryCopy(dptr, sptr, rowlen);
            dptr += dstride;
            sptr += sstride;
        }
    }
}

// NOTE: absolute index
void *GetContentsAbsoluteIndex(void *contents, const Vec2u &p, u32 width, u32 height,
                               u32 stride, u32 bytesPerPixel)
{
    Assert(p.x < width && p.y < height);
    return (u8 *)contents + stride * p.y + p.x * bytesPerPixel;
}

// NOTE: all values in terms of blocks/texels, not bytes
void Copy(void *src, const Vec2u &srcIndex, u32 srcWidth, u32 srcHeight, void *dst,
          const Vec2u &dstIndex, u32 dstWidth, u32 dstHeight, u32 vRes, u32 rowLen,
          u32 bytesPerBlock)
{
    u32 srcStride = srcWidth * bytesPerBlock;
    u32 dstStride = dstWidth * bytesPerBlock;
    rowLen *= bytesPerBlock;
    src = Utils::GetContentsAbsoluteIndex(src, srcIndex, srcWidth, srcHeight, srcStride,
                                          bytesPerBlock);
    dst = Utils::GetContentsAbsoluteIndex(dst, dstIndex, dstWidth, dstHeight, dstStride,
                                          bytesPerBlock);

    vRes   = Min(srcHeight, vRes);
    rowLen = Min(rowLen, srcStride);
    Copy(src, srcStride, dst, dstStride, vRes, rowLen);
}

} // namespace Utils

namespace rt
{

Ptex::PtexCache *cache;
PtexErrHandler errorHandler;
PtexInpHandler ptexInputHandler;

void InitializePtex(u32 maxFiles, u64 maxMem)
{
    cache = Ptex::PtexCache::create(maxFiles, maxMem, true, &ptexInputHandler, &errorHandler);
}

Vec2u PaddedImage::ConvertRelativeToAbsolute(const Vec2u &p)
{
    return Vec2u(p.x + borderSize, p.y + borderSize);
}

// NOTE: absolute index
u8 *PaddedImage::GetContentsAbsoluteIndex(const Vec2u &p)
{
    return (u8 *)Utils::GetContentsAbsoluteIndex(
        contents, p, GetPaddedWidth(), GetPaddedHeight(), strideWithBorder, bytesPerPixel);
}

// NOTE: ignores border
u8 *PaddedImage::GetContentsRelativeIndex(const Vec2u &p)
{
    Assert(p.x < width && p.y < height);
    return GetContentsAbsoluteIndex(ConvertRelativeToAbsolute(p));
}

u32 PaddedImage::GetPaddedWidth() const { return width + 2 * borderSize; }

u32 PaddedImage::GetPaddedHeight() const { return height + 2 * borderSize; }

// NOTE: start is an absolute index
void PaddedImage::WriteRotated(PaddedImage &other, Vec2u srcStart, Vec2u dstStart, int rotate,
                               int srcVLen, int srcRowLen, int dstVLen, int dstRowLen,
                               Vec2u scale)
{
    int uStart = srcStart.x;
    int vStart = srcStart.y;

    Assert(bytesPerPixel == other.bytesPerPixel);
    Assert(borderSize == other.borderSize);

    u8 *src       = other.GetContentsAbsoluteIndex(srcStart);
    u32 srcStride = other.strideWithBorder;

    ScratchArena scratch;

    int pixelsToWriteX = 1 << scale.x;
    int pixelsToWriteY = 1 << scale.y;

    u32 size       = dstVLen * dstRowLen * bytesPerPixel;
    u8 *tempBuffer = PushArray(scratch.temp.arena, u8, size);
    for (int v = 0; v < srcVLen; v++)
    {
        for (int u = 0; u < srcRowLen; u++)
        {
            Vec2i dstAddress;
            switch (rotate)
            {
                case 0:
                {
                    dstAddress.x = u;
                    dstAddress.y = v;
                }
                break;
                case 1:
                {
                    dstAddress.x = srcVLen - 1 - v;
                    dstAddress.y = u;
                }
                break;
                case 2:
                {
                    dstAddress.x = srcRowLen - 1 - u;
                    dstAddress.y = srcVLen - 1 - v;
                }
                break;
                case 3:
                {
                    dstAddress.y = srcRowLen - 1 - u;
                    dstAddress.x = v;
                }
                break;
                default: Assert(0); break;
            }

            Vec2i dstPos(dstAddress);
            dstPos.x <<= scale.x;
            dstPos.y <<= scale.y;

            Vec2u srcPos(u, v);
            srcPos += srcStart;

            u32 offset = (dstPos.y * dstRowLen + dstPos.x) * bytesPerPixel;

            for (int y = 0; y < pixelsToWriteY; y++)
            {
                u32 dstOffset = offset + y * dstRowLen * bytesPerPixel;
                for (int x = 0; x < pixelsToWriteX; x++)
                {
                    Assert(dstOffset < size);
                    MemoryCopy(tempBuffer + dstOffset, other.GetContentsAbsoluteIndex(srcPos),
                               bytesPerPixel);
                    dstOffset += bytesPerPixel;
                }
            }
        }
    }

    Utils::Copy(tempBuffer, dstRowLen * bytesPerPixel, GetContentsAbsoluteIndex(dstStart),
                strideWithBorder, dstVLen, dstRowLen * bytesPerPixel);
}

template <typename T>
RingBuffer<T>::RingBuffer(Arena *arena, u32 max) : max(max)
{
    entries = StaticArray<T>(arena, max);
    // mutex       = {};
    readOffset  = 0;
    writeOffset = 0;
}

template <typename T>
bool RingBuffer<T>::Write(T *vals, u32 num)
{
    bool result  = true;
    u32 capacity = entries.capacity;
    if (writeOffset + num >= readOffset + capacity)
    {
        result = false;
    }
    else
    {
        u32 writeIndex = writeOffset & (capacity - 1);
        u32 numToEnd   = capacity - writeIndex;
        MemoryCopy(entries.data + writeIndex, vals, sizeof(T) * Min(num, numToEnd));
        if (num > numToEnd)
        {
            MemoryCopy(entries.data, vals + numToEnd, sizeof(T) * (num - numToEnd));
        }
        writeOffset += num;
    }

    return result;
}

template <typename T>
void RingBuffer<T>::WriteWithOverwrite(T *vals, u32 num)
{
    u32 capacity = entries.capacity;
    if (writeOffset + num >= readOffset + capacity)
    {
        readOffset = writeOffset + num - capacity;
    }
    u32 writeIndex = writeOffset & (capacity - 1);
    u32 numToEnd   = capacity - writeIndex;
    MemoryCopy(entries.data + writeIndex, vals, sizeof(T) * Min(num, numToEnd));
    if (num > numToEnd)
    {
        MemoryCopy(entries.data, vals + numToEnd, sizeof(T) * (num - numToEnd));
    }
    writeOffset += num;
}

template <typename T>
void RingBuffer<T>::SynchronizedWrite(Mutex *mutex, T *vals, u32 num)
{
    for (;;)
    {
        BeginMutex(mutex);
        bool result = Write(vals, num);

        if (result)
        {
            EndMutex(mutex);
            break;
        }

        EndMutex(mutex);

        std::this_thread::yield();
    }
}

template <typename T>
T *RingBuffer<T>::Read(Arena *arena, u32 &num)
{
    // BeginMutex(&mutex);
    Assert(readOffset <= writeOffset);
    u32 numToRead = writeOffset - readOffset;
    numToRead     = num == 0 ? numToRead : Min(numToRead, num);
    u32 capacity  = entries.capacity;
    T *vals       = 0;
    if (numToRead)
    {
        vals          = (T *)PushArrayNoZero(arena, u8, sizeof(T) * numToRead);
        u32 readIndex = readOffset & (capacity - 1);
        u32 numToEnd  = capacity - readIndex;
        MemoryCopy(vals, entries.data + readIndex, sizeof(T) * Min(numToRead, numToEnd));
        if (numToRead > numToEnd)
        {
            MemoryCopy(vals + numToEnd, entries.data, sizeof(T) * (numToRead - numToEnd));
        }
        readOffset += numToRead;
    }
    num = numToRead;
    // EndMutex(&mutex);
    return vals;
}

template <typename T>
T *RingBuffer<T>::SynchronizedRead(Mutex *mutex, Arena *arena, u32 &num)
{
    BeginMutex(mutex);
    T *result = Read(arena, num);
    EndMutex(mutex);
    return result;
}

VirtualTextureManager::VirtualTextureManager(Arena *arena, u32 maxSize, u32 slabSize,
                                             VkFormat format)
    : format(format), slabAllocInfoRingBuffer(arena, 1u << 20u), slabSize(slabSize),
      submissionReadIndex(0), submissionWriteIndex(0),
      feedbackRequestRingBuffer(arena, 1u << 20u), evictRequestRingBuffer(arena, 1u << 20u),
      slabAllocInfoRingBuffer2(arena, 1u << 20u)
{
    string shaderName = "../src/shaders/update_page_tables.spv";
    string data       = OS_ReadFile(arena, shaderName);
    shader            = device->CreateShader(ShaderStage::Compute, "update page tables", data);

    string clearPageTableName = "../src/shaders/clear_page_table.spv";
    string clearPageTableData = OS_ReadFile(arena, clearPageTableName);
    Shader clearPageTableShader =
        device->CreateShader(ShaderStage::Compute, "clear page table", clearPageTableData);

    descriptorSetLayout = {};
    descriptorSetLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    descriptorSetLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    push     = PushConstant(ShaderStage::Compute, 0, sizeof(PageTableUpdatePushConstant));
    pipeline = device->CreateComputePipeline(&shader, &descriptorSetLayout, &push);

    clearPageTableLayout = {};
    clearPageTableLayout.AddBinding(0, DescriptorType::StorageImage,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
    clearPageTablePipeline =
        device->CreateComputePipeline(&clearPageTableShader, &clearPageTableLayout);

    // ImageLimits limits = device->GetImageLimits();

    // Allocate page table
    pageHashTableBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               sizeof(Vec4u) * MAX_LEVEL * (1u << 20u));

    // Allocate cpu page table
    cpuPageHashTable =
        StaticArray<HashTableEntry>(arena, MAX_LEVEL * (1u << 20u), MAX_LEVEL * (1u << 20u));
    MemorySet(cpuPageHashTable.data, 0xff, sizeof(HashTableEntry) * MAX_LEVEL * (1u << 20u));

    gpuSubmissionStates = StaticArray<GPUSubmissionState>(arena, 1024);
    for (u32 i = 0; i < 1024; i++)
    {
        GPUSubmissionState state    = {};
        state.arena                 = ArenaAlloc();
        state.status                = SubmissionStatus::Empty;
        state.semaphore             = device->CreateSemaphore();
        state.semaphore.signalValue = 1;
        state.uploadBuffer.size     = {};
        gpuSubmissionStates.Push(state);
    }

    // Slabs
    {
        u32 numSlabs = maxSize / slabSize;
        slabs        = StaticArray<Slab>(arena, numSlabs);
        caches       = StaticArray<StaticArray<int>>(arena, log2MaxTileDim + 1);
        for (int i = 0; i <= log2MaxTileDim; i++)
        {
            caches.Push(StaticArray<int>(arena, log2MaxTileDim + 1));
            for (int j = 0; j <= log2MaxTileDim; j++)
            {
                caches[i].Push(-1);
            }
        }

        texArena = ArenaAlloc();

        lruHead = 0;
        lruTail = 1;

        HashTableEntry entry      = {};
        entry.lruNext             = lruTail;
        cpuPageHashTable[lruHead] = entry;
        entry.lruPrev             = lruHead;
        cpuPageHashTable[lruTail] = entry;

        arenaMutex.count     = 0;
        slabInfoMutex.count  = 0;
        slabInfoMutex2.count = 0;
        feedbackMutex.count  = 0;
        evictMutex.count     = 0;
    }

    // Instantiate streaming system
    {
        feedbackBuffers =
            FixedArray<TransferBuffer, numPendingSubmissions>(numPendingSubmissions);
        feedbackBuffers[0] =
            device->GetReadbackBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, megabytes(8));
        feedbackBuffers[1] =
            device->GetReadbackBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, megabytes(8));

        highestMipUploadBuffer = {};
        requestUploadBuffer    = {};
        requestBuffer          = {};
    }
}

u32 VirtualTextureManager::AllocateVirtualPages(Arena *arena, string filename, u8 *faceData,
                                                u32 numFaces, CommandBuffer *cmd)
{
    TextureInfo texInfo;
    texInfo.filename      = PushStr8Copy(arena, filename);
    texInfo.faceDataSizes = faceData;
    texInfo.numFaces      = numFaces;
    textureInfo.push_back(texInfo);

    u32 textureIndex = textureInfo.size() - 1;

    ScratchArena scratch;

    Ptex::String error;
    Ptex::PtexTexture *t = cache->get((char *)texInfo.filename.str, error);

    BufferImageCopy *copies = PushArrayNoZero(scratch.temp.arena, BufferImageCopy, numFaces);
    StaticArray<PageTableUpdateRequest> updateRequests(scratch.temp.arena, numFaces);
    u32 numCopies = 0;
    int slabIndex = -1;

    if (numFaces * GetFormatSize(format) > highestMipUploadBuffer.size)
    {
        if (highestMipUploadBuffer.size)
        {
            device->DestroyBuffer(&highestMipUploadBuffer);
        }
        highestMipUploadBuffer =
            device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 numFaces * GetFormatSize(format), MemoryUsage::CPU_TO_GPU);
    }
    u32 bufferOffset = 0;
    for (u32 faceID = 0; faceID < numFaces; faceID++)
    {
        Ptex::FaceInfo fi = t->getFaceInfo(faceID);
        Ptex::Res res(0, 0);
        u32 mipLevel = Max(fi.res.ulog2, fi.res.vlog2);
        Ptex::PtexPtr<Ptex::PtexFaceData> d(t->getData(faceID, res));
        Assert(d->isConstant());

        char *buffer = PushArrayNoZero(scratch.temp.arena, char, GetFormatSize(format));

        u32 numChannels     = t->numChannels();
        u32 bytesPerChannel = Ptex::DataSize(t->dataType());
        u32 bytesPerPixel   = numChannels * bytesPerChannel;

        int stride = res.u() * bytesPerPixel;
        int rowlen = stride;

        Ptex::PtexUtils::fill(d->getData(), buffer, stride, 1, 1, bytesPerPixel);

        if (t->alphaChannel() == -1)
        {
            buffer[3] = 255;
        }

        MemoryCopy((u8 *)highestMipUploadBuffer.mappedPtr + bufferOffset, buffer,
                   GetFormatSize(format));

        SlabAllocInfo info;
        bool allocated = AllocateMemory(0, 0, info);
        HashTableEntry entry(textureIndex, faceID, 0, mipLevel, info.slabIndex,
                             info.entryIndex, info.offset, info.bindlessIndex);
        u32 hashIndex = UpdateHash(entry);
        if (slabIndex == -1)
        {
            slabIndex = info.slabIndex;
        }
        if (info.slabIndex != slabIndex)
        {
            cmd->Barrier(&slabs[slabIndex].image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_NONE,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT);
            cmd->FlushBarriers();
            cmd->CopyImage(&highestMipUploadBuffer, &slabs[slabIndex].image, copies,
                           numCopies);
            cmd->Barrier(&slabs[slabIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_NONE);
            cmd->FlushBarriers();
            numCopies = 0;
            slabIndex = info.slabIndex;
        }

        u32 copyIndex = numCopies++;

        BufferImageCopy copy = {};
        copy.bufferOffset    = bufferOffset;
        copy.baseLayer       = info.layerIndex;
        copy.layerCount      = 1;
        copy.offset          = Vec3i(info.offset.x, info.offset.y, 0);
        copy.extent          = Vec3u(1, 1, 1);
        copies[copyIndex]    = copy;

        bufferOffset += GetFormatSize(format);

        PageTableUpdateRequest request;
        request.packed    = PackHashTableEntry(entry);
        request.hashIndex = hashIndex;
        updateRequests.Push(request);

        Assert(allocated);
    }
    cmd->Barrier(&slabs[slabIndex].image, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_NONE,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT);
    cmd->FlushBarriers();
    cmd->CopyImage(&highestMipUploadBuffer, &slabs[slabIndex].image, copies, numCopies);
    cmd->Barrier(&slabs[slabIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_ACCESS_2_NONE);
    cmd->FlushBarriers();

    if (updateRequests.Length() * sizeof(PageTableUpdateRequest) > requestUploadBuffer.size)
    {
        if (requestUploadBuffer.size)
        {
            device->DestroyBuffer(&requestUploadBuffer);
            device->DestroyBuffer(&requestBuffer);
        }
        requestUploadBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sizeof(PageTableUpdateRequest) * updateRequests.Length(), MemoryUsage::CPU_TO_GPU);
        requestBuffer =
            device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 sizeof(PageTableUpdateRequest) * updateRequests.Length());
    }

    BufferToBufferCopy copy = {};
    copy.size               = sizeof(PageTableUpdateRequest) * updateRequests.Length();
    MemoryCopy(requestUploadBuffer.mappedPtr, updateRequests.data, copy.size);

    cmd->CopyBuffer(&requestBuffer, &requestUploadBuffer);
    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    PageTableUpdatePushConstant pc;
    pc.numRequests = updateRequests.Length();

    cmd->StartBindingCompute(pipeline, &descriptorSetLayout)
        .Bind(&requestBuffer)
        .Bind(&pageHashTableBuffer)
        .PushConstants(&push, &pc)
        .End();
    cmd->Dispatch((pc.numRequests + 63) >> 6, 1, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    return textureInfo.size() - 1;
}

bool VirtualTextureManager::AllocateMemory(int logWidth, int logHeight, SlabAllocInfo &info)
{
    int indexX = logWidth;
    int indexY = logHeight;

    Assert(indexX >= indexY);

    int slabHead  = caches[indexX][indexY];
    int slabIndex = slabHead;

    while (slabIndex != -1)
    {
        Slab &slab = slabs[slabIndex];
        if (slab.freeList.Length())
        {
            u32 entryIndex = slab.freeList.Pop();

            Vec3u offset = slab.entries[entryIndex].offset;

            SlabAllocInfo result;
            result.offset        = offset.xy;
            result.bindlessIndex = slab.entries[entryIndex].bindlessIndex;
            result.slabIndex     = slabIndex;
            result.entryIndex    = entryIndex;
            result.layerIndex    = offset.z;

            info = result;

            return true;
        }
        slabIndex = slab.next;
    }
    Assert(slabIndex == -1);

    int blockShift   = GetBlockShift(format);
    u32 formatSize   = GetFormatSize(format);
    int newLogWidth  = Max(logWidth - blockShift, 0);
    int newLogHeight = Max(logHeight - blockShift, 0);

    u32 sizeX = Max((1u << newLogWidth), 16u);
    u32 sizeY = Max((1u << newLogHeight), 16u);

    u32 entrySizeX = 1u << newLogWidth;
    u32 entrySizeY = 1u << newLogHeight;

    if (slabs.Length() == slabs.capacity) return false;

    slabIndex = slabs.Length();

    u32 numLayers  = slabSize / (sizeX * sizeY * formatSize);
    u32 entriesX   = sizeX / (1u << logWidth);
    u32 entriesY   = sizeY / (1u << logHeight);
    u32 numEntries = entriesX * entriesY * numLayers;

    Slab slab;
    slab.log2Width  = logWidth;
    slab.log2Height = logHeight;

    slab.entries      = StaticArray<SlabEntry>(texArena, numEntries);
    slab.freeList     = StaticArray<int>(texArena, numEntries);
    slab.layerLayouts = StaticArray<VkImageLayout>(texArena, numLayers);

    ImageDesc desc(
        ImageType::Array2D, sizeX, sizeY, 1, 1, numLayers, format, MemoryUsage::GPU_ONLY,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_OPTIMAL);

    slab.image = device->CreateImage(desc, numLayers);

    for (u32 layer = 0; layer < numLayers; layer++)
    {
        int subresourceIndex = device->CreateSubresource(&slab.image, 0, ~0u, layer, 1);
        int bindlessIndex    = device->BindlessIndex(
            &slab.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceIndex);
        slab.layerLayouts.Push(VK_IMAGE_LAYOUT_UNDEFINED);

        for (u32 entryY = 0; entryY < entriesY; entryY++)
        {
            for (u32 entryX = 0; entryX < entriesX; entryX++)
            {
                Vec3u offset(entryX * entrySizeX, entryY * entrySizeY, layer);

                SlabEntry entry = {offset, bindlessIndex};
                slab.entries.Push(entry);
                slab.freeList.Push(slab.entries.Length() - 1);
            }
        }
    }
    slab.next              = slabHead;
    caches[indexX][indexY] = slabIndex;

    u32 freeListIndex = slab.freeList.Pop();
    Vec3u offset      = slab.entries[freeListIndex].offset;

    SlabAllocInfo result;
    result.offset        = offset.xy;
    result.bindlessIndex = slab.entries[freeListIndex].bindlessIndex;
    result.slabIndex     = slabIndex;
    result.entryIndex    = freeListIndex;
    result.layerIndex    = offset.z;

    info = result;
    slabs.Push(slab);

    return true;
}

u64 VirtualTextureManager::CalculateHash(u32 textureIndex, u32 faceIndex, u32 mipLevel,
                                         u32 tileIndex)
{
    u64 bitsToHash = (u64)textureIndex | ((u64)tileIndex << 16) | ((u64)faceIndex << 32) |
                     ((u64)mipLevel << 60);
    u64 hash = MixBits(bitsToHash);
    return hash;
}

u32 VirtualTextureManager::UpdateHash(HashTableEntry entry)
{
    u64 hash =
        CalculateHash(entry.textureIndex, entry.faceID, entry.mipLevel, entry.tileIndex);
    u64 hashIndex = hash % u64(cpuPageHashTable.Length());
    for (;;)
    {
        if (cpuPageHashTable[hashIndex].textureIndex == ~0u)
        {
            cpuPageHashTable[hashIndex] = entry;
            return hashIndex;
        }
        hashIndex = (hashIndex + 1) % cpuPageHashTable.Length();
    }
}

int VirtualTextureManager::GetInHash(u32 textureIndex, u32 tileIndex, u32 faceID, u32 mipLevel)
{
    u64 hash      = CalculateHash(textureIndex, faceID, mipLevel, tileIndex);
    u64 hashIndex = hash % cpuPageHashTable.Length();
    for (;;)
    {
        if (cpuPageHashTable[hashIndex].textureIndex != ~0u)
        {
            HashTableEntry &testEntry = cpuPageHashTable[hashIndex];
            if (testEntry.faceID == faceID && testEntry.textureIndex == textureIndex &&
                testEntry.mipLevel == mipLevel && testEntry.tileIndex == tileIndex)
            {
                return hashIndex;
            }
        }
        else
        {
            return -1;
        }
        hashIndex = (hashIndex + 1) % cpuPageHashTable.Length();
    }
}

void VirtualTextureManager::ClearTextures(CommandBuffer *cmd)
{
    cmd->ClearBuffer(&pageHashTableBuffer, ~0u);
}

///////////////////////////////////////
// Streaming/Feedback
//

Vec4u VirtualTextureManager::PackHashTableEntry(HashTableEntry &entry)
{
    Vec4u packed;
    Assert(entry.textureIndex < (1u << 16u));
    Assert(entry.tileIndex < (1u << 16u));
    Assert(entry.faceID < (1u << 28u));
    Assert(entry.mipLevel < (1u << 4u));
    Assert(entry.offset.x < 16 && entry.offset.y < 16);

    packed.x = entry.textureIndex | (entry.tileIndex << 16u);
    packed.y = entry.faceID | (entry.mipLevel << 28u);
    packed.z = entry.bindlessIndex;
    packed.w = entry.offset.x | (entry.offset.y << 4u);
    return packed;
}
// Executes on main thread
void VirtualTextureManager::Update(CommandBuffer *computeCmd)
{
    // Update page table
    ScratchArena scratch;
    u32 currentBuffer = !(device->frameCount & 1);

    u32 numFeedbackRequests = ((u32 *)feedbackBuffers[currentBuffer].mappedPtr)[0];
    Vec2u *feedbackRequests = (Vec2u *)((u32 *)feedbackBuffers[currentBuffer].mappedPtr + 1);

    BeginMutex(&feedbackMutex);
    feedbackRequestRingBuffer.WriteWithOverwrite(feedbackRequests, numFeedbackRequests);
    EndMutex(&feedbackMutex);

    // Process feedback requests
    if (counter.count.load() == 0)
    {
        scheduler.Schedule(&counter, [&](u32 jobID) {
            ScratchArena scratch;

            u32 numFeedbackRequests = 0;
            Vec2u *feedbackRequests = feedbackRequestRingBuffer.SynchronizedRead(
                &feedbackMutex, scratch.temp.arena, numFeedbackRequests);
            StaticArray<FeedbackRequest> compactedFeedbackRequests(scratch.temp.arena,
                                                                   numFeedbackRequests);
            u32 hashMapSize = NextPowerOfTwo(numFeedbackRequests);
            HashIndex pageHashMap(scratch.temp.arena, hashMapSize, hashMapSize);

            // Compact feedback
            for (int requestIndex = 0; requestIndex < numFeedbackRequests; requestIndex++)
            {
                Vec2u &feedbackRequest = feedbackRequests[requestIndex];
                u32 textureIndex       = BitFieldExtractU32(feedbackRequest.x, 16, 0);
                u32 tileIndex          = BitFieldExtractU32(feedbackRequest.x, 16, 16);
                u32 faceID             = BitFieldExtractU32(feedbackRequest.y, 28, 0);
                u32 mipLevel           = BitFieldExtractU32(feedbackRequest.y, 4, 28);

                u8 faceSizes = textureInfo[textureIndex].faceDataSizes[faceID];
                u32 maxMip   = Max(BitFieldExtractU32(faceSizes, 4, 0),
                                   BitFieldExtractU32(faceSizes, 4, 4));

                if (mipLevel == maxMip) continue;

                u64 packed = ((u64)feedbackRequest.y << 32u) | ((u64)feedbackRequest.x);
                i32 hash   = (i32)MixBits(packed);

                bool found = false;
                for (int index = pageHashMap.FirstInHash(hash); index != -1;
                     index     = pageHashMap.NextInHash(index))
                {
                    FeedbackRequest &fr = compactedFeedbackRequests[index];

                    if (fr.packedFeedbackInfo == feedbackRequest)
                    {
                        compactedFeedbackRequests[index].count++;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    u32 index = compactedFeedbackRequests.size();
                    FeedbackRequest request;
                    request.packedFeedbackInfo = feedbackRequest;
                    request.count              = 1;
                    compactedFeedbackRequests.push_back(request);
                    Assert(index == 0 ||
                           request.packedFeedbackInfo !=
                               compactedFeedbackRequests[index - 1].packedFeedbackInfo);
                    pageHashMap.AddInHash(hash, index);
                }
            }

            u32 numCompacted           = compactedFeedbackRequests.size();
            u32 numNonResidentFeedback = 0;

            int lruUsed = -1;
            // If requested page is resident Update LRU. Otherwise
            for (int requestIndex = 0; requestIndex < compactedFeedbackRequests.size();
                 requestIndex++)
            {
                FeedbackRequest fr    = compactedFeedbackRequests[requestIndex];
                Vec2u feedbackRequest = fr.packedFeedbackInfo;
                u32 textureIndex      = BitFieldExtractU32(feedbackRequest.x, 16, 0);
                u32 tileIndex         = BitFieldExtractU32(feedbackRequest.x, 16, 16);
                u32 faceID            = BitFieldExtractU32(feedbackRequest.y, 28, 0);
                u32 mipLevel          = BitFieldExtractU32(feedbackRequest.y, 4, 28);

                TextureInfo &texInfo = textureInfo[textureIndex];
                int entryIndex       = GetInHash(textureIndex, tileIndex, faceID, mipLevel);

                // Move to head of LRU if requested page is already resident
                if (entryIndex != -1)
                {
                    HashTableEntry &entry = cpuPageHashTable[entryIndex];
                    lruUsed               = lruUsed == -1 ? entryIndex : lruUsed;
                    if (entry.lruNext != -1 && entry.lruPrev != -1)
                    {
                        UnlinkLRU(entryIndex);
                        LinkLRU(entryIndex);
                    }
                }
                // Otherwise, page needs to be mapped
                else
                {
                    compactedFeedbackRequests[numNonResidentFeedback++] = fr;
                }
            }
            compactedFeedbackRequests.size() = numNonResidentFeedback;

            Print("Num feedback: %u, num compacted: %u, num non resident: %u\n",
                  numFeedbackRequests, numCompacted, numNonResidentFeedback);

            // compactedFeedbackRequestsRingBuffer.SynchronizedWrite(
            //     &feedbackMutex2, compactedFeedbackRequests.data,
            //     compactedFeedbackRequests.size());

            Array<PageTableUpdateRequest> updateRequests(scratch.temp.arena,
                                                         numNonResidentFeedback);
            StaticArray<SlabAllocInfo> newSlabAllocInfo(scratch.temp.arena,
                                                        numNonResidentFeedback);
            u32 numEvicted = 0;

            for (FeedbackRequest fr : compactedFeedbackRequests)
            {
                Vec2u feedbackRequest = fr.packedFeedbackInfo;
                u32 textureIndex      = BitFieldExtractU32(feedbackRequest.x, 16, 0);
                u32 tileIndex         = BitFieldExtractU32(feedbackRequest.x, 16, 16);
                u32 faceID            = BitFieldExtractU32(feedbackRequest.y, 28, 0);
                u32 mipLevel          = BitFieldExtractU32(feedbackRequest.y, 4, 28);

                TextureInfo &texInfo = textureInfo[textureIndex];
                u8 faceSizes         = texInfo.faceDataSizes[faceID];

                int ulog2 = (int)BitFieldExtractU32(faceSizes, 4, 0);
                int vlog2 = (int)BitFieldExtractU32(faceSizes, 4, 4);

                Assert(faceID < texInfo.numFaces);
                ErrorExit(mipLevel <= ulog2 || mipLevel <= vlog2, "%S %u %u %u %u %u %u\n",
                          texInfo.filename, mipLevel, ulog2, vlog2, textureIndex, tileIndex,
                          faceID);
                int log2Width  = Clamp((int)ulog2 - (int)mipLevel, 0, (int)log2MaxTileDim);
                int log2Height = Clamp((int)vlog2 - (int)mipLevel, 0, (int)log2MaxTileDim);
                int minDim     = Min(log2Width, log2Height);
                int maxDim     = Max(log2Width, log2Height);

                SlabAllocInfo info;
                bool allocated = AllocateMemory(maxDim, minDim, info);

                if (!allocated)
                {
                    for (;;)
                    {
                        int entryToEvict = cpuPageHashTable[lruTail].lruPrev;

                        if (entryToEvict == lruUsed)
                        {
                            ErrorExit(0, "Ran out of entries in the LRU for : %u %u\n",
                                      log2Width, log2Height);
                            break;
                        }

                        UnlinkLRU(entryToEvict);
                        HashTableEntry &entry = cpuPageHashTable[entryToEvict];
                        Slab &slab            = slabs[entry.slabIndex];
                        slab.freeList.Push(entry.slabEntryIndex);

                        PageTableUpdateRequest evictRequest;
                        evictRequest.packed    = Vec4u(~0u);
                        evictRequest.hashIndex = entryToEvict;
                        updateRequests.Push(evictRequest);

                        cpuPageHashTable[entryToEvict].textureIndex = ~0u;

                        numEvicted++;

                        if (slab.log2Width >= maxDim && slab.log2Height >= minDim)
                        // if (slab.log2Width == maxDim && slab.log2Height == minDim)
                        {
                            allocated = AllocateMemory(slab.log2Width, slab.log2Height, info);
                            Assert(allocated);
                            break;
                        }
                    }
                }

                // TODO: what if the LRU gets evicted before the data is uploaded?
                HashTableEntry entry(textureIndex, faceID, tileIndex, mipLevel, info.slabIndex,
                                     info.entryIndex, info.offset, info.bindlessIndex);
                u32 hashIndex = UpdateHash(entry);
                LinkLRU(hashIndex);

                info.hashIndex       = hashIndex;
                info.feedbackRequest = feedbackRequest;
                newSlabAllocInfo.Push(info);
            }

            Print("Evicting %u entries\n", numEvicted);

            if (updateRequests.Length())
            {
                Assert(updateRequests.Length() < evictRequestRingBuffer.max);
                evictRequestRingBuffer.SynchronizedWrite(&evictMutex, updateRequests.data,
                                                         updateRequests.Length());
            }
            if (newSlabAllocInfo.Length())
            {
                slabAllocInfoRingBuffer2.SynchronizedWrite(
                    &slabInfoMutex2, newSlabAllocInfo.data, newSlabAllocInfo.Length());
            }
        });
    }

    u32 numUpdateRequests                 = 0;
    PageTableUpdateRequest *evictRequests = evictRequestRingBuffer.SynchronizedRead(
        &evictMutex, scratch.temp.arena, numUpdateRequests);

    u32 numInfos         = 0;
    SlabAllocInfo *infos = slabAllocInfoRingBuffer2.SynchronizedRead(
        &slabInfoMutex2, scratch.temp.arena, numInfos);
    Print("num infos read: %u\n", numInfos);

    slabAllocInfoRingBuffer.SynchronizedWrite(&slabInfoMutex, infos, numInfos);

    Array<PageTableUpdateRequest> updateRequests(scratch.temp.arena,
                                                 Max(2u * numUpdateRequests, 4096u));
    updateRequests.size = numUpdateRequests;
    MemoryCopy(updateRequests.data, evictRequests,
               sizeof(PageTableUpdateRequest) * numUpdateRequests);

    int textureFeedback2 = TIMED_CPU_RANGE_NAME_BEGIN("texture update");

    Array<SmallAllocation> smallAllocations(scratch.temp.arena, 4096);

    for (;;)
    {
        BeginMutex(&submissionMutex);
        u32 submissionIndex               = submissionReadIndex % gpuSubmissionStates.Length();
        u64 readIndex                     = submissionReadIndex;
        u64 writeIndex                    = submissionWriteIndex;
        GPUSubmissionState &state         = gpuSubmissionStates[submissionIndex];
        SubmissionStatus submissionStatus = state.status;
        EndMutex(&submissionMutex);

        if (readIndex >= writeIndex) break;
        Assert(submissionStatus == SubmissionStatus::Pending);

        u64 semValue =
            device->GetSemaphoreValue(gpuSubmissionStates[submissionIndex].semaphore);

        if (state.semaphore.signalValue == semValue)
        {
            state.status = SubmissionStatus::Empty;
            state.semaphore.signalValue++;

            // Update hash table
            for (SlabAllocInfo &info : state.slabAllocInfos)
            {
                Vec2u fr = info.feedbackRequest;

                u32 textureIndex = BitFieldExtractU32(fr.x, 16, 0);
                u32 tileIndex    = BitFieldExtractU32(fr.x, 16, 16);
                u32 faceID       = BitFieldExtractU32(fr.y, 28, 0);
                u32 mipLevel     = BitFieldExtractU32(fr.y, 4, 28);

                HashTableEntry entry(textureIndex, faceID, tileIndex, mipLevel, info.slabIndex,
                                     info.entryIndex, info.offset, info.bindlessIndex);

                int hashIndex = info.hashIndex;
                Assert(hashIndex != -1);

                PageTableUpdateRequest request;
                request.packed    = PackHashTableEntry(entry);
                request.hashIndex = hashIndex;
                updateRequests.Push(request);

                Slab &slab = slabs[info.slabIndex];
                computeCmd->Barrier(
                    &slab.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                    QueueType_Ignored, QueueType_Ignored, 0, ~0u, info.layerIndex, 1);
            }

            // Allocate small entries
            for (SmallAllocation &info : state.smallAllocs)
            {
                SmallAllocation newAlloc = info;
                newAlloc.buffer          = PushArrayNoZero(scratch.temp.arena, u8, info.size);
                MemoryCopy(newAlloc.buffer, info.buffer, info.size);
                smallAllocations.Push(newAlloc);
            }

            computeCmd->FlushBarriers();

            GPUBuffer &uploadBuffer = state.uploadBuffer;
            if (uploadBuffer.size)
            {
                device->DestroyBuffer(&uploadBuffer);
                uploadBuffer.size = 0;
            }

            ArenaClear(state.arena);

            BeginMutex(&submissionMutex);
            submissionReadIndex++;
            EndMutex(&submissionMutex);
        }
        else
        {
            break;
        }
    }

    for (SmallAllocation &info : smallAllocations)
    {
        Slab &slab            = slabs[info.info.slabIndex];
        SlabEntry &entry      = slab.entries[info.info.entryIndex];
        VkImageLayout &layout = slab.layerLayouts[info.info.layerIndex];
        if (layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            computeCmd->Barrier(&slab.image, layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_NONE,
                                VK_ACCESS_2_TRANSFER_WRITE_BIT, QueueType_Ignored,
                                QueueType_Ignored, 0, ~0u, info.info.layerIndex, 1);
            layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        }
    }
    computeCmd->FlushBarriers();

    if (smallAllocations.Length())
    {
        struct Handle
        {
            u32 sortKey;
            int index;
        };

        // Generate radix sort keys
        Handle *handles =
            PushArrayNoZero(scratch.temp.arena, Handle, smallAllocations.Length());

        for (int index = 0; index < smallAllocations.Length(); index++)
        {
            SmallAllocation &info  = smallAllocations[index];
            handles[index].sortKey = info.info.slabIndex;
            handles[index].index   = index;
        }

        SortHandles<Handle, false>(handles, smallAllocations.Length());

        u32 maxSmallUploadSize = smallAllocations.Length() * 8 * 16 * GetFormatSize(format);
        if (highestMipUploadBuffer.size < maxSmallUploadSize)
        {
            if (highestMipUploadBuffer.size)
            {
                device->DestroyBuffer(&highestMipUploadBuffer);
            }
            highestMipUploadBuffer = device->CreateBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, maxSmallUploadSize, MemoryUsage::CPU_TO_GPU);
        }

        int slabIndex = smallAllocations[handles[0].index].info.slabIndex;

        StaticArray<BufferImageCopy> copies(scratch.temp.arena, smallAllocations.Length());
        u32 bufferOffset = 0;

        for (int index = 0; index < smallAllocations.Length(); index++)
        {
            SmallAllocation &info = smallAllocations[handles[index].index];
            if (info.info.slabIndex != slabIndex)
            {
                Slab &slab = slabs[slabIndex];
                computeCmd->CopyImage(&highestMipUploadBuffer, &slab.image, copies.data,
                                      copies.Length());

                slabIndex = info.info.slabIndex;
                copies.Clear();
            }

            MemoryCopy((u8 *)highestMipUploadBuffer.mappedPtr + bufferOffset, info.buffer,
                       info.size);

            BufferImageCopy copy = {};
            copy.bufferOffset    = bufferOffset;
            copy.baseLayer       = info.info.layerIndex;
            copy.layerCount      = 1;
            copy.offset          = Vec3i(info.info.offset.x, info.info.offset.y, 0);
            copy.extent          = Vec3u(info.width, info.height, 1);

            copies.Push(copy);

            bufferOffset += info.size;
        }

        Slab &slab = slabs[slabIndex];
        computeCmd->CopyImage(&highestMipUploadBuffer, &slab.image, copies.data,
                              copies.Length());

        Print("num small allocs: %u, size: %u\n", smallAllocations.Length(), bufferOffset);

        for (SmallAllocation &info : smallAllocations)
        {
            Slab &slab = slabs[info.info.slabIndex];

            Vec2u fr = info.info.feedbackRequest;

            u32 textureIndex = BitFieldExtractU32(fr.x, 16, 0);
            u32 tileIndex    = BitFieldExtractU32(fr.x, 16, 16);
            u32 faceID       = BitFieldExtractU32(fr.y, 28, 0);
            u32 mipLevel     = BitFieldExtractU32(fr.y, 4, 28);

            HashTableEntry entry(textureIndex, faceID, tileIndex, mipLevel,
                                 info.info.slabIndex, info.info.entryIndex, info.info.offset,
                                 info.info.bindlessIndex);

            int hashIndex = info.info.hashIndex;
            Assert(hashIndex != -1);

            PageTableUpdateRequest request;
            request.packed    = PackHashTableEntry(entry);
            request.hashIndex = hashIndex;
            updateRequests.Push(request);

            VkImageLayout &layout = slab.layerLayouts[info.info.layerIndex];

            if (layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            {
                computeCmd->Barrier(
                    &slab.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT, QueueType_Ignored, QueueType_Ignored, 0, ~0u,
                    info.info.layerIndex, 1);

                layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }
        computeCmd->FlushBarriers();
    }

    if (updateRequests.Length() * sizeof(PageTableUpdateRequest) > requestUploadBuffer.size)
    {
        if (requestUploadBuffer.size)
        {
            device->DestroyBuffer(&requestUploadBuffer);
            device->DestroyBuffer(&requestBuffer);
        }
        requestUploadBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sizeof(PageTableUpdateRequest) * updateRequests.Length(), MemoryUsage::CPU_TO_GPU);
        requestBuffer =
            device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 sizeof(PageTableUpdateRequest) * updateRequests.Length());
    }

    if (updateRequests.Length())
    {
        BufferToBufferCopy copy = {};
        copy.size               = sizeof(PageTableUpdateRequest) * updateRequests.Length();
        MemoryCopy(requestUploadBuffer.mappedPtr, updateRequests.data, copy.size);

        computeCmd->CopyBuffer(&requestBuffer, &requestUploadBuffer, &copy, 1);

        computeCmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->FlushBarriers();

        PageTableUpdatePushConstant pc;
        pc.numRequests = updateRequests.Length();

        computeCmd->StartBindingCompute(pipeline, &descriptorSetLayout)
            .Bind(&requestBuffer)
            .Bind(&pageHashTableBuffer)
            .PushConstants(&push, &pc)
            .End();
        computeCmd->Dispatch((pc.numRequests + 63) >> 6, 1, 1);
        computeCmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        computeCmd->FlushBarriers();
    }

    TIMED_RANGE_END(textureFeedback2);

    u32 totalNumRequests = numInfos;

    // Dispatch worker threads to upload data to GPU
    u32 maxThreads = OS_NumProcessors();
    u32 numThreads = Min(maxThreads, (totalNumRequests + 1023) / 1024);
    for (u32 i = 0; i < numThreads; i++)
    {
        scheduler.Schedule([&](u32 jobID) {
            for (;;)
            {
                ScratchArena scratch;
                u32 num              = 1024;
                SlabAllocInfo *infos = slabAllocInfoRingBuffer.SynchronizedRead(
                    &slabInfoMutex, scratch.temp.arena, num);

                if (!infos) return;

                u32 submissionIndex = ~0u;
                for (;;)
                {
                    BeginMutex(&submissionMutex);
                    if (submissionWriteIndex + 1 <
                        submissionReadIndex + gpuSubmissionStates.Length())
                    {
                        submissionIndex =
                            submissionWriteIndex++ % gpuSubmissionStates.Length();
                        GPUSubmissionState &state = gpuSubmissionStates[submissionIndex];
                        SubmissionStatus submissionStatus = state.status;
                        Assert(submissionStatus == SubmissionStatus::Empty);
                        state.status = SubmissionStatus::Pending;
                        EndMutex(&submissionMutex);
                        break;
                    }
                    EndMutex(&submissionMutex);
                    std::this_thread::yield();
                }
                GPUSubmissionState &state = gpuSubmissionStates[submissionIndex];

                state.status         = SubmissionStatus::Pending;
                state.smallAllocs    = StaticArray<SmallAllocation>(state.arena, num);
                state.slabAllocInfos = StaticArray<SlabAllocInfo>(state.arena, num);

                ChunkedLinkedList<u8> data(scratch.temp.arena);

                CommandBuffer *cmd = device->BeginCommandBuffer(QueueType_Copy);

                struct GPUTextureSubmitInfo
                {
                    u32 slabIndex;
                    u32 layerIndex;
                    u32 bufferOffset;
                    u32 width;
                    u32 height;
                };

                StaticArray<GPUTextureSubmitInfo> submitInfos(scratch.temp.arena, num);
                for (u32 requestIndex = 0; requestIndex < num; requestIndex++)
                {
                    SlabAllocInfo &slabAllocInfo = infos[requestIndex];
                    Vec2u fr                     = slabAllocInfo.feedbackRequest;

                    u32 textureIndex = BitFieldExtractU32(fr.x, 16, 0);
                    u32 tileIndex    = BitFieldExtractU32(fr.x, 16, 16);
                    u32 faceID       = BitFieldExtractU32(fr.y, 28, 0);
                    u32 mipLevel     = BitFieldExtractU32(fr.y, 4, 28);

                    TextureInfo &texInfo = textureInfo[textureIndex];

                    Ptex::String error;
                    Ptex::PtexTexture *t = cache->get((char *)texInfo.filename.str, error);
                    Ptex::FaceInfo fi    = t->getFaceInfo(faceID);

                    Assert(mipLevel <= fi.res.ulog2 || mipLevel <= fi.res.vlog2);

                    int log2Width  = Max((int)fi.res.ulog2 - (int)mipLevel, 0);
                    int log2Height = Max((int)fi.res.vlog2 - (int)mipLevel, 0);
                    Ptex::Res res(log2Width, log2Height);

                    Assert(t);
                    Ptex::PtexPtr<Ptex::PtexFaceData> d(t->getData(faceID, res));

                    u32 numChannels     = t->numChannels();
                    u32 bytesPerChannel = Ptex::DataSize(t->dataType());
                    u32 bytesPerPixel   = numChannels * bytesPerChannel;
                    int stride          = res.u() * bytesPerPixel;
                    int rowlen          = stride;

                    int size = stride * res.v();

                    char *buffer = 0;

                    u32 numTilesU = Max(res.u() >> log2MaxTileDim, 1);
                    u32 numTilesV = Max(res.v() >> log2MaxTileDim, 1);

                    u32 offsetU = maxTileDim * (tileIndex % numTilesU);
                    u32 offsetV = maxTileDim * (tileIndex / numTilesU);

                    int dataResU = res.u();
                    int dataResV = res.v();

                    Assert(tileIndex < numTilesU * numTilesV);
                    Assert(offsetU < dataResU);
                    Assert(offsetV < dataResV);

                    if (d->isConstant())
                    {
                        buffer = PushArrayNoZero(scratch.temp.arena, char, size);
                        Ptex::PtexUtils::fill(d->getData(), buffer, stride, res.u(), res.v(),
                                              bytesPerPixel);
                    }
                    else if (d->isTiled())
                    {
                        // loop over tiles
                        Ptex::Res tileres = d->tileRes();
                        int ntilesu       = res.ntilesu(tileres);
                        int ntilesv       = res.ntilesv(tileres);
                        int tileures      = tileres.u();
                        int tilevres      = tileres.v();

                        u32 ptexTileIndex =
                            ntilesu * (offsetV / tilevres) + offsetU / tileures;

                        dataResU = tileures;
                        dataResV = tilevres;

                        stride = tileures * bytesPerPixel;
                        size   = stride * tilevres;
                        rowlen = stride;
                        buffer = PushArrayNoZero(scratch.temp.arena, char, size);

                        int tilerowlen   = bytesPerPixel * tileures;
                        int tile         = 0;
                        char *dsttilerow = (char *)buffer;
                        char *dsttile    = dsttilerow;

                        Ptex::PtexPtr<Ptex::PtexFaceData> t(d->getTile(ptexTileIndex));
                        if (t->isConstant())
                            Ptex::PtexUtils::fill(t->getData(), dsttile, stride, tileures,
                                                  tilevres, bytesPerPixel);
                        else
                            Ptex::PtexUtils::copy(t->getData(), tilerowlen, dsttile, stride,
                                                  tilevres, tilerowlen);

                        offsetU = offsetU % tileures;
                        offsetV = offsetV % tilevres;
                    }
                    else
                    {
                        buffer = PushArrayNoZero(scratch.temp.arena, char, size);
                        Ptex::PtexUtils::copy(d->getData(), rowlen, buffer, stride, res.v(),
                                              rowlen);
                    }

                    if (dataResU > maxTileDim || dataResV > maxTileDim)
                    {
                        int dstStride = Min(stride, int(maxTileDim * bytesPerPixel));
                        int vres      = Min(dataResV, int(maxTileDim));
                        size          = dstStride * vres;

                        char *newBuffer = PushArrayNoZero(scratch.temp.arena, char, size);
                        char *src       = buffer + stride * offsetV + offsetU * bytesPerPixel;

                        Utils::Copy(src, stride, newBuffer, dstStride, vres, dstStride);

                        stride = dstStride;
                        rowlen = dstStride;

                        dataResU = Min(dataResU, (int)maxTileDim);
                        dataResV = Min(dataResV, (int)maxTileDim);
                        buffer   = newBuffer;
                    }

                    PaddedImage currentFaceImg;
                    currentFaceImg.contents      = 0;
                    currentFaceImg.width         = dataResU;
                    currentFaceImg.height        = dataResV;
                    currentFaceImg.log2Width     = Log2Int(dataResU);
                    currentFaceImg.log2Height    = Log2Int(dataResV);
                    currentFaceImg.bytesPerPixel = GetFormatSize(format);
                    currentFaceImg.strideNoBorder =
                        currentFaceImg.width * currentFaceImg.bytesPerPixel;
                    currentFaceImg.borderSize       = 0;
                    currentFaceImg.strideWithBorder = currentFaceImg.strideNoBorder;

                    if (t->alphaChannel() == -1)
                    {
                        u32 dstBytesPerPixel = GetFormatSize(format);
                        size = currentFaceImg.width * currentFaceImg.height * dstBytesPerPixel;
                        currentFaceImg.contents =
                            PushArrayNoZero(scratch.temp.arena, u8, size);

                        u32 dstOffset = 0;
                        u32 srcOffset = 0;
                        u8 alpha      = 255;
                        for (int i = 0; i < currentFaceImg.GetPaddedWidth() *
                                                currentFaceImg.GetPaddedHeight();
                             i++)
                        {
                            MemoryCopy(currentFaceImg.contents + dstOffset, buffer + srcOffset,
                                       bytesPerPixel);
                            MemorySet(currentFaceImg.contents + dstOffset + bytesPerPixel,
                                      alpha, bytesPerChannel);
                            dstOffset += dstBytesPerPixel;
                            srcOffset += bytesPerPixel;
                        }
                        bytesPerPixel = dstBytesPerPixel;
                    }
                    else
                    {
                        currentFaceImg.contents = (u8 *)buffer;
                    }

                    if (dataResU < dataResV)
                    {
                        PaddedImage img = currentFaceImg;
                        img.contents    = PushArray(scratch.temp.arena, u8, size);
                        Swap(img.width, img.height);
                        Swap(img.log2Width, img.log2Height);
                        Assert(img.width > img.height);
                        img.strideNoBorder   = img.width * img.bytesPerPixel;
                        img.strideWithBorder = img.GetPaddedWidth() * img.bytesPerPixel;
                        img.WriteRotated(currentFaceImg, Vec2u(0, 0), Vec2u(0, 0), 1,
                                         currentFaceImg.GetPaddedHeight(),
                                         currentFaceImg.GetPaddedWidth(),
                                         currentFaceImg.GetPaddedWidth(),
                                         currentFaceImg.GetPaddedHeight(), Vec2u(0, 0));

                        currentFaceImg = img;
                    }
                    t->release();

                    if (currentFaceImg.height < 16)
                    {
                        char *out = PushArrayNoZero(state.arena, char, size);
                        MemoryCopy(out, currentFaceImg.contents, size);

                        SmallAllocation smallAlloc;
                        smallAlloc.info   = infos[requestIndex];
                        smallAlloc.buffer = (u8 *)out;
                        smallAlloc.size   = size;
                        smallAlloc.width  = currentFaceImg.width;
                        smallAlloc.height = currentFaceImg.height;

                        state.smallAllocs.Push(smallAlloc);
                    }
                    else
                    {
                        u32 bufferOffset = data.Length();
                        auto *node       = data.AddNode(size);
                        MemoryCopy(node->values, currentFaceImg.contents, size);

                        GPUTextureSubmitInfo submitInfo;
                        submitInfo.slabIndex    = slabAllocInfo.slabIndex;
                        submitInfo.layerIndex   = slabAllocInfo.layerIndex;
                        submitInfo.bufferOffset = bufferOffset;
                        submitInfo.width        = currentFaceImg.width;
                        submitInfo.height       = currentFaceImg.height;

                        submitInfos.Push(submitInfo);
                        state.slabAllocInfos.Push(slabAllocInfo);
                    }
                }

                for (GPUTextureSubmitInfo &info : submitInfos)
                {
                    Slab &slab = slabs[info.slabIndex];
                    cmd->Barrier(&slab.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                 VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 QueueType_Ignored, QueueType_Ignored, 0, ~0u, info.layerIndex,
                                 1);
                }
                cmd->FlushBarriers();

                GPUBuffer &uploadBuffer = state.uploadBuffer;
                if (uploadBuffer.size)
                {
                    device->DestroyBuffer(&uploadBuffer);
                    uploadBuffer.size = 0;
                }
                if (data.Length())
                {
                    uploadBuffer =
                        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data.Length(),
                                             MemoryUsage::CPU_TO_GPU);
                    data.Flatten((u8 *)uploadBuffer.mappedPtr);

                    for (GPUTextureSubmitInfo &info : submitInfos)
                    {
                        BufferImageCopy copy = {};
                        copy.bufferOffset    = info.bufferOffset;
                        copy.baseLayer       = info.layerIndex;
                        copy.layerCount      = 1;
                        copy.offset          = Vec3i(0);
                        copy.extent          = Vec3u(info.width, info.height, 1);
                        cmd->CopyImage(&uploadBuffer, &slabs[info.slabIndex].image, &copy, 1);
                    }
                }

                cmd->SignalOutsideFrame(state.semaphore);
                device->SubmitCommandBuffer(cmd, false, true);
            }
        });
    }
}

void VirtualTextureManager::UnlinkLRU(int entryIndex)
{
    HashTableEntry &entry = cpuPageHashTable[entryIndex];
    Assert(entry.lruPrev != -1 && entry.lruNext != -1);

    int prevEntry                       = entry.lruPrev;
    int nextEntry                       = entry.lruNext;
    cpuPageHashTable[prevEntry].lruNext = nextEntry;
    cpuPageHashTable[nextEntry].lruPrev = prevEntry;
}

void VirtualTextureManager::LinkLRU(int index)
{
    int nextEntry                   = cpuPageHashTable[lruHead].lruNext;
    cpuPageHashTable[index].lruNext = nextEntry;
    cpuPageHashTable[index].lruPrev = lruHead;

    cpuPageHashTable[nextEntry].lruPrev = index;
    cpuPageHashTable[lruHead].lruNext   = index;
}

} // namespace rt
