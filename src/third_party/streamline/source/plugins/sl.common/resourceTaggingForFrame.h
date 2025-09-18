#pragma once

#include "source/plugins/sl.common/commonInterface.h"

namespace sl
{
namespace common
{

struct ProtectedResourceTagContainer
{
    std::unordered_map<uint64_t, CommonResource> resourceTagContainer;
    mutable std::shared_timed_mutex resourceTagContainerMutex{};

    void setFrameIndex(uint32_t frameIndex)
    {
        // before changing the index - must release all tags
        assert(m_frameIndex == frameIndex || resourceTagContainer.size() == 0);
        m_frameIndex = frameIndex;
    }
    uint32_t getFrameIndex() const
    {
        return m_frameIndex;
    }

private:
    uint32_t m_frameIndex = 0;
};

// Forward declarations for RAII wrappers
class ScopedResourceTagContainerReadAccess;
class ScopedResourceTagContainerWriteAccess;

struct ResourceTaggingForFrame : public ResourceTaggingBase
{
    ResourceTaggingForFrame(chi::ICompute* pCompute, chi::IResourcePool* pPool);
    virtual sl::Result setTag(const sl::Resource* resource,
                              BufferType tag,
                              uint32_t id,
                              const Extent* ext,
                              ResourceLifecycle lifecycle,
                              CommandBuffer* cmdBuffer,
                              bool localTag,
                              const PrecisionInfo* pi,
                              const sl::FrameToken& frame) override final;
    virtual void getTag(BufferType tagType,
                        uint32_t frameId,
                        uint32_t viewportId,
                        CommonResource& res,
                        const sl::BaseStructure** inputs,
                        uint32_t numInputs,
                        bool optional) override final;

    void recycleTags();

    ~ResourceTaggingForFrame();

  private:
    void recycleTagsForFrame(ScopedResourceTagContainerWriteAccess &frameAccess);
    void recycleTagsInternal(uint32_t currAppFrameIndex);
    std::mutex m_recyclingMutex{};
    uint32_t m_prevSeenAppFrameIndex = 0;

    std::mutex requiredTagMutex{};

    // frame-aware nested container of resources for each type of input resource tagged
    std::array<ProtectedResourceTagContainer, 32> m_frames{};

    // Returns a RAII object with read-only access to the ProtectedResourceTagContainer for the specified frame
    // Returns nullptr if the frame is not found
    // TODO: change to std::optional<> when the support of VS2017 is deprecated
    std::unique_ptr<ScopedResourceTagContainerReadAccess>
        findFrameForReading(uint32_t frameIndex);

    // Returns a RAII object with write access to the ProtectedResourceTagContainer for the specified frame
    // Creates the container if it doesn't exist
    // TODO: change to std::optional<> when the support of VS2017 is deprecated
    std::unique_ptr<ScopedResourceTagContainerWriteAccess>
        findFrameForWriting(uint32_t frameIndex);

    chi::ICompute* m_pCompute = nullptr;
    chi::IResourcePool* m_pPool = nullptr;
    sl::RenderAPI m_platform{};
};

} // namespace common
} // namespace sl
