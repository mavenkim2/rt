#include "base.h"
#include "math/vec2.h"
#include "memory.h"

namespace rt
{
#if 0
Sampler *Sampler::Create(Arena *arena, const ScenePacket *packet, const Vec2i fullResolution)
{
    i32 samplesPerPixel        = 16;
    RandomizeStrategy strategy = RandomizeStrategy::FastOwen;
    i32 seed                   = 0;

    b32 isHalton     = (packet->type == "halton"_sid);
    b32 isStratified = (packet->type == "stratified"_sid);

    // stratified sampler only
    bool jitter  = true;
    i32 xSamples = 4;
    i32 ySamples = 4;

    constexpr auto help = "pixelsamples"_sid;
    for (u32 i = 0; i < packet->parameterCount; i++)
    {
        switch (packet->parameterNames[i])
        {
            case "pixelsamples"_sid:
            {
                samplesPerPixel = packet->GetInt(i);
            }
            break;
            case "randomization"_sid:
            {
                if (Compare(packet->bytes[i], "none"))
                {
                    strategy = RandomizeStrategy::FastOwen;
                }
                else if (Compare(packet->bytes[i], "permutedigits"))
                {
                    strategy = RandomizeStrategy::PermuteDigits;
                }
                else if (Compare(packet->bytes[i], "owen"))
                {
                    Assert(!isHalton);
                    strategy = RandomizeStrategy::Owen;
                }
            }
            break;
            case "seed"_sid:
            {
                seed = packet->GetInt(i);
            }
            break;
            case "jitter"_sid:
            {
                Assert(isStratified);
                jitter = packet->GetBool(i);
            }
            break;
            case "xsamples"_sid:
            {
                Assert(isStratified);
                xSamples = packet->GetInt(i);
            }
            break;
            case "ysamples"_sid:
            {
                Assert(isStratified);
                ySamples = packet->GetInt(i);
            }
            break;
            default:
            {
                ErrorExit(0, "Invalid option encountered during Sampler creation\n");
            }
        }
    }
    switch (packet->type)
    {
        case "independent"_sid:
            return PushStructConstruct(arena, IndependentSampler)(samplesPerPixel);
        case "paddedsobol"_sid:
            return PushStructConstruct(arena, PaddedSobolSampler)(samplesPerPixel, strategy,
                                                                  seed);
        case "sobol"_sid:
            return PushStructConstruct(arena, SobolSampler)(samplesPerPixel, fullResolution,
                                                            strategy, seed);
        case "stratified"_sid:
            return PushStructConstruct(arena, StratifiedSampler)(xSamples, ySamples, jitter,
                                                                 seed);
        case "halton"_sid: ErrorExit(0, "Halton sampler not implemented.");
        default:
            return PushStructConstruct(arena, ZSobolSampler)(samplesPerPixel, fullResolution,
                                                             strategy);
    }
}
#endif
} // namespace rt
