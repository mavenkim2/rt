uint WaveGetActiveLaneIndexLast()
{
    uint2 activeMask = WaveActiveBallot(true).xy;
    return firstbithigh(activeMask.y ? activeMask.y : activeMask.x) + (activeMask.y ? 32 : 0);
}

#define WaveReadLaneLast(x) WaveReadLaneAt(x, WaveGetActiveLaneIndexLast())

#define WaveInterlockedAddScalarTest(dest, test, value, outputIndex) \
{ \
    uint __numToAdd__ = WaveActiveCountBits(test) * value; \
    outputIndex = 0; \
    if (WaveIsFirstLane() && __numToAdd__ > 0) \
    { \
        InterlockedAdd(dest, __numToAdd__, outputIndex); \
    } \
    outputIndex = WaveReadLaneFirst(outputIndex) + WavePrefixCountBits(test) * value; \
}

#define WaveInterlockedAdd(dest, value, outputIndex) \
{ \
    uint __localIndex__ = WavePrefixSum(value); \
    uint __numToAdd__ = WaveReadLaneLast(__localIndex__ + value); \
    outputIndex = 0; \
    if (WaveIsFirstLane()) \
    { \
        InterlockedAdd(dest, __numToAdd__, outputIndex); \
    } \
    outputIndex = WaveReadLaneFirst(outputIndex) + __localIndex__; \
}
