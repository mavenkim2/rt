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
