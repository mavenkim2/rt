#ifndef PAYLOAD_HLSLI_
#define PAYLOAD_HLSLI_

#if 1
struct [raypayload] RayPayload 
{
    float3x4 objectToWorld : write(closesthit) : read(caller);
    float3 objectRayDir : write(closesthit) : read(caller);
    float2 bary : write(closesthit) : read(caller);
    float rayT : write(closesthit) : read(caller);
    uint hitKind : write(closesthit) : read(caller);

    uint primitiveIndex : write(closesthit) : read(caller);
    uint instanceID : write(closesthit) : read(caller);
    bool miss : write(closesthit, miss) : read(caller);
};
#endif

#endif
