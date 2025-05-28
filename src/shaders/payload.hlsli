#ifndef PAYLOAD_HLSLI_
#define PAYLOAD_HLSLI_

struct [raypayload] RayPayload 
{
    float3x4 objectToWorld : write(closesthit) : read(caller);
    float3 objectRayDir : write(closesthit) : read(caller);
    float2 bary : write(closesthit) : read(caller);
    float rayT : write(closesthit) : read(caller);
    uint hitKind : write(closesthit) : read(caller);
};

#endif
