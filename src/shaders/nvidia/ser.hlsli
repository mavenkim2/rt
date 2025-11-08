#ifndef SER_HLSLI_
#define SER_HLSLI_

#include "../payload.hlsli"

#define HitObjectAttributeNV 5385
#define OpHitObjectRecordHitMotionNV 5249
#define OpHitObjectRecordHitWithIndexMotionNV 5250
#define OpHitObjectRecordMissMotionNV 5251
#define OpHitObjectGetWorldToObjectNV 5252
#define OpHitObjectGetObjectToWorldNV 5253
#define OpHitObjectGetObjectRayDirectionNV 5254
#define OpHitObjectGetObjectRayOriginNV 5255
#define OpHitObjectTraceRayMotionNV 5256
#define OpHitObjectGetShaderRecordBufferHandleNV 5257
#define OpHitObjectGetShaderBindingTableRecordIndexNV 5258
#define OpHitObjectRecordEmptyNV 5259
#define OpHitObjectTraceRayNV 5260
#define OpHitObjectRecordHitNV 5261
#define OpHitObjectRecordHitWithIndexNV 5262
#define OpHitObjectRecordMissNV 5263
#define OpHitObjectExecuteShaderNV 5264
#define OpHitObjectGetCurrentTimeNV 5265
#define OpHitObjectGetAttributesNV 5266
#define OpHitObjectGetHitKindNV 5267
#define OpHitObjectGetPrimitiveIndexNV 5268
#define OpHitObjectGetGeometryIndexNV 5269
#define OpHitObjectGetInstanceIdNV 5270
#define OpHitObjectGetInstanceCustomIndexNV 5271
#define OpHitObjectGetWorldRayDirectionNV 5272
#define OpHitObjectGetWorldRayOriginNV 5273
#define OpHitObjectGetRayTMaxNV 5274
#define OpHitObjectGetRayTMinNV 5275
#define OpHitObjectIsEmptyNV 5276
#define OpHitObjectIsHitNV 5277
#define OpHitObjectIsMissNV 5278
#define OpReorderThreadWithHitObjectNV 5279
#define OpReorderThreadWithHintNV 5280
#define OpTypeHitObjectNV 5281

#define RayPayloadKHR 5338

[[vk::ext_storage_class(RayPayloadKHR)]] static RayPayload payload;

[[vk::ext_instruction(OpReorderThreadWithHintNV)]]
void NvReorderThread(int hint, int bits);

[[vk::ext_type_def(HitObjectAttributeNV, OpTypeHitObjectNV)]]
void CreateHitObjectNV();
#define HitObjectNV vk::ext_type<HitObjectAttributeNV>

[[vk::ext_instruction(OpReorderThreadWithHitObjectNV)]]
void ReorderThreadWithHitNV([[vk::ext_reference]] HitObjectNV hitObject, uint hint, uint bits);

[[vk::ext_instruction(OpHitObjectIsMissNV)]]
bool IsMissNV([[vk::ext_reference]] HitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectIsHitNV)]]
bool IsHitNV([[vk::ext_reference]] HitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectTraceRayNV)]]
void TraceRayHitObjectNV(
    [[vk::ext_reference]] HitObjectNV hitObject,
    RaytracingAccelerationStructure as,
    uint RayFlags,
    uint CullMask,
    uint SBTOffset,
    uint SBTStride,
    uint MissIndex,
    float3 RayOrigin,
    float RayTmin,
    float3 RayDirection,
    float RayTMax,
    [[vk::ext_reference]] [[vk::ext_storage_class(RayPayloadKHR)]] RayPayload payload
  );

[[vk::ext_instruction(OpHitObjectExecuteShaderNV)]]
void InvokeHitObjectNV(
    [[vk::ext_reference]] HitObjectNV hitObject, 
    [[vk::ext_reference]] [[vk::ext_storage_class(RayPayloadKHR)]] RayPayload payload
);

[[vk::ext_instruction(OpHitObjectGetPrimitiveIndexNV)]]
uint GetPrimitiveIndexNV([[vk::ext_reference]] HitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetInstanceIdNV)]]
uint GetInstanceIndexNV([[vk::ext_reference]] HitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetInstanceCustomIndexNV)]]
uint GetInstanceIDNV([[vk::ext_reference]] HitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetObjectRayDirectionNV)]]
float3 GetObjectRayDirectionNV([[vk::ext_reference]] HitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetObjectToWorldNV)]]
float4x3 GetObjectToWorldNV([[vk::ext_reference]] HitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetHitKindNV)]]
uint GetHitKindNV([[vk::ext_reference]] HitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectRecordHitNV)]]
void SERMakeHit(
    [[vk::ext_reference]] HitObjectNV hitObject, 
    RaytracingAccelerationStructure as,
    int instanceIndex,
    int geometryIndex, 
    int primitiveIndex,
    uint hitKind,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    float3 rayOrigin,
    float tMin,
    float3 rayDir,
    float tMax,
    [[vk::ext_reference]] [[vk::ext_storage_class(HitObjectAttributeNV)]] BuiltInTriangleIntersectionAttributes attr
);


#endif
