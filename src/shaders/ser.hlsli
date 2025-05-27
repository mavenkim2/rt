#ifndef SER_HLSLI_
#define SER_HLSLI_

#define ShaderInvocationReorderNV 5383
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

[[vk::ext_capability(ShaderInvocationReorderNV)]]
[[vk::ext_extension("SPV_NV_shader_invocation_reorder")]]

#define RayPayloadKHR 5338
//[[vk::ext_storage_class(RayPayloadKHR)]] static RayPayload payload;

[[vk::ext_instruction(OpReorderThreadWithHintNV)]]
void NvReorderThread(int hint, int bits);

[[vk::ext_type_def(HitObjectAttributeNV, OpTypeHitObjectNV)]]
void NvCreateHitObject();
#define NvHitObject vk::ext_type<HitObjectAttributeNV>

[[vk::ext_instruction(OpReorderThreadWithHitObjectNV)]]
void NvReorderThreadWithHit([[vk::ext_reference]] NvHitObject hitObject, uint hint, uint bits);

#if 0
[[vk::ext_instruction(OpHitObjectTraceRayNV)]]
void NvTraceRayHitObject(
    [[vk::ext_reference]] NvHitObject hitObject,
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
#endif

#endif
