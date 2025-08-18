#ifndef CLAS_HLSLI
#define CLAS_HLSLI

#include "ser.hlsli"

#define OpRayQueryGetIntersectionClusterIdNV 5345
#define OpHitObjectGetClusterIdNV 5346

#define RayQueryCandidateIntersectionKHR 0
#define RayQueryCommittedIntersectionKHR 1

[[vk::ext_instruction(OpRayQueryGetIntersectionClusterIdNV)]]
uint GetClusterIDNV([[vk::ext_reference]] RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query, uint intersection);

[[vk::ext_instruction(OpRayQueryGetIntersectionClusterIdNV)]]
uint GetClusterIDNV([[vk::ext_reference]] RayQuery<RAY_FLAG_NONE | RAY_FLAG_FORCE_OPAQUE> query, uint intersection);

[[vk::ext_instruction(OpHitObjectGetClusterIdNV)]]
uint GetClusterIDNV([[vk::ext_reference]] HitObjectNV hitObject);

#endif
