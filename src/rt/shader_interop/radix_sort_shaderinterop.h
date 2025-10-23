#ifndef RADIX_SORT_SHADERINTEROP_H_
#define RADIX_SORT_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define SORT_WORKGROUP_SIZE 256

struct SortKey
{
    uint key;
    uint index;
};

struct RadixSortPushConstant
{
    uint g_shift;
    uint queueIndex;
};

#ifdef __cplusplus
}
#endif

#endif
