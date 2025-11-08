#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt 
{
#endif

static const uint RADIX = 256;
static const uint SORT_WORKGROUP_SIZE = 512;
static const uint PARTITION_DIVISION = 8;
static const uint PARTITION_SIZE = PARTITION_DIVISION * SORT_WORKGROUP_SIZE;
static const uint MAX_SUBGROUP_SIZE = 128;
#ifdef __cplusplus
}
#endif
