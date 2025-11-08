#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define KD_TREE_WORKGROUP_SIZE 32

#define KD_TREE_REDUCTION_BITS 9
#define KD_TREE_REDUCTION_SIZE (1u << KD_TREE_REDUCTION_BITS)

#define KD_TREE_INDIRECT_X 0
#define KD_TREE_INDIRECT_Y 1
#define KD_TREE_INDIRECT_Z 2

#define SORT_KEYS_INDIRECT_X 3
#define SORT_KEYS_INDIRECT_Y 4
#define SORT_KEYS_INDIRECT_Z 5

#define KD_TREE_INDIRECT_SIZE 6

#ifdef __cplusplus
}
#endif
