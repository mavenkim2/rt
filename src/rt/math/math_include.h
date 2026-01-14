#ifndef MATH_INCLUDE_H
#define MATH_INCLUDE_H

#ifndef __CUDACC__
#include "basemath.h"
#include "dual.h"
#include "simd_include.h"
#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include "bounds.h"
#include "matx.h"
#include "math.h"
#else
// TODO: unify these code paths
#include "../gpu/helper_math.h"
#endif

#endif
