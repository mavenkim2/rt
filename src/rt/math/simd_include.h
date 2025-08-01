#ifndef SIMD_INCLUDE_H
#define SIMD_INCLUDE_H

#include "../base.h"
#include "simd_base.h"
#include "lane4u32.h"
#include "lane4f32.h"

#if defined(__AVX__)
#include "lane8u32.h"
#include "lane8f32.h"
#endif

#endif
