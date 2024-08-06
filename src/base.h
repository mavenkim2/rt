#include <emmintrin.h>
#include <stdint.h>
#include <windows.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <intrin.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef float f32;
typedef double f64;
typedef uintptr_t uintptr;

typedef i8 b8;
typedef i16 b16;
typedef i32 b32;

#define SSE42

#define ArrayLength(array) sizeof(array) / sizeof(array[0])
#define kilobytes(value)   ((value)*1024LL)
#define megabytes(value)   (kilobytes(value) * 1024LL)
#define gigabytes(value)   (megabytes(value) * 1024LL)
#define terabytes(value)   (gigabytes(value) * 1024LL)

#define MemoryCopy            memcpy
#define MemorySet             memset
#define MemoryZero(ptr, size) MemorySet((ptr), 0, (size))
#define AlignPow2(x, b)       (((x) + (b)-1) & (~((b)-1)))

#define IsPow2(x) (((x) & ((x) - 1)) == 0)
