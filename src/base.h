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

#define IsPow2(x) (((x) & ((x)-1)) == 0)

#define ENUM_CLASS_FLAGS(Enum)                                                                                                                            \
    inline Enum &operator|=(Enum &lhs, Enum rhs)                                                                                                          \
    {                                                                                                                                                     \
        lhs = (Enum)((std::underlying_type<Enum>::type)lhs | (std::underlying_type<Enum>::type)rhs);                                                      \
        return lhs;                                                                                                                                       \
    }                                                                                                                                                     \
    inline Enum &operator&=(Enum &lhs, Enum rhs)                                                                                                          \
    {                                                                                                                                                     \
        lhs = (Enum)((std::underlying_type<Enum>::type)lhs & (std::underlying_type<Enum>::type)rhs);                                                      \
        return lhs;                                                                                                                                       \
    }                                                                                                                                                     \
    inline Enum &operator^=(Enum &lhs, Enum rhs)                                                                                                          \
    {                                                                                                                                                     \
        lhs = (Enum)((std::underlying_type<Enum>::type)lhs ^ (std::underlying_type<Enum>::type)rhs);                                                      \
        return lhs;                                                                                                                                       \
    }                                                                                                                                                     \
    inline constexpr Enum operator|(Enum lhs, Enum rhs) { return (Enum)((std::underlying_type<Enum>::type)lhs | (std::underlying_type<Enum>::type)rhs); } \
    inline constexpr Enum operator&(Enum lhs, Enum rhs) { return (Enum)((std::underlying_type<Enum>::type)lhs & (std::underlying_type<Enum>::type)rhs); } \
    inline constexpr Enum operator^(Enum lhs, Enum rhs) { return (Enum)((std::underlying_type<Enum>::type)lhs ^ (std::underlying_type<Enum>::type)rhs); } \
    inline constexpr bool operator!(Enum e) { return !(std::underlying_type<Enum>::type)e; }                                                              \
    inline constexpr Enum operator~(Enum e) { return (Enum) ~(std::underlying_type<Enum>::type)e; }

template <typename Enum>
constexpr bool EnumHasAllFlags(Enum Flags, Enum Contains)
{
    using UnderlyingType = std::underlying_type<Enum>::type;
    return ((UnderlyingType)Flags & (UnderlyingType)Contains) == (UnderlyingType)Contains;
}

template <typename Enum>
constexpr bool EnumHasAnyFlags(Enum Flags, Enum Contains)
{
    using UnderlyingType = std::underlying_type<Enum>::type;
    return ((UnderlyingType)Flags & (UnderlyingType)Contains) != 0;
}

template <typename Enum>
void EnumAddFlags(Enum &Flags, Enum FlagsToAdd)
{
    using UnderlyingType = std::underlying_type<Enum>::type;
    Flags                = (Enum)((UnderlyingType)Flags | (UnderlyingType)FlagsToAdd);
}

template <typename Enum>
void EnumRemoveFlags(Enum &Flags, Enum FlagsToRemove)
{
    using UnderlyingType = std::underlying_type<Enum>::type;
    Flags                = (Enum)((UnderlyingType)Flags & ~(UnderlyingType)FlagsToRemove);
}
