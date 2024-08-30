#include <emmintrin.h>
#include <stdint.h>
#include <windows.h>
#include <iostream>
#include <vector>
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

#ifdef _MSC_VER
#define Trap() __debugbreak()
#else
#error compiler not implemented
#endif

void Print(char *fmt, ...);
void Print(char *fmt, va_list args);

#define Glue(a, b) a##b
#if DEBUG
// #define Assert(expression) (!(expression) ? (*(volatile int *)0 = 0, 0) : 0)
#define Assert(expression)                                                                  \
    if (expression)                                                                         \
    {                                                                                       \
    }                                                                                       \
    else                                                                                    \
    {                                                                                       \
        Print("Expression: %s\nFile: %s\nLine Num: %u\n", #expression, __FILE__, __LINE__); \
        Trap();                                                                             \
    }
#define Error(expression, str, ...)                                                         \
    if (expression)                                                                         \
    {                                                                                       \
    }                                                                                       \
    else                                                                                    \
    {                                                                                       \
        Print(str, __VA_ARGS__);                                                            \
        Print("Expression: %s\nFile: %s\nLine Num: %u\n", #expression, __FILE__, __LINE__); \
        Trap();                                                                             \
    }
#define StaticAssert(expr, ID) static u8 Glue(ID, __LINE__)[(expr) ? 1 : -1]
#else
#define Assert(expression)
#define Error(expression, str, ...)
#define StaticAssert(expr, ID)
#endif

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

template <typename T>
void Swap(T a, T b)
{
    T temp = std::move(a);
    a      = std::move(b);
    b      = std::move(temp);
}

////////////////////////////////////////////////////////////////////////////////////////
// Linked list helpers
//
#define CheckNull(p) ((p) == 0)
#define SetNull(p)   ((p) = 0)
#define QueuePush_NZ(f, l, n, next, zchk, zset) \
    (zchk(f) ? (((f) = (l) = (n)), zset((n)->next)) : ((l)->next = (n), (l) = (n), zset((n)->next)))
#define SLLStackPop_N(f, next)     ((f) = (f)->next)
#define SLLStackPush_N(f, n, next) ((n)->next = (f), (f) = (n))

#define DLLInsert_NPZ(f, l, p, n, next, prev, zchk, zset)                                                      \
    (zchk(f)   ? (((f) = (l) = (n)), zset((n)->next), zset((n)->prev))                                         \
     : zchk(p) ? (zset((n)->prev), (n)->next = (f), (zchk(f) ? (0) : ((f)->prev = (n))), (f) = (n))            \
               : ((zchk((p)->next) ? (0) : (((p)->next->prev) = (n))), (n)->next = (p)->next, (n)->prev = (p), \
                  (p)->next = (n), ((p) == (l) ? (l) = (n) : (0))))
#define DLLPushBack_NPZ(f, l, n, next, prev, zchk, zset) DLLInsert_NPZ(f, l, l, n, next, prev, zchk, zset)
#define DLLRemove_NPZ(f, l, n, next, prev, zchk, zset)                           \
    (((f) == (n))   ? ((f) = (f)->next, (zchk(f) ? (zset(l)) : zset((f)->prev))) \
     : ((l) == (n)) ? ((l) = (l)->prev, (zchk(l) ? (zset(f)) : zset((l)->next))) \
                    : ((zchk((n)->next) ? (0) : ((n)->next->prev = (n)->prev)),  \
                       (zchk((n)->prev) ? (0) : ((n)->prev->next = (n)->next))))

#define QueuePush(f, l, n) QueuePush_NZ(f, l, n, next, CheckNull, SetNull)
#define StackPop(f)        SLLStackPop_N(f, next)
#define StackPush(f, n)    SLLStackPush_N(f, n, next)

#define DLLPushBack(f, l, n)  DLLPushBack_NPZ(f, l, n, next, prev, CheckNull, SetNull)
#define DLLPushFront(f, l, n) DLLPushBack_NPZ(l, f, n, prev, next, CheckNull, SetNull)
#define DLLInsert(f, l, p, n) DLLInsert_NPZ(f, l, p, n, next, prev, CheckNull, SetNull)
#define DLLRemove(f, l, n)    DLLRemove_NPZ(f, l, n, next, prev, CheckNull, SetNull)

//////////////////////////////
// Mutexes
//

struct TicketMutex
{
    std::atomic<u64> ticket;
    std::atomic<u64> serving;
    void Init()
    {
        ticket.store(0);
        serving.store(0);
    }
    // u64 volatile ticket;
    // u64 volatile serving;
};

inline void BeginTicketMutex(TicketMutex *mutex)
{
    u64 ticket = mutex->ticket.fetch_add(1);
    while (ticket != mutex->serving)
    {
        _mm_pause();
    }
}

inline void EndTicketMutex(TicketMutex *mutex)
{
    mutex->serving.fetch_add(1);
}

struct Mutex
{
    std::atomic<u32> count;
};

inline void BeginMutex(Mutex *mutex)
{
    u32 expected = 0;
    while (!mutex->count.compare_exchange_weak(expected, 1))
    {
        expected = 0;
        _mm_pause();
    }
}

// TODO: use memory barrier instead, _mm_sfence()?
inline void EndMutex(Mutex *mutex)
{
    mutex->count.store(0);
}

inline void BeginRMutex(Mutex *mutex)
{
    for (;;)
    {
        u32 oldValue = (mutex->count.load() & 0x7fffffff);
        u32 newValue = oldValue + 1;
        if (mutex->count.compare_exchange_weak(oldValue, newValue))
        {
            break;
        }
        _mm_pause();
    }
}

inline void EndRMutex(Mutex *mutex)
{
    mutex->count.fetch_sub(1);
    Assert(mutex->count >= 0);
}

inline void BeginWMutex(Mutex *mutex)
{
    u32 expected = 0;
    while (!mutex->count.compare_exchange_weak(expected, 0x80000000))
    {
        expected = 0;
        _mm_pause();
    }
}

inline void EndWMutex(Mutex *mutex)
{
    u32 expected = 0x80000000;
    b32 result   = mutex->count.compare_exchange_strong(expected, 0);
    Assert(result);
}
