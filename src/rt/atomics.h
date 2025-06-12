#ifndef ATOMICS_H_
#define ATOMICS_H_

namespace rt
{
__forceinline int AtomicCompareExchange(volatile int *dst, int src, int comperand)
{
    return _InterlockedCompareExchange((volatile long *)dst, (long)src, (long)comperand);
}

__forceinline int AtomicExchange(volatile i32 *dst, i32 value)
{
    return _InterlockedExchange((volatile long *)dst, (long)value);
}
} // namespace rt
#endif
