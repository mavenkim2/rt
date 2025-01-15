#ifndef DUAL_H
#define DUAL_H
namespace rt
{
////////////////////////////////////////////////////
// Dual numbers

template <typename T, i32 N>
struct Dual
{
    T real;   // real
    T eps[N]; // infinitesimal
    Dual() {}
    Dual(const T &real) : real(real), eps{} {}
    template <typename... Args>
    Dual(const T &real, Args... args) : real(real)
    {
        Append<0>(args...);
    }

    template <i32 X>
    void Append(const T &a)
    {
        eps[X] = a;
    }

    template <i32 X, typename... Args>
    void Append(const T &a, Args... args)
    {
        eps[X] = a;
        Append<X + 1>(args...);
    }
};

// Add
template <typename T, i32 N>
__forceinline Dual<T, N> operator+(const Dual<T, N> &a, const Dual<T, N> &b)
{
    Dual<T, N> result;
    result.real = a.real + b.real;
    for (u32 i = 0; i < N; i++)
    {
        result.eps[i] = a.eps[i] + b.eps[i];
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> operator+(const Dual<T, N> &a, const T &b)
{
    Dual<T, N> result;
    result.real = a.real + b;
    for (u32 i = 0; i < N; i++)
    {
        result.eps[i] = a.eps[i];
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> operator+(const T &a, const Dual<T, N> &b)
{
    return b + a;
}

template <typename T, i32 N>
__forceinline Dual<T, N> &operator+=(Dual<T, N> &a, const Dual<T, N> &b)
{
    a = a + b;
    return a;
}

template <typename T, i32 N>
__forceinline Dual<T, N> &operator+=(Dual<T, N> &a, const T &b)
{
    a = a + b;
    return a;
}

// Subtract
template <typename T, i32 N>
__forceinline Dual<T, N> operator-(const Dual<T, N> &a, const Dual<T, N> &b)
{
    Dual<T, N> result;
    result.real = a.real - b.real;
    for (u32 i = 0; i < N; i++)
    {
        result.eps[i] = a.eps[i] - b.eps[i];
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> operator-(const Dual<T, N> &a, const T &b)
{
    Dual<T, N> result;
    result.real = a.real - b;
    for (u32 i = 0; i < N; i++)
    {
        result.eps[i] = a.eps[i];
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> operator-(const T &a, const Dual<T, N> &b)
{
    return b - a;
}

template <typename T, i32 N>
__forceinline Dual<T, N> &operator-=(Dual<T, N> &a, const Dual<T, N> &b)
{
    a = a - b;
    return a;
}

template <typename T, i32 N>
__forceinline Dual<T, N> &operator-=(Dual<T, N> &a, const T &b)
{
    a = a - b;
    return a;
}

// Mult
template <typename T, i32 N>
__forceinline Dual<T, N> operator*(const Dual<T, N> &a, const Dual<T, N> &b)
{
    Dual<T, N> result;
    result.real = a.real * b.real;
    for (u32 index = 0; index < N; index++)
    {
        result.eps[index] = a.real * b.eps[index] + a.eps[index] * b.real;
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> operator*(const Dual<T, N> &a, const T &b)
{
    Dual<T, N> result;
    result.real = a.real * b;
    for (u32 index = 0; index < N; index++)
    {
        result.eps[index] = a.eps[index] * b;
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> operator*(const T &a, const Dual<T, N> &b)
{
    return b * a;
}

template <typename T, i32 N>
__forceinline Dual<T, N> &operator*=(Dual<T, N> &a, const Dual<T, N> &b)
{
    a = a * b;
    return a;
}

template <typename T, i32 N>
__forceinline Dual<T, N> &operator*=(Dual<T, N> &a, const T &b)
{
    a = a * b;
    return a;
}

// Divide
template <typename T, i32 N>
__forceinline Dual<T, N> operator/(const Dual<T, N> &a, const Dual<T, N> &b)
{
    Dual<T, N> result;
    T bInvReal  = T(1) / b.real;
    T aOverB    = a.real / b.real;
    result.real = aOverB;
    for (u32 index = 0; index < N; index++)
    {
        result.eps[index] = bInvReal * (a.eps[index] - aOverB * b.eps[index]);
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> operator/(const Dual<T, N> &a, const T &b)
{
    Dual<T, N> result;
    T bInvReal  = T(1) / b;
    T aOverB    = a.real / b;
    result.real = aOverB;
    for (u32 index = 0; index < N; index++)
    {
        result.eps[index] = bInvReal * a.eps[index];
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> operator/(const T &a, const Dual<T, N> &b)
{
    Dual<T, N> result;
    T bInvReal  = T(1) / b.real;
    T aOverB    = a / b.real;
    result.real = aOverB;
    for (u32 index = 0; index < N; index++)
    {
        result.eps[index] = bInvReal * (-aOverB * b.eps[index]);
    }
    return result;
}

template <typename T, i32 N>
__forceinline Dual<T, N> &operator/=(Dual<T, N> &a, const Dual<T, N> &b)
{
    a = a / b;
    return a;
}

template <typename T, i32 N>
__forceinline Dual<T, N> &operator/=(Dual<T, N> &a, const T &b)
{
    a = a / b;
    return a;
}

template <typename T, i32 N>
__forceinline Dual<T, N> DualFunc(const Dual<T, N> &dual, const T &f, const T &dfdu)
{
    Dual<T, N> result;
    result.real = f;
    for (u32 index = 0; index < N; index++)
    {
        result.eps[index] = dual.eps[index] * dfdu;
    }
}

template <typename T, i32 N>
struct HyperDual
{
    T real; // real
    T eps[N * 2];
    T dd[N];

    HyperDual() {}
    HyperDual(const T &real) : real(real), eps{}, dd{} {}
    template <typename... Args>
    HyperDual(const T &real, Args... args) : real(real)
    {
        Assert(sizeof...(args) == N * 2);
        Append<0>(args...);
    }

    template <i32 X>
    void Append(const T &a)
    {
        eps[X] = a;
    }

    template <i32 X, typename... Args>
    void Append(const T &a, Args... args)
    {
        eps[X] = a;
        Append<X + 1>(args...);
    }
};

// Add
template <typename T, i32 N>
__forceinline HyperDual<T, N> operator+(const HyperDual<T, N> &a, const HyperDual<T, N> &b)
{
    HyperDual<T, N> result;
    result.real = a.real + b.real;
    for (u32 i = 0; i < N * 2; i++)
    {
        result.eps[i] = a.eps[i] + b.eps[i];
    }
    for (u32 i = 0; i < N; i++)
    {
        result.dd[i] = a.dd[i] + b.dd[i];
    }
    return result;
}

template <typename T, i32 N>
__forceinline HyperDual<T, N> operator+(const T &a, const HyperDual<T, N> &b)
{
    return b + a;
}

template <typename T, i32 N>
__forceinline HyperDual<T, N> &operator+=(HyperDual<T, N> &a, const HyperDual<T, N> &b)
{
    a = a + b;
    return a;
}

template <typename T, i32 N>
__forceinline HyperDual<T, N> &operator+=(HyperDual<T, N> &a, const T &b)
{
    a = a + b;
    return a;
}

// Mult
template <typename T, i32 N>
__forceinline HyperDual<T, N> operator*(const HyperDual<T, N> &a, const HyperDual<T, N> &b)
{
    HyperDual<T, N> result;
    result.real = a.real * b.real;
    for (u32 i = 0; i < N * 2; i++)
    {
        result.eps[i] = a.real * b.eps[i] + a.eps[i] * b.real;
    }
    for (u32 i = 0; i < N; i++)
    {
        result.dd[i] = a.real * b.dd[i] + a.eps[2 * i] * b.eps[2 * i + 1] +
                       a.eps[2 * i + 1] * b.eps[2 * i] + b.real * a.dd[i];
    }
    return result;
}

#if 0 
template <typename T, i32 N>
__forceinline HyperDual<T, N> operator*(const HyperDual<T, N> &a, const HyperDual<T, N> &b)
{
    return HyperDual<T, N>(a.real - b, a.ep0, a.ep1, a.ep0ep1);

    HyperDual<T, N> result;
    result.real = a.real * b.real;
    for (u32 index = 0; index < N; index++)
    {
        result.eps[index] = a.real * b.eps[index] + a.eps[index] * b.real;
    }
    return result;
}

template <typename T, i32 N>
__forceinline HyperDual<T, N> operator*(const HyperDual<T, N> &a, const T &b)
{
    HyperDual<T, N> result;
    result.real = a.real * b;
    for (u32 index = 0; index < N; index++)
    {
        result.eps[index] = a.eps[index] * b;
    }
    return result;
}

template <typename T, i32 N>
__forceinline HyperDual<T, N> operator*(const T &a, const HyperDual<T, N> &b)
{
    return b * a;
}

template <typename T, i32 N>
__forceinline HyperDual<T, N> &operator*=(HyperDual<T, N> &a, const HyperDual<T, N> &b)
{
    a = a * b;
    return a;
}

template <typename T, i32 N>
__forceinline HyperDual<T, N> &operator*=(HyperDual<T, N> &a, const T &b)
{
    a = a * b;
    return a;
}
#endif

} // namespace rt
#endif
