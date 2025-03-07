#ifndef TEMPLATE_H
#define TEMPLATE_H

#include "macros.h"
namespace rt
{

#define DispatchHelp(x, ...)                                                                  \
    template <typename F, DispatchTmplHelper(x, __VA_ARGS__)>                                 \
    auto Dispatch(F &&func, u32 index)                                                        \
    {                                                                                         \
        Assert(index >= 0 && index < x);                                                      \
        switch (index)                                                                        \
        {                                                                                     \
            DispatchSwitchHelper(x, __VA_ARGS__)                                              \
        }                                                                                     \
    }

#define COMMA                      ,
#define DispatchTmplHelper(x, ...) EXPAND(CONCAT(RECURSE__, x)(TMPL, __VA_ARGS__))
#define TMPL(x, ...)               typename CONCAT(T, x)

#define DispatchSwitchHelper(x, ...) CASES(x, __VA_ARGS__)
#define CASE(x)                                                                               \
    case x: return func(CONCAT(T, x)());
#define CASES(n, ...) EXPAND(CONCAT(RECURSE_, n)(CASE, __VA_ARGS__))

template <typename F, typename T0>
auto Dispatch(F &&func, u32 index)
{
    Assert(index == 0);
    return func(T0());
}

DispatchHelp(2, 0, 1);
DispatchHelp(3, 0, 1, 2);
DispatchHelp(4, 0, 1, 2, 3);
DispatchHelp(5, 0, 1, 2, 3, 4);
DispatchHelp(6, 0, 1, 2, 3, 4, 5);
DispatchHelp(7, 0, 1, 2, 3, 4, 5, 6);

template <typename... Ts>
struct TypePack
{
    static constexpr size_t count = sizeof...(Ts);
};

template <typename F, typename... Ts>
auto Dispatch(F &&closure, TypePack<Ts...>, u32 index)
{
    Dispatch<F, Ts...>(std::move(closure), index);
}

template <typename T, typename... Ts>
struct IndexOf
{
    static constexpr i32 count = 0;
    static_assert(!std::is_same_v<T, T>, "Type not present in TypePack");
};

template <typename T, typename... Ts>
struct IndexOf<T, TypePack<T, Ts...>>
{
    static constexpr i32 count = 0;
};

template <typename T, typename U, typename... Ts>
struct IndexOf<T, TypePack<U, Ts...>>
{
    static constexpr size_t count = 1 + IndexOf<T, TypePack<Ts...>>::count;
};

template <typename... Ts>
struct TaggedPointer
{
    using Types = TypePack<Ts...>;

    u64 bits                      = 0;
    static constexpr i32 tagShift = 57;
    static constexpr i32 tagBits  = 64 - tagShift;
    static constexpr u64 tagMask  = ((1ull << tagBits) - 1) << tagShift;
    static constexpr u64 ptrMask  = ~tagMask;

    TaggedPointer() = default;
    template <typename T>
    TaggedPointer(T *ptr)
    {
        u64 iPtr           = reinterpret_cast<u64>(ptr);
        constexpr u32 type = TypeIndex<T>();
        bits               = iPtr | ((u64)(type) << tagShift);
    }
    template <typename T>
    static constexpr u32 TypeIndex()
    {
        using Type = typename std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Type, std::nullptr_t>)
        {
            Assert(0);
            return 0;
        }
        else return IndexOf<Type, Types>::count;
    }
    inline u32 GetTag() const
    {
        u32 tag = (bits & tagMask) >> tagShift;
        return tag;
    }
    inline void *GetPtr() const
    {
        void *ptr = reinterpret_cast<void *>(bits & ptrMask);
        return ptr;
    }
    static constexpr inline u32 MaxTag() { return sizeof...(Ts); }
    template <typename T>
    T *Cast()
    {
        return reinterpret_cast<T *>(GetPtr());
    }
    explicit operator bool() { return (bits & ptrMask) != 0; }
};

template <typename... Ts>
struct Tuple;
template <>
struct Tuple<>
{
    template <size_t>
    using type = void;
};

template <typename T, typename... Ts>
struct Tuple<T, Ts...> : Tuple<Ts...>
{
    using Base = Tuple<Ts...>;

    Tuple()                         = default;
    Tuple(const Tuple &)            = default;
    Tuple(Tuple &&)                 = default;
    Tuple &operator=(Tuple &&)      = default;
    Tuple &operator=(const Tuple &) = default;

    Tuple(const T &value, const Ts &...values) : Base(values...), value(value) {}

    Tuple(T &&value, Ts &&...values) : Base(std::move(values)...), value(std::move(value)) {}

    T value;
};

template <typename... Ts>
Tuple(Ts &&...) -> Tuple<std::decay_t<Ts>...>;

template <size_t I, typename T, typename... Ts>
auto &Get(Tuple<T, Ts...> &t)
{
    if constexpr (I == 0) return t.value;
    else return Get<I - 1>((Tuple<Ts...> &)t);
}

template <size_t I, typename T, typename... Ts>
const auto &Get(const Tuple<T, Ts...> &t)
{
    if constexpr (I == 0) return t.value;
    else return Get<I - 1>((const Tuple<Ts...> &)t);
}

template <typename Req, typename T, typename... Ts>
auto &Get(Tuple<T, Ts...> &t)
{
    if constexpr (std::is_same_v<Req, T>) return t.value;
    else return Get<Req>((Tuple<Ts...> &)t);
}

template <typename Req, typename T, typename... Ts>
const auto &Get(const Tuple<T, Ts...> &t)
{
    if constexpr (std::is_same_v<Req, T>) return t.value;
    else return Get<Req>((const Tuple<Ts...> &)t);
}

template <typename F, typename... Ts>
void ForEachType(F func, TypePack<Ts...>);
template <typename F, typename T, typename... Ts>
void ForEachType(F func, TypePack<T, Ts...>)
{
    func.template operator()<T>();
    ForEachType(func, TypePack<Ts...>());
}

template <typename F>
void ForEachType(F func, TypePack<>)
{
}

template <typename... Ts>
struct Prepend;

template <typename T, typename... Ts>
struct Prepend<T, TypePack<Ts...>>
{
    using type = TypePack<T, Ts...>;
};

template <typename... Ts>
struct Prepend<void, TypePack<Ts...>>
{
    using type = TypePack<Ts...>;
};

template <i32 index, typename T, typename... Ts>
struct RemoveFirstN;

template <i32 index, typename T, typename... Ts>
struct RemoveFirstN<index, TypePack<T, Ts...>>
{
    using type = typename RemoveFirstN<index - 1, TypePack<Ts...>>::type;
};

template <typename... Ts>
struct ArrayTuple;

template <typename... Ts>
struct ArrayTuple<TypePack<Ts...>>
{
    using Types = TypePack<Ts...>;
    Tuple<Ts *...> arrays; // Add *;
    // Tuple<typename AddPointer<TypePack<Ts...>>::type> arrays;
    u32 counts[Types::count];

    // __forceinline operator const Tuple<Ts...> &() const { return arrays; }
    // __forceinline operator Tuple<Ts...> &() { return arrays; }

    template <typename T>
    __forceinline T *Get()
    {
        return ::Get<T *>(arrays);
    }
    template <typename T>
    __forceinline const T *Get() const
    {
        return ::Get<T *>(arrays);
    }
    template <typename T>
    __forceinline const u32 GetCount() const
    {
        return counts[IndexOf<T, Types>::count];
    }

    // __forceinline const auto &Get(u32 type, u32 index) const
    // {
    //     return ::Get<ArrayTuple<TypePack<Ts...>>, Ts...>(arrays, type, index);
    // }

    // __forceinline const void *GetPtr(u32 index) const
    // {
    //     auto getPtr = [&](auto array) { return (void *)&array[index]; };
    //     return Dispatch<decltype(getPtr), Ts...>(getPtr, index);
    // }

    template <typename F>
    __forceinline auto Dispatch(F &func, u32 index) const
    {
        return Dispatch<F, Ts...>(func, index);
    }
    __forceinline u32 ConvertIndexToType(u32 index, u32 *outIndex) const
    {
        for (u32 i = 0; i < Types::count; i++)
        {
            if (index < counts[i])
            {
                *outIndex = index;
                return i;
            }
            index -= counts[i];
        }
        return 0xffffffff;
    }
    template <typename T>
    __forceinline void Set(T *array, u32 count)
    {
        ::Get<T *>(arrays)               = array;
        counts[IndexOf<T, Types>::count] = count;
    }
};

template <typename F, typename ArrayType, typename... Ts>
void ForEachType(ArrayType arrays, F func, TypePack<Ts...>);

template <typename F, typename ArrayType>
void ForEachType(ArrayType arrays, F func, TypePack<>)
{
}

template <typename F, typename ArrayType, typename T, typename... Ts>
void ForEachType(ArrayType &arrays, F func, TypePack<T, Ts...>)
{
    func.template operator()<T>(Get<T *>(arrays.arrays),
                                arrays.counts[IndexOf<T, typename ArrayType::Types>::count]);
    ForEachType(arrays, func, TypePack<Ts...>());
}

template <typename F, typename... Ts>
void ForEachType(ArrayTuple<TypePack<Ts...>> &arrays, F func)
{
    ForEachType(arrays, func, TypePack<Ts...>());
}

template <typename F, typename ArrayType, typename... Ts>
void ForEachTypeSubset(ArrayType &arrays, F func, TypePack<Ts...> types)
{
    ForEachType(arrays, func, TypePack<Ts...>());
}

// template <template <typename> class M, typename... Ts>
// struct MapType;

template <template <typename> class F, typename... Ts>
struct MapType
{
    using type = TypePack<typename F<Ts>::type...>;
};

// template <template <typename> class M, typename T>
// struct MapType<M, TypePack<T>>
// {
//     using type = TypePack<M<T>>;
// };
//
// template <template <typename> class M, typename T, typename... Ts>
// struct MapType<M, TypePack<T, Ts...>>
// {
//     using type = typename Prepend<M<T>, typename MapType<M, TypePack<Ts...>>::type>::type;
// };

} // namespace rt

#endif
