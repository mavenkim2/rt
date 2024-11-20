#ifndef TEMPLATE_H
#define TEMPLATE_H
namespace rt
{

template <typename... Ts>
struct TypePack
{
    static constexpr size_t count = sizeof...(Ts);
};

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
            assert("Null");
            return 0;
        }
        else
            return IndexOf<Type, Types>::count;
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
    static constexpr inline u32 MaxTag()
    {
        return sizeof...(Ts);
    }
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

    Tuple(T &&value, Ts &&...values)
        : Base(std::move(values)...), value(std::move(value)) {}

    T value;
};

template <typename... Ts>
Tuple(Ts &&...) -> Tuple<std::decay_t<Ts>...>;

template <size_t I, typename T, typename... Ts>
auto &Get(Tuple<T, Ts...> &t)
{
    if constexpr (I == 0)
        return t.value;
    else
        return Get<I - 1>((Tuple<Ts...> &)t);
}

template <size_t I, typename T, typename... Ts>
const auto &Get(const Tuple<T, Ts...> &t)
{
    if constexpr (I == 0)
        return t.value;
    else
        return Get<I - 1>((const Tuple<Ts...> &)t);
}

template <typename Req, typename T, typename... Ts>
auto &Get(Tuple<T, Ts...> &t)
{
    if constexpr (std::is_same_v<Req, T>)
        return t.value;
    else
        return Get<Req>((Tuple<Ts...> &)t);
}

template <typename Req, typename T, typename... Ts>
const auto &Get(const Tuple<T, Ts...> &t)
{
    if constexpr (std::is_same_v<Req, T>)
        return t.value;
    else
        return Get<Req>((const Tuple<Ts...> &)t);
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
void ForEachType(F func, TypePack<>) {}

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

template <typename... Ts>
struct AddPointer;

template <typename T>
struct AddPointer<TypePack<T>>
{
    using type = T *;
};

template <typename T, typename... Ts>
struct AddPointer<T, TypePack<Ts...>>
{
    using type = typename Prepend<T *, typename AddPointer<TypePack<Ts...>>::type>::type;
};

template <typename T>
struct ArrayTuple;

template <typename... Ts>
struct ArrayTuple<TypePack<Ts...>>
{
    using Types = TypePack<Ts...>;
    Tuple<typename AddPointer<Types>::type> arrays;
    u32 counts[Types::count];

    // __forceinline operator const Tuple<Ts...> &() const { return arrays; }
    // __forceinline operator Tuple<Ts...> &() { return arrays; }

    template <typename T>
    __forceinline const T *Get() const
    {
        return Get<T>(arrays);
    }
    template <typename T>
    __forceinline void Set(T *array, u32 count)
    {
        Get<T>(arrays)                   = array;
        counts[IndexOf<T, Types>::count] = count;
    }
};

template <typename F, typename ArrayType, typename... Ts>
void ForEachType(ArrayType arrays, F func, TypePack<Ts...>);

template <typename F, typename ArrayType>
void ForEachType(ArrayType arrays, F func, TypePack<>) {}

template <typename F, typename ArrayType, typename T, typename... Ts>
void ForEachType(ArrayType &arrays, F func, TypePack<T, Ts...>)
{
    func.template operator()<T>(Get<T>(arrays.arrays), arrays.counts[IndexOf<T, ArrayType::Types>::count]);
    ForEachType(arrays, func, TypePack<Ts...>());
}

template <typename F, typename... Ts>
void ForEachType(ArrayTuple<Ts...> &arrays, F func)
{
    ForEachType(arrays, func, TypePack<Ts...>());
}

template <typename F, typename ArrayType, typename... Ts>
void ForEachTypeSubset(ArrayType &arrays, F func, TypePack<Ts...> types)
{
    ForEachType(arrays, func, TypePack<Ts...>());
}

// template <typename F, typename ArrayType>
// struct EvaluateArrayCallback
// {
//     F &func;
//     ArrayType *arrays;
//     template <typename Type>
//     __forceinline void operator()()
//     {
//         Type *array = Get<Type>(arrays->arrays);
//         func(array, IndexOf<Type, ArrayType::Types>::count);
//     }
// };
//
// template <typename F, typename ArrayType>
// __forceinline void ForEachType(ArrayType &arrays, const F &func)
// {
//     ForEachType(EvaluateArrayCallback{func, &arrays}, ArrayType::Types);
// }

} // namespace rt

#endif
