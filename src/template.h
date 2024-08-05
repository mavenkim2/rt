#ifndef TEMPLATE_H
#define TEMPLATE_H

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
        // if constexpr (std::is_same_v<Type, std::nullptr_t>)
        //     return 0;
        // else
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
};

#endif
