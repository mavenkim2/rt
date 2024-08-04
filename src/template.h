// template <typename... Ts>
// struct TypePack
// {
//     static constexpr size_t count = sizeof...(Ts);
// };
//
// template <typename T, typename... Ts>
// struct IndexOf
// {
//     static constexpr i32 count = 0;
//     static_assert(!std::is_same_v<T, T>, "Type not present in TypePack");
// };
//
// template <typename T, typename... Ts>
// struct IndexOf<T, TypePack<T, Ts...>>
// {
//     static constexpr i32 count = 0;
// };
//
// template <typename T, typename U, typename... Ts>
// struct IndexOf<T, TypePack<U, Ts...>>
// {
//     static constexpr size_t count = 1 + IndexOf<T, TypePack<Ts...>>::count;
// };
//
// template <typename... Ts>
// struct TaggedPointer
// {
//     using Types     = TypePack<Ts...>;
//     TaggedPointer() = default;
//     template <typename T>
//     TaggedPointer(T *ptr)
//     {
//         u64 iPtr           = reinterpret_cast<uintptr>(ptr);
//         constexpr u32 type = TypeIndex<T>();
//         bits               = iPtr | ((u64)type << tagShift);
//     }
//
//     template <typename T>
//     static constexpr u32 TypeIndex()
//     {
//         using Type = typename std::remove_cv_t<T>;
//         if constexpr (std::is_same_v<Type, std::nullptr_t>)
//             return 0;
//         else
//             return 1 + IndexOf<Type, Types>::count;
//     }
//
//     static constexpr size_t MaxTag() { return sizeof...(Ts); }
//
//     static constexpr i32 tagShift = 57;
//     static constexpr i32 tagBits  = 64 - tagShift;
//     static constexpr u64 tagMask  = ((1ull << tagBits) - 1) << tagShift;
//     static constexpr u64 ptrMask  = ~tagMask;
//     u64 bits                      = 0;
// };
//
// class Derived1
// {
// };
//
// class Derived2
// {
// };
//
// class Base : public TaggedPointer<Derived1, Derived2>
// {
//     using TaggedPointer::TaggedPointer;
//     void Print(char *text)
//     {
//     }
// };

// using fn_Print = void (*)(const Base &b, char *text);
// static fn_Print printTable[TaggedPointer<Derived1, Derived2>::MaxTag()];

// template <class T, i32 i>
// struct BaseCRTP : Base
// {
//     static const i32 id;
//     static void CallPrint(const Base &b, char *text)
//     {
//         static_cast<const T &>(b).Print(text);
//     }
//     static constexpr i32 Register()
//     {
//         return printTable[id] = CallPrint, id;
//     }
// };

// template <class T, i32 i>
// const i32 BaseCRTP::<T, i>::id = BaseCRTP::<T, i>::register();

// struct Derived1 : BaseCRTP<Derived1, IndexOf<Derived1, Base::Type>>
// {
//     void Print(char *text)
//     {
//         printf("Derived1");
//     }
// };
//
// struct Derived2 : BaseCRTP<Derived2, IndexOf<Derived2, Base::Type>>
// {
//     void Print(char *text)
//     {
//         printf("Derived2");
//     }
// };
