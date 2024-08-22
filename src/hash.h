u64 MixBits(u64 v)
{
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33);
    return v;
}

LaneU32 MixBits(LaneU32 v)
{
    v ^= (v >> 31u);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27u);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33u);
    return v;
}

inline u64 MurmurHash64A(const u8 *key, size_t len, u64 seed)
{
    const u64 m = 0xc6a4a7935bd1e995ull;
    const int r = 47;

    u64 h = seed ^ (len * m);

    const u8 *end = key + 8 * (len / 8);

    while (key != end)
    {
        u64 k;
        std::memcpy(&k, key, sizeof(u64));
        key += 8;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    switch (len & 7)
    {
        case 7:
            h ^= u64(key[6]) << 48;
        case 6:
            h ^= u64(key[5]) << 40;
        case 5:
            h ^= u64(key[4]) << 32;
        case 4:
            h ^= u64(key[3]) << 24;
        case 3:
            h ^= u64(key[2]) << 16;
        case 2:
            h ^= u64(key[1]) << 8;
        case 1:
            h ^= u64(key[0]);
            h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

template <typename... Args>
inline u64 Hash(Args... args);

template <typename... Args>
inline void HashRecursiveCopy(u8 *buf, Args...);

template <>
inline void HashRecursiveCopy(u8 *buf) {}

template <typename T, typename... Args>
inline void HashRecursiveCopy(u8 *buf, T v, Args... args)
{
    std::memcpy(buf, &v, sizeof(T));
    HashRecursiveCopy(buf + sizeof(T), args...);
}

template <typename... Args>
inline u64 Hash(Args... args)
{
    constexpr size_t sz = (sizeof(Args) + ... + 0);
    constexpr size_t n  = (sz + 7) / 8;
    u64 buf[n];
    HashRecursiveCopy((u8 *)buf, args...);
    return MurmurHash64A((const u8 *)buf, sz, 0);
}

inline i32 PermutationElement(u32 i, u32 l, u32 p)
{
    u32 w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    do
    {
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}
