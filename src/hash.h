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

// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
// https://github.com/AntonJohansson/StaticMurmur/blob/master/StaticMurmur.hpp
constexpr uint32_t get_block(const char *p, unsigned i)
{
    const uint32_t block =
        static_cast<uint32_t>(p[0 + i * 4]) << 0 |
        static_cast<uint32_t>(p[1 + i * 4]) << 8 |
        static_cast<uint32_t>(p[2 + i * 4]) << 16 |
        static_cast<uint32_t>(p[3 + i * 4]) << 24;
    return block;
}

constexpr uint32_t rotl32(uint32_t x, int8_t r)
{
    return (x << r) | (x >> (32 - r));
}

constexpr uint32_t fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

constexpr u32 MurmurHash32(const char *key,
                           const u32 len,
                           const u32 seed)
{
    const u32 nblocks = len / 4;

    u32 h1 = seed;

    const u32 c1 = 0xcc9e2d51;
    const u32 c2 = 0x1b873593;

    //----------
    // body
    for (u32 i = 0; i < nblocks; ++i)
    {
        u32 k1 = get_block(key, i);

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail

    u32 k1 = 0;

    // The "tail" of key are the bytes that do not fit into any block,
    // len%4 is the size of the tail and
    // 	(tail_start + i)
    // returns the i:th tail byte.
    const unsigned tail_start = len - (len % 4);
    switch (len & 3)
    {
        case 3: k1 ^= key[tail_start + 2] << 16;
        case 2: k1 ^= key[tail_start + 1] << 8;
        case 1:
            k1 ^= key[tail_start + 0];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
    };

    //----------
    // finalization

    h1 ^= len;

    h1 = fmix32(h1);

    return h1;
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

inline u32 Hash(string arg)
{
    return MurmurHash32((const char *)arg.str, (int)arg.size, 0);
}

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
