template <typename T, i32 logBlockSize = 10>
struct MPMC
{
    struct Cell
    {
        T data;
        std::atomic<u8> a;
    };
    enum
    {
        EMPTY   = 0,
        WRITTEN = 1,
    };
    const size_t blockSize = 1 << logBlockSize;
    const size_t blockMask = blockSize - 1;
    Cell array[blockSize];
    alignas(CACHE_LINE_SIZE * 2) std::atomic<u64> readPos;
    alignas(CACHE_LINE_SIZE * 2) std::atomic<u64> writePos;

    bool Push(T data)
    {
        for (;;)
        {
            u64 pos    = writePos.load(std::memory_order_relaxed);
            Cell &cell = array[pos & blockMask];
            u8 a       = cell.a.load(std::memory_order_acquire);
            if (a == EMPTY && writePos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
            {
                cell.data = data;
                cell.a.store.(WRITTEN, std::memory_order_release);
                return true;
            }
        }
    }
    T Pop()
    {
        for (;;)
        {
            u64 pos    = readPos.load(std::memory_order_relaxed);
            Cell &cell = array[pos & blockMask];
            u8 a       = cell.a.load(std::memory_order_acquire);
            if (a == WRITTEN && readPos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
            {
                T out = cell.data;
                cell.a.store(EMPTY, std::memory_order_release);
                return out;
            }
        }
    }
};

struct JobDeque
{
    alignas(CACHE_LINE_SIZE * 2) std::atomic<u64> top;
    alignas(CACHE_LINE_SIZE * 2) std::atomic<u64> bottom;
};
