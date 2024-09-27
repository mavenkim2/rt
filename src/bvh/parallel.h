#include <atomic>
namespace rt
{
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

// https://fzn.fr/readings/ppopp13.pdf
// https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf
template <typename T>
struct JobDeque
{
    static const i32 logBlockSize = 10;
    static const i32 size         = 1 << 10;
    static const i32 mask         = size - 1;
    alignas(CACHE_LINE_SIZE * 2) std::atomic<u64> top;
    alignas(CACHE_LINE_SIZE * 2) std::atomic<u64> bottom;
    alignas(CACHE_LINE_SIZE * 2) std::atomic<T> buffer[size];

    bool Push(T item)
    {
        u64 b = bottom.load(std::memory_order_relaxed);
        u64 t = top.load(std::memory_order_acquire);
        if (b >= t + size - 1) return false;
        buffer[b & mask].store(item, std::memory_order_relaxed);
        bottom.store(b + 1, std::memory_order_release);
        return true;
    }
    bool Pop(T &out)
    {
        u64 b = bottom.load(std::memory_order_relaxed) - 1;
        bottom.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        u64 t = top.load(std::memory_order_relaxed);

        bool result = 1;
        if (t <= b)
        {
            out = buffer[b & mask].load(std::memory_order_relaxed);
            if (t == b)
            {
                if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
                {
                    result = false;
                }
                bottom.store(b + 1, std::memory_order_relaxed);
            }
        }
        else
        {
            result = false;
            bottom.store(b + 1, std::memory_order_relaxed);
        }
        return result;
    }
    bool Steal(T &out)
    {
        u64 t = top.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        u64 b       = bottom.load(std::memory_order_acquire);
        bool result = false;
        if (t < b)
        {
            result = true;
            out    = buffer[t & mask].load(std::memory_order_relaxed);
            if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
            {
                result = false;
            }
        }
        return result;
    }
};

struct Worker
{
    JobDeque queue;
};
} // namespace rt
