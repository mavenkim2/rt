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

// Lock-free condition variable
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/ThreadPool/EventCount.h
struct EventCount
{
    static const u32 MAX_THREAD_COUNT = 128;
    static const u64 waiterBits       = 14;
    static const u64 stackMask        = (1ull << kWaiterBits) - 1;
    static const u64 waiterShift      = kWaiterBits;
    static const u64 waiterMask       = ((1ull << kWaiterBits) - 1) << kWaiterShift;
    static const u64 waiterInc        = 1ull << kWaiterShift;
    static const u64 signalShift      = 2 * kWaiterBits;
    static const u64 signalMask       = ((1ull << kWaiterBits) - 1) << kSignalShift;
    static const u64 signalInc        = 1ull << kSignalShift;
    static const u64 epochShift       = 3 * kWaiterBits;
    static const u64 epochBits        = 64 - kEpochShift;
    static const u64 epochMask        = ((1ull << kEpochBits) - 1) << kEpochShift;
    static const u64 epochInc         = 1ull << kEpochShift;

    struct Waiter
    {
        enum
        {
            NOT_SIGNALED,
            WAITING,
            SIGNALED,
        };
        std::mutex mutex;
        std::condition_variable cv;
        alignas(2 * CACHE_LINE_SIZE) std::atomic<u64> next{stackMask};
        u64 epoch = 0;
        u32 state = NOT_SIGNALED;
    };

    Waiter waiters[MAX_THREAD_COUNT];
    alignas(CACHE_LINE_SIZE * 2) std::atomic<u64> state{stackMask};

    void Prewait()
    {
        u64 s = state.load(std::memory_order_relaxed);
        for (;;)
        {
            u64 newState = s + waiterInc;
            if (state.compare_exchange_weak(s, newState, std::memory_order_seq_cst)) return;
        }
    }
    void CommitWait(Waiter *w)
    {
        w->state     = Waiter::NOT_SIGNALED;
        const u64 me = (w - waiters) | w->epoch;
        u64 s        = state.load(std::memory_order_seq_cst);
        for (;;)
        {
            u64 newState;
            if ((s & signalMask) != 0)
            {
                newState = s - waiterInc - signalInc;
                if (state.compare_exchange_weak(s, newState, std::memory_order_acq_rel)) return;
            }
            else
            {
                newState = ((s & waiterMask) - waiterInc) | me;
                w->next.store(s & (stackMask | epochMask), std::memory_order_relaxed);
                if (state.compare_exchange_weak(s, newState, std::memory_order_acq_rel))
                {
                    w->epoch += kEpochInc;
                    Park(w);
                    return;
                }
            }
        }
    }
    void CancelWait()
    {
        u64 s = state.load(std::memory_order_relaxed);
        for (;;)
        {
            u64 newState = s - waiterInc;
            if (((s & waiterMask) >> waiterShift) == ((s & signalMask) >> signalShift)) newState -= signalInc;
            if (state.compare_exchange_weak(s, newState, std::memory_order_acq_rel)) return;
        }
    }
    void NotifyAll()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        u64 s = state.load(std::memory_order_acquire);
        for (;;)
        {
            const u64 numWaiters = (state & waiterMask) >> waiterShift;
            const u64 numSignals = (state & signalMask) >> signalShift;
            if ((state & stackMask) == stackMask && numWaiters == numSignals) return;
            // Sets the number of signals = to the number of waiters
            u64 newState = (state & waiterMask) | (numWaiters << signalShift) | stackMask;
            if (state.compare_exchange_weak(state, newState, std::memory_order_acq_rel))
            {
                u64 stackTop = s & stackMask;
                if (stackTop == stackMask) return;
                Waiter *w = &waiters[stackTop];

                Unpark(w);
                return;
            }
        }
    };

    void NotifyOne()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        u64 s = state.load(std::memory_order_acquire);
        for (;;)
        {
            const u64 numWaiters = (state & waiterMask) >> waiterShift;
            const u64 numSignals = (state & signalMask) >> signalShift;
            if ((state & stackMask) == stackMask && numWaiters == numSignals) return;
            u64 newState;
            if (numSignals < numWaiters)
            {
                newState = s + signalInc;
                if (state.compare_exchange_weak(state, newState, std::memory_order_acq_rel)) return;
            }
            else
            {
                Waiter *w = &waiters[s & stackMask];
                u64 next  = w->next.load(std::memory_order_relaxed);
                newState  = (s & (waiterMask | signalMask)) | next;
                if (state.compare_exchange_weak(state, newState, std::memory_order_acq_rel))
                {
                    u64 stackTop = s & stackMask;
                    if (stackTop == stackMask) return;
                    Waiter *w = &waiters[stackTop];
                    // Next = stackMask means there's no next
                    w->next.store(stackMask, std::memory_order_relaxed);
                    Unpark(w);
                    return;
                }
            }
        }
    }

    void Park(Waiter *w)
    {
        std::unique_lock<std::mutex> lock(w->mutex);
        while (w->state != Waiter::SIGNALED)
        {
            w->state = Waiter::WAITING;
            w->cv.wait(lock);
        }
    }

    void Unpark(Waiter *w)
    {
        for (Waiter *next; w; w = next)
        {
            u64 wNext = w->next.load(std::memory_order_relaxed) & stackMask;
            next      = wNext == stackMask ? 0 : &waiters[wNext];

            u32 state;
            {
                std::unique_lock<std::mutex> lock(w->mutex);
                state    = w->state;
                w->state = Waiter::SIGNALED;
            }

            if (state == Waiter::WAITING) w->cv.notify_one();
        }
    }
};

struct Worker
{
    JobDeque queue;
};
} // namespace rt
