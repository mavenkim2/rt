#include <atomic>
#include <mutex>
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
    static const size_t blockSize = 1 << logBlockSize;
    static const size_t blockMask = blockSize - 1;
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
    bool TryPop(T &out)
    {
        u64 pos    = readPos.load(std::memory_order_relaxed);
        Cell &cell = array[pos & blockMask];
        u8 a       = cell.a.load(std::memory_order_acquire);
        if (a == WRITTEN && readPos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
        {
            out = cell.data;
            cell.a.store(EMPTY, std::memory_order_release);
            return true;
        }
        return false;
    }
};

// https://fzn.fr/readings/ppopp13.pdf
// https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf
template <typename T>
struct JobDeque
{
    static const i32 logBlockSize = 10;
    static const i32 size         = 1 << logBlockSize;
    static const i32 mask         = size - 1;
    alignas(CACHE_LINE_SIZE * 2) std::atomic<i64> top{0};
    alignas(CACHE_LINE_SIZE * 2) std::atomic<i64> bottom{0};
    alignas(CACHE_LINE_SIZE * 2) T buffer[size];

    JobDeque() {}
    bool Push(T item)
    {
        i64 b = bottom.load(std::memory_order_relaxed);
        i64 t = top.load(std::memory_order_acquire);
        if (b >= t + size - 1) return false;
        buffer[b & mask] = item;
        bottom.store(b + 1, std::memory_order_release);
        return true;
    }
    bool Pop(T &out)
    {
        i64 b = bottom.load(std::memory_order_relaxed) - 1;
        bottom.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        i64 t = top.load(std::memory_order_relaxed);

        bool result = 1;
        if (t <= b)
        {
            out = buffer[b & mask];
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
        i64 t = top.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        i64 b       = bottom.load(std::memory_order_acquire);
        bool result = false;
        if (t < b)
        {
            result = true;
            out    = buffer[t & mask];
            if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
            {
                result = false;
            }
        }
        return result;
    }
    bool Empty() const
    {
        return bottom.load(std::memory_order_relaxed) <= top.load(std::memory_order_relaxed);
    }
};

// Lock-free condition variable
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/ThreadPool/EventCount.h
static const u32 MAX_THREAD_COUNT = 64;
struct EventCount
{
    static const u64 waiterBits  = 14;
    static const u64 stackMask   = (1ull << waiterBits) - 1;
    static const u64 waiterShift = waiterBits;
    static const u64 waiterMask  = ((1ull << waiterBits) - 1) << waiterShift;
    static const u64 waiterInc   = 1ull << waiterShift;
    static const u64 signalShift = 2 * waiterBits;
    static const u64 signalMask  = ((1ull << waiterBits) - 1) << signalShift;
    static const u64 signalInc   = 1ull << signalShift;
    static const u64 epochShift  = 3 * waiterBits;
    static const u64 epochBits   = 64 - epochShift;
    static const u64 epochMask   = ((1ull << epochBits) - 1) << epochShift;
    static const u64 epochInc    = 1ull << epochShift;

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

    EventCount() {}

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
                    w->epoch += epochInc;
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
            if ((s & stackMask) == stackMask && numWaiters == numSignals) return;
            // Sets the number of signals = to the number of waiters
            u64 newState = (s & waiterMask) | (numWaiters << signalShift) | stackMask;
            if (state.compare_exchange_weak(s, newState, std::memory_order_acq_rel))
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
            if ((s & stackMask) == stackMask && numWaiters == numSignals) return;
            u64 newState;
            if (numSignals < numWaiters)
            {
                newState = s + signalInc;
                if (state.compare_exchange_weak(s, newState, std::memory_order_acq_rel)) return;
            }
            else
            {
                u64 stackTop = s & stackMask;
                if (stackTop == stackMask) return;
                Waiter *w = &waiters[stackTop];
                u64 next  = w->next.load(std::memory_order_relaxed);
                newState  = (s & (waiterMask | signalMask)) | next;
                if (state.compare_exchange_weak(s, newState, std::memory_order_acq_rel))
                {
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

            u32 s;
            {
                std::unique_lock<std::mutex> lock(w->mutex);
                s        = w->state;
                w->state = Waiter::SIGNALED;
            }

            if (s == Waiter::WAITING) w->cv.notify_one();
        }
    }
};

THREAD_ENTRY_POINT(WorkerLoop);
using TaskFunction = std::function<void(u32)>;
struct Scheduler
{
    struct Counter
    {
        std::atomic<u32> count{0};
    };
    struct Task
    {
        static const u32 INVALID_ID = 0xffffffff;
        TaskFunction func;
        u32 id           = INVALID_ID;
        Counter *counter = 0;
        Task() {}
        // Task(Counter *counter, TaskFunction inFunc, u32 jobID) : counter(counter), func(inFunc), id(jobID) {}
    };

    struct Worker
    {
        JobDeque<Task> queue;
        EventCount::Waiter *waiter;
        u32 victim;
    };
    EventCount notifier;
    alignas(CACHE_LINE_SIZE * 2) Worker workers[MAX_THREAD_COUNT];
    u32 numWorkers;

    Scheduler() {}

    void Init(u32 numThreads)
    {
        numWorkers        = numThreads;
        workers[0].waiter = &notifier.waiters[0];
        workers[0].victim = 0;
        for (u32 i = 1; i < numThreads; i++)
        {
            workers[i].waiter = &notifier.waiters[i];
            workers[i].victim = i;
            OS_ThreadStart(WorkerLoop, (void *)&workers[i]);
        }
    }
    void ExploitTask(Worker *w, Task *t)
    {
        do
        {
            if (t->id != Task::INVALID_ID)
            {
                ExecuteTask(t);
            }
        } while (w->queue.Pop(*t));
    }
    void ExecuteTask(Task *t)
    {
        t->func(t->id);
        if (t->counter)
        {
            t->counter->count.fetch_sub(1, std::memory_order_acq_rel); //, std::memory_order_release);
        }
    }
    template <typename Pred>
    bool ExploreTask(Worker *w, Task *t, Pred predicate)
    {
        const u32 stealBound = 2 * (numWorkers + 1);
        const u32 yieldBound = 100;
        u32 numFailedSteals  = 0;
        u32 numYields        = 0;
        const u32 id         = (u32)(w - workers);
        for (;;)
        {
            // TODO: spread out distribution
            if (w->victim == id ? w->queue.Steal(*t) : workers[w->victim].queue.Steal(*t)) return true;
            if (!predicate())
            {
                numFailedSteals++;
                if (numFailedSteals > stealBound)
                {
                    std::this_thread::yield();
                    numYields++;
                    if (numYields == yieldBound) return false;
                }
                // w->victim = RandomInt(0, numWorkers);
                w->victim = (w->victim + 1) % numWorkers;
            }
            else
            {
                return false;
            }
        }
    }
    bool WaitForTask(Worker *w, Task *t, Counter *counter = 0)
    {
    begin:
        if (counter)
        {
            if (ExploreTask(w, t, [&]() { return counter->count.load(std::memory_order_relaxed) == 0; })) return true;
        }
        else
        {
            if (ExploreTask(w, t, [&]() { return false; })) return true;
        }
        notifier.Prewait();
        if (!w->queue.Empty())
        {
            w->victim = (u32)(w - workers);
            notifier.CancelWait();
            goto begin;
        }
        for (u32 i = 0; i < numWorkers; i++)
        {
            if (!workers[i].queue.Empty())
            {
                w->victim = i;
                notifier.CancelWait();
                goto begin;
            }
        }
        if (counter)
        {
            notifier.CancelWait();
            if (counter->count.load(std::memory_order_relaxed) == 0)
                return false;
        }
        else
        {
            notifier.CommitWait(w->waiter);
        }
        goto begin;
    }
    void Schedule(Counter *counter, const TaskFunction &func)
    {
        Worker *worker = &workers[GetThreadIndex()];
        Task task;
        counter->count.fetch_add(1, std::memory_order_acq_rel);
        task.counter = counter;
        task.func    = func;
        task.id      = 0;
        worker->queue.Push(task);
        notifier.NotifyOne();
    }
    void Schedule(Counter *counter, u32 numJobs, u32 groupSize, const TaskFunction &func)
    {
        Worker *worker = &workers[GetThreadIndex()];
        u32 numGroups  = (numJobs + groupSize - 1) / groupSize;
        // numGroups      = Min(numGroups, numWorkers);
        counter->count.fetch_add(numGroups, std::memory_order_acq_rel);
        Task task;
        task.counter = counter;
        task.func    = func;
        Assert(numGroups <= JobDeque<Task>::size);
        for (u32 i = 0; i < numGroups; i++)
        {
            task.id = i;
            worker->queue.Push(task);
        }
        if (numGroups >= numWorkers)
        {
            notifier.NotifyAll();
        }
        else
        {
            for (u32 i = 0; i < numGroups; i++)
            {
                notifier.NotifyOne();
            }
        }
    }
    void ScheduleAndWait(u32 numJobs, u32 groupSize, const TaskFunction &func)
    {
        Counter counter = {};
        Schedule(&counter, numJobs, groupSize, func);
        Wait(&counter);
    }
    void Wait(Counter *counter)
    {
        Worker *worker = &workers[GetThreadIndex()];
        Task t;
        for (;;)
        {
            ExploitTask(worker, &t);
            if (!WaitForTask(worker, &t, counter)) break;
        }
    }
};
static Scheduler scheduler;
THREAD_ENTRY_POINT(WorkerLoop)
{
    Scheduler::Worker *w = (Scheduler::Worker *)parameter;

    SetThreadContext(ctx);
    u64 threadIndex = (u64)(w - scheduler.workers);
    SetThreadIndex((u32)threadIndex);

    TempArena temp = ScratchStart(0, 0);
    SetThreadName(PushStr8F(temp.arena, "[Jobsystem] Worker %u", threadIndex));
    ScratchEnd(temp);

    Scheduler::Task t;

    for (;;)
    {
        scheduler.ExploitTask(w, &t);
        if (!scheduler.WaitForTask(w, &t)) break;
    }
}

template <typename T, typename Func, typename Reduce, typename... Args>
T ParallelReduce(u32 start, u32 count, u32 groupSize, Func func, Reduce reduce, Args... inArgs)
{
    TempArena temp    = ScratchStart(0, 0);
    temp.arena->align = 32;
    u32 taskCount     = (count + groupSize - 1) / groupSize;
    taskCount         = Min(taskCount, scheduler.numWorkers);
    T *values         = (T *)PushArray(temp.arena, u8, sizeof(T) * taskCount);
    for (u32 i = 0; i < taskCount; i++)
    {
        new (&values[i]) T(std::forward<Args>(inArgs)...);
    }

    u32 end      = start + count;
    u32 stepSize = count / taskCount;
    scheduler.ScheduleAndWait(taskCount, 1, [&](u32 jobID) {
        T &val          = values[jobID];
        u32 threadStart = start + stepSize * jobID;
        u32 size        = stepSize;
        if (jobID == taskCount - 1)
        {
            size = end - threadStart;
        }
        func(val, threadStart, size);
    });

    T out = T(std::forward<Args>(inArgs)...);
    for (u32 i = 0; i < taskCount; i++)
    {
        reduce(out, values[i]);
    }
    ScratchEnd(temp);
    return out;
}

// template <i32 N>
// struct SchedulingInfo
// {
// };

#if 0
struct DynamicPoolScheduler
{
    struct Counter
    {
        std::atomic<u32> count{0};
    };
    struct Task
    {
        static const u32 INVALID_ID = 0xffffffff;
        TaskFunction func;
        u32 id           = INVALID_ID;
        Counter *counter = 0;
        Task() {}
    };

    struct Worker
    {
        // EventCount::Waiter *waiter;
        // u32 victim;
        std::atomic<u32> poolIndex;
    };
    struct ThreadPool
    {
        JobDeque<Task> queue;
        u32 start;
        u32 count;
    };
    alignas(CACHE_LINE_SIZE) Worker workers[MAX_THREAD_COUNT];
    u32 numWorkers;
    std::atomic<ThreadPool> pools[MAX_THREAD_COUNT];
    std::atomic<u32> numPools{1};

    DynamicPoolScheduler() {}
    // for Dynamic Load balancing

    template <i32 N>
    bool SplitThreadPool(u32 *starts, u32 *counts)
    {
        Assert(N == 4);
        u32 orderedStarts[N];
        u32 orderedCounts[N];
        u32 totalCount = 0;
        for (u32 i = 0; i < N; i++)
        {
            orderedStarts[i] = starts[i];
            orderedCounts[i] = counts[i];
            totalCount += counts[i];
        }
        if (orderedStarts[2] < orderedStarts[1])
        {
            Swap(orderedStarts[2], orderedStarts[1]);
            Swap(orderedCounts[2], orderedCounts[1]);
        }
        if (orderedStarts[3] < orderedStarts[2])
        {
            Swap(orderedStarts[3], orderedStarts[2]);
            Swap(orderedCounts[3], orderedCounts[2]);
        }
        if (orderedStarts[2] < orderedStarts[1])
        {
            Swap(orderedStarts[2], orderedStarts[1]);
            Swap(orderedCounts[2], orderedCounts[1]);
        }

        u32 threadIndex = GetThreadIndex();
        u32 poolIndex   = workers[threadIndex].poolIndex.load(std::memory_order_acquire);
        ThreadPool pool = pools[poolIndex].load(std::memory_order_relaxed);
        if (pool.count == 1)
        {
            return false;
        }

        u32 newNumThreads[N];
        u32 total = 0;
        for (u32 i = 0; i < N - 1; i++)
        {
            newNumThreads[i] = u32(pool.count * (f32)orderedCounts[i] / (totalCount));
            total += newNumThreads[i];
        }
        newNumThreads[N - 1] = pool.count - total;

        u32 offset      = 0;
        u32 numNewPools = 0;
        ThreadPool newPools[N];
        for (u32 i = 0; i < N; i++)
        {
            u32 numThreads = newNumThreads[i];
            if (numThreads == 0) continue;
            ThreadPool &pool = newPools[numNewPools++];
            pool.start       = offset;
            pool.count       = numThreads;
            offset += numThreads;
        }

        if (numNewPools == 1) return true;

        // pool[poolIndex].store(left, std::memory_order_relaxed);
        pools[poolIndex].store(newPools[0], std::memory_order_relaxed);
        u32 newPoolIndex = numPools.fetch_add(numNewPools - 1, std::memory_order_relaxed);
        for (u32 i = 0; i < numNewPools - 1; i++)
        {
            pool[newPoolIndex + i].store(newPools[i + 1], std::memory_order_relaxed);
        }
        for (u32 i = 1; i < numNewPools; i++)
        {
            ThreadPool &pool = newPools[i];
            for (u32 j = pool.start; j < pool.start + pool.count; j++)
            {
                workers[j].poolIndex.store(newPoolIndex + i - 1, std::memory_order_release);
            }
        }
        return true;
    }

    void WorkerLoop(Worker *w)
    {
        Scheduler::Task t;
        const u32 stealBound = 2 * (numWorkers + 1);
        const u32 yieldBound = 100;
        u32 numFailedSteals  = 0;
        u32 numYields        = 0;
        const u32 id         = (u32)(w - workers);
        // Initial dynamic thread pool phase
        for (;;)
        {
            if (t.id != Task::INVALID_ID)
            {
                ExecuteTask(&t);
                numFailedSteals = 0;
            }
            u32 poolIndex    = w->poolIndex.load(std::memory_order_relaxed);
            ThreadPool &pool = pools[poolIndex].load(std::memory_order_relaxed);
            // if (pool.count == 1) break;
            if (!pool.queue.Steal(*t))
            {
                numFailedSteals++;
                if (numFailedSteals > stealBounds)
                {
                    std::this_thread_yield();
                }
            }
        }
        // Now start stealing (?)
        // for (;;)
        // {
        //     ? for (;;)
        //     {
        //         u32 poolIndex    = w->poolIndex.load(std::memory_order_relaxed);
        //         ThreadPool &pool = pools[poolIndex];
        //         if (!pool.queue.SinglePop(*t)) break;
        //     }
        //     WaitForTask(w, &t);
        // }
    }

    void ExecuteTask(Task *t)
    {
        t->func(t->id);
        if (t->counter)
        {
            t->counter->count.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    void Schedule(Counter *counter, u32 numJobs, u32 groupSize, const TaskFunction &func)
    {
        Worker *worker = &workers[GetThreadIndex()];
        u32 numGroups  = (numJobs + groupSize - 1) / groupSize;
        counter->count.fetch_add(numGroups, std::memory_order_acq_rel);
        Task task;
        task.counter = counter;
        task.func    = func;
        Assert(numGroups <= JobDeque<Task>::size);

        ThreadPool &pool = pools[worker->poolIndex.load(std::memory_order_relaxed)];
        for (u32 i = 0; i < numGroups; i++)
        {
            pool.queue.
        }
    }

    // template <typename Pred>
    // bool ExploreTask(Worker *w, Task *t, Pred predicate)
    // {
    //     const u32 stealBound = 2 * (numWorkers + 1);
    //     const u32 yieldBound = 100;
    //     u32 numFailedSteals  = 0;
    //     u32 numYields        = 0;
    //     const u32 id         = (u32)(w - workers);
    //     ThreadPool &pool     = pools[w->poolIndex.load(std::memory_order_relaxed)];
    //     Assert(pool.count == 1);
    //     for (;;)
    //     {
    //         // TODO: spread out distribution
    //         if (w->victim == id ? pools->queue.Steal(*t) : pools[w->victim].queue.Steal(*t)) return true;
    //         if (!predicate())
    //         {
    //             numFailedSteals++;
    //             if (numFailedSteals > stealBound)
    //             {
    //                 std::this_thread::yield();
    //                 numYields++;
    //                 if (numYields == yieldBound) return false;
    //             }
    //             // w->victim = RandomInt(0, numWorkers);
    //             w->victim = (w->victim + 1) % numWorkers;
    //         }
    //         else
    //         {
    //             return false;
    //         }
    //     }
    // }
    //
    // void WaitForTask(Worker *w, Task *t, Counter *counter = 0)
    // {
    // begin:
    //     if (counter)
    //     {
    //         if (ExploreTask(w, t, [&]() { return counter->count.load(std::memory_order_relaxed) == 0; })) return true;
    //     }
    //     else
    //     {
    //         if (ExploreTask(w, t, [&]() { return false; })) return true;
    //     }
    //     notifier.Prewait();
    //     if (!w->queue.Empty())
    //     {
    //         w->victim = (u32)(w - workers);
    //         notifier.CancelWait();
    //         goto begin;
    //     }
    //     for (u32 i = 0; i < numWorkers; i++)
    //     {
    //         if (!workers[i].queue.Empty())
    //         {
    //             w->victim = i;
    //             notifier.CancelWait();
    //             goto begin;
    //         }
    //     }
    //     if (counter)
    //     {
    //         notifier.CancelWait();
    //         if (counter->count.load(std::memory_order_relaxed) == 0)
    //             return false;
    //     }
    //     else
    //     {
    //         notifier.CommitWait(w->waiter);
    //     }
    //     goto begin;
    // }
};
#endif

} // namespace rt