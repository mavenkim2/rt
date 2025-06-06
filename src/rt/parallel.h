#ifndef PARALLEL_H_
#define PARALLEL_H_
#include <atomic>
#include <mutex>
#include <functional>

#include "random.h"
#include "thread_context.h"

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
            if (a == EMPTY &&
                writePos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
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
            if (a == WRITTEN &&
                readPos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
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
        if (a == WRITTEN &&
            readPos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
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
        for (;;)
        {
            i64 b = bottom.load(std::memory_order_relaxed);
            i64 t = top.load(std::memory_order_acquire);
            if (b >= t + size - 1) continue;
            // {
            //     Assert(false);
            //     return false;
            // }
            buffer[b & mask] = item;
            bottom.store(b + 1, std::memory_order_release);
            return true;
        }
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
                if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                                 std::memory_order_relaxed))
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
        i64 b = bottom.load(std::memory_order_acquire);
        if (t < b)
        {
            if (top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                            std::memory_order_relaxed))
            {
                out = buffer[t & mask];
                return true;
            }
        }
        return false;
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
                if (state.compare_exchange_weak(s, newState, std::memory_order_acq_rel))
                    return;
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
            if (((s & waiterMask) >> waiterShift) == ((s & signalMask) >> signalShift))
                newState -= signalInc;
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
                if (state.compare_exchange_weak(s, newState, std::memory_order_acq_rel))
                    return;
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
        // Task(Counter *counter, TaskFunction inFunc, u32 jobID) : counter(counter),
        // func(inFunc), id(jobID) {}
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
        // OS_SetThreadAffinity(GetMainThreadHandle(), 0);
        for (u32 i = 1; i < numThreads; i++)
        {
            workers[i].waiter      = &notifier.waiters[i];
            workers[i].victim      = i;
            OS_Handle threadHandle = OS_ThreadStart(WorkerLoop, (void *)&workers[i]);
            // OS_SetThreadAffinity(threadHandle, i);
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
            t->counter->count.fetch_sub(1, std::memory_order_acq_rel);
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
            if (predicate()) return false;
            if (w->victim == id ? w->queue.Steal(*t) : workers[w->victim].queue.Steal(*t))
                return true;
            numFailedSteals++;
            if (numFailedSteals > stealBound)
            {
                std::this_thread::yield();
                numYields++;
                if (numYields == yieldBound) return false;
            }
            w->victim = RandomInt(0, numWorkers);
            // w->victim = (w->victim + 1) % numWorkers;
        }
    }
    bool WaitForTask(Worker *w, Task *t)
    {
    begin:
        if (ExploreTask(w, t, [&]() { return false; })) return true;
        notifier.Prewait();
        for (u32 i = 0; i < numWorkers; i++)
        {
            if (!workers[i].queue.Empty())
            {
                w->victim = i;
                notifier.CancelWait();
                goto begin;
            }
        }
        notifier.CommitWait(w->waiter);
        goto begin;
    }

    bool StealUntilEmpty(Worker *w, Task *t)
    {
        if (ExploreTask(w, t, [&]() { return false; })) return true;
        return false;
    }

    bool WaitForTask(Worker *w, Task *t, Counter *counter)
    {
    begin:
        if (ExploreTask(w, t,
                        [&]() { return counter->count.load(std::memory_order_acquire) == 0; }))
            return true;
        if (counter->count.load(std::memory_order_acquire) == 0) return false;
        for (u32 i = 0; i < numWorkers; i++)
        {
            if (!workers[i].queue.Empty())
            {
                w->victim = i;
                goto begin;
            }
        }
        goto begin;
    }
    void Schedule(const TaskFunction &func)
    {
        Worker *worker = &workers[GetThreadIndex()];
        Task task;
        task.counter = 0;
        task.func    = func;
        task.id      = 0;
        worker->queue.Push(task);
        notifier.NotifyOne();
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
    void ScheduleAndWait(const TaskFunction &func)
    {
        Counter counter = {};
        Schedule(&counter, func);
        Wait(&counter);
    }

    // NOTE: use when want to delay, instead of needing jobs to run to completion
    void WaitUntilEmpty(Counter *counter)
    {
        if (counter->count.load(std::memory_order_acquire) == 0) return;
        Worker *worker = &workers[GetThreadIndex()];
        Task t;
        for (;;)
        {
            ExploitTask(worker, &t);
            if (!StealUntilEmpty(worker, &t)) break;
        }
    }

    void Wait(Counter *counter)
    {
        if (counter->count.load(std::memory_order_acquire) == 0) return;
        Worker *worker = &workers[GetThreadIndex()];
        Task t;
        for (;;)
        {
            ExploitTask(worker, &t);
            if (!WaitForTask(worker, &t, counter)) break;
        }
    }
};

extern Scheduler scheduler;

template <typename Func>
void ParallelFor(u32 start, u32 count, u32 groupSize, const Func &func)
{
    u32 taskCount = (count + groupSize - 1) / groupSize;
    taskCount     = Min(taskCount, 768u);
    u32 end       = start + count;
    u32 stepSize  = count / taskCount;
    scheduler.ScheduleAndWait(taskCount, 1, [&](u32 jobID) {
        u32 tStart = start + jobID * stepSize;
        u32 size   = jobID == taskCount - 1 ? end - tStart : stepSize;
        func(jobID, tStart, size);
    });
}

template <typename Func>
void ParallelFor(u32 start, u32 count, u32 threshold, u32 groupSize, const Func &func)
{
    if (count > threshold) ParallelFor(start, count, groupSize, func);
    else func(0, start, count);
}

template <typename Func>
void ParallelFor2D(Vec2i start, Vec2i end, Vec2i tileSize, const Func &func)
{
    Assert(end[0] > start[0] && end[1] > start[1]);
    Vec2i width = end - start;

    int tileCountX = (width[0] + tileSize[0] - 1) / tileSize[0];
    int tileCountY = (width[1] + tileSize[1] - 1) / tileSize[1];

    int taskCount = tileCountX * tileCountY;
    int offset    = 0;
    for (; taskCount;)
    {
        int jobsToRun = Min(taskCount, JobDeque<Scheduler::Task>::size - 1);
        taskCount -= jobsToRun;

        scheduler.ScheduleAndWait(jobsToRun, 1, [&](int jobID) {
            jobID += offset;
            int tileX = jobID % tileCountX;
            int tileY = jobID / tileCountX;
            Vec2i startBounds(start[0] + tileSize[0] * tileX, start[1] + tileSize[1] * tileY);
            Vec2i endBounds(Min(start[0] + tileSize[0] * (tileX + 1), end[0]),
                            Min(start[1] + tileSize[1] * (tileY + 1), end[1]));
            func(jobID, startBounds, endBounds);
        });
        offset += jobsToRun;
    }
}

struct ParallelForOutput
{
    void *out;
    u32 num;
    u32 groupSize;
};

template <typename T, typename Func, typename... Args>
ParallelForOutput ParallelFor(TempArena temp, u32 start, u32 count, u32 groupSize, Func func,
                              Args... inArgs)
{
    u32 taskCount = (count + groupSize - 1) / groupSize;
    taskCount     = Min(taskCount, 768u);
    T *values     = (T *)PushArrayNoZero(temp.arena, u8, sizeof(T) * taskCount);
    for (u32 i = 0; i < taskCount; i++)
    {
        new (&values[i]) T(std::forward<Args>(inArgs)...);
    }

    u32 end      = start + count;
    u32 stepSize = count / taskCount;
    scheduler.ScheduleAndWait(taskCount, 1, [&](u32 jobID) {
        T &val          = values[jobID];
        u32 threadStart = start + stepSize * jobID;
        Assert(end > threadStart);
        u32 size = jobID == taskCount - 1 ? end - threadStart : stepSize;
        func(val, jobID, threadStart, size);
    });

    ParallelForOutput out;
    out.out       = (void *)values;
    out.num       = taskCount;
    out.groupSize = stepSize;

    return out;
}

template <typename T, typename ReduceFunc, typename... Args>
void Reduce(T &out, ParallelForOutput output, ReduceFunc reduce, Args... inArgs)
{
    new (&out) T(std::forward<Args>(inArgs)...);
    for (u32 i = 0; i < output.num; i++)
    {
        reduce(out, ((T *)output.out)[i]);
    }
}

template <typename T, typename Func, typename ReduceFunc, typename... Args>
void ParallelReduce(T *out, u32 start, u32 count, u32 groupSize, Func func, ReduceFunc reduce,
                    Args... inArgs)
{
    if (count == 0) return;

    if (count < groupSize)
    {
        func(*out, 0, 0, count);
        return;
    }

    TempArena temp = ScratchStart(0, 0);
    ParallelForOutput output =
        ParallelFor<T>(temp, start, count, groupSize, func, std::forward<Args>(inArgs)...);
    Reduce<T>(*out, output, reduce, std::forward<Args>(inArgs)...);
    ScratchEnd(temp);
}

} // namespace rt
#endif
