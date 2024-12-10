#include <thread>
#include <utility>
namespace rt
{
namespace jobsystem
{
static JobSystem jobSystem;

static const u32 TASK_STACK_SIZE = 1024;

struct ThreadQueue
{
    Job taskStack[TASK_STACK_SIZE];
    u32 stackPtr;
};

ThreadQueue *threadQueues;
thread_local ThreadQueue *threadLocalQueue;
static u32 numProcessors;

void InitializeJobsystem()
{
    numProcessors           = OS_NumProcessors();
    jobSystem.threadCount   = Min<u32>(ArrayLength(jobSystem.threads), numProcessors);
    jobSystem.readSemaphore = OS_CreateSemaphore(jobSystem.threadCount);

    // threadQueues = (ThreadQueue *)malloc(numProcessors * sizeof(ThreadQueue));

    for (size_t i = 0; i < numProcessors; i++)
    {
        jobSystem.threads[i] = OS_ThreadStart(jobsystem::JobThreadEntryPoint, (void *)i);
        OS_SetThreadAffinity(jobSystem.threads[i], (u32)i);
    }
}

// why is the const necessary here?
void KickJob(Counter *counter, const JobFunction &func, Priority priority)
{
    JobQueue *queue = 0;
    switch (priority)
    {
        case Priority::Low: queue = &jobSystem.lowPriorityQueue; break;
        case Priority::High: queue = &jobSystem.highPriorityQueue; break;
    }
    for (;;)
    {
        u64 writePos = queue->writePos.load();
        u64 readPos  = queue->readPos.load();

        u64 availableSpots = JOB_QUEUE_LENGTH - (writePos - readPos);
        if (availableSpots >= 1)
        {
            if (queue->commitWritePos.compare_exchange_weak(writePos, writePos + 1))
            {
                Job *job           = &queue->jobs[writePos & (JOB_QUEUE_LENGTH - 1)];
                job->func          = func;
                job->counter       = counter;
                job->groupId       = 0;
                job->groupJobStart = 0;
                job->groupJobSize  = 1;
                if (job->counter)
                {
                    job->counter->count.fetch_add(1);
                }

                queue->writePos.store(writePos + 1);
                // OS_ReleaseSemaphore(jobSystem.readSemaphore);
                break;
            }
        }
    }
}

void KickJobs(Counter *counter, u32 numJobs, u32 groupSize, const JobFunction &func,
              Priority priority)
{
    JobQueue *queue = 0;
    switch (priority)
    {
        case Priority::Low: queue = &jobSystem.lowPriorityQueue; break;
        case Priority::High: queue = &jobSystem.highPriorityQueue; break;
    }

    u32 numGroups = (numJobs + groupSize - 1) / groupSize;

    for (;;)
    {
        u64 writePos = queue->writePos.load();
        u64 readPos  = queue->readPos.load();

        u64 availableSpots = JOB_QUEUE_LENGTH - (writePos - readPos);
        if (availableSpots >= numGroups)
        {
            if (queue->commitWritePos.compare_exchange_weak(writePos, writePos + numGroups))
            {
                for (u32 i = 0; i < numGroups; i++)
                {
                    Job *job           = &queue->jobs[(writePos + i) & (JOB_QUEUE_LENGTH - 1)];
                    job->func          = func;
                    job->counter       = counter;
                    job->groupId       = i;
                    job->groupJobStart = i * groupSize;
                    job->groupJobSize  = Min(groupSize, numJobs - job->groupJobStart);
                }
                counter->count.fetch_add(numGroups);
                queue->writePos.store(writePos + numGroups);
                // OS_ReleaseSemaphores(jobSystem.readSemaphore, Min(numGroups,
                // numProcessors));
                break;
            }
        }
    }
}

template <typename T, typename Func, typename Reduce, typename... Args>
T ParallelReduce(u32 count, u32 blockSize, Func func, Reduce reduce, Args... inArgs)
{
    TempArena temp             = ScratchStart(0, 0);
    jobsystem::Counter counter = {};

    // TODO: this is kind of redundant. maybe just want to pass in the number of groups
    // directly
    u32 taskCount = (count + blockSize - 1) / blockSize;
    taskCount     = Min(taskCount, numProcessors);
    u32 groupSize = (count + taskCount - 1) / taskCount;

    T *objs = (T *)PushArray(temp.arena, u8, sizeof(T) * taskCount);
    for (u32 i = 0; i < taskCount; i++)
    {
        new (&objs[i]) T(std::forward<Args>(inArgs)...);
    }
    // TODO: maybe going to need task stealing, and have the thread work on something while it
    // waits
    jobsystem::KickJobs(&counter, taskCount, 1, [&](jobsystem::JobArgs args) {
        T &obj    = objs[args.jobId];
        u32 start = groupSize * args.jobId;
        func(obj, start, Min(groupSize, count - start));
    });

    jobsystem::WaitJobs(&counter);

    T result = T(std::forward<Args>(inArgs)...);

    for (u32 i = 0; i < taskCount; i++)
    {
        reduce(result, objs[i]);
    }

    ScratchEnd(temp);
    return result;
}

void WaitJobs(Counter *counter)
{
    while (counter->count.load() != 0)
    {
        if (Pop(jobSystem.highPriorityQueue, 0) && Pop(jobSystem.lowPriorityQueue, 0))
        {
            _mm_pause();
            // std::this_thread::yield();
        }
    }
}

b32 Pop(JobQueue &queue, u64 threadIndex)
{
    b32 result         = 0;
    u64 writePos       = queue.writePos.load();
    u64 readPos        = queue.readPos.load();
    u64 commitReadPos  = queue.commitReadPos.load();
    u64 availableSpots = writePos - readPos;
    if (availableSpots >= 1)
    {
        if (queue.commitReadPos.compare_exchange_weak(readPos, readPos + 1))
        {
            Job *readJob = &queue.jobs[(readPos) & (JOB_QUEUE_LENGTH - 1)];

            Job job;
            job.groupJobStart = readJob->groupJobStart;
            job.groupJobSize  = readJob->groupJobSize;
            job.groupId       = readJob->groupId;
            job.func          = readJob->func;
            job.counter       = readJob->counter;

            queue.readPos.store(readPos + 1);

            JobArgs args;
            for (u32 i = job.groupJobStart; i < job.groupJobStart + job.groupJobSize; i++)
            {
                args.threadId  = (u32)threadIndex;
                args.jobId     = i;
                args.idInGroup = i - job.groupJobStart;
                args.isLastJob = i == (job.groupJobSize + job.groupJobSize - 1);
                job.func(args);
            }
            if (job.counter)
            {
                job.counter->count.fetch_sub(1);
            }
        }
    }
    else
    {
        result = 1;
    }
    return result;
}

THREAD_ENTRY_POINT(JobThreadEntryPoint)
{
    SetThreadContext(ctx);
    u64 threadIndex = (u64)parameter;
    SetThreadIndex((u32)threadIndex);
    TempArena temp = ScratchStart(0, 0);
    SetThreadName(PushStr8F(temp.arena, "[Jobsystem] Worker %u", threadIndex));
    ScratchEnd(temp);

    // threadLocalQueue           = &threadQueues[threadIndex];
    // threadLocalQueue->stackPtr = 0;

    for (; !gTerminateJobs;)
    {
        if (Pop(jobSystem.highPriorityQueue, threadIndex) &&
            Pop(jobSystem.lowPriorityQueue, threadIndex))
        {
            OS_Yield();
        }
    }
}

void EndJobsystem()
{
    while (!Pop(jobSystem.highPriorityQueue, 0) || !Pop(jobSystem.lowPriorityQueue, 0))
        continue;

    gTerminateJobs = 1;
    std::atomic_thread_fence(std::memory_order_release);
    OS_ReleaseSemaphores(jobSystem.readSemaphore, jobSystem.threadCount);
    for (size_t i = 0; i < jobSystem.threadCount; i++)
    {
        OS_ThreadJoin(jobSystem.threads[i]);
    }
}
} // namespace jobsystem
} // namespace rt
