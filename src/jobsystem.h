#include <functional>
namespace rt
{
namespace jobsystem
{

volatile b32 gTerminateJobs = 0;
const i32 JOB_QUEUE_LENGTH  = 256;

using std::atomic;

THREAD_ENTRY_POINT(JobThreadEntryPoint);

struct Counter
{
    atomic<u32> count;
};

enum class Priority
{
    Low,
    High,
};

struct JobArgs
{
    u32 jobId;
    u32 idInGroup;
    b32 isLastJob;
    u32 threadId;
};

using JobFunction = std::function<void(JobArgs)>;

struct Job
{
    Counter *counter;
    JobFunction func;
    u32 groupId;
    u32 groupJobStart;
    u32 groupJobSize;
};

// Jobs cannot be spawned from within a thread (for now)
struct JobQueue
{
    Job jobs[JOB_QUEUE_LENGTH];
    atomic<u64> writePos;
    atomic<u64> readPos;
    atomic<u64> commitReadPos;
    atomic<u64> commitWritePos;
};

struct JobSystem
{
    JobQueue highPriorityQueue;
    JobQueue lowPriorityQueue;

    OS_Handle threads[16]; // TODO: ?
    u32 threadCount;
    OS_Handle readSemaphore;
};

void InitializeJobsystem();
void KickJob(Counter *counter, const JobFunction &func, Priority priority = Priority::Low);
void KickJobs(Counter *counter, u32 numJobs, u32 groupSize, const JobFunction &func, Priority priority = Priority::Low);
void WaitJobs(Counter *counter);
b32 Pop(JobQueue &queue, u64 threadIndex);
THREAD_ENTRY_POINT(JobThreadEntryPoint);
void EndJobsystem();

} // namespace jobsystem
} // namespace rt
