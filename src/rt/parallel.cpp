#include "parallel.h"

namespace rt
{
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

} // namespace rt
