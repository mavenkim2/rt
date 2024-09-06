#ifndef DEBUG_H
#define DEBUG_H

namespace rt
{
struct Event
{
    u32 offset;
    PerformanceCounter counter;
    Event(u32 offset);
    ~Event();
};

#define TIMED_FUNCTION(arg) Event event_##__LINE__##arg(OffsetOf(ThreadStatistics, arg) / 8)

} // namespace rt
#endif
