#ifndef DEBUG_H
#define DEBUG_H

struct Event
{
    u32 offset;
    PerformanceCounter counter;
    Event(u32 offset);
    ~Event();
};

#define TIMED_FUNCTION(arg) Event event_##__LINE__(OffsetOf(ThreadStatistics, arg) / 8)

#endif
