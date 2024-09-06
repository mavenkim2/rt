namespace rt
{
Event::Event(u32 offset) : offset(offset)
{
    counter = OS_StartCounter();
}
Event::~Event()
{
    f32 time = OS_GetMilliseconds(counter);
    ((u64 *)(&threadLocalStatistics[GetThreadIndex()]))[offset] += (u64)time;
}
} // namespace rt
