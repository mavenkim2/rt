namespace rt
{
Event::Event(u32 offset) : offset(offset) { counter = OS_StartCounter(); }
Event::~Event()
{
    f32 time = OS_GetMilliseconds(counter);
    ((f64 *)(&threadLocalStatistics[GetThreadIndex()]))[offset] += (f64)time;
}
} // namespace rt
