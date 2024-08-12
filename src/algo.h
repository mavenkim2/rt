template <typename Predicate>
u32 FindInterval(u32 sz, const Predicate &pred)
{
    i32 size = sz - 2, first = 1;
    while (size > 0)
    {
        u32 half = size >> 1, middle = first + half;
        bool predResult = pred(middle);
        first           = predResult ? middle + 1 : first;
        size            = predResult ? size - (half + 1) : half;
    }
    return (u32)Clamp(first - 1, 0, (i32)sz - 2);
}
