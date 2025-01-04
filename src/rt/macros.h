#ifndef MACROS_H
#define MACROS_H

#define RECURSE_1(macro, first)      macro(first)
#define RECURSE_2(macro, first, ...) macro(first) EXPAND(RECURSE_1(macro, __VA_ARGS__))
#define RECURSE_3(macro, first, ...) macro(first) EXPAND(RECURSE_2(macro, __VA_ARGS__))
#define RECURSE_4(macro, first, ...) macro(first) EXPAND(RECURSE_3(macro, __VA_ARGS__))
#define RECURSE_5(macro, first, ...) macro(first) EXPAND(RECURSE_4(macro, __VA_ARGS__))
#define RECURSE_6(macro, first, ...) macro(first) EXPAND(RECURSE_5(macro, __VA_ARGS__))
#define RECURSE_7(macro, first, ...) macro(first) EXPAND(RECURSE_6(macro, __VA_ARGS__))

#define RECURSE__1(macro, first)      macro(first)
#define RECURSE__2(macro, first, ...) macro(first), EXPAND(RECURSE__1(macro, __VA_ARGS__))
#define RECURSE__3(macro, first, ...) macro(first), EXPAND(RECURSE__2(macro, __VA_ARGS__))
#define RECURSE__4(macro, first, ...) macro(first), EXPAND(RECURSE__3(macro, __VA_ARGS__))
#define RECURSE__5(macro, first, ...) macro(first), EXPAND(RECURSE__4(macro, __VA_ARGS__))
#define RECURSE__6(macro, first, ...) macro(first), EXPAND(RECURSE__5(macro, __VA_ARGS__))
#define RECURSE__7(macro, first, ...) macro(first), EXPAND(RECURSE__6(macro, __VA_ARGS__))

#define RECURSE2_1(macro, first)      macro(first, 0)
#define RECURSE2_2(macro, first, ...) macro(first, 1) EXPAND(RECURSE2_1(macro, __VA_ARGS__))
#define RECURSE2_3(macro, first, ...) macro(first, 1) EXPAND(RECURSE2_2(macro, __VA_ARGS__))
#define RECURSE2_4(macro, first, ...) macro(first, 1) EXPAND(RECURSE2_3(macro, __VA_ARGS__))
#define RECURSE2_5(macro, first, ...) macro(first, 1) EXPAND(RECURSE2_4(macro, __VA_ARGS__))
#define RECURSE2_6(macro, first, ...) macro(first, 1) EXPAND(RECURSE2_5(macro, __VA_ARGS__))
#define RECURSE2_7(macro, first, ...) macro(first, 1) EXPAND(RECURSE2_6(macro, __VA_ARGS__))

#define EXPAND(x)    x
#define CONCAT(a, b) a##b

#define COMMA_SEPARATED_LIST(...)                                                             \
    COMMA_SEPARATED_LIST_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)
#define COMMA_SEPARATED_LIST_HELPER(x, ...) EXPAND(CONCAT(RECURSE__, x)(EXPAND, __VA_ARGS__))

#define COUNT_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15,     \
                        _16, _17, _18, _19, _20, N, ...)                                      \
    N
#define COUNT_ARGS(...)                                                                       \
    EXPAND(COUNT_ARGS_IMPL(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,  \
                           6, 5, 4, 3, 2, 1))

#define BOOL_ARGS(...)                                                                        \
    EXPAND(COUNT_ARGS_IMPL(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                           1, 0))
#endif
