namespace rt
{
static const i32 MAX_OS_PATH = 256;

struct string
{
    u8 *str;
    u64 size;

    string() {}
    string(const char *c);
    b32 operator==(const string &a) const;
    b32 operator==(const string &a);
    b32 operator==(const char *text);
};

#define STB_SPRINTF_IMPLEMENTATION
#include "third_party/stb_sprintf.h"

//////////////////////////////
// Char
//
inline b32 CharIsWhitespace(u8 c);
b32 CharIsAlpha(u8 c);
b32 CharIsAlphaUpper(u8 c);
b32 CharIsAlphaLower(u8 c);
b32 CharIsDigit(u8 c);
u8 CharToLower(u8 c);
u8 CharToUpper(u8 c);

//////////////////////////////
// Creating Strings
//
string Str8(u8 *str, u64 size);
inline string Substr8(string str, u64 min, u64 max);
u64 CalculateCStringLength(const char *cstr);
string PushStr8F(Arena *arena, char *fmt, ...);
string PushStr8FV(Arena *arena, char *fmt, va_list args);
string PushStr8Copy(Arena *arena, string str);
string StrConcat(Arena *arena, string s1, string s2);

#define Str8Lit(s)     Str8((u8 *)(s), sizeof(s) - 1)
#define Str8C(cstring) Str8((u8 *)(cstring), CalculateCStringLength(cstring))

//////////////////////////////
// Finding strings
//
typedef u32 MatchFlags;
enum
{
    MatchFlag_CaseInsensitive  = (1 << 0),
    MatchFlag_RightSideSloppy  = (1 << 1),
    MatchFlag_SlashInsensitive = (1 << 2),
    MatchFlag_FindLast         = (1 << 3),
    MatchFlag_KeepEmpties      = (1 << 4),
};

string SkipWhitespace(string str);
b32 StartsWith(string a, string b);
b32 MatchString(string a, string b, MatchFlags flags);
u64 FindSubstring(string haystack, string needle, u64 startPos, MatchFlags flags);
b8 Contains(string haystack, string needle, MatchFlags flags = MatchFlag_CaseInsensitive);
u32 ConvertToUint(string word);
string SkipToNextWord(string line);
string GetFirstWord(string line);

//////////////////////////////
// File Path Helpers
//
string PathSkipLastSlash(string str);
string GetFileExtension(string path);
string Str8PathChopPastLastSlash(string string);
string Str8PathChopLastSlash(string string);

//////////////////////////////
// Hash
//
i32 HashFromString(string string);
u32 HashCombine(u32 hash1, u32 hash2);
u64 HashStruct_(void *ptr, u64 size);
#define HashStruct(ptr) HashStruct_((ptr), sizeof(*(ptr)))

//////////////////////////////
// String Ids
//
typedef u32 StringId;
constexpr StringId operator""_sid(const char *ptr, size_t count);

//////////////////////////////
// String token building/reading
//

struct StringBuilderNode
{
    string str;
};

struct StringBuilderChunkNode
{
    StringBuilderNode *values;
    StringBuilderChunkNode *next;

    u32 count;
    u32 cap;
};

struct StringBuilder
{
    StringBuilderChunkNode *first;
    StringBuilderChunkNode *last;

    u64 totalSize;
    Arena *arena;
};

struct Tokenizer
{
    string input;
    u8 *cursor;
};

inline void Advance(Tokenizer *tokenizer, u32 size);
inline b32 Advance(Tokenizer *tokenizer, string check);
string ReadLine(Tokenizer *tokenizer);
inline u32 ReadUint(Tokenizer *iter);
inline f32 ReadFloat(Tokenizer *iter);
inline void SkipToNextLine(Tokenizer *iter);
inline u8 *GetCursor_(Tokenizer *tokenizer);
inline b32 EndOfBuffer(Tokenizer *tokenizer);
string ReadLine(Tokenizer *tokenizer);
void Get(Tokenizer *tokenizer, void *ptr, u32 size);
inline u8 *GetPointer_(Tokenizer *tokenizer);

u64 Put(StringBuilder *builder, void *data, u64 size);
u64 Put(StringBuilder *builder, string str);
u64 Put(StringBuilder *builder, u32 value);
string CombineBuilderNodes(StringBuilder *builder);
b32 WriteEntireFile(StringBuilder *builder, string filename);
inline u64 PutPointer(StringBuilder *builder, u64 address);
inline void ConvertPointerToOffset(u8 *buffer, u64 location, u64 offset);
inline u8 *ConvertOffsetToPointer(u8 *base, u64 offset);
void PutLine(StringBuilder *builder, u32 indents, char *fmt, ...);

#define PutPointerValue(builder, ptr)    Put(builder, ptr, sizeof(*ptr))
#define PutStruct(builder, s)            PutPointerValue(builder, &s);
#define AppendArray(builder, ptr, count) Put(builder, ptr, sizeof(*ptr) * count)
#define PutArray(builder, array, count)  Put((builder), (array), sizeof((array)[0]) * (count));

#define GetPointerValue(tokenizer, ptr) Get(tokenizer, ptr, sizeof(*ptr))
#define GetPointer(tokenizer, type)     (type *)GetPointer_(tokenizer)
#define GetArray(tokenizer, array, count_)                            \
    do                                                                \
    {                                                                 \
        array.count = count_;                                         \
        Get(tokenizer, array.items, sizeof(array.items[0]) * count_); \
    } while (0)

#define GetTokenCursor(tokenizer, type) (type *)GetCursor_(tokenizer)
} // namespace rt