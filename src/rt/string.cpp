namespace rt
{
string::string(const char *c) { *this = Str8C(c); }

//////////////////////////////
// Char
//

u8 CharGetPair(const u8 ch)
{
    if (ch == '(') return ')';
    else if (ch == '[') return ']';
    else if (ch == '"') return '"';

    Assert(0);
    return 'x';
}

inline b32 CharIsWhitespace(u8 c) { return c == ' '; }

inline b32 CharIsBlank(u8 c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

b32 CharIsAlpha(u8 c) { return CharIsAlphaUpper(c) || CharIsAlphaLower(c); }

b32 CharIsAlphaUpper(u8 c) { return c >= 'A' && c <= 'Z'; }

b32 CharIsAlphaLower(u8 c) { return c >= 'a' && c <= 'z'; }
b32 CharIsDigit(u8 c) { return (c >= '0' && c <= '9'); }

u8 CharToLower(u8 c)
{
    u8 result = (c >= 'A' && c <= 'Z') ? ('a' + (c - 'A')) : c;
    return result;
}
u8 CharToUpper(u8 c)
{
    u8 result = (c >= 'a' && c <= 'z') ? ('A' + (c - 'a')) : c;
    return result;
}
b32 CharIsSlash(u8 c) { return (c == '/' || c == '\\'); }
u8 CharCorrectSlash(u8 c)
{
    if (CharIsSlash(c))
    {
        c = '/';
    }
    return c;
}

//////////////////////////////
// Creating Strings
//
string Str8(u8 *str, u64 size)
{
    string result;
    result.str  = str;
    result.size = size;
    return result;
}
inline string Substr8(string str, u64 min, u64 max)
{
    if (max > str.size)
    {
        max = str.size;
    }
    if (min > str.size)
    {
        min = str.size;
    }
    if (min > max)
    {
        Swap(min, max);
    }
    str.size = max - min;
    str.str += min;
    return str;
}
u64 CalculateCStringLength(const char *cstr)
{
    u64 length = 0;
    for (; cstr[length]; length++)
    {
    }
    return length;
}

string PushStr8FV(Arena *arena, const char *fmt, va_list args)
{
    string result = {};
    va_list args2;
    va_copy(args2, args);
    u64 neededBytes = stbsp_vsnprintf(0, 0, fmt, args) + 1;
    result.str      = PushArray(arena, u8, neededBytes);
    result.size     = neededBytes - 1;
    stbsp_vsnprintf((char *)result.str, (int)neededBytes, fmt, args2);
    return result;
}

string PushStr8Copy(Arena *arena, string str)
{
    string res;
    res.size = str.size;
    res.str  = PushArrayTagged(arena, u8, str.size + 1, MemoryType_String);
    MemoryCopy(res.str, str.str, str.size);
    res.str[str.size] = 0;
    return res;
}

string PushStr8F(Arena *arena, const char *fmt, ...)
{
    string result = {};
    va_list args;
    va_start(args, fmt);
    result = PushStr8FV(arena, fmt, args);
    va_end(args);
    return result;
}

string StrConcat(Arena *arena, string s1, string s2)
{
    string result;
    result.size = s1.size + s2.size;
    result.str  = PushArrayTagged(arena, u8, result.size + 1, MemoryType_String);
    MemoryCopy(result.str, s1.str, s1.size);
    MemoryCopy(result.str + s1.size, s2.str, s2.size);
    result.str[result.size] = 0;
    return result;
}

//////////////////////////////
// Finding Strings
//

// NOTE: assumes string a already has a backing buffer of size at least b.size. If need to
// copy, use PushStr8Copy()
void StringCopy(string *out, string in)
{
    u8 *ptr = out->str;
    for (u64 i = 0; i < in.size; i++)
    {
        *ptr++ = *in.str++;
    }
    out->size = in.size;
}

b32 string::operator==(const string &a) const
{
    b32 result = false;
    if (size == a.size)
    {
        for (int i = 0; i < a.size; i++)
        {
            result = (str[i] == a.str[i]);
            if (!result)
            {
                break;
            }
        }
    }
    return result;
}

b32 string::operator==(const string &a)
{
    b32 result = false;
    if (size == a.size)
    {
        for (int i = 0; i < a.size; i++)
        {
            result = (str[i] == a.str[i]);
            if (!result)
            {
                break;
            }
        }
    }
    return result;
}

b32 string::operator==(const char *text)
{
    const string test = Str8C(text);
    return *this == test;
}

string SkipWhitespace(string str)
{
    u32 start = 0;
    for (u32 i = 0; i < str.size; i++)
    {
        start = i;
        if (!CharIsWhitespace(str.str[i]))
        {
            break;
        }
    }
    string result = Substr8(str, start, str.size);
    return result;
}

b32 StartsWith(string a, string b)
{
    b32 result = true;
    if (a.size >= b.size)
    {
        for (i32 i = 0; i < b.size; i++)
        {
            if (a.str[i] != b.str[i])
            {
                result = false;
                break;
            }
        }
    }
    return result;
}

b32 MatchString(string a, string b, MatchFlags flags)
{
    b32 result = 0;
    if (a.size == b.size || flags & MatchFlag_RightSideSloppy)
    {
        result               = 1;
        u64 size             = Min(a.size, b.size);
        b32 caseInsensitive  = (flags & MatchFlag_CaseInsensitive);
        b32 slashInsensitive = (flags & MatchFlag_SlashInsensitive);
        for (u64 i = 0; i < size; i++)
        {
            u8 charA = a.str[i];
            u8 charB = b.str[i];
            if (caseInsensitive)
            {
                charA = CharToLower(charA);
                charB = CharToLower(charB);
            }
            if (slashInsensitive)
            {
                charA = CharCorrectSlash(charA);
                charB = CharCorrectSlash(charB);
            }
            if (charA != charB)
            {
                result = 0;
                break;
            }
        }
    }
    return result;
}

u64 FindSubstring(string haystack, string needle, u64 startPos, MatchFlags flags)
{
    u64 foundIndex = haystack.size;
    for (u64 i = startPos; i < haystack.size; i++)
    {
        if (i + needle.size <= haystack.size)
        {
            string substr = Substr8(haystack, i, i + needle.size);
            if (MatchString(substr, needle, flags))
            {
                foundIndex = i;
                if (!(flags & MatchFlag_FindLast))
                {
                    break;
                }
                break;
            }
        }
    }
    return foundIndex;
}

b8 Contains(string haystack, string needle, MatchFlags flags)
{
    u64 index = FindSubstring(haystack, needle, 0, flags);
    return index != haystack.size;
}

u32 ConvertToUint(string word)
{
    u32 result = 0;
    while (CharIsDigit(*word.str))
    {
        result *= 10;
        result += *word.str++ - '0';
    }
    return result;
}

string SkipToNextWord(string line)
{
    u32 index;
    for (index = 0; index < line.size && line.str[index] != ' '; index++) continue;
    Assert(index + 1 < line.size && line.str[index + 1] != ' ');
    line = Substr8(line, index + 1, line.size);
    return line;
}

string GetFirstWord(string line)
{
    u32 index;
    u32 startIndex = 0;
    for (; startIndex < line.size && line.str[startIndex] == ' '; startIndex++) continue;
    for (index = startIndex; index < line.size && line.str[index] != ' '; index++) continue;
    line = Substr8(line, startIndex, index);
    return line;
}

// NOTE: assumes words are separated by one space, and that lines don't start with a space
// NOT zero-indexed. returns last word if n is too big
string GetNthWord(string line, u32 n)
{
    u32 index;
    u32 start = 0;
    i32 end   = -1;
    for (index = 0; index < line.size && n; index++)
    {
        if (line.str[index] == ' ')
        {
            n--;
            start = end + 1;
            end   = index;
        }
    }
    if (n)
    {
        start = end + 1;
        end   = index;
    }

    return Substr8(line, start, end);
}

//////////////////////////////
// File path helpers
//
string GetFileExtension(string str)
{
    for (u64 size = str.size; size > 0;)
    {
        size--;
        if (str.str[size] == '.')
        {
            u64 amt = Min(size + 1, str.size);
            str.str += amt;
            str.size -= amt;
            break;
        }
    }
    return str;
}

string RemoveFileExtension(string str)
{
    for (u64 size = str.size; size > 0;)
    {
        size--;
        if (str.str[size] == '.')
        {
            str.size = Min(size, str.size);
            break;
        }
    }
    return str;
}

string Str8PathChopLastSlash(string string)
{
    u64 onePastLastSlash = string.size;
    for (u64 count = 0; count < string.size; count++)
    {
        if (CharIsSlash(string.str[count]))
        {
            onePastLastSlash = count;
        }
    }
    string.size = onePastLastSlash;
    return string;
}

string Str8PathChopPastLastSlash(string string)
{
    // TODO: implement find substring
    u64 onePastLastSlash = string.size;
    for (u64 count = 0; count < string.size; count++)
    {
        if (CharIsSlash(string.str[count]))
        {
            onePastLastSlash = count + 1;
        }
    }
    string.size = onePastLastSlash;
    return string;
}

string PathSkipLastSlash(string str)
{
    for (u64 size = str.size; size > 0;)
    {
        size--;
        if (CharIsSlash(str.str[size]))
        {
            u64 amt = Min(size + 1, str.size);
            str.str += amt;
            str.size -= amt;
            break;
        }
    }
    return str;
}

bool IsInt(string str)
{
    for (u64 i = 0; i < str.size; i++)
    {
        if (!CharIsDigit(str.str[i])) return false;
    }
    return true;
}

//////////////////////////////
// Hash
//
i32 HashFromString(string string)
{
#if 0
    i32 result = 5381;
    for (u64 i = 0; i < string.size; i += 1)
    {
        result = ((result << 5) + result) + string.str[i];
    }
#endif
    i32 result = 0;
    for (u64 i = 0; i < string.size; i++)
    {
        result += (string.str[i]) * ((i32)i + 119);
    }
    return result;
}

u64 HashStruct_(void *ptr, u64 size)
{
    string str;
    str.str    = (u8 *)(ptr);
    str.size   = size;
    u64 result = HashFromString(str);
    return result;
}

//////////////////////////////
// String reading
//
inline b32 Advance(Tokenizer *tokenizer, string check)
{
    string token;
    if ((u64)(tokenizer->input.str + tokenizer->input.size - tokenizer->cursor) < check.size)
    {
        return false;
    }
    token.size = check.size;
    token.str  = tokenizer->cursor;

    if (token == check)
    {
        tokenizer->cursor += check.size;
        return true;
    }
    return false;
}

inline void Advance(Tokenizer *tokenizer, size_t size)
{
    if (tokenizer->cursor + size <= tokenizer->input.str + tokenizer->input.size)
    {
        tokenizer->cursor += size;
    }
    else
    {
        tokenizer->cursor = tokenizer->input.str + tokenizer->input.size;
    }
}

inline u8 *GetCursor_(Tokenizer *tokenizer)
{
    u8 *result = tokenizer->cursor;
    return result;
}

inline b32 EndOfBuffer(Tokenizer *tokenizer)
{
    b32 result = tokenizer->cursor >= tokenizer->input.str + tokenizer->input.size;
    return result;
}

inline b32 IsAlpha(Tokenizer *tokenizer) { return CharIsAlpha(tokenizer->cursor[0]); }

inline b32 IsDigit(Tokenizer *tokenizer) { return CharIsDigit(tokenizer->cursor[0]); }

inline b32 IsAlphaNumeric(Tokenizer *tokenizer)
{
    return IsAlpha(tokenizer) || IsDigit(tokenizer);
}

inline b32 IsBlank(Tokenizer *tokenizer) { return CharIsBlank(tokenizer->cursor[0]); }

// TODO: carriage returns?
string ReadLine(Tokenizer *tokenizer)
{
    string result;
    result.str  = tokenizer->cursor;
    result.size = 0;

    while (!EndOfBuffer(tokenizer) &&
           (tokenizer->cursor++,
            (*(tokenizer->cursor - 1) != '\n' && (*(tokenizer->cursor - 1) != '\r'))))
    {
        result.size++;
    }
    return result;
}

string ReadWord(Tokenizer *tokenizer)
{
    string result;
    result.str  = tokenizer->cursor;
    result.size = 0;

    while (!EndOfBuffer(tokenizer) &&
           (tokenizer->cursor++,
            (*(tokenizer->cursor - 1) != ' ' && *(tokenizer->cursor - 1) != '\n' &&
             (*(tokenizer->cursor - 1) != '\r'))))
    {
        result.size++;
    }
    return result;
}

inline void SkipToNextChar(Tokenizer *tokenizer)
{
    while (!EndOfBuffer(tokenizer) && IsBlank(tokenizer)) tokenizer->cursor++;
}

string ReadWordAndSkipToNextWord(Tokenizer *tokenizer)
{
    Assert(CharIsAlpha(*tokenizer->cursor));
    string result;
    result.str  = tokenizer->cursor;
    result.size = 0;

    while (!EndOfBuffer(tokenizer) && !CharIsBlank(*tokenizer->cursor))
    {
        tokenizer->cursor++;
        result.size++;
    }
    SkipToNextChar(tokenizer);
    return result;
}

string ReadBytes(Tokenizer *tokenizer, u64 numBytes)
{
    string result;
    result.str = tokenizer->cursor;
    result.size =
        Min(numBytes, (u64)(tokenizer->input.str + tokenizer->input.size - tokenizer->cursor));
    tokenizer->cursor += result.size;
    return result;
}

string CheckWord(Tokenizer *tokenizer)
{
    string result;
    result.str  = tokenizer->cursor;
    result.size = 0;

    while (!EndOfBuffer(tokenizer) && !CharIsBlank(*tokenizer->cursor))
    {
        tokenizer->cursor++;
        result.size++;
    }
    return result;
}

b8 GetBetweenPair(string &out, const string line, const u8 ch)
{
    u8 left  = ch;
    u8 right = CharGetPair(ch);

    i32 startIndex = -1;
    u32 endIndex   = (u32)line.size;
    for (u32 i = 0; i < line.size; i++)
    {
        if (line.str[i] == left && startIndex == -1)
        {
            startIndex = i;
        }
        else if (line.str[i] == right && startIndex != -1)
        {
            endIndex = i;
            break;
        }
    }
    if (startIndex != -1)
    {
        out = Substr8(line, (u32)startIndex + 1, endIndex - 1);
        return 1;
    }
    return 0;
}

inline f32 ReadFloat(Tokenizer *iter)
{
    f32 value    = 0;
    i32 exponent = 0;
    u8 c;
    b32 valueSign = (*iter->cursor == '-');
    if (valueSign || *iter->cursor == '+')
    {
        iter->cursor++;
    }
    while (CharIsDigit((c = *iter->cursor++)))
    {
        value = value * 10.0f + (c - '0');
    }
    if (c == '.')
    {
        while (CharIsDigit((c = *iter->cursor++)))
        {
            value = value * 10.0f + (c - '0');
            exponent -= 1;
        }
    }
    if (c == 'e' || c == 'E')
    {
        i32 sign = 1;
        i32 i    = 0;
        c        = *iter->cursor++;
        sign     = c == '+' ? 1 : -1;
        c        = *iter->cursor++;
        while (CharIsDigit(c))
        {
            i = i * 10 + (c - '0');
            c = *iter->cursor++;
        }
        exponent += i * sign;
    }
    while (exponent > 0)
    {
        value *= 10.0f;
        exponent--;
    }
    while (exponent < 0)
    {
        value *= 0.1f;
        exponent++;
    }
    if (valueSign)
    {
        value = -value;
    }
    return value;
}

inline i32 ReadInt(Tokenizer *iter)
{
    b32 valueSign = (*iter->cursor == '-');
    if (valueSign) iter->cursor++;

    i32 result = 0;
    while (CharIsDigit(*iter->cursor))
    {
        result *= 10;
        result += *iter->cursor++ - '0';
    }
    return valueSign ? -result : result;
}

inline u32 ReadUint(Tokenizer *iter)
{
    u32 result = 0;
    while (CharIsDigit(*iter->cursor))
    {
        result *= 10;
        result += *iter->cursor++ - '0';
    }
    return result;
}

inline void SkipToNextDigit(Tokenizer *tokenizer)
{
    while (!EndOfBuffer(tokenizer) && (!IsDigit(tokenizer) && *tokenizer->cursor != '-'))
        tokenizer->cursor++;
}

inline void SkipToNextLine(Tokenizer *iter)
{
    while (!EndOfBuffer(iter) && *iter->cursor != '\n')
    {
        iter->cursor++;
    }
    iter->cursor++;
}

inline void SkipToNextWord(Tokenizer *iter)
{
    bool findChar = false;
    while (!EndOfBuffer(iter) && !(findChar && CharIsAlpha(*iter->cursor)))
    {
        if (!findChar && *iter->cursor == ' ') findChar = true;
        iter->cursor++;
    }
}

void Get(Tokenizer *tokenizer, void *ptr, u32 size)
{
    Assert(tokenizer->cursor + size <= tokenizer->input.str + tokenizer->input.size);
    MemoryCopy(ptr, tokenizer->cursor, size);
    Advance(tokenizer, size);
}

inline u8 *GetPointer_(Tokenizer *tokenizer)
{
    u64 offset;
    GetPointerValue(tokenizer, &offset);
    u8 *result = tokenizer->input.str + offset;
    return result;
}

b32 Compare(u8 *ptr, string str)
{
    Assert(ptr);
    return memcmp(ptr, str.str, str.size) == 0;
}

// NOTE: counts the number of lines starting with ch, ignoring whitespace
// TODO: hardcoded
inline u32 CountLinesStartWith(Tokenizer *tokenizer, u8 ch)
{
    u8 *cursor = tokenizer->cursor;
    while (CharIsBlank(*cursor)) cursor++;
    u32 count      = 0;
    bool isComment = false;

    b8 left  = false;
    b8 right = false; //']';
    while (cursor[0] == ch || cursor[0] == '#' || left != right)
    {
        isComment = cursor[0] == '#';
        while (*cursor != '\n')
        {
            if (*cursor == ']')
            {
                right = true;
            }
            else if (*cursor == '[')
            {
                left = true;
            }
            cursor++;
            continue;
        }
        if (!isComment && (left == right))
        {
            count++;
            left = right = false;
        }
        while (CharIsBlank(*cursor)) cursor++;
    }
    return count;
}

b8 GetBetweenPair(string &out, Tokenizer *tokenizer, const u8 ch)
{
    u8 left  = ch;
    u8 right = CharGetPair(ch);

    u8 *cursor = tokenizer->cursor;
    for (; cursor < tokenizer->input.str + tokenizer->input.size && CharIsBlank(*cursor) &&
           *cursor != '#' && *cursor != left;)
    {
        if (*cursor == '\n') return 0;
        cursor++;
    }
    if (*cursor == '#') return 2;
    if (cursor >= tokenizer->input.str + tokenizer->input.size || *cursor != left) return 0;
    cursor++;

    u8 *start = cursor;
    u32 count = 0;
    for (; cursor < tokenizer->input.str + tokenizer->input.size && *cursor != right; count++)
    {
        // if (*cursor == '\n') return 0;
        cursor++;
    }
    cursor++;

    if (cursor > tokenizer->input.str + tokenizer->input.size) return 0;

    out               = Str8(start, count);
    tokenizer->cursor = cursor;
    return 1;
}

u32 CountBetweenPair(Tokenizer *tokenizer, const u8 ch)
{
    u8 left  = ch;
    u8 right = CharGetPair(ch);

    u8 *cursor = tokenizer->cursor;
    for (; *cursor != '\n' && *cursor++ != left;) continue;
    if (*(cursor - 1) != left || *cursor == right) return 0;

    for (; CharIsBlank(*cursor);) cursor++;

    u32 count = 1;
    for (; *cursor != right;)
    {
        if (*cursor == ' ')
        {
            if (*(cursor + 1) != right) count++;
            while (CharIsBlank(*cursor)) cursor++;
        }
        else
        {
            cursor++;
        }
    }

    return count;
}

b32 IsEndOfLine(Tokenizer *tokenizer) { return (*tokenizer->cursor == '\n'); }

//////////////////////////////
// Global string table
//
constexpr u32 operator""_sid(const char *ptr, size_t count)
{
    return MurmurHash32((const char *)ptr, (int)count, 0);
}

#if 0
u32 stringIds[1024];
std::atomic<u32> stringIdCount;

u32 Hash(string str)
{
    u32 result = 0;
    for (u64 i = 0; i < str.size; i++)
    {
        result += (str.str[i]) * ((i32)i + 119);
    }
    return result;
}

u32 HashCombine(u32 hash1, u32 hash2)
{
    hash1 ^= hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2);
    return hash1;
}

u32 AddSID(string str)
{
    u32 sid = Hash(str);
#ifdef INTERNAL
    for (u32 i = 0; i < stringIdCount.load(); i++)
    {
        if (sid == stringIds[i])
        {
            Printf("Collision: string %S, sid %i\n", str, sid);
            Assert(!"Hash collision");
        }
    }
#endif
    u32 index        = stringIdCount.fetch_add(1);
    stringIds[index] = sid;
    return sid;
}

u32 GetSID(string str)
{
    u32 sid = Hash(str);
    return sid;
}
#endif

//////////////////////////////
// String writing
//

u64 Put(StringBuilder *builder, void *data, u64 size)
{
    u64 cursor                        = builder->totalSize;
    StringBuilderChunkNode *chunkNode = builder->last;
    if (chunkNode == 0 || chunkNode->count >= chunkNode->cap)
    {
        chunkNode = PushStruct(builder->arena, StringBuilderChunkNode);
        QueuePush(builder->first, builder->last, chunkNode);
        chunkNode->cap    = 256;
        chunkNode->values = PushArray(builder->arena, StringBuilderNode, chunkNode->cap);
    }
    StringBuilderNode *node = &chunkNode->values[chunkNode->count++];
    node->str.str           = PushArray(builder->arena, u8, size);
    node->str.size          = size;

    builder->totalSize += size;

    MemoryCopy(node->str.str, data, size);
    return cursor;
}

u64 Put(StringBuilder *builder, string str)
{
    Assert((u32)str.size == str.size);
    u64 result = Put(builder, str.str, (u32)str.size);
    return result;
}

u64 Put(StringBuilder *builder, u32 value)
{
    u64 result = PutPointerValue(builder, &value);
    return result;
}

u64 PutU64(StringBuilder *builder, u64 value)
{
    u64 result = PutPointerValue(builder, &value);
    return result;
}

void PutLine(StringBuilder *builder, u32 indents, char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    for (u32 i = 0; i < indents; i++)
    {
        Put(builder, Str8Lit("\t"));
    }
    string result = PushStr8FV(builder->arena, fmt, args);
    Put(builder, result);
    Put(builder, Str8Lit("\n"));
    va_end(args);
}

StringBuilder ConcatBuilders(Arena *arena, StringBuilder *a, StringBuilder *b)
{
    StringBuilder result = {};
    result.first         = a->first;
    result.last          = b->last;
    a->last->next        = b->first;
    result.totalSize     = a->totalSize + b->totalSize;
    result.arena         = arena;
    return result;
}

string CombineBuilderNodes(StringBuilder *builder)
{
    string result;
    result.str  = PushArray(builder->arena, u8, builder->totalSize);
    result.size = builder->totalSize;

    u8 *cursor = result.str;
    for (StringBuilderChunkNode *node = builder->first; node != 0; node = node->next)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            StringBuilderNode *n = &node->values[i];
            MemoryCopy(cursor, n->str.str, n->str.size);
            cursor += n->str.size;
        }
    }
    return result;
}

b32 WriteEntireFile(StringBuilder *builder, string filename)
{
    string result = CombineBuilderNodes(builder);
    b32 success   = OS_WriteFile(filename, result);
    return success;
}

inline u64 PutPointer(StringBuilder *builder, u64 address)
{
    u64 offset = builder->totalSize;
    offset += sizeof(offset) + address;
    PutPointerValue(builder, &offset);
    return offset;
}

inline void ConvertPointerToOffset(u8 *buffer, u64 location, u64 offset)
{
    // MemoryCopy(buffer + location, &offset, sizeof(offset));
    *(u64 *)(buffer + location) = offset;
}

inline u8 *ConvertOffsetToPointer(u8 *base, u64 offset)
{
    u8 *ptr = base + offset;
    return ptr;
}
} // namespace rt
