#ifndef SCENE_LOAD_H
#define SCENE_LOAD_H

namespace rt
{
struct FileOffsets
{
    u64 metaOffset;
    u64 infoOffset;
    u64 dataOffset;
};

bool GetSectionOffsets(Tokenizer *tokenizer, FileOffsets *offsets)
{
    bool result = Advance(tokenizer, "META");
    if (!result) return false;
    GetPointerValue(tokenizer, &offsets->metaOffset);
    result = Advance(tokenizer, "INFO");
    if (!result) return false;
    GetPointerValue(tokenizer, &offsets->infoOffset);
    result = Advance(tokenizer, "DATA");
    if (!result) return false;
    GetPointerValue(tokenizer, &offsets->dataOffset);
    return true;
}

enum class DataType
{
    Float,
    Vec2,
    Vec3,
    Int,
    Bool,
    String,
    Blackbody,
    Spectrum,
};

struct ScenePacket
{
    StringId *parameterNames;
    u8 **bytes;
    u32 *sizes;
    DataType *types;

    StringId type;

    // const string **parameterNames;
    // SceneByteType *types;
    u32 parameterCount;

    void Initialize(Arena *arena, u32 count)
    {
        // parameterCount = count;
        // parameterCount = 0;
        parameterNames = PushArray(arena, StringId, count);
        bytes          = PushArray(arena, u8 *, count);
        sizes          = PushArray(arena, u32, count);
        // types          = PushArray(arena, SceneByteType, count);
    }

    inline i32 GetInt(i32 i) const { return *(i32 *)(bytes[i]); }
    inline bool GetBool(i32 i) const { return *(bool *)(bytes[i]); }
    inline f32 GetFloat(i32 i) const { return *(f32 *)(bytes[i]); }
    inline i32 FindKey(StringID parameterName)
    {
        for (u32 i = 0; i < parameterCount; i++)
        {
            if (parameterNames[i] == parameterName) return i;
        }
        return -1;
    }
    // inline u8 *GetByParamName(const string &name) const
    // {
    //     for (u32 i = 0; i < parameterCount; i++)
    //     {
    //         if (*parameterNames[i] == name)
    //         {
    //             return bytes[i];
    //         }
    //     }
    //     return 0;
    // }
};
} // namespace rt

#endif
