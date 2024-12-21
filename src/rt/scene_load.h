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
} // namespace rt

#endif
