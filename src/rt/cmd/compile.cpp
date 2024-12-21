#include "../memory.h"
#include "../string.h"
#include "../memory.cpp"
#include "../string.cpp"

namespace detail
{
using namespace rt;

struct HashNode
{
    string str;
    u32 id;
    HashNode *next;
};

struct SceneLoadState
{
    HashNode *map;
    u32 materialID;
};

void CheckNewMaterial(SceneLoadState *state, Arena *arena, StringBuilder *builder,
                      string materialType, string *args, u32 count)
{
    TempArena temp = ScratchStart(&arena, 1);
    i32 hash       = HashFromString(fullMaterialType);
    HashNode *node = &map[hash];
    HashNode *prev = 0;
    while (node)
    {
        if (node->str == fullMaterialType) break;
        prev = node;
        node = node->next;
    }
    if (!node)
    {
        prev->next = PushStruct(arena, HashNode);
        prev->str  = str;
        u32 matID  = state->materialID++;
        prev->id   = matID;

        string build = PushStr8F(temp.arena, "%S(%u", materialType, matID);
        for (u32 i = 0; i < count; i++)
        {
            build = PushStr8F(temp.arena, build, ", %S");
        }
        build = StrConcat(temp.arena, build, ")");
        Put(builder, build);
    }
}

void ParseTexture(SceneLoadState *state, Arena *arena, StringBuilder *builder,
                  Tokenizer *tokenizer, string materialType)
{
    while (tokenizer->cursor[0] != '_')
    {
        if (Advance(tokenizer, "Const$"))
        {
            string strType = StrConcat(arena, materialType, "Const");
            HashNode *node = GetHashNode(strType);
        }
        else if (Advance(tokenizer, "Ptex$"))
        {
        }
    }
}
void CreateDiffuseMaterial(Arena *arena, StringBuilder *builder, Tokenizer *tokenizer)
{
    TempArena temp = ScratchStart(0, 0);
    ParseTexture(arena, builder, tokenizer, "DIFFUSE_MATERIAL");
    ScratchEnd(temp);
};

void CreateDielectricMaterial(StringBuilder *builder, string textureType) {}

} // namespace detail

int main(int argc, char **argv)
{
    TempArena temp  = ScratchStart(0, 0);
    string filename = Str8C(argv[0]);
    string file     = OS_MapFileRead(filename);
    Tokenizer tokenizer;
    tokenizer.cursor = file.str;
    tokenizer.input  = file;

    StringBuilder builder;
    HashNode *map  = PushArray(temp.arena, HashNode, 1024);
    u32 materialID = 0;

    // NOTE: Material, and then texture types
    // e.g. DiffuseMaterial$Const$Ptex#
    Tokenizer tempTokenizer;
    tempTokenizer.cursor = tokenizer.cursor;
    tempTokenizer.input  = file.str;
    while (tempTokenizer.cursor[0]++ != '_');
    string str(tokenizer.cursor, u64(tokenizer.cursor - tokenizer.cursor));
    u32 matID = node->id;
    if (Advance("DiffuseMaterial$"))
    {
        CreateDiffuseMaterial(builder, tokenizer);
    }
}
