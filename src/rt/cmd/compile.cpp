#include "../base.h"
#include "../math/basemath.h"
#include "../memory.h"
#include "../string.h"
#include "../win32.h"
#include "../thread_context.h"
#include "../hash.h"
#include "../scene_load.h"
#include "../win32.cpp"
#include "../memory.cpp"
#include "../string.cpp"
#include "../thread_context.cpp"

namespace detail
{
using namespace rt;

#if 0
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
    }
}

struct DictNode
{
    string key[8];
    string value[8];
    u32 count;
    DictNode *next;
};

void ParseKeyValue(SceneLoadState *state, DictNode *node, Arena *arena, StringBuilder *builder,
                   Tokenizer *tokenizer)
{
    Assert(tokenizer->cursor[0] == '#');

    while (*tokenizer->cursor != '_')
    {
        tokenizer->cursor++;
        u8 *start = tokenizer->cursor;
        while (*tokenizer->cursor != '$')
        {
            tokenizer->cursor++;
        }

        string key(start, u64(tokenizer->cursor++ - start));

        start = tokenizer->cursor;
        while (*tokenizer->cursor != '#' && *tokenizer->cursor != '_')
        {
            tokenizer->cursor++;
        }

        string value(start, u64(tokenizer->cursor - start));

        DictNode *node = state->node;
        if (node->count > ArrayLength(node->key))
        {
            node->next = PushStruct(arena, DictNode);
            node       = node->next;
        }
        u32 index          = node->count++;
        node->key[index]   = key;
        node->value[index] = value;
    }
    tokenizer->cursor++;
}
#endif

bool GetString(Tokenizer *tokenizer, string *str, char start, char end)
{
    if (*tokenizer->cursor != start)
    {
        return false;
    }
    tokenizer->cursor++;
    u8 *begin = tokenizer->cursor;
    while (*tokenizer->cursor != end)
    {
        tokenizer->cursor++;
    }
    *str = string(begin, u64(tokenizer->cursor - begin));
    return true;
}

void ProcessMetaMaterials(StringBuilder *builder, Arena *arena, Tokenizer *tokenizer,
                          u64 checkOffset)
{
    u32 materialID = 0;
    Put(builder, "namespace rt {");
    u8 *start = tokenizer->cursor;
    while (!Advance(tokenizer, "MATERIALS_END"))
    {
        if (u64(tokenizer->cursor - start) > checkOffset)
        {
            printf("Malformed material section.\n");
            return;
        }
        string materialType;
        GetString(tokenizer, &materialType, '#', '$');
        string matString = PushStr8F(arena, "%S(%u", materialType, materialID++);
        for (;;)
        {
            string concat;
            bool result = GetString(tokenizer, &concat, '$', '$');
            Assert(result);
            if (concat == "_") break;
            matString = PushStr8F(arena, "%S, %S", matString, concat);
        }
        matString = StrConcat(arena, matString, ")");
        Put(builder, matString);
    }
    Put(builder, "}");
}

} // namespace detail

using namespace detail;
using namespace rt;

// Steps:
// 1. Convert PBRT file to my file format.
// - my file format will have a meta section containing information needed to allocate,
// as well as instantiate vtables and types used in  the program
// - sections: section offsets, meta, info, data. data contains vertices/indices/other data
// that needs to remain permanently allocated. see above for meta. info contains information
// regarding textures, materials, lights, triangle meshes in the scene. maybe contain
// another section for data that is only temporarily allocated/used to derive other data.
// 2. Compile, passing in my file format.
// - parses the meta section, generates the material structs used
// 3. Run rt
// - parses the other sections
int main(int argc, char **argv)
{
    TempArena temp  = ScratchStart(0, 0);
    string filename = Str8C(argv[0]);
    // TODO: get current working directory and append
    string file = OS_MapFileRead(filename);
    if (GetFileExtension(file) != ".rtscene")
    {
        printf("Input file extension is not .rtscene. Aborting...\n");
        return 1;
    }
    Tokenizer tokenizer;
    tokenizer.cursor = file.str;
    tokenizer.input  = file;

    // Start of file
    bool result = Advance(&tokenizer, "RTF_START");
    if (!result)
    {
        printf("Magic number is not RTF_START. Aborting... \n");
        return 1;
    }
    // NOTE: offsets from start of file
    FileOffsets offsets;
    if (!GetSectionOffsets(&tokenizer, &offsets))
    {
        printf("File missing section offset data. Aborting...\n");
        return 1;
    }

    tokenizer.cursor = file.str + offsets.metaOffset;
    u32 materialID   = 0;
    // Meta section
    Assert(offsets.metaOffset != 0);
    result = Advance(&tokenizer, "META_START");
    if (!result)
    {
        printf("Meta section specified, but META_START tag not found. Aborting... \n");
        return 1;
    }

    while (!Advance(&tokenizer, "META_END"))
    {
        if (tokenizer.cursor - file.str > offsets.infoOffset)
        {
            printf("Meta section badly formed. Aborting... \n");
            return 1;
        }

        if (Advance(&tokenizer, "MATERIALS_START"))
        {
            StringBuilder builder;
            builder.arena = temp.arena;
            ProcessMetaMaterials(&builder, temp.arena, &tokenizer, offsets.infoOffset);
            WriteEntireFile(&builder, "../src/gen/shaders.cpp");
        }
        else if (Advance(&tokenizer, "TEXTURES_START"))
        {
        }
        else if (Advance(&tokenizer, "LIGHTS_START"))
        {
        }
    }

#if 0
    // Generate material file

    StringBuilder builder;
    HashNode *map = PushArray(temp.arena, HashNode, 1024);

    // Parse meta fields
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
#endif
}
