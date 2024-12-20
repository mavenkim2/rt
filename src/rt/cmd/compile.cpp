#include "../string.h"
#include "../string.cpp"

namespace rt
{
void CreateDiffuseMaterial(StringBuilder *builder, string textureType, string textureFunc,
                           u32 &numDiffuseTypes)
{
    TempArena temp  = ScratchStart(0, 0);
    string function = PushStr8F(temp.arena, "DIFFUSE_MATERIAL(%u, %S, %S)", numDiffuseTypes++,
                                textureType, textureFunc);
    Put(builder, function);
    ScratchEnd(temp);
};

void CreateDielectricMaterial(StringBuilder *builder, string textureType) {}

void CreateTextures()
{
    while (tokenizer.cursor[0] != '#')
    {
        if (Advance(&tokenizer, "Const$"))
        {
        }
        else if (Advance(&tokenizer, "Ptex$"))
        {
        }
    }
}

} // namespace rt

using namespace rt;
int main(int argc, char **argv)
{
    string filename = Str8C(argv[0]);
    string file     = OS_MapFileRead(filename);
    Tokenizer tokenizer;
    tokenizer.cursor = file.str;
    tokenizer.input  = file;

    StringBuilder builder;

    // NOTE: Material, and then texture types
    // e.g. DiffuseMaterial$Const$Ptex#
    while (tokenizer.cursor[0] != '_')
    {
        if (Advance(&tokenizer, "DiffuseMaterial$"))
        {
            CreateTextures();
        }
        else if (Advance(&toknizer, "DielectricMaterial"))
        {
            CreateTextures();
        }
        else if (Advance(&toknizer, "DiffuseTransmission"))
        {
            CreateTextures();
        }
        else if (Advance(&toknizer, "CoatedDiffuse"))
        {
            CreateTextures();
        }
    }
}
