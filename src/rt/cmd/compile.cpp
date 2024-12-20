#include "../string.h"
#include "../string.cpp"

namespace rt
{
void CreateDiffuseMaterial(StringBuilder *builder, u32 materialIndex, u32 textureIndex)
{
    TempArena temp = ScratchStart(0, 0);
    // clang-format off
    string function =
        PushStr8F(temp.arena, "struct Material%u {"
                              "DiffuseBxDF GetBxDF(SurfaceInteractions &intr, Vec4lfn &filterWidths, SampledWavelengthsN &lambda) { "
                              "SampledSpectrumN sampledSpectra = textureFuncs[%u](intr, ???, filterWidths, lambda);"
                              "return DiffuseBxDF(sampledSpectra); } ", materialIndex, textureIndex);
    // clang-format on
    Put(builder, function);
    ScratchEnd(temp);
};

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
