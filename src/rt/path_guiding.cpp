#include "path_guiding.h"
#include "../thread_context.h"
#include "shader_interop/path_guiding_shaderinterop.h"

namespace rt
{

PathGuider::PathGuider(Arena *arena)
{
    string shaderName = "../src/shaders/wem.spv";
    string shaderData = OS_ReadFile(arena, shaderName);
    Shader shader =
        device->CreateShader(ShaderStage::Compute, "weighted expectation step", shaderData);
}

void PathGuiding()
{
    ScratchArena scratch;
    StaticArray<PathGuidingSample> samples(scratch.temp.arena, 32);
    for (u32 i = 0; i < 64; i++)
    {
        PathGuidingSample sample;
        sample.vmmIndex = i % 32;
        samples.Push(sample);
    }
}

} // namespace rt
