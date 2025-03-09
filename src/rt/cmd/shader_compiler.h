#ifndef SHADER_COMPILER_H
#define SHADER_COMPILER_H

#include "../platform.h"
#include "../../third_party/dxcapi.h"

#ifdef _WIN32
#include <wrl/client.h>
#define CComPtr Microsoft::WRL::ComPtr
#endif

namespace rt
{

struct Arena;

static string shaderDirectory = "src/shaders/\0";
const char *functionTable[]   = {"DxcCreateInstance"};

struct ShaderCompiler
{
    CComPtr<IDxcCompiler3> dxcCompiler;
    CComPtr<IDxcVersionInfo> info;
    CComPtr<IDxcUtils> dxcUtils;
    CComPtr<IDxcIncludeHandler> defaultIncludeHandler;

    OS_DLL dll;

    struct FunctionTable
    {
        DxcCreateInstanceProc DxcCreateInstance;
    } functions;

    void Destroy()
    {
        // dxcCompiler->Release();
        // info->Release();
        // dxcUtils->Release();
        // defaultIncludeHandler->Release();
    }

    ShaderCompiler() {}
    ~ShaderCompiler() {}
};

struct CompileInput
{
    ShaderStage stage;
    string shaderName;
    string outName;
    struct ShaderDefine
    {
        string val;
    };
    ShaderDefine *defines;
    u32 numDefines;
};

struct CompileOutput
{
    string shaderData;
    std::vector<string> dependencies;
};

void InitShaderCompiler();
wchar_t *ConvertToWide(Arena *arena, string input);
string ConvertFromWide(Arena *arena, const wchar_t *str);
void CompileShader(Arena *arena, CompileInput *input, CompileOutput *output);

} // namespace rt

#endif
