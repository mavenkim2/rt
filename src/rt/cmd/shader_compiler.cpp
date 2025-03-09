#include "../base.h"
#include "../memory.h"
#include "../string.h"
#include "shader_compiler.h"

#include "../string.cpp"

namespace shadercompiler
{

static ShaderCompiler *compiler;

enum class ShaderStage
{
    Vertex,
    Geometry,
    Fragment,
    Compute,
    Raygen,
    Miss,
    Hit,
    Intersect,
    Count,
};

void InitShaderCompiler()
{
    compiler                     = new ShaderCompiler();
    compiler->dll.mSource        = "src/third_party/dxcompiler.dll";
    compiler->dll.mFunctions     = (void **)&compiler->functions;
    compiler->dll.mFunctionNames = (char **)functionTable;
    compiler->dll.mFunctionCount = ArrayLength(functionTable);

    platform.LoadDLLNoTemp(&compiler->dll);
    Assert(compiler->dll.mValid);

    HRESULT hr = compiler->functions.DxcCreateInstance(CLSID_DxcCompiler,
                                                       IID_PPV_ARGS(&compiler->dxcCompiler));
    Assert(SUCCEEDED(hr));

    hr = compiler->dxcCompiler->QueryInterface(IID_PPV_ARGS(&compiler->info));
    Assert(SUCCEEDED(hr));

    hr = compiler->functions.DxcCreateInstance(CLSID_DxcUtils,
                                               IID_PPV_ARGS(&compiler->dxcUtils));
    Assert(SUCCEEDED(hr));
    hr = compiler->functions.DxcCreateInstance(CLSID_DxcCompiler,
                                               IID_PPV_ARGS(&compiler->dxcCompiler));
    Assert(SUCCEEDED(hr));
    Assert(compiler->dxcCompiler);

    hr = compiler->dxcUtils->CreateDefaultIncludeHandler(&compiler->defaultIncludeHandler);
    Assert(SUCCEEDED(hr));
}

wchar_t *ConvertToWide(Arena *arena, string input)
{
    // u32 wideSize    = (u32)(input.size * 2);
    i32 wideSize = MultiByteToWideChar(CP_UTF8, 0, (char *)input.str, -1, 0, 0);
    Assert(wideSize > 0);
    wchar_t *out   = (wchar_t *)PushArray(arena, wchar_t, wideSize);
    u32 resultSize = MultiByteToWideChar(CP_UTF8, 0, (char *)input.str, -1, out, wideSize);
    return out;
}

string ConvertFromWide(Arena *arena, const wchar_t *str)
{
    i32 size = WideCharToMultiByte(CP_UTF8, 0, str, -1, 0, 0, 0, 0);
    Assert(size > 0);
    string out;
    out.size = size - 1;
    WideCharToMultiByte(CP_UTF8, 0, str, -1, (char *)out.str, size, 0, 0);
    return out;
}

void CompileShader(Arena *arena, CompileInput *input, CompileOutput *output)
{
    TempArena temp = ScratchStart(0, 0);
    CComPtr<IDxcBlobUtf8> blob;

    CComPtr<IDxcIncludeHandler> dxcIncludeHandler;
    compiler->dxcUtils->CreateDefaultIncludeHandler(&dxcIncludeHandler);

    std::vector<const wchar_t *> args;
    args.push_back(L"nologo");
    args.push_back(L"-spirv");
    args.push_back(L"-fspv-target-env=vulkan1.3");
    // Shift register slots
    // b, s, t, u
    // args.push_back(L"-I");
    // args.push_back(ConvertToWide(temp.arena, shaderDirectory));
    for (u32 i = 0; i < input->numDefines; i++)
    {
        CompileInput::ShaderDefine *define = &input->defines[i];
        args.push_back(L"-D");
        args.push_back(ConvertToWide(temp.arena, define->val));
    }
    args.push_back(L"-fvk-s-shift");
    args.push_back(L"100"); // NOTE: if changed, the respective constants in mkgraphicsvulkan
                            // must be changed as well
    args.push_back(L"0");

    args.push_back(L"-fvk-t-shift");
    args.push_back(L"200");
    args.push_back(L"0");

    args.push_back(L"-fvk-u-shift");
    args.push_back(L"300");
    args.push_back(L"0");

    args.push_back(L"-T");
    switch (input->stage)
    {
        case ShaderStage::Vertex:
        {
            args.push_back(L"vs_6_6");
        }
        break;
        case ShaderStage::Fragment:
        {
            args.push_back(L"ps_6_6");
        }
        break;
        case ShaderStage::Compute:
        {
            args.push_back(L"cs_6_6");
        }
        break;
        case ShaderStage::Raygen:
        {
            args.push_back(L"lib_6_6");
        }
        break;
        default: Assert(0);
    }

    // Output target name
    args.push_back(L"-Fo");
    string outputName;
    if (input->outName.size != 0)
    {
        outputName = PushStr8F(temp.arena, "%S.spv\0", input->outName);
    }
    else
    {
        outputName = PushStr8F(temp.arena, "%S.spv\0", RemoveFileExtension(input->shaderName));
    }
    args.push_back(ConvertToWide(temp.arena, outputName));

    // Final argument is the source filename
    string filePath = PushStr8F(temp.arena, "%S%S\0", shaderDirectory, input->shaderName);
    args.push_back(ConvertToWide(temp.arena, filePath));

    string shaderCode = platform.ReadEntireFile(temp.arena, filePath);
    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr      = shaderCode.str;
    sourceBuffer.Size     = shaderCode.size;
    sourceBuffer.Encoding = 0;

    struct CustomIncludeHandler : public IDxcIncludeHandler
    {
        Arena *arena;
        CompileInput *input;
        CompileOutput *output;
        CComPtr<IDxcIncludeHandler> dxcIncludeHandler;
        HRESULT STDMETHODCALLTYPE
        LoadSource(_In_ LPCWSTR pFilename,
                   _COM_Outptr_result_maybenull_ IDxcBlob **ppIncludeSource) override
        {
            HRESULT hr = dxcIncludeHandler->LoadSource(pFilename, ppIncludeSource);
            if (SUCCEEDED(hr))
            {
                string dependency = ConvertFromWide(arena, pFilename);
                output->dependencies.push_back(dependency);
            }
            return hr;
        }

        HRESULT STDMETHODCALLTYPE
        QueryInterface(REFIID riid, _COM_Outptr_ void __RPC_FAR *__RPC_FAR *ppvObject) override
        {
            return dxcIncludeHandler->QueryInterface(riid, ppvObject);
        }

        ULONG STDMETHODCALLTYPE AddRef(void) override { return 0; }
        ULONG STDMETHODCALLTYPE Release(void) override { return 0; }
    } customIncludeHandler;
    customIncludeHandler.arena             = arena;
    customIncludeHandler.dxcIncludeHandler = compiler->defaultIncludeHandler;
    customIncludeHandler.input             = input;
    customIncludeHandler.output            = output;

    CComPtr<IDxcResult> compileResult;
    compiler->dxcCompiler->Compile(&sourceBuffer, args.data(), (u32)args.size(),
                                   &customIncludeHandler, IID_PPV_ARGS(&compileResult));

    CComPtr<IDxcBlobUtf8> errors = 0;
    HRESULT hr = compileResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), 0);
    Assert(SUCCEEDED(hr));
    if (errors != 0 && errors->GetStringLength() != 0)
    {
        printf("Shader compile error: %s\n", errors->GetStringPointer());
        Assert(0);
    }

    CComPtr<IDxcBlob> shader = 0;
    hr = compileResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader), 0);
    Assert(SUCCEEDED(hr));
    if (shader != 0)
    {
        output->dependencies.push_back(input->shaderName);
        output->shaderData.size = shader->GetBufferSize();
        output->shaderData.str  = PushArray(arena, u8, output->shaderData.size);
        MemoryCopy(output->shaderData.str, shader->GetBufferPointer(),
                   output->shaderData.size);
    }
    // customIncludeHandler.dxcIncludeHandler->Release();
}

using namespace rt;

void main(int argc, char *argv[])
{
    Arena *arena = ArenaAlloc();
    InitShaderCompiler();
    CompileInput input;
    input.stage      = ShaderStage::Raygen;
    input.shaderName = "render_raytrace_rgen.hlsl";
    input.outName    = PushStr8F(arena, "%S.spv", RemoveFileExtension(input.shaderName));
    CompileOutput output;

    input.CompileShader(arena, &input, &output);
}

} // namespace shadercompiler
