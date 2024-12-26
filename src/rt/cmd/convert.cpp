#include "../base.h"
#include "../template.h"
#include "../math/basemath.h"
#include "../math/simd_include.h"
#include "../math/vec2.h"
#include "../math/vec3.h"
#include "../math/vec4.h"
#include "../math/bounds.h"
#include "../math/matx.h"
#include "../math/math.h"

#include "../memory.h"
#include "../containers.h"
#include "../string.h"
#include "../win32.h"
#include "../thread_context.h"
#include "../hash.h"
#include <functional>
#include "../random.h"
#include "../bvh/parallel.h"
// #include "../handles.h"
#include "../base.cpp"
#include "../win32.cpp"
#include "../memory.cpp"
#include "../string.cpp"
#include "../thread_context.cpp"

namespace rt
{

struct TriangleMesh
{
    Vec3f *p;
    Vec3f *n;
    // Vec3f *t;
    Vec2f *uv;
    u32 *indices;
    u32 numVertices;
    u32 numIndices;
};
struct QuadMesh
{
    Vec3f *p     = 0;
    Vec3f *n     = 0;
    Vec2f *uv    = 0;
    u32 *indices = 0;
    u32 numIndices;
    u32 numVertices;
};
} // namespace rt

#include "../scene_load.h"

namespace rt
{
struct Instance
{
    // TODO: materials
    u32 id;
    // GeometryID geomID;
    u32 transformIndex;
};

#define CREATE_ENUM_AND_TYPE_PACK(packName, enumName, ...)                                    \
    using packName = TypePack<COMMA_SEPARATED_LIST(__VA_ARGS__)>;                             \
    enum class enumName                                                                       \
    {                                                                                         \
        COMMA_SEPARATED_LIST(__VA_ARGS__),                                                    \
        Max,                                                                                  \
    };                                                                                        \
    ENUM_CLASS_FLAGS(enumName)

#define COMMA_SEPARATED_LIST(...)                                                             \
    COMMA_SEPARATED_LIST_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)
#define COMMA_SEPARATED_LIST_HELPER(x, ...) EXPAND(CONCAT(RECURSE__, x)(EXPAND, __VA_ARGS__))

#define COUNT_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15,     \
                        _16, _17, _18, _19, _20, N, ...)                                      \
    N
#define COUNT_ARGS(...)                                                                       \
    EXPAND(COUNT_ARGS_IMPL(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,  \
                           6, 5, 4, 3, 2, 1))

CREATE_ENUM_AND_TYPE_PACK(PrimitiveTypes, GeometryType, QuadMesh, TriangleMesh, Instance);

struct SceneInstance
{
    StringId name;
    u32 transformIndex;
};

struct InstanceType
{
    string filename;
    u32 transformIndexStart;
    u32 transformIndexEnd;
};

struct PBRTFileInfo
{
    enum Type
    {
        Film,
        Camera,
        Sampler,
        Integrator,
        Accelerator,
        MAX,
    };
    Arena *arena;
    string filename;
    ScenePacket packets[MAX] = {};
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape> shapes;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Material> materials;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Texture> textures;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Light> lights;

    ChunkedLinkedList<InstanceType, 1024, MemoryType_Instance> fileInstances;
    u32 numInstances;

    ChunkedLinkedList<AffineSpace, 16384, MemoryType_Transform> transforms;

    PBRTFileInfo *imports[32];
    u32 numImports;

    void Init(string inFilename)
    {
        arena     = ArenaAlloc(8);
        filename  = PushStr8Copy(arena, inFilename);
        shapes    = decltype(shapes)(arena);
        materials = decltype(materials)(arena);
        textures  = decltype(textures)(arena);
        lights    = decltype(lights)(arena);

        fileInstances = decltype(fileInstances)(arena);

        transforms   = decltype(transforms)(arena);
        numInstances = 0;
    }
    void Merge(PBRTFileInfo *import)
    {
        numInstances += import->numInstances;
        shapes.Merge(&import->shapes);
        u32 transformOffset = transforms.totalCount;
        for (auto *node = import->fileInstances.first; node != 0; node = node->next)
        {
            for (u32 j = 0; j < node->count; j++)
            {
                InstanceType *instance = &node->values[j];
                instance->transformIndexStart += transformOffset;
                instance->transformIndexEnd += transformOffset;
            }
        }
        fileInstances.Merge(&import->fileInstances);
        transforms.Merge(&import->transforms);
    }
};

struct GraphicsState
{
    StringId materialId = 0;
    i32 materialIndex   = -1;
    // Mat4 transform      = Mat4::Identity();
    AffineSpace transform = AffineSpace::Identity();

    i32 transformIndex = -1;
    i32 areaLightIndex = -1;
    i32 mediaIndex     = -1;

    // ObjectInstanceType *instanceType = 0;
};

void PBRTSkipToNextChar(Tokenizer *tokenizer)
{
    for (;;)
    {
        while (!EndOfBuffer(tokenizer) && CharIsBlank(*tokenizer->cursor))
        {
            tokenizer->cursor++;
        }
        if (*tokenizer->cursor != '#') break;
        SkipToNextLine(tokenizer);
    }
}

void ReadParameters(Arena *arena, ScenePacket *packet, Tokenizer *tokenizer,
                    MemoryType memoryType, u32 additionalParameters = 0);
// NOTE: sets the camera, film, sampler, etc.
void CreateScenePacket(Arena *arena, string word, ScenePacket *packet, Tokenizer *tokenizer,
                       MemoryType memoryType, u32 additionalParameters = 0)
{
    string type;
    b32 result = GetBetweenPair(type, tokenizer, '"');
    Assert(result);
    packet->type = Hash(type);
    PBRTSkipToNextChar(tokenizer);

    ReadParameters(arena, packet, tokenizer, memoryType, additionalParameters);
}

inline void SkipToNextDigitArray(Tokenizer *tokenizer)
{
    while (!EndOfBuffer(tokenizer) &&
           (!IsDigit(tokenizer) && *tokenizer->cursor != '-' && *tokenizer->cursor != ']'))
        tokenizer->cursor++;
}

inline void AdvanceToNextParameter(Tokenizer *tokenizer)
{
    for (;;)
    {
        while (!EndOfBuffer(tokenizer) &&
               (CharIsBlank(*tokenizer->cursor) || *tokenizer->cursor == ']'))
        {
            tokenizer->cursor++;
        }
        if (*tokenizer->cursor != '#') break;
        SkipToNextLine(tokenizer);
    }
}

string ReadWordAndSkipToNextChar(Tokenizer *tokenizer)
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
    PBRTSkipToNextChar(tokenizer);
    return result;
}

void ReadParameters(Arena *arena, ScenePacket *packet, Tokenizer *tokenizer,
                    MemoryType memoryType, u32 additionalParameters)
{
    static const u32 MAX_PARAMETER_COUNT = 16;

    string infoType;
    b8 result;
    u32 numVertices = 0;
    u32 numIndices  = 0;

    u32 parameterCount = 0;

    StringId parameterNames[MAX_PARAMETER_COUNT];
    u8 *bytes[MAX_PARAMETER_COUNT];
    u32 sizes[MAX_PARAMETER_COUNT];

    for (;;)
    {
        Assert(packet->parameterCount < MAX_PARAMETER_COUNT);
        result = GetBetweenPair(infoType, tokenizer, '"');
        if (!result) break;
        string dataType      = GetFirstWord(infoType);
        u32 currentParam     = packet->parameterCount++;
        string parameterName = GetNthWord(infoType, 2);

        PBRTSkipToNextChar(tokenizer);

        u32 numValues = CountBetweenPair(tokenizer, '[');
        numValues     = numValues ? numValues : 1;
        u8 *out       = 0;
        u32 size      = 0;
        if (dataType == "float")
        {
            f32 *floats = PushArrayNoZeroTagged(arena, f32, numValues, memoryType);

            Advance(tokenizer, "[");
            SkipToNextDigit(tokenizer);
            if (numValues == 1)
            {
                floats[0] = ReadFloat(tokenizer);
            }
            else
            {
                for (u32 i = 0; i < numValues; i++)
                {
                    floats[i] = ReadFloat(tokenizer);
                    SkipToNextDigitArray(tokenizer);
                }
            }
            out  = (u8 *)floats;
            size = sizeof(f32) * numValues;
            AdvanceToNextParameter(tokenizer);
        }
        else if (dataType == "point2" || dataType == "vector2")
        {
            Assert((numValues & 1) == 0);
            Vec2f *vectors = PushArrayNoZeroTagged(arena, Vec2f, numValues / 2, memoryType);

            b32 brackets = Advance(tokenizer, "[");
            Advance(tokenizer, "[");
            SkipToNextDigit(tokenizer);
            if (numValues == 2)
            {
                for (u32 i = 0; i < numValues; i++)
                {
                    vectors[i / 2][i & 1] = ReadFloat(tokenizer);
                }
            }
            else
            {
                for (u32 i = 0; i < numValues; i++)
                {
                    vectors[i / 2][i & 1] = ReadFloat(tokenizer);
                    SkipToNextDigitArray(tokenizer);
                }
            }
            out  = (u8 *)vectors;
            size = sizeof(f32) * numValues;
            AdvanceToNextParameter(tokenizer);
        }
        else if (dataType == "rgb" || dataType == "point3" || dataType == "vector3" ||
                 dataType == "normal3" || dataType == "normal" || dataType == "vector")
        {
            Assert(numValues % 3 == 0);
            Vec3f *vectors = PushArrayNoZeroTagged(arena, Vec3f, numValues / 3, memoryType);

            b32 brackets = Advance(tokenizer, "[");
            SkipToNextDigit(tokenizer);
            if (numValues == 3)
            {
                for (u32 i = 0; i < numValues; i++)
                {
                    vectors[i / 3][i % 3] = ReadFloat(tokenizer);
                }
            }
            else
            {
                for (u32 i = 0; i < numValues; i++)
                {
                    vectors[i / 3][i % 3] = ReadFloat(tokenizer);
                    SkipToNextDigitArray(tokenizer);
                }
            }
            out  = (u8 *)vectors;
            size = sizeof(f32) * numValues;
            AdvanceToNextParameter(tokenizer);
        }
        else if (dataType == "integer")
        {
            i32 *ints    = PushArrayNoZeroTagged(arena, i32, numValues, memoryType);
            b32 brackets = Advance(tokenizer, "[");
            SkipToNextDigit(tokenizer);
            if (numValues == 1)
            {
                ints[0] = ReadInt(tokenizer);
            }
            else
            {
                for (u32 i = 0; i < numValues; i++)
                {
                    ints[i] = ReadInt(tokenizer);
                    SkipToNextDigitArray(tokenizer);
                }
            }
            out  = (u8 *)ints;
            size = sizeof(i32) * numValues;
            AdvanceToNextParameter(tokenizer);
        }
        else if (dataType == "bool")
        {
            out  = PushStructNoZeroTagged(arena, u8, memoryType);
            size = sizeof(u8);
            Advance(tokenizer, "[");
            SkipToNextChar(tokenizer);
            // NOTE: this assumes that the bool is true or false (and not garbage and not
            // capitalized)
            if (*tokenizer->cursor == 'f')
            {
                *out = 0;
            }
            else
            {
                *out = 1;
            }
            AdvanceToNextParameter(tokenizer);
        }
        else if (dataType == "string" || dataType == "texture")
        {
            Assert(numValues == 1);
            Advance(tokenizer, "[");
            SkipToNextChar(tokenizer);

            string str;
            b32 pairResult = GetBetweenPair(str, tokenizer, '"');
            Assert(pairResult);

            string copy = PushStr8Copy(arena, str);
            out         = copy.str;
            size        = (u32)copy.size;
            AdvanceToNextParameter(tokenizer);
        }
        else if (dataType == "blackbody")
        {
            Assert(numValues == 1);
            SkipToNextDigit(tokenizer);
            i32 val = ReadInt(tokenizer);
            tokenizer->cursor++;

            i32 *ints = PushArrayNoZeroTagged(arena, i32, 1, memoryType);
            ints[0]   = val;
            out       = (u8 *)ints;
            size      = (u32)sizeof(i32);
        }

        // NOTE: either a series of wavelength value pairs or the name of a file with
        // wavelength value pairs
        else if (dataType == "spectrum")
        {
            if (numValues > 1)
            {
                string str;
                b32 pairResult = GetBetweenPair(str, tokenizer, '"');
                Assert(pairResult);

                out  = str.str;
                size = (u32)str.size;
            }
            else
            {
                Advance(tokenizer, "[");
                Assert((numValues & 1) == 0);
                out = PushArrayNoZeroTagged(arena, u8, sizeof(f32) * numValues, memoryType);
                for (u32 i = 0; i < numValues / 2; i++)
                {
                    *((i32 *)out + 2 * i)     = ReadInt(tokenizer);
                    *((f32 *)out + 2 * i + 1) = ReadFloat(tokenizer);
                }
                size = sizeof(f32) * numValues;
                AdvanceToNextParameter(tokenizer);
            }
        }
        else
        {
            Error(0, "Invalid data type: %S\n", dataType);
        }
        parameterNames[currentParam] = Hash(parameterName);
        bytes[currentParam]          = out;
        sizes[currentParam]          = size;
    }
    packet->Initialize(arena, packet->parameterCount + additionalParameters);
    MemoryCopy(packet->parameterNames, parameterNames,
               sizeof(StringId) * packet->parameterCount);
    MemoryCopy(packet->bytes, bytes, sizeof(u8 *) * packet->parameterCount);
    MemoryCopy(packet->sizes, sizes, sizeof(u32) * packet->parameterCount);
}

void WriteFile(string directory, PBRTFileInfo *info);

string ConvertPBRTToRTScene(Arena *arena, string file)
{
    Assert(GetFileExtension(file) == "pbrt");
    string out = RemoveFileExtension(file);
    return PushStr8F(arena, "%S.rtscene", out);
}

struct IncludeHashNode
{
    string filename;
    IncludeHashNode *next;
};

struct IncludeMap
{
    IncludeHashNode *map;
    Mutex *mutexes;
    u32 count;

    bool FindOrAddFile(Arena *arena, string filename)
    {
        u32 hash  = Hash(filename);
        u32 index = hash & (count - 1);
        BeginRMutex(&mutexes[index]);
        IncludeHashNode *node = &map[index];
        IncludeHashNode *prev;
        while (node)
        {
            if (node->filename == filename)
            {
                EndRMutex(&mutexes[index]);
                return true;
            }
            prev = node;
            node = node->next;
        }
        Assert(!node);
        EndRMutex(&mutexes[index]);
        BeginWMutex(&mutexes[index]);
        prev->filename = PushStr8Copy(arena, filename);
        prev->next     = PushStruct(arena, IncludeHashNode);
        EndWMutex(&mutexes[index]);
        return false;
    }
};

PBRTFileInfo *LoadPBRT(Arena **arenas, string directory, string filename,
                       IncludeMap *includeMap, GraphicsState graphicsState = {},
                       bool inWorldBegin = false, bool imported = false, bool write = true)
{
    enum class ScopeType
    {
        None,
        Attribute,
        Object,
    };
    ScopeType scope[32] = {};
    u32 scopeCount      = 0;

    Scheduler::Counter counter = {};
    TempArena temp             = ScratchStart(0, 0);
    u32 threadIndex            = GetThreadIndex();

    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(StrConcat(temp.arena, directory, filename));
    tokenizer.cursor = tokenizer.input.str;

    Arena *threadArena = arenas[GetThreadIndex()];

    PBRTFileInfo *state = PushStruct(threadArena, PBRTFileInfo);
    state->Init(ConvertPBRTToRTScene(temp.arena, filename));

    Arena *tempArena = state->arena;

    auto *shapes     = &state->shapes;
    auto *materials  = &state->materials;
    auto *textures   = &state->textures;
    auto *lights     = &state->lights;
    auto *transforms = &state->transforms;

    bool worldBegin = inWorldBegin;
    bool writeFile  = write;

    struct FileStackEntry
    {
        Tokenizer tokenizer;
        bool writeFile;
    };
    FileStackEntry fileStack[32];
    u32 numFileStackEntries = 0;
    PBRTFileInfo *fileInfoStack[32];
    u32 numFileInfoStackEntries = 0;

    GraphicsState graphicsStateStack[64];
    u32 graphicsStateCount = 0;

    GraphicsState currentGraphicsState = graphicsState;

    auto AddTransform = [&]() {
        if (currentGraphicsState.transformIndex == transforms->Length())
        {
            transforms->Push(currentGraphicsState.transform);
        }
    };

    auto SetNewState = [&](PBRTFileInfo *newState) {
        state      = newState;
        shapes     = &state->shapes;
        materials  = &state->materials;
        textures   = &state->textures;
        lights     = &state->lights;
        transforms = &state->transforms;
        tempArena  = state->arena;
    };

    // TODO: media
    for (;;)
    {
    loop_start:
        PBRTSkipToNextChar(&tokenizer);
        if (EndOfBuffer(&tokenizer))
        {
            OS_UnmapFile(tokenizer.input.str);
            scheduler.Wait(&counter);

            for (u32 i = 0; i < state->numImports; i++)
            {
                state->Merge(state->imports[i]);
            }
            if (writeFile)
            {
                // TODO: allocate per file arenas and handle them appropriately here
                WriteFile(directory, state);
                ArenaRelease(state->arena);
                for (u32 i = 0; i < state->numImports; i++)
                {
                    ArenaRelease(state->imports[i]->arena);
                }
                if (numFileInfoStackEntries)
                {
                    Assert(graphicsStateCount);
                    SetNewState(fileInfoStack[--numFileInfoStackEntries]);
                    currentGraphicsState = graphicsStateStack[--graphicsStateCount];
                }
            }

            if (numFileStackEntries == 0) break;
            FileStackEntry entry = fileStack[--numFileStackEntries];
            tokenizer            = entry.tokenizer;
            writeFile            = entry.writeFile;
            continue;
        }

        string word = ReadWordAndSkipToNextChar(&tokenizer);
        // Comments/Blank lines
        Assert(word.size && word.str[0] != '#');

        StringId sid = Hash(word);
        switch (sid)
        {
            case "Accelerator"_sid:
            {
                Error(!worldBegin,
                      "%S cannot be specified after WorldBegin "
                      "statement\n",
                      word);
                PBRTFileInfo::Type type = PBRTFileInfo::Type::Accelerator;
                ScenePacket *packet     = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Other);
                continue;
            }
            break;
            case "AttributeBegin"_sid:
            {
                Error(worldBegin,
                      "%S cannot be specified before WorldBegin "
                      "statement\n",
                      word);
                Assert(graphicsStateCount < ArrayLength(graphicsStateStack));
                GraphicsState *gs   = &graphicsStateStack[graphicsStateCount++];
                *gs                 = currentGraphicsState;
                scope[scopeCount++] = ScopeType::Attribute;
            }
            break;
            case "AttributeEnd"_sid:
            {
                Error(worldBegin,
                      "%S cannot be specified before WorldBegin "
                      "statement\n",
                      word);
                Error(scopeCount, "Unmatched AttributeEnd statement.\n");
                ScopeType type = scope[--scopeCount];
                Error(type == ScopeType::Attribute,
                      "Unmatched AttributeEnd statement. Aborting...\n");
                Assert(graphicsStateCount > 0);

                // Pop stack
                currentGraphicsState = graphicsStateStack[--graphicsStateCount];
            }
            break;
            case "AreaLightSource"_sid:
            {
                Error(worldBegin,
                      "%S cannot be specified before WorldBegin "
                      "statement\n",
                      word);
                currentGraphicsState.areaLightIndex = lights->Length();
                ScenePacket *packet                 = &lights->AddBack();
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Light);
            }
            break;
            case "Attribute"_sid:
            {
                Error(0, "Not implemented Attribute");
            }
            break;
            case "Camera"_sid:
            {
                Error(!worldBegin,
                      "%S cannot be specified after WorldBegin "
                      "statement\n",
                      word);
                PBRTFileInfo::Type type = PBRTFileInfo::Type::Camera;
                ScenePacket *packet     = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Other);
                continue;
            }
            case "ConcatTransform"_sid:
            {
                currentGraphicsState.transformIndex = transforms->Length();
                SkipToNextDigit(&tokenizer);
                f32 r0c0 = ReadFloat(&tokenizer);
                f32 r0c1 = ReadFloat(&tokenizer);
                f32 r0c2 = ReadFloat(&tokenizer);
                f32 r0c3 = ReadFloat(&tokenizer);

                f32 r1c0 = ReadFloat(&tokenizer);
                f32 r1c1 = ReadFloat(&tokenizer);
                f32 r1c2 = ReadFloat(&tokenizer);
                f32 r1c3 = ReadFloat(&tokenizer);

                f32 r2c0 = ReadFloat(&tokenizer);
                f32 r2c1 = ReadFloat(&tokenizer);
                f32 r2c2 = ReadFloat(&tokenizer);
                f32 r2c3 = ReadFloat(&tokenizer);

                f32 r3c0 = ReadFloat(&tokenizer);
                f32 r3c1 = ReadFloat(&tokenizer);
                f32 r3c2 = ReadFloat(&tokenizer);
                f32 r3c3 = ReadFloat(&tokenizer);

                currentGraphicsState.transform =
                    currentGraphicsState.transform *
                    AffineSpace(Vec3f(r0c0, r0c1, r0c2), Vec3f(r1c0, r1c1, r1c2),
                                Vec3f(r2c0, r2c1, r2c2), Vec3f(r3c0, r3c1, r3c2));
                SkipToNextChar(&tokenizer);
                bool result = Advance(&tokenizer, "]\n");
                Assert(result);
            }
            break;
            case "CoordinateSystem"_sid:
            case "CoordSysTransform"_sid:
            {
                Error(0, "Not implemented %S\n", word);
            }
            break;
            case "Film"_sid:
            {
                Error(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
                PBRTFileInfo::Type type = PBRTFileInfo::Type::Film;
                ScenePacket *packet     = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Other);
                continue;
            }
            case "Integrator"_sid:
            {
                Error(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
                PBRTFileInfo::Type type = PBRTFileInfo::Type::Integrator;
                ScenePacket *packet     = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Other);
                continue;
            }
            case "Identity"_sid:
            {
                currentGraphicsState.transform = AffineSpace::Identity();
            }
            break;
            case "Import"_sid:
            {
                string importedFilename;
                b32 result = GetBetweenPair(importedFilename, &tokenizer, '"');
                Assert(result);

                u64 popPos         = ArenaPos(tempArena);
                string newFilename = ConvertPBRTToRTScene(tempArena, importedFilename);

                bool checkFileInstance =
                    graphicsStateCount &&
                    (currentGraphicsState.transform != AffineSpace::Identity() &&
                     currentGraphicsState.transformIndex != -1) &&
                    (scopeCount && scope[scopeCount - 1] == ScopeType::Attribute);

                bool importWriteFile = false;
                if (checkFileInstance)
                {
                    state->numInstances++;
                    if (state->fileInstances.totalCount &&
                        state->fileInstances.Last().filename == newFilename)
                    {
                        state->fileInstances.Last().transformIndexEnd =
                            currentGraphicsState.transformIndex;
                        AddTransform();
                        ArenaPopTo(tempArena, popPos);
                        goto loop_start;
                    }

                    InstanceType &inst       = state->fileInstances.AddBack();
                    inst.filename            = newFilename;
                    inst.transformIndexStart = currentGraphicsState.transformIndex;
                    inst.transformIndexEnd   = currentGraphicsState.transformIndex;
                    AddTransform();
                    importWriteFile = true;

                    if (includeMap->FindOrAddFile(threadArena, newFilename)) goto loop_start;
                }

                string copiedFilename = PushStr8Copy(temp.arena, importedFilename);

                GraphicsState importedState  = currentGraphicsState;
                importedState.transform      = AffineSpace::Identity();
                importedState.transformIndex = -1;

                if (!checkFileInstance)
                {
                    u32 index = state->numImports++;

                    scheduler.Schedule(&counter, [=](u32 jobID) {
                        state->imports[index] =
                            LoadPBRT(arenas, directory, copiedFilename, includeMap,
                                     importedState, worldBegin, true, importWriteFile);
                    });
                }
                else
                {
                    scheduler.Schedule(&counter, [=](u32 jobID) {
                        LoadPBRT(arenas, directory, copiedFilename, includeMap, importedState,
                                 worldBegin, true, importWriteFile);
                    });
                }
            }
            break;
            case "Include"_sid:
            {
                string importedFilename;
                b32 result = GetBetweenPair(importedFilename, &tokenizer, '"');
                Assert(result);

                string importedFullPath = StrConcat(temp.arena, directory, importedFilename);
                u64 popPos              = ArenaPos(tempArena);
                string newFilename      = ConvertPBRTToRTScene(tempArena, importedFilename);

                bool checkFileInstance =
                    graphicsStateCount &&
                    (currentGraphicsState.transform != AffineSpace::Identity() &&
                     currentGraphicsState.transformIndex != -1) &&
                    (scopeCount && scope[scopeCount - 1] == ScopeType::Attribute);
                bool skipFile = false;
                if (checkFileInstance)
                {
                    state->numInstances++;
                    if (state->fileInstances.totalCount &&
                        state->fileInstances.Last().filename == newFilename)
                    {
                        state->fileInstances.Last().transformIndexEnd =
                            currentGraphicsState.transformIndex;
                        AddTransform();
                        ArenaPopTo(tempArena, popPos);
                        goto loop_start;
                    }

                    InstanceType &inst       = state->fileInstances.AddBack();
                    inst.filename            = newFilename;
                    inst.transformIndexStart = currentGraphicsState.transformIndex;
                    inst.transformIndexEnd   = currentGraphicsState.transformIndex;
                    AddTransform();

                    if (includeMap->FindOrAddFile(threadArena, newFilename)) goto loop_start;

                    PBRTFileInfo *newState = PushStruct(tempArena, PBRTFileInfo);
                    newState->Init(newFilename);

                    Assert(numFileStackEntries < ArrayLength(fileStack));
                    Assert(numFileInfoStackEntries < ArrayLength(fileInfoStack));
                    fileStack[numFileStackEntries++]         = {tokenizer, writeFile};
                    fileInfoStack[numFileInfoStackEntries++] = state;
                    SetNewState(newState);
                    writeFile = true;

                    GraphicsState *gs              = &graphicsStateStack[graphicsStateCount++];
                    *gs                            = currentGraphicsState;
                    currentGraphicsState.transform = AffineSpace::Identity();
                    currentGraphicsState.transformIndex = -1;
                }
                else
                {
                    Assert(numFileStackEntries < ArrayLength(fileStack));
                    fileStack[numFileStackEntries++] = {tokenizer, writeFile};
                    writeFile                        = false;
                }

                tokenizer.input  = OS_MapFileRead(importedFullPath);
                tokenizer.cursor = tokenizer.input.str;
            }
            break;
            case "LookAt"_sid:
            {
                currentGraphicsState.transformIndex = transforms->Length();
                f32 posX                            = ReadFloat(&tokenizer);
                f32 posY                            = ReadFloat(&tokenizer);
                f32 posZ                            = ReadFloat(&tokenizer);
                SkipToNextDigit(&tokenizer);
                f32 lookX = ReadFloat(&tokenizer);
                f32 lookY = ReadFloat(&tokenizer);
                f32 lookZ = ReadFloat(&tokenizer);
                SkipToNextDigit(&tokenizer);
                f32 upX = ReadFloat(&tokenizer);
                f32 upY = ReadFloat(&tokenizer);
                f32 upZ = ReadFloat(&tokenizer);

                currentGraphicsState.transform =
                    currentGraphicsState.transform *
                    AffineSpace::LookAt(Vec3f(posX, posY, posZ), Vec3f(lookX, lookY, lookZ),
                                        Normalize(Vec3f(upX, upY, upZ)));
            }
            break;
            case "LightSource"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                ScenePacket *packet = &lights->AddBack();
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Light);
            }
            break;
            case "Material"_sid:
            case "MakeNamedMaterial"_sid:
            {
                bool isNamedMaterial = (sid == "MakeNamedMaterial"_sid);
                string materialNameOrType;
                b32 result = GetBetweenPair(materialNameOrType, &tokenizer, '"');
                Assert(result);

                u32 materialIndex   = materials->Length();
                ScenePacket *packet = &materials->AddBack();
                packet->type        = Hash(materialNameOrType);
                PBRTSkipToNextChar(&tokenizer);
                ReadParameters(tempArena, packet, &tokenizer, MemoryType_Material);

                if (isNamedMaterial)
                {
                    currentGraphicsState.materialId    = Hash(materialNameOrType);
                    currentGraphicsState.materialIndex = -1;
                }
                else
                {
                    currentGraphicsState.materialIndex = materialIndex;
                    currentGraphicsState.materialId    = 0;
                }
            }
            break;
            case "MakeNamedMedium"_sid:
            case "MediumInterface"_sid:
            {
                // not implemented yet
                Error(0, "Not implemented %S\n", word);
            }
            break;
            case "NamedMaterial"_sid:
            {
                string materialName;
                b32 result = GetBetweenPair(materialName, &tokenizer, '"');
                Assert(result);

                currentGraphicsState.materialId    = Hash(materialName);
                currentGraphicsState.materialIndex = -1;
            }
            break;
            case "ObjectBegin"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                Error(!scopeCount || scope[scopeCount - 1] != ScopeType::Object,
                      "ObjectBegin cannot be called recursively.");
                Error(currentGraphicsState.areaLightIndex == -1,
                      "Area lights instancing not supported.");
                scope[scopeCount++] = ScopeType::Object;

                string objectName;

                b32 result = GetBetweenPair(objectName, &tokenizer, '"');
                Assert(result);

                PBRTFileInfo *newState = PushStruct(tempArena, PBRTFileInfo);
                string objectFileName =
                    PushStr8F(tempArena, "objects/%S_obj.rtscene", objectName);

                newState->Init(objectFileName);

                Assert(numFileInfoStackEntries < ArrayLength(fileInfoStack));
                fileInfoStack[numFileInfoStackEntries++] = state;

                SetNewState(newState);
                AddTransform();
            }
            break;
            case "ObjectEnd"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                Error(scopeCount, "Unmatched AttributeEnd statement. Aborting...\n");
                ScopeType type = scope[--scopeCount];
                Error(type == ScopeType::Object,
                      "Unmatched AttributeEnd statement. Aborting...\n");

                WriteFile(directory, state);
                Assert(numFileInfoStackEntries > 0);
                PBRTFileInfo *oldState = state;
                SetNewState(fileInfoStack[--numFileInfoStackEntries]);
            }
            break;
            case "ObjectInstance"_sid:
            {
                Error(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                Error(!scopeCount || scope[scopeCount - 1] != ScopeType::Object,
                      "Cannot have object instance in object definition block.\n");
                string objectName;
                b32 result = GetBetweenPair(objectName, &tokenizer, '"');
                u64 popPos = ArenaPos(temp.arena);
                string objectFileName =
                    PushStr8F(temp.arena, "objects/%S_obj.rtscene", objectName);
                Assert(result);

                if (state->fileInstances.totalCount &&
                    state->fileInstances.Last().filename == objectFileName)
                {
                    state->fileInstances.Last().transformIndexEnd =
                        currentGraphicsState.transformIndex;
                }
                else
                {
                    InstanceType &inst       = state->fileInstances.AddBack();
                    inst.filename            = PushStr8Copy(tempArena, objectFileName);
                    inst.transformIndexStart = currentGraphicsState.transformIndex;
                    inst.transformIndexEnd   = currentGraphicsState.transformIndex;
                }
                state->numInstances++;
                ArenaPopTo(temp.arena, popPos);
                AddTransform();
            }
            break;
            case "PixelFilter"_sid:
            {
                // TODO: actually parse
                SkipToNextLine(&tokenizer);
            }
            break;
            case "Rotate"_sid:
            {
                currentGraphicsState.transformIndex = transforms->Length();
                f32 angle                           = ReadFloat(&tokenizer);
                f32 axisX                           = ReadFloat(&tokenizer);
                f32 axisY                           = ReadFloat(&tokenizer);
                f32 axisZ                           = ReadFloat(&tokenizer);
                AffineSpace rotationMatrix =
                    AffineSpace::Rotate(Vec3f(axisX, axisY, axisZ), angle);
                currentGraphicsState.transform =
                    currentGraphicsState.transform * rotationMatrix;
            }
            break;
            case "Sampler"_sid:
            {
                Error(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
                PBRTFileInfo::Type type = PBRTFileInfo::Type::Sampler;
                ScenePacket *packet     = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Other);
                continue;
            }
            break;
            case "Scale"_sid:
            {
                currentGraphicsState.transformIndex = transforms->Length();
                f32 s0                              = ReadFloat(&tokenizer);
                f32 s1                              = ReadFloat(&tokenizer);
                f32 s2                              = ReadFloat(&tokenizer);

                AffineSpace scale              = AffineSpace::Scale(Vec3f(s0, s1, s2));
                currentGraphicsState.transform = currentGraphicsState.transform * scale;
            }
            break;
            case "Shape"_sid:
            {
                Error(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                ScenePacket *packet = &shapes->AddBack();
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Shape, 1);

                // TODO: temp
                if (packet->type == "curve"_sid)
                {
                    shapes->last->count--;
                    shapes->totalCount--;
                    continue;
                }

                u32 numVertices = 0;
                u32 numIndices  = 0;
                for (u32 i = 0; i < packet->parameterCount; i++)
                {
                    if (packet->parameterNames[i] == "P"_sid)
                    {
                        numVertices = packet->sizes[i] / sizeof(Vec3f);
                    }
                    else if (packet->parameterNames[i] == "indices"_sid)
                    {
                        numIndices = packet->sizes[i] / sizeof(u32);
                    }
                    else if (packet->parameterNames[i] == "filename"_sid)
                    {
                        string plyMeshFile;
                        plyMeshFile.str  = packet->bytes[i];
                        plyMeshFile.size = packet->sizes[i];

                        if (CheckQuadPLY(StrConcat(temp.arena, directory, plyMeshFile)))
                            packet->type = "quadmesh"_sid;
                    }
                }
                if (packet->type == "trianglemesh"_sid && numVertices && numIndices &&
                    numVertices / 2 == numIndices / 3)
                {
                    packet->type = "quadmesh"_sid;
                }

                i32 *indices = PushArray(tempArena, i32, 4);
                // ORDER: Light, Medium, Transform, Material Index,
                // Material StringID (if present)
                indices[0] = currentGraphicsState.areaLightIndex;
                indices[1] = currentGraphicsState.mediaIndex;
                indices[2] = currentGraphicsState.transformIndex;
                // NOTE: the highest bit is set if it's an index
                indices[3] = currentGraphicsState.materialIndex == -1
                                 ? i32(currentGraphicsState.materialId)
                                 : (u32)currentGraphicsState.materialIndex | 0x80000000;

                u32 currentParameter                     = packet->parameterCount++;
                packet->parameterNames[currentParameter] = "Indices"_sid;
                packet->bytes[currentParameter]          = (u8 *)indices;
                packet->sizes[currentParameter]          = sizeof(i32) * 4;

                AddTransform();
            }
            break;
            case "Translate"_sid:
            {
                currentGraphicsState.transformIndex = transforms->Length();
                f32 t0                              = ReadFloat(&tokenizer);
                f32 t1                              = ReadFloat(&tokenizer);
                f32 t2                              = ReadFloat(&tokenizer);

                AffineSpace t                  = AffineSpace::Translate(Vec3f(t0, t1, t2));
                currentGraphicsState.transform = currentGraphicsState.transform * t;
            }
            break;
            case "Transform"_sid:
            {
                currentGraphicsState.transformIndex = transforms->Length();
                SkipToNextDigit(&tokenizer);
                f32 r0c0 = ReadFloat(&tokenizer);
                f32 r0c1 = ReadFloat(&tokenizer);
                f32 r0c2 = ReadFloat(&tokenizer);
                f32 r0c3 = ReadFloat(&tokenizer);

                f32 r1c0 = ReadFloat(&tokenizer);
                f32 r1c1 = ReadFloat(&tokenizer);
                f32 r1c2 = ReadFloat(&tokenizer);
                f32 r1c3 = ReadFloat(&tokenizer);

                f32 r2c0 = ReadFloat(&tokenizer);
                f32 r2c1 = ReadFloat(&tokenizer);
                f32 r2c2 = ReadFloat(&tokenizer);
                f32 r2c3 = ReadFloat(&tokenizer);

                f32 r3c0 = ReadFloat(&tokenizer);
                f32 r3c1 = ReadFloat(&tokenizer);
                f32 r3c2 = ReadFloat(&tokenizer);
                f32 r3c3 = ReadFloat(&tokenizer);

                // NOTE: this transposes the matrix
                currentGraphicsState.transform = AffineSpace(
                    r0c0, r1c0, r2c0, r3c0, r0c1, r1c1, r2c1, r3c1, r0c2, r1c2, r2c2, r3c2);

                AdvanceToNextParameter(&tokenizer);
            }
            break;
            case "Texture"_sid:
            {
                string textureName;
                b32 result = GetBetweenPair(textureName, &tokenizer, '"');
                Assert(result);
                PBRTSkipToNextChar(&tokenizer);

                string textureType;
                result = GetBetweenPair(textureType, &tokenizer, '"');
                Assert(result);
                PBRTSkipToNextChar(&tokenizer);

                string textureClass;
                result = GetBetweenPair(textureClass, &tokenizer, '"');
                Assert(result);
                PBRTSkipToNextChar(&tokenizer);

                ScenePacket *packet = &textures->AddBack();
                packet->type        = Hash(StrConcat(tempArena, textureType, textureClass));

                PBRTSkipToNextChar(&tokenizer);

                ReadParameters(tempArena, packet, &tokenizer, MemoryType_Texture, 1);

                u32 currentParameter                     = packet->parameterCount++;
                packet->parameterNames[currentParameter] = "name"_sid;
                packet->bytes[currentParameter] = PushStr8Copy(tempArena, textureName).str;
                packet->sizes[currentParameter] = textureName.size;
            }
            break;
            case "WorldBegin"_sid:
            {
                // NOTE: this assumes "WorldBegin" only occurs in one
                // file
                worldBegin = true;

                const ScenePacket *filmPacket = &state->packets[PBRTFileInfo::Type::Film];
                Vec2i fullResolution;
                for (u32 i = 0; i < filmPacket->parameterCount; i++)
                {
                    switch (filmPacket->parameterNames[i])
                    {
                        case "xresolution"_sid:
                        {
                            fullResolution.x = filmPacket->GetInt(i);
                        }
                        break;
                        case "yresolution"_sid:
                        {
                            fullResolution.y = filmPacket->GetInt(i);
                        }
                        break;
                    }
                }

                const ScenePacket *samplerPacket =
                    &state->packets[PBRTFileInfo::Type::Sampler];
                // state->scene->sampler =
                //     Sampler::Create(state->mainArena, samplerPacket,
                //     fullResolution);

                AddTransform();
                // TODO: instantiate the camera with the current
                // transform
                currentGraphicsState.transform      = AffineSpace::Identity();
                currentGraphicsState.transformIndex = -1;
            }
            break;
            default:
            {
                string line = ReadLine(&tokenizer);
                Error(0, "Error while parsing scene. Buffer: %S", line);
            }
                // TODO IMPORTANT: the indices are clockwise since PBRT
                // uses a left-handed coordinate system. either need to
                // revert the winding or use a left handed system as
                // well
        }
    }

    ScratchEnd(temp);
    return state;
}

struct MaterialHashNode
{
    u64 hash;
    u32 id;
    string buffer;
    MaterialHashNode *next;
};

struct TextureHashNode
{
    u64 hash;
    u32 id;
    string name;
    TextureHashNode *next;
};

void CheckDuplicateMaterial(Arena *arena, ScenePacket *packet, MaterialHashNode *map,
                            u32 hashMask, string materialType, const StringId *parameterNames,
                            u32 count, u32 &materialCount)
{
    TempArena temp = ScratchStart(&arena, 1);

    u32 typeSize  = (u32)materialType.size;
    u32 totalSize = typeSize;

    for (u32 c = 0; c < count; c++)
    {
        for (u32 i = 0; i < packet->parameterCount; i++)
        {
            if (packet->parameterNames[i] == parameterNames[c])
            {
                totalSize += packet->sizes[i];
            }
        }
    }
    u8 *buffer = PushArrayNoZero(temp.arena, u8, totalSize);
    MemoryCopy(buffer, materialType.str, typeSize);
    u32 offset = typeSize;
    for (u32 c = 0; c < count; c++)
    {
        for (u32 i = 0; i < packet->parameterCount; i++)
        {
            if (packet->parameterNames[i] == parameterNames[c])
            {
                MemoryCopy(buffer + offset, packet->bytes[i], packet->sizes[i]);
                offset += packet->sizes[i];
            }
        }
    }
    u64 hash = MurmurHash64A(buffer, totalSize, 0);

    MaterialHashNode *node = &map[hash & hashMask];
    MaterialHashNode *prev;
    while (node)
    {
        if (node->hash == hash && node->buffer.size == totalSize)
        {
            if (memcmp(node->buffer.str, buffer, totalSize) == 0) break;
        }
        prev = node;
        node = node->next;
    }
    if (!node)
    {
        prev->hash       = hash;
        prev->buffer.str = PushArrayNoZero(arena, u8, totalSize);
        MemoryCopy(prev->buffer.str, buffer, totalSize);
        prev->buffer.size = totalSize;
        prev->next        = PushStruct(arena, MaterialHashNode);
        materialCount++;
    }
    ScratchEnd(temp);
}

enum class TextureType
{
    Ptex,
    Max,
};
ENUM_CLASS_FLAGS(TextureType)

static const StringId cDiffuseParameterIds[] = {
    "reflectance"_sid,
    "displacement"_sid,
};
static const string cDiffuseParameterNames[] = {
    "reflectance",
    "displacement",
};

// TODO: need to consolidate object types, object instances, textures, materials, shapes,
// transforms, etc.
#if 0
void WriteMeta(StringBuilder *builder, string filename, SceneLoadState *state)
{
    TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();

    u32 tentativeCount = 0;
    u32 textureCount   = 0;
    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        auto *list = &state->materials[pIndex];
        tentativeCount += list->totalCount;
        textureCount += &state->textures[pIndex].totalCount;
    }

    u32 hashTableSize = Max(1024, NextPowerOfTwo(tentativeCount));
    u32 hashMask      = hashTableSize - 1;

    u32 textureTableSize        = Max(1024, NextPowerOfTwo(textureCount));
    MaterialHashNode *map       = PushArray(temp.arena, MaterialHashNode, hashTableSize);
    TextureHashNode *textureMap = PushArray(temp.arena, TextureHashNode, textureTableSize);
    u32 materialCount           = 0;

    u32 textureCount = 0;

    // ways of doing this:
    // 1. all the textures are in 1 file. the problem with this is that there's a lot...
    // 2. split the textures between files. the problem with THIS is that how do I
    // allocate?
    //      - duplicate the material/texture
    //          - this probably doesn't work because transforms would need to be
    //          duplicated?
    //      - index into global array
    //          - keep track of running total using atomics. wouldn't be horrible because
    //          we could batch per file
    Put(builder, "TEXTURE_START");
    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        auto *list = &state->textures[pIndex];
        for (auto *node = list->first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ScenePacket *packet = &node->values[i];
                switch (packet->type)
                {
                    case "floatptex"_sid:
                    case "spectrumptex"_sid:
                    {
                        i32 index = packet->FindKey("name"_sid);
                        if (index == -1) Error(0, "No texture name speciied for material.\n");

                        u64 hash =
                            MurmurHash64A(packet->bytes[index], packet->sizes[index], 0);

                        string textureName = Str8(packet->bytes[index], packet->sizes[index]);
                        TextureHashNode *node = textureMap[hash & (textureTableSize - 1)];
                        TextureHashNode *prev;
                        while (node)
                        {
                            if (node->hash == hash && node->name == textureName)
                            {
                                break;
                            }
                            prev = node;
                            node = node->next;
                        }
                        if (!node)
                        {
                            prev->hash = hash;
                            prev->id   = textureCount++;
                            prev->name = PushStr8Copy(temp.arena, textureName);
                            prev->next = PushStruct(temp.arena, TextureHashNode);
                            StringId parameterNames[] = {
                                "filename"_sid,
                                "scale"_sid,
                                "encoding"_sid,
                            };
                            u32 count = ArrayLength(parameterNames);
                            for (u32 c = 0; c < count; c++)
                            {
                                for (u32 i = 0; i < packet->parameterCount; i++)
                                {
                                    if (packet->parameterNames[i] == parameterNames[c])
                                    {
                                        Put(builder, "$");
                                        Put(builder, packet->bytes[i], packet->sizes[i]);
                                    }
                                }
                            }
                        }
                    }
                    break;
                    default:
                        Error(0, "Texture type string is invalid or currently unsupported. "
                                 "Aborting...\n");
                }
            }
        }
    }
    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        auto *list = &state->materials[pIndex];
        for (auto *node = list->first; node != 0; node = node->next)
        {
            // Check for duplicate materials
            for (u32 i = 0; i < node->count; i++)
            {
                ScenePacket *packet = &node->values[i];
                switch (packet->type)
                {
                    case "diffuse"_sid:
                    {
                        CheckDuplicateMaterial(
                            temp.arena, packet, map, hashMask, "diffuse", cDiffuseParameterIds,
                            ArrayLength(cDiffuseParameterIds), materialCount);
                    }
                    break;
                    case "diffusetransmission"_sid:
                    {
                        const StringId parameterNames[] = {
                            "reflectance"_sid,
                            "transmittance"_sid,
                            "scale"_sid,
                        };
                        CheckDuplicateMaterial(temp.arena, packet, map, hashMask,
                                               "diffusetransmission", parameterNames,
                                               ArrayLength(parameterNames), materialCount);
                    }
                    break;
                    case "dielectric"_sid:
                    {
                        const StringId parameterNames[] = {
                            "roughness"_sid,      "uroughness"_sid, "vroughness"_sid,
                            "remaproughness"_sid, "eta"_sid,
                        };
                        CheckDuplicateMaterial(temp.arena, packet, map, hashMask, "dielectric",
                                               parameterNames, ArrayLength(parameterNames),
                                               materialCount);
                    }
                    break;
                    case "coateddiffuse"_sid:
                    {
                        const StringId parameterNames[] = {
                            "roughness"_sid,      "uroughness"_sid,  "vroughness"_sid,
                            "remaproughness"_sid, "reflectance"_sid, "displacement"_sid,
                            "albedo"_sid,         "g"_sid,           "maxdepth"_sid,
                            "nsamples"_sid,       "thickness"_sid,
                        };
                        CheckDuplicateMaterial(temp.arena, packet, map, hashMask,
                                               "coateddiffuse", parameterNames,
                                               ArrayLength(parameterNames), materialCount);
                    }
                    break;
                    default: Error(0, "Material type string is invalid. Aborting...\n");
                }
            }
        }
    }
    printf("Total # unique materials: %u\n", materialCount);
}

void WriteMaterial(StringBuilder *builder, PBRTFileInfo *fileInfo, ScenePacket *packet,
                   StringId *parameterIDs, string *parameterNames, u32 count)
{
    for (u32 c = 0; c < count; c++)
    {
        for (u32 i = 0; i < packet->parameterCount; i++)
        {
            if (packet->parameterNames[i] == parameterNames[c])
            {
                u64 hash = MurmurHash64A(packet->bytes[i], packet->sizes[i], 0);
                TextureHashNode *node =
                    fileInfo->textureMap[hash & (fileInfo->textureMapSize - 1)];
                TextureHashNode *prev;
                while (node)
                {
                    if (node->hash == hash && node->name == name) break;
                    prev = node;
                    node = node->next;
                }
                Error(node, "Material references an unknown texture.\n");
                Put(builder, "t %u ", node->id);
            }
        }
    }
}
#endif

void WriteFile(string directory, PBRTFileInfo *info)
{
    if (info->shapes.totalCount == 0 && info->numInstances == 0) return;
    TempArena temp = ScratchStart(0, 0);
    Assert(GetFileExtension(info->filename) == "rtscene");
    string outFile = StrConcat(temp.arena, directory, info->filename);

    StringBuilder builder  = {};
    builder.arena          = temp.arena;
    u32 totalMaterialCount = 0;

    StringBuilder offsetBuilder = {};
    offsetBuilder.arena         = temp.arena;

    Put(&offsetBuilder, "RTSCENE_START ");

    if (info->fileInstances.totalCount && info->shapes.totalCount)
    {
        Print("%S has both instances and shapes\n", info->filename);
    }

    struct BuilderNode
    {
        string filename;
        StringBuilder builder = {};
        BuilderNode *next;
    };

    BuilderNode bNode = {};

    if (info->fileInstances.totalCount)
    {
        Put(&builder, "INCLUDE_START ");
        Assert(info->numInstances);
        Put(&builder, "Count: %u ", info->numInstances);
        u32 count = 0;
        for (auto *instNode = info->fileInstances.first; instNode != 0;
             instNode       = instNode->next)
        {
            for (u32 i = 0; i < instNode->count; i++)
            {
                InstanceType *fileInst = &instNode->values[i];
                BuilderNode *node      = &bNode;
                while (!(node->filename == fileInst->filename) && node->next)
                    node = node->next;
                if (!node->next)
                {
                    node->filename      = fileInst->filename;
                    node->builder.arena = temp.arena;
                    node->next          = PushStruct(temp.arena, BuilderNode);
                    Put(&node->builder, "File: %S ", node->filename);
                }
                Put(&node->builder, "%u-%u ", fileInst->transformIndexStart,
                    fileInst->transformIndexEnd);
            }
        }
        BuilderNode *node = &bNode;
        while (node->next)
        {
            builder = ConcatBuilders(&builder, &node->builder);
            node    = node->next;
        }
        Put(&builder, "INCLUDE_END ");
    }

    StringBuilder dataBuilder = {};
    dataBuilder.arena         = temp.arena;
    Put(&dataBuilder, "DATA_START ");

    if (info->transforms.totalCount)
    {
        Put(&dataBuilder, "TRANSFORM_START ");
        Put(&dataBuilder, "Count %u ", info->transforms.totalCount);
        for (auto *node = info->transforms.first; node != 0; node = node->next)
        {
            Put(&dataBuilder, node->values, sizeof(node->values[0]) * node->count);
        }
        Put(&dataBuilder, "TRANSFORM_END");
    }
    if (info->shapes.totalCount)
    {
        Put(&builder, "SHAPE_START ");
        for (auto *node = info->shapes.first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ScenePacket *packet = node->values + i;
                switch (packet->type)
                {
                    case "quadmesh"_sid:
                    {
                        Put(&builder, "Quad ");
                        for (u32 c = 0; c < packet->parameterCount; c++)
                        {
                            if (packet->parameterNames[c] == "filename"_sid)
                            {
                                QuadMesh mesh = LoadQuadPLY(
                                    temp.arena,
                                    StrConcat(temp.arena, directory,
                                              Str8(packet->bytes[c], packet->sizes[c])));
                                Put(&builder, "c %u ", mesh.numVertices);
                                u64 pOffset = dataBuilder.totalSize;
                                Put(&dataBuilder, mesh.p, mesh.numVertices * sizeof(Vec3f));
                                u64 nOffset = dataBuilder.totalSize;
                                Put(&dataBuilder, mesh.n, mesh.numVertices * sizeof(Vec3f));
                                Put(&builder, "p %llu n %llu ", pOffset, nOffset);
                            }
                            else if (packet->parameterNames[c] == "P"_sid)
                            {
                                u64 pOffset = dataBuilder.totalSize;
                                Put(&dataBuilder, packet->bytes[c], packet->sizes[c]);
                                Put(&builder, "p %llu ", pOffset);
                                Put(&builder, "c %u ", packet->sizes[c] / sizeof(Vec3f));
                            }
                            else if (packet->parameterNames[c] == "N"_sid)
                            {
                                u64 nOffset = dataBuilder.totalSize;
                                Put(&dataBuilder, packet->bytes[c], packet->sizes[c]);
                                Put(&builder, "n %llu ", nOffset);
                            }
                        }
                    }
                    break;
                    case "curve"_sid:
                    {
                    }
                    break;
                    default: Assert(0);
                }
            }
        }
        Put(&builder, "SHAPE_END ");
        Put(&dataBuilder, "DATA_END");
    }

    Put(&builder, "RTSCENE_END");
    u64 dataOffset = builder.totalSize + offsetBuilder.totalSize + sizeof(u64);
    PutPointerValue(&offsetBuilder, &dataOffset);

    StringBuilder finalBuilder = ConcatBuilders(&offsetBuilder, &builder);
    finalBuilder               = ConcatBuilders(&finalBuilder, &dataBuilder);
    WriteFileMapped(&finalBuilder, outFile);
    ScratchEnd(temp);
}

void LoadPBRT(Arena *arena, string filename)
{
    TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16);
    }

    IncludeMap map;
    map.count   = 1024;
    map.map     = PushArray(arena, IncludeHashNode, map.count);
    map.mutexes = PushArray(arena, Mutex, map.count);

    string directory = "../data/island/pbrt-v4/";
    OS_CreateDirectory(StrConcat(temp.arena, directory, "objects"));

    PerformanceCounter counter = OS_StartCounter();
    LoadPBRT(arenas, "../data/island/pbrt-v4/", filename, &map);
    f32 time = OS_GetMilliseconds(counter);
    printf("convert time: %fms\n", time);

    for (u32 i = 0; i < numProcessors; i++)
    {
        ArenaClear(arenas[i]);
    }
    ScratchEnd(temp);
}
} // namespace rt

using namespace rt;
int main(int argc, char **argv)
{
    Arena *arena = ArenaAlloc();
    InitThreadContext(arena, "[Main Thread]", 1);
    OS_Init();
    u32 numProcessors = OS_NumProcessors();
    scheduler.Init(numProcessors);
    TempArena temp        = ScratchStart(0, 0);
    StringBuilder builder = {};
    builder.arena         = arena;

    if (argc != 2)
    {
        printf("You must pass in a valid PBRT file to convert. Aborting... \n");
        return 1;
    }
    Assert(argc == 2);
    string filename = Str8C(argv[1]);
    if (!(GetFileExtension(filename) == "pbrt"))
    {
        printf("You must pass in a valid PBRT file to convert. Aborting... \n");
        return 1;
    }
    LoadPBRT(arena, filename);

    // read pbrt as i've done before, getting scene packets
    // list of things to do
    // 1. reduce material count by hashing?
    return 0;
}
