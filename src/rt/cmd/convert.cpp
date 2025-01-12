#include "../base.h"
#include "../macros.h"
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
#include "../string.h"
#include "../containers.h"
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

struct Mesh
{
    Vec3f *p     = 0;
    Vec3f *n     = 0;
    Vec2f *uv    = 0;
    u32 *indices = 0;
    u32 numIndices;
    u32 numVertices;
    u32 numFaces;
};
} // namespace rt

#include "../scene_load.h"

namespace rt
{
struct Instance
{
    u32 id;
    u32 transformIndex;
};

enum class GeometryType
{
    QuadMesh,
    TriangleMesh,
    Instance,
    Max,
};

struct ShapeType
{
    ScenePacket packet;
    u32 *transformIndex;

    string materialName;
};

struct InstanceType
{
    string filename;
    u32 transformIndexStart;
    u32 transformIndexEnd;
};

// NOTE: for materials and textures
struct NamedPacket
{
    ScenePacket packet;
    string name;
    string type;

    u32 Hash() const { return rt::Hash(name); }
    bool operator==(const NamedPacket &other) const { return name == other.name; }
    bool operator==(string str) const { return str == name; }
};

typedef HashMap<NamedPacket> SceneHashMap;

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
    ChunkedLinkedList<ShapeType, 1024, MemoryType_Shape> shapes;
    ChunkedLinkedList<InstanceType, 1024, MemoryType_Instance> fileInstances;
    u32 numInstances;

    ChunkedLinkedList<AffineSpace, 16384, MemoryType_Transform> transforms;

    PBRTFileInfo *imports[32];
    u32 numImports;
    Scheduler::Counter counter = {};

    void Init(string inFilename)
    {
        arena    = ArenaAlloc(8);
        filename = PushStr8Copy(arena, inFilename);
        shapes   = decltype(shapes)(arena);

        fileInstances = decltype(fileInstances)(arena);
        transforms    = decltype(transforms)(arena);
        numInstances  = 0;
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

struct SceneLoadState
{
    Arena **arenas;
    u32 numProcessors;
    ChunkedLinkedList<NamedPacket, 1024, MemoryType_Material> *materials;
    ChunkedLinkedList<NamedPacket, 1024, MemoryType_Light> *lights;

    SceneHashMap *textureHashMaps;
    const u32 hashMapSize = 8192;

    IncludeMap includeMap;

    void Init(Arena *arena)
    {
        u32 threadIndex = GetThreadIndex();
        numProcessors   = OS_NumProcessors();
        arenas          = PushArray(arena, Arena *, numProcessors);
        materials       = PushArray(
            arena, ChunkedLinkedList<NamedPacket COMMA 1024 COMMA MemoryType_Material>,
            numProcessors);
        textureHashMaps = PushArray(arena, SceneHashMap, numProcessors);
        lights =
            PushArray(arena, ChunkedLinkedList<NamedPacket COMMA 1024 COMMA MemoryType_Light>,
                      numProcessors);

        for (u32 i = 0; i < numProcessors; i++)
        {
            arenas[i]    = ArenaAlloc(16);
            materials[i] = ChunkedLinkedList<NamedPacket, 1024, MemoryType_Material>(arena);
            lights[i]    = ChunkedLinkedList<NamedPacket, 1024, MemoryType_Light>(arena);
            textureHashMaps[i] = SceneHashMap(arena, hashMapSize);
        }

        includeMap.count   = 1024;
        includeMap.map     = PushArray(arena, IncludeHashNode, includeMap.count);
        includeMap.mutexes = PushArray(arena, Mutex, includeMap.count);
    }
};

// NOTE: this is kind of a hack
#if 0
string CheckDuplicateMaterial(Arena *arena, MaterialLL &matList, MaterialHashMap &map,
                              u32 hashMask, ScenePacket *packet, string materialName,
                              string materialType, const StringId *parameterNames, u32 count)
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

    MaterialHashNode *node = map[hash & hashMask];
    while (node)
    {
        if (node->hash == hash && node->buffer.size == totalSize)
        {
            if (memcmp(node->buffer.str, buffer, totalSize) == 0)
            {
                ScratchEnd(temp);
                return node->packet->id;
            }
            else
            {
                Error(0, "Hash is equal but contents are different.");
            }
        }
        node = node->next;
    }
    if (!node)
    {
        MaterialPacket &mat = matList.AddBack();
        mat.packet          = packet;
        mat.id              = materialName;

        MaterialHashNode *newNode = PushStruct(arena, MaterialHashNode);
        newNode->hash             = hash;
        newNode->buffer.str       = PushArrayNoZero(arena, u8, totalSize);
        newNode->buffer.size      = totalSize;
        MemoryCopy(newNode->buffer.str, buffer, totalSize);
        newNode->packet = &mat;

        // add to front
        newNode->next        = map[hash & hashMask];
        map[hash & hashMask] = newNode;
    }
    ScratchEnd(temp);
    return materialName;
}
#endif

struct GraphicsState
{
    string materialName = {};
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
                    MemoryType memoryType);
// NOTE: sets the camera, film, sampler, etc.
void CreateScenePacket(Arena *arena, string word, ScenePacket *packet, Tokenizer *tokenizer,
                       MemoryType memoryType)
{
    string type;
    b32 result = GetBetweenPair(type, tokenizer, '"');
    Assert(result);
    packet->type = Hash(type);
    PBRTSkipToNextChar(tokenizer);

    ReadParameters(arena, packet, tokenizer, memoryType);
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
                    MemoryType memoryType)
{
    string infoType;
    b8 result;
    u32 numVertices = 0;
    u32 numIndices  = 0;

    u32 parameterCount = 0;

    StringId parameterNames[MAX_PARAMETER_COUNT];
    u8 *bytes[MAX_PARAMETER_COUNT];
    u32 sizes[MAX_PARAMETER_COUNT];
    DataType dataTypes[MAX_PARAMETER_COUNT];

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
        DataType dt;
        if (dataType == "float")
        {
            dt          = DataType::Float;
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
            dt = DataType::Vec2;
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
            dt = DataType::Vec3;
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
            dt           = DataType::Int;
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
            dt   = DataType::Bool;
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
            if (dataType == "string") dt = DataType::String;
            else dt = DataType::Texture;
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
            dt = DataType::Blackbody;
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
            if (numValues == 1)
            {
                dt = DataType::String;
                string str;
                b32 pairResult = GetBetweenPair(str, tokenizer, '"');
                Assert(pairResult);

                out  = str.str;
                size = (u32)str.size;
            }
            else
            {
                dt = DataType::Floats;
                Advance(tokenizer, "[");
                Assert((numValues & 1) == 0);
                out = PushArrayNoZeroTagged(arena, u8, sizeof(f32) * numValues, memoryType);
                for (u32 i = 0; i < numValues / 2; i++)
                {
                    SkipToNextDigit(tokenizer);
                    *((i32 *)out + 2 * i) = ReadInt(tokenizer);
                    SkipToNextDigit(tokenizer);
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
        dataTypes[currentParam]      = dt;
    }
    packet->Initialize(arena, packet->parameterCount);
    MemoryCopy(packet->parameterNames, parameterNames,
               sizeof(StringId) * packet->parameterCount);
    MemoryCopy(packet->bytes, bytes, sizeof(u8 *) * packet->parameterCount);
    MemoryCopy(packet->sizes, sizes, sizeof(u32) * packet->parameterCount);
    MemoryCopy(packet->types, dataTypes, sizeof(DataType) * packet->parameterCount);
}

void WriteFile(string directory, PBRTFileInfo *info, SceneLoadState *state = 0);

string ConvertPBRTToRTScene(Arena *arena, string file)
{
    Assert(GetFileExtension(file) == "pbrt");
    string out = RemoveFileExtension(file);
    return PushStr8F(arena, "%S.rtscene", out);
}

string ReplaceColons(Arena *arena, string str)
{
    string newString = PushStr8Copy(arena, str);
    for (u64 i = 0; i < newString.size; i++)
    {
        if (newString.str[i] == ':')
        {
            newString.str[i] = '-';
        }
    }
    return newString;
}

PBRTFileInfo *LoadPBRT(SceneLoadState *sls, string directory, string filename,
                       GraphicsState graphicsState = {}, bool originFile = true,
                       bool inWorldBegin = false, bool write = true)
{
    enum class ScopeType
    {
        None,
        Attribute,
        Object,
    };

    ScopeType scope[32] = {};
    u32 scopeCount      = 0;

    TempArena temp  = ScratchStart(0, 0);
    u32 threadIndex = GetThreadIndex();

    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(StrConcat(temp.arena, directory, filename));
    tokenizer.cursor = tokenizer.input.str;

    string currentFilename = filename;

    Arena *threadArena = sls->arenas[threadIndex];

    PBRTFileInfo *state = PushStruct(threadArena, PBRTFileInfo);
    state->Init(ConvertPBRTToRTScene(threadArena, filename));

    Arena *tempArena = state->arena;
    auto *shapes     = &state->shapes;
    auto *transforms = &state->transforms;

    auto *materials      = &sls->materials[threadIndex];
    auto &textureHashMap = sls->textureHashMaps[threadIndex];
    auto *lights         = &sls->lights[threadIndex];

    bool worldBegin = inWorldBegin;
    bool writeFile  = write;

    PBRTFileInfo *tempStateHolder = 0;

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
            scheduler.Wait(&state->counter);

            for (u32 i = 0; i < state->numImports; i++)
            {
                state->Merge(state->imports[i]);
            }
            if (writeFile)
            {
                WriteFile(directory, state, originFile ? sls : 0);
                ArenaRelease(state->arena);
                for (u32 i = 0; i < state->numImports; i++)
                {
                    ArenaRelease(state->imports[i]->arena);
                }
            }
            break;
        }

        string word = ReadWordAndSkipToNextChar(&tokenizer);
        // Comments/Blank lines
        Assert(word.size && word.str[0] != '#');

        StringId sid  = Hash(word);
        bool isImport = false;
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
                GraphicsState *gs = &graphicsStateStack[graphicsStateCount++];
                *gs               = currentGraphicsState;
                Assert(scopeCount < ArrayLength(scope));
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
                NamedPacket *packet                 = &lights->AddBack();
                CreateScenePacket(tempArena, word, &packet->packet, &tokenizer,
                                  MemoryType_Light);
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
                isImport = true;
            }
            case "Include"_sid:
            {
                string importedFilename;
                b32 result = GetBetweenPair(importedFilename, &tokenizer, '"');
                Assert(result);

                string newFilename = ConvertPBRTToRTScene(tempArena, importedFilename);

                bool checkFileInstance =
                    graphicsStateCount &&
                    (currentGraphicsState.transform != AffineSpace::Identity() &&
                     currentGraphicsState.transformIndex != -1) &&
                    (scopeCount && scope[scopeCount - 1] == ScopeType::Attribute);

                if (checkFileInstance)
                {
                    state->numInstances++;
                    if (state->fileInstances.totalCount &&
                        state->fileInstances.Last().filename == newFilename)
                    {
                        state->fileInstances.Last().transformIndexEnd =
                            currentGraphicsState.transformIndex;
                        AddTransform();
                        goto loop_start;
                    }

                    InstanceType &inst       = state->fileInstances.AddBack();
                    inst.filename            = newFilename;
                    inst.transformIndexStart = currentGraphicsState.transformIndex;
                    inst.transformIndexEnd   = currentGraphicsState.transformIndex;
                    AddTransform();

                    if (sls->includeMap.FindOrAddFile(threadArena, newFilename))
                        goto loop_start;
                }

                string copiedFilename = PushStr8Copy(threadArena, importedFilename);

                GraphicsState importedState  = currentGraphicsState;
                importedState.transform      = AffineSpace::Identity();
                importedState.transformIndex = -1;

                u32 index = state->numImports;
                state->numImports += !checkFileInstance;
                if (isImport)
                {
                    scheduler.Schedule(&state->counter, [=](u32 jobID) {
                        PBRTFileInfo *newState =
                            LoadPBRT(sls, directory, copiedFilename, importedState, false,
                                     worldBegin, checkFileInstance);
                        if (!checkFileInstance) state->imports[index] = newState;
                    });
                }
                else
                {
                    PBRTFileInfo *newState =
                        LoadPBRT(sls, directory, copiedFilename, importedState, false,
                                 worldBegin, checkFileInstance);
                    if (!checkFileInstance) state->imports[index] = newState;
                }
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
                NamedPacket *packet = &lights->AddBack();
                CreateScenePacket(tempArena, word, &packet->packet, &tokenizer,
                                  MemoryType_Light);
            }
            break;
            case "Material"_sid:
            case "MakeNamedMaterial"_sid:
            {
                bool isNamedMaterial = (sid == "MakeNamedMaterial"_sid);
                string materialNameOrType;
                b32 result = GetBetweenPair(materialNameOrType, &tokenizer, '"');
                Assert(result);

                string materialName =
                    isNamedMaterial ? PushStr8Copy(threadArena, materialNameOrType)
                                    : PushStr8F(threadArena, "%S%S%llu", materialNameOrType,
                                                RemoveFileExtension(state->filename),
                                                (u64)(tokenizer.cursor - tokenizer.input.str));

                NamedPacket *nPacket = &materials->AddBack();
                ScenePacket *packet  = &nPacket->packet;
                packet->type         = "material"_sid; // Hash(materialNameOrType);

                PBRTSkipToNextChar(&tokenizer);
                ReadParameters(threadArena, packet, &tokenizer, MemoryType_Material);
                if (isNamedMaterial)
                {
                    bool found = false;
                    for (u32 i = 0; i < packet->parameterCount; i++)
                    {
                        if (packet->parameterNames[i] == "type"_sid)
                        {
                            nPacket->type = PushStr8Copy(
                                threadArena, Str8(packet->bytes[i], packet->sizes[i]));
                            found = true;
                            break;
                        }
                    }
                    Error(found, "Named material must have a type\n");
                }
                else
                {
                    nPacket->type = PushStr8Copy(threadArena, materialNameOrType);
                }
                nPacket->name = materialName;

                currentGraphicsState.materialName = materialName;
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

                currentGraphicsState.materialName = PushStr8Copy(tempArena, materialName);
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

                PBRTFileInfo *newState = PushStruct(threadArena, PBRTFileInfo);
                string objectFileName  = PushStr8F(threadArena, "objects/%S_obj.rtscene",
                                                   ReplaceColons(tempArena, objectName));

                newState->Init(objectFileName);

                Assert(tempStateHolder == 0);
                tempStateHolder = state;

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

                scheduler.Wait(&state->counter);
                for (u32 i = 0; i < state->numImports; i++)
                {
                    state->Merge(state->imports[i]);
                }
                WriteFile(directory, state);
                ArenaRelease(state->arena);
                for (u32 i = 0; i < state->numImports; i++)
                    ArenaRelease(state->imports[i]->arena);

                Assert(tempStateHolder);
                SetNewState(tempStateHolder);
                tempStateHolder = 0;
            }
            break;
            case "ObjectInstance"_sid:
            {
                Error(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                Error(!scopeCount || scope[scopeCount - 1] != ScopeType::Object,
                      "Cannot have object instance in object definition block.\n");
                string objectName;
                b32 result            = GetBetweenPair(objectName, &tokenizer, '"');
                string objectFileName = PushStr8F(tempArena, "objects/%S_obj.rtscene",
                                                  ReplaceColons(tempArena, objectName));
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
                ShapeType *shape    = &shapes->AddBack();
                ScenePacket *packet = &shape->packet;
                CreateScenePacket(tempArena, word, packet, &tokenizer,
                                  MemoryType_Shape); //, 1);

                // TODO: temp
                if (packet->type == "curve"_sid)
                {
                    shapes->Last() = {};
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
                        else packet->type = "trianglemesh"_sid;
                    }
                }
                if (packet->type == "trianglemesh"_sid && numVertices && numIndices &&
                    numVertices / 2 == numIndices / 3)
                {
                    packet->type = "quadmesh"_sid;
                }

                shape->materialName = currentGraphicsState.materialName;

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

                NamedPacket nPacket = {};
                ScenePacket *packet = &nPacket.packet;
                packet->type =
                    "texture"_sid; // Hash(StrConcat(tempArena, textureType, textureClass));

                PBRTSkipToNextChar(&tokenizer);

                ReadParameters(threadArena, packet, &tokenizer, MemoryType_Texture);
                nPacket.name = PushStr8Copy(threadArena, textureName);
                nPacket.type = PushStr8Copy(threadArena, textureClass);
                textureHashMap.Add(threadArena, nPacket);
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

void WriteTexture(StringBuilder *builder, const NamedPacket *packet)
{
    const ScenePacket *scenePacket = &packet->packet;
    // Put(builder, "t name %S type %S ", packet->name, packet->type);
    Put(builder, "%S ", packet->type);
    TextureType type = ConvertStringToTextureType(packet->type);
    switch (type)
    {
        case TextureType::ptex:
        {
            const string parameterNames[] = {"filename", "scale"};
            const StringId parameterIds[] = {"filename"_sid, "scale"_sid};
            u32 count                     = 2;
            Assert(parameterNames);
            for (u32 i = 0; i < count; i++)
            {
                for (u32 j = 0; j < scenePacket->parameterCount; j++)
                {
                    if (scenePacket->parameterNames[j] == parameterIds[i])
                    {
                        Put(builder, "%S ", parameterNames[i]);
                        Put(builder, scenePacket->bytes[j], scenePacket->sizes[j]);
                        Put(builder, " ");
                    }
                }
            }
        }
        break;
        default: Assert(0);
    }
}

void WriteMaterials(StringBuilder *builder, SceneHashMap *textureHashMap, NamedPacket &packet,
                    u32 hashMask)
{
    MaterialTypes type = ConvertStringToMaterialType(packet.type);
    if (type == MaterialTypes::Interface) return;
    u32 index = (u32)type;
    Put(builder, "m %S ", packet.name);
    Put(builder, "%S ", packet.type);
    const string *names = materialParameterNames[index];
    u32 count           = materialParameterCounts[index];
    const StringId *ids = materialParameterIDs[index];
    for (u32 i = 0; i < count; i++)
    {
        ScenePacket *scenePacket = &packet.packet;
        for (u32 p = 0; p < scenePacket->parameterCount; p++)
        {
            if (scenePacket->parameterNames[p] == ids[i])
            {
                Put(builder, "%S ", names[i]);
                Put(builder, "%u ", scenePacket->types[p]);
                switch (scenePacket->types[p])
                {
                    case DataType::Float:
                    {
                        u32 count = scenePacket->sizes[p] / sizeof(f32);
                        Assert(count == 1);

                        Put(builder, scenePacket->bytes[p], scenePacket->sizes[p]);
                        Put(builder, " ");
                    }
                    break;
                    case DataType::Vec3:
                    {
                        Put(builder, scenePacket->bytes[p], scenePacket->sizes[p]);
                        Put(builder, " ");
                    }
                    break;
                    case DataType::Spectrum:
                    {
                        Assert(0);
                    }
                    break;
                    case DataType::Texture:
                    {
                        string textureName =
                            Str8(scenePacket->bytes[p], scenePacket->sizes[p]);
                        const NamedPacket *packet = textureHashMap->Get(textureName);
                        WriteTexture(builder, packet);
                    }
                    break;
                    default: Error(0, "not supported yet\n");
                }
                break;
            }
        }
    }
}

void WriteData(StringBuilder *builder, StringBuilderMapped *dataBuilder, void *ptr, u64 size,
               string out)
{
    Assert(ptr);
    u64 offset = dataBuilder->totalSize;
    Put(dataBuilder, ptr, size);
    Put(builder, "%S %llu ", out, offset);
}

i32 WriteData(ScenePacket *packet, StringBuilder *builder, StringBuilderMapped *dataBuilder,
              StringId matchId, string out)
{
    for (u32 c = 0; c < packet->parameterCount; c++)
    {
        if (packet->parameterNames[c] == matchId)
        {
            WriteData(builder, dataBuilder, packet->bytes[c], packet->sizes[c], out);
            return c;
        }
    }
    return -1;
}

void WriteFile(string directory, PBRTFileInfo *info, SceneLoadState *state)
{
    if (info->shapes.totalCount == 0 && info->numInstances == 0) return;
    TempArena temp = ScratchStart(0, 0);
    Assert(GetFileExtension(info->filename) == "rtscene");
    string outFile = StrConcat(temp.arena, directory, info->filename);

    StringBuilder preBuilder = {};
    preBuilder.arena         = temp.arena;
    StringBuilder builder    = {};
    builder.arena            = temp.arena;

    u32 totalMaterialCount = 0;
    Put(&preBuilder, "RTSCENE_START ");

    if (info->fileInstances.totalCount && info->shapes.totalCount)
    {
        Print("%S has both instances and shapes\n", info->filename);
    }

    string dataBuilderFile =
        PushStr8F(temp.arena, "%S%S.rtdata", directory, RemoveFileExtension(info->filename));
    StringBuilderMapped dataBuilder(dataBuilderFile);

    Put(&dataBuilder, "DATA_START ");

    struct BuilderNode
    {
        string filename;
        StringBuilder builder = {};
        u64 prevEnd;
        BuilderNode *next;
    };

    BuilderNode bNode = {};

    Scheduler::Counter counter = {};
    StringBuilder *builders    = 0;

    if (state)
    {
        builders = PushArray(temp.arena, StringBuilder, state->numProcessors);

        SceneHashMap *textureHashMap = &state->textureHashMaps[0];
        for (u32 i = 1; i < state->numProcessors; i++)
        {
            textureHashMap->Merge(state->textureHashMaps[i]);
        }

        scheduler.Schedule(&counter, state->numProcessors, 1, [=](u32 jobID) {
            StringBuilder *builder = &builders[jobID];
            Arena *arena           = state->arenas[jobID];
            builder->arena         = arena;
            auto &list             = state->materials[jobID];

            for (auto *node = list.first; node != 0; node = node->next)
            {
                for (u32 j = 0; j < node->count; j++)
                {
                    auto &packet = node->values[j];
                    WriteMaterials(builder, textureHashMap, packet, state->hashMapSize - 1);
                }
            }
        });
    }

    if (info->shapes.totalCount && info->fileInstances.totalCount == 0)
    {
        Put(&builder, "SHAPE_START ");
        for (auto *node = info->shapes.first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ShapeType *shapeType = node->values + i;
                ScenePacket *packet  = &shapeType->packet;

                switch (packet->type)
                {
                    case "quadmesh"_sid:
                    {
                        Put(&builder, "Quad ");
                        bool plyMesh = false;
                        for (u32 c = 0; c < packet->parameterCount; c++)
                        {
                            if (packet->parameterNames[c] == "filename"_sid)
                            {
                                Mesh mesh = LoadQuadPLY(
                                    temp.arena,
                                    StrConcat(temp.arena, directory,
                                              Str8(packet->bytes[c], packet->sizes[c])));
                                WriteData(&builder, &dataBuilder, mesh.p,
                                          mesh.numVertices * sizeof(Vec3f), "p");
                                WriteData(&builder, &dataBuilder, mesh.n,
                                          mesh.numVertices * sizeof(Vec3f), "n");
                                // if (mesh.uv)
                                //     WriteData(&builder, &dataBuilder, mesh.uv,
                                //               mesh.numVertices * sizeof(Vec2f), "uv");
                                Put(&builder, "v %u ", mesh.numVertices);
                                plyMesh = true;
                                break;
                            }
                        }
                        if (!plyMesh)
                        {
                            i32 numVertices = 0;
                            i32 c = WriteData(packet, &builder, &dataBuilder, "P"_sid, "p");
                            if (c != -1) numVertices = packet->sizes[c] / sizeof(Vec3f);
                            WriteData(packet, &builder, &dataBuilder, "N"_sid, "n");
                            // WriteData(packet, &builder, &dataBuilder, "uv"_sid, "uv");
                            if (numVertices) Put(&builder, "v %u ", numVertices);
                        }
                    }
                    break;
                    case "trianglemesh"_sid:
                    {
                        Put(&builder, "Tri ");

                        bool plyMesh = false;
                        for (u32 c = 0; c < packet->parameterCount; c++)
                        {
                            if (packet->parameterNames[c] == "filename"_sid)
                            {
                                Mesh mesh = LoadPLY(
                                    temp.arena,
                                    StrConcat(temp.arena, directory,
                                              Str8(packet->bytes[c], packet->sizes[c])));
                                WriteData(&builder, &dataBuilder, mesh.p,
                                          mesh.numVertices * sizeof(Vec3f), "p");
                                WriteData(&builder, &dataBuilder, mesh.n,
                                          mesh.numVertices * sizeof(Vec3f), "n");
                                if (mesh.uv)
                                    WriteData(&builder, &dataBuilder, mesh.uv,
                                              mesh.numVertices * sizeof(Vec2f), "uv");
                                if (mesh.indices)
                                    WriteData(&builder, &dataBuilder, mesh.indices,
                                              mesh.numIndices * sizeof(u32), "indices");
                                Put(&builder, "v %u ", mesh.numVertices);
                                if (mesh.indices) Put(&builder, "i %u ", mesh.numIndices);
                                plyMesh = true;
                                break;
                            }
                        }
                        if (!plyMesh)
                        {
                            u32 numVertices = 0;
                            u32 numIndices  = 0;
                            i32 c           = -1;
                            c = WriteData(packet, &builder, &dataBuilder, "P"_sid, "p");
                            if (c != -1) numVertices = packet->sizes[c] / sizeof(Vec3f);
                            WriteData(packet, &builder, &dataBuilder, "N"_sid, "n");
                            WriteData(packet, &builder, &dataBuilder, "uv"_sid, "uv");
                            c = WriteData(packet, &builder, &dataBuilder, "indices"_sid,
                                          "indices");
                            if (c != -1) numIndices = packet->sizes[c] / sizeof(u32);
                            if (numVertices) Put(&builder, "v %u ", numVertices);
                            if (numIndices) Put(&builder, "i %u ", numIndices);
                        }
                    }
                    break;
                    case "curve"_sid:
                    {
                    }
                    break;
                    default: Assert(0);
                }
                if (shapeType->materialName.size)
                    Put(&builder, "m %S ", shapeType->materialName);
            }
        }
        Put(&builder, "SHAPE_END ");
        Put(&dataBuilder, "DATA_END");
    }
    else if (info->shapes.totalCount)
    {
        PBRTFileInfo *shapeInfo = PushStruct(temp.arena, PBRTFileInfo);
        shapeInfo->filename =
            PushStr8F(temp.arena, "%S_shape.rtscene", RemoveFileExtension(info->filename));
        shapeInfo->shapes = info->shapes;
        WriteFile(directory, shapeInfo);

        u32 transformIndex            = info->transforms.Length();
        info->transforms.AddBack()    = AffineSpace::Identity();
        info->fileInstances.AddBack() = {shapeInfo->filename, transformIndex, transformIndex};
    }

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
                    node->prevEnd       = fileInst->transformIndexEnd;
                    Put(&node->builder, "File: %S %u-", node->filename,
                        fileInst->transformIndexStart);
                }
                else
                {
                    if (node->prevEnd + 1 == fileInst->transformIndexStart)
                        node->prevEnd = fileInst->transformIndexEnd;
                    else
                    {
                        Put(&node->builder, "%u %u-", node->prevEnd,
                            fileInst->transformIndexStart);
                        node->prevEnd = fileInst->transformIndexEnd;
                    }
                }
            }
        }
        BuilderNode *node = &bNode;
        while (node->next)
        {
            Put(&node->builder, "%u ", node->prevEnd);
            builder = ConcatBuilders(&builder, &node->builder);
            node    = node->next;
        }
        Put(&builder, "INCLUDE_END ");
    }

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

    if (state)
    {
        Assert(builders);
        Put(&preBuilder, "MATERIALS_START ");
        for (u32 i = 0; i < state->numProcessors; i++)
        {
            preBuilder = ConcatBuilders(&preBuilder, &builders[i]);
        }
        Put(&preBuilder, "MATERIALS_END ");
    }

    Put(&builder, "RTSCENE_END");
    builder = ConcatBuilders(&preBuilder, &builder);
    WriteFileMapped(&builder, outFile);
    OS_UnmapFile(dataBuilder.ptr);
    OS_ResizeFile(dataBuilder.filename, dataBuilder.totalSize);
    ScratchEnd(temp);
}

void LoadPBRT(Arena *arena, string filename)
{
    TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();

    SceneLoadState sls;
    sls.Init(arena);

    string directory = "../data/island/pbrt-v4/";
    OS_CreateDirectory(StrConcat(temp.arena, directory, "objects"));

    PerformanceCounter counter = OS_StartCounter();
    LoadPBRT(&sls, "../data/island/pbrt-v4/", filename);
    f32 time = OS_GetMilliseconds(counter);
    printf("convert time: %fms\n", time);

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
