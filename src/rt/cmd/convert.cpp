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
#include "../scene_load.h"
#include "../win32.cpp"
#include "../memory.cpp"
#include "../string.cpp"
#include "../thread_context.cpp"

namespace rt
{

struct QuadMesh;
struct TriangleMesh;
struct Instance;

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

bool CheckQuadPLY(string filename)
{
    string buffer = OS_MapFileRead(filename); // OS_ReadFile(arena, filename);
    Tokenizer tokenizer;
    tokenizer.input  = buffer;
    tokenizer.cursor = buffer.str;

    string line = ReadLine(&tokenizer);
    Assert(line == "ply");
    line = ReadLine(&tokenizer);
    // TODO: really need to handle big endian, used in some of the stanford models
    Assert(line == "format binary_little_endian 1.0");

    u32 numVertices = 0;
    u32 numFaces    = 0;

    for (;;)
    {
        string word = ReadWord(&tokenizer);
        if (word == "element")
        {
            string elementType = ReadWord(&tokenizer);
            if (elementType == "vertex")
            {
                numVertices = ConvertToUint(ReadWord(&tokenizer));
                for (;;)
                {
                    word = CheckWord(&tokenizer);
                    if (word == "element") break;
                    ReadWord(&tokenizer);
                    word = ReadWord(&tokenizer);
                    Assert(word == "float");
                    word = ReadWord(&tokenizer);
                }
            }
            else if (elementType == "face")
            {
                numFaces = ConvertToUint(ReadWord(&tokenizer));
                for (;;)
                {
                    word = CheckWord(&tokenizer);
                    if (word == "element" || word == "end_header") break;
                    // Skips the word "property"
                    ReadWord(&tokenizer);
                    word = ReadWord(&tokenizer);
                    if (word == "list")
                    {
                        ReadWord(&tokenizer);
                        ReadWord(&tokenizer);
                        ReadWord(&tokenizer);
                    }
                    else
                    {
                        ReadWord(&tokenizer);
                    }
                }
            }
            else
            {
                Error(0, "elementType: %s\n", (char *)elementType.str);
            }
        }
        else if (word == "end_header") break;
    }

    // 2 triangles/1 quad for every 4 vertices. If this condition isn't met, it isn't a quad
    // mesh
    return numFaces == numVertices / 2;
}

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

template <typename T, i32 numPerChunk, i32 memoryTag = 0>
struct ChunkedLinkedList
{
    Arena *arena;
    struct ChunkNode
    {
        T values[numPerChunk];
        u32 count;
        ChunkNode *next;
    };
    ChunkNode *first;
    ChunkNode *last;
    u32 totalCount;

    struct Iterator
    {
        ChunkNode *node;
        u32 localIndex;
        u32 numRemaining;

        bool End() { return numRemaining == 0; }

        void Next()
        {
            localIndex++;
            numRemaining--;
            if (localIndex = numPerChunk)
            {
                node = node->next;
            }
        }
        T *Get() { return node->values[localIndex]; }
    };

    Iterator Itr(u32 start, u32 end)
    {
        Iterator itr;
        itr.node         = first;
        itr.numRemaining = end - start;
        u32 index        = start;
        while (index > numPerChunk)
        {
            itr.node = itr.node->next;
            index -= numPerChunk;
        }
        itr.localIndex = index;
        return itr;
    }

    ChunkedLinkedList(Arena *arena) : arena(arena), first(0), last(0), totalCount(0)
    {
        AddNode();
    }
    T &AddBack()
    {
        if (last->count >= numPerChunk)
        {
            AddNode();
        }
        T &result = last->values[last->count++];
        totalCount++;
        return result;
    }
    T &operator[](u32 i)
    {
        Assert(i < totalCount);
        ChunkNode *node = first;
        for (;;)
        {
            Assert(node);
            if (i >= node->count)
            {
                i -= node->count;
                node = node->next;
            }
            else
            {
                return node->values[i];
            }
        }
    }
    inline void Push(T &val) { AddBack() = val; }
    inline void Push(T &&val) { AddBack() = std::move(val); }
    inline const T &Last() const { return last->values[last->count - 1]; }

    inline void AddNode()
    {
        ChunkNode *newNode = PushStructTagged(arena, ChunkNode, memoryTag);
        QueuePush(first, last, newNode);
    }
    inline u32 Length() const { return totalCount; }
};

template <i32 numNodes, i32 chunkSize, i32 numStripes>
struct InternedStringCache
{
    StaticAssert(IsPow2(numNodes), CachePow2N);
    StaticAssert(IsPow2(numStripes), CachePow2Stripes);
    struct ChunkNode
    {
        StringId stringIds[chunkSize];
#ifdef DEBUG
        string str[chunkSize];
#endif
        u32 count;
        ChunkNode *next;
    };
    ChunkNode *nodes;
    Mutex *mutexes;

    InternedStringCache() {}
    InternedStringCache(Arena *arena)
    {
        nodes   = PushArrayTagged(arena, ChunkNode, numNodes, MemoryType_String);
        mutexes = PushArray(arena, Mutex, numStripes);
    }
    StringId GetOrCreate(Arena *arena, string value);
#ifdef DEBUG
    string Get(StringId id);
#endif
};

template <i32 numNodes, i32 chunkSize, i32 numStripes>
StringId InternedStringCache<numNodes, chunkSize, numStripes>::GetOrCreate(Arena *arena,
                                                                           string value)
{
    StringId sid = Hash(value);
    Assert(sid != StringId::Invalid);
    ChunkNode *node = &nodes[sid & (numNodes - 1)];
    ChunkNode *prev = 0;

    u32 stripe = sid & (numStripes - 1);
    BeginRMutex(&mutexes[stripe]);
    while (node)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (sid == node->stringIds[i])
            {
#ifdef DEBUG
                Error(node->str[i] == value, "Hash collision between %S and %S\n", value,
                      node->str[i]);
#endif
                EndRMutex(&mutexes[stripe]);
                return node->stringIds[i];
            }
        }
        prev = node;
        node = node->next;
    }
    EndRMutex(&mutexes[stripe]);

    StringId result = 0;
    BeginWMutex(&mutexes[stripe]);
    if (prev->count == ArrayLength(prev->stringIds))
    {
        node       = PushStructTagged(arena, ChunkNode, MemoryType_String);
        prev->next = node;
        prev       = node;
    }
    prev->stringIds[prev->count] = sid;
#ifdef DEBUG
    prev->str[prev->count] = PushStr8Copy(arena, value);
#endif
    prev->count++;
    EndWMutex(&mutexes[stripe]);

    return sid;
}

struct ObjectInstanceType
{
    StringId name;
    // string name;
    u32 transformIndex  = 0;
    u32 shapeIndexStart = 0xffffffff;
    u32 shapeIndexEnd   = 0xffffffff;
    u32 shapeTypeFlags  = 0;
    u32 shapeTypeCount[(u32)GeometryType::Max];

    bool Invalid() const { return shapeIndexStart == 0xffffffff; }
    void Invalidate() { shapeIndexStart = 0xffffffff; }
};

struct SceneInstance
{
    StringId name;
    u32 transformIndex;
};

struct SceneLoadState
{
    // Map<ScenePacket> caches[64];
    // std::atomic<u32> numCaches;
    enum Type
    {
        Film,
        Camera,
        Sampler,
        Integrator,
        Accelerator,
        MAX,
    };

    ScenePacket packets[MAX] = {};

    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape> *shapes;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape> *instanceShapes;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Material> *materials;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Texture> *textures;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Light> *lights;
    ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance> *instanceTypes;
    ChunkedLinkedList<SceneInstance, 1024, MemoryType_Instance> *instances;

    ChunkedLinkedList<const AffineSpace *, 16384, MemoryType_Transform> *transforms;

    // TODO: other shapes?
    u32 *numQuadMeshes;
    u32 *numTriMeshes;
    u32 *totalNumInstTypes;
    u32 **shapeTypeCount;
    // u32 *numCurves;

    InternedStringCache<16384, 8, 64> stringCache;
    HashSet<AffineSpace, 1048576, 8, 1024, MemoryType_Transform> transformCache;

    Arena **tempArenas;
    Arena *mainArena;

    Scheduler::Counter counter = {};
};

struct GraphicsState
{
    StringId materialId = 0;
    i32 materialIndex   = -1;
    // Mat4 transform      = Mat4::Identity();
    AffineSpace transform = AffineSpace::Identity();
    u32 transformIndex    = 0;

    i32 areaLightIndex = -1;
    i32 mediaIndex     = -1;

    ObjectInstanceType *instanceType = 0;
};

// NOTE: sets the camera, film, sampler, etc.
template <i32 numNodes, i32 chunkSize, i32 numStripes>
void CreateScenePacket(Arena *arena, string word, ScenePacket *packet, Tokenizer *tokenizer,
                       InternedStringCache<numNodes, chunkSize, numStripes> *stringCache,
                       MemoryType memoryType, u32 additionalParameters = 0)
{
    string type;
    b32 result = GetBetweenPair(type, tokenizer, '"');
    Assert(result);
    packet->type = stringCache->GetOrCreate(arena, type);
    if (IsEndOfLine(tokenizer))
    {
        SkipToNextLine(tokenizer);
    }
    else
    {
        SkipToNextChar(tokenizer);
    }

    ReadParameters(arena, packet, tokenizer, stringCache, memoryType, additionalParameters);
}

inline void SkipToNextDigitArray(Tokenizer *tokenizer)
{
    while (!EndOfBuffer(tokenizer) &&
           (!IsDigit(tokenizer) && *tokenizer->cursor != '-' && *tokenizer->cursor != ']'))
        tokenizer->cursor++;
}

inline void AdvanceToNextLine(Tokenizer *tokenizer)
{
    bool nextLine = false;
    while (CharIsBlank(*tokenizer->cursor))
    {
        tokenizer->cursor++;
    }
    if (*tokenizer->cursor == ']')
    {
        tokenizer->cursor++;
    }
    while (CharIsBlank(*tokenizer->cursor))
    {
        tokenizer->cursor++;
    }
}

template <i32 numNodes, i32 chunkSize, i32 numStripes>
void ReadParameters(Arena *arena, ScenePacket *packet, Tokenizer *tokenizer,
                    InternedStringCache<numNodes, chunkSize, numStripes> *stringCache,
                    MemoryType memoryType, u32 additionalParameters = 0)
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
        if (result == 0) break;
        if (result == 2)
        {
            SkipToNextLine(tokenizer);
            continue;
        }
        string dataType      = GetFirstWord(infoType);
        u32 currentParam     = packet->parameterCount++;
        string parameterName = GetNthWord(infoType, 2);

        SkipToNextChar(tokenizer);

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
            AdvanceToNextLine(tokenizer);
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
            AdvanceToNextLine(tokenizer);
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
            AdvanceToNextLine(tokenizer);
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
            AdvanceToNextLine(tokenizer);
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
            AdvanceToNextLine(tokenizer);
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
            AdvanceToNextLine(tokenizer);
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
                AdvanceToNextLine(tokenizer);
            }
        }
        else
        {
            Error(0, "Invalid data type: %S\n", dataType);
        }
        parameterNames[currentParam] = stringCache->GetOrCreate(arena, parameterName);
        bytes[currentParam]          = out;
        sizes[currentParam]          = size;
    }
    packet->Initialize(arena, packet->parameterCount + additionalParameters);
    MemoryCopy(packet->parameterNames, parameterNames,
               sizeof(StringId) * packet->parameterCount);
    MemoryCopy(packet->bytes, bytes, sizeof(u8 *) * packet->parameterCount);
    MemoryCopy(packet->sizes, sizes, sizeof(u32) * packet->parameterCount);
}

void LoadPBRT(string filename, string directory, SceneLoadState *state,
              GraphicsState graphicsState = {}, bool inWorldBegin = false,
              bool imported = false)
{
    Scheduler::Counter counter = {};
    TempArena temp             = ScratchStart(0, 0);
    u32 threadIndex            = GetThreadIndex();
    Arena *tempArena           = state->tempArenas[threadIndex];

    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(filename);
    tokenizer.cursor = tokenizer.input.str;

    auto &shapes         = state->shapes[threadIndex];
    auto &instanceShapes = state->instanceShapes[threadIndex];
    auto &materials      = state->materials[threadIndex];
    auto &textures       = state->textures[threadIndex];
    auto &lights         = state->lights[threadIndex];
    auto &instanceTypes  = state->instanceTypes[threadIndex];
    auto &instances      = state->instances[threadIndex];

    auto &transforms  = state->transforms[threadIndex];
    auto &stringCache = state->stringCache;

    bool worldBegin = inWorldBegin;

    // Stack variables
    Tokenizer oldTokenizers[32];
    u32 numTokenizers = 0;

    GraphicsState graphicsStateStack[64];
    u32 graphicsStateCount = 0;

    GraphicsState currentGraphicsState = graphicsState;
    ObjectInstanceType *&currentObject = currentGraphicsState.instanceType;

    if (transforms.totalCount == 0)
    {
        transforms.Push(state->transformCache.GetOrCreate(tempArena, AffineSpace::Identity()));
    }
    if (currentObject && imported)
    {
        StringId name                  = currentObject->name;
        currentObject                  = &instanceTypes.AddBack();
        currentObject->name            = name;
        currentObject->shapeIndexStart = shapes.Length();
    }

    auto AddTransform = [&]() {
        if (currentGraphicsState.transformIndex != 0)
        {
            if (currentGraphicsState.transformIndex == transforms.Length())
            {
                // Assert(currentGraphicsState.transformIndex ==
                // transforms.Length());
                const AffineSpace *transform = state->transformCache.GetOrCreate(
                    tempArena, currentGraphicsState.transform);
                transforms.Push(transform);
            }
            else
            {
                const AffineSpace *t = transforms.Last();
                Assert(*t == currentGraphicsState.transform);
            }
        }
    };

    // TODO: media
    for (;;)
    {
        if (EndOfBuffer(&tokenizer))
        {
            if (currentObject && imported)
            {
                currentObject->shapeIndexEnd = shapes.Length();
                Assert(currentObject->shapeIndexEnd >= currentObject->shapeIndexStart);
            }
            OS_UnmapFile(tokenizer.input.str);
            if (numTokenizers == 0) break;
            tokenizer = oldTokenizers[--numTokenizers];
            continue;
        }
        if (graphicsStateCount != 0) SkipToNextChar(&tokenizer);

        string word = ReadWordAndSkipToNextWord(&tokenizer);
        // Comments/Blank lines
        if (word.size == 0)
        {
            continue;
        }
        if (word.str[0] == '#')
        {
            SkipToNextLine(&tokenizer);
            continue;
        }

        StringId sid = stringCache.GetOrCreate(tempArena, word);
        switch (sid)
        {
            case "Accelerator"_sid:
            {
                Error(!worldBegin,
                      "%S cannot be specified after WorldBegin "
                      "statement\n",
                      word);
                SceneLoadState::Type type = SceneLoadState::Type::Accelerator;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
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
            }
            break;
            case "AttributeEnd"_sid:
            {
                Error(worldBegin,
                      "%S cannot be specified before WorldBegin "
                      "statement\n",
                      word);
                Assert(graphicsStateCount > 0);

                // AddTransform();

                // Pop stack
                currentGraphicsState = graphicsStateStack[--graphicsStateCount];
            }
            break;
            // TODO: area light count is reported as 23 when there's 22
            case "AreaLightSource"_sid:
            {
                Error(worldBegin,
                      "%S cannot be specified before WorldBegin "
                      "statement\n",
                      word);
                currentGraphicsState.areaLightIndex = lights.Length();
                ScenePacket *packet                 = &lights.AddBack();
                CreateScenePacket(tempArena, word, packet, &tokenizer, &stringCache,
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
                SceneLoadState::Type type = SceneLoadState::Type::Camera;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
                continue;
            }
            case "ConcatTransform"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
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
                SceneLoadState::Type type = SceneLoadState::Type::Film;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
                continue;
            }
            case "Integrator"_sid:
            {
                Error(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
                SceneLoadState::Type type = SceneLoadState::Type::Integrator;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
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
                string importedFullPath = StrConcat(tempArena, directory, importedFilename);

                scheduler.Schedule(&counter, [importedFullPath, directory, state,
                                              currentGraphicsState, worldBegin](u32 jobID) {
                    LoadPBRT(importedFullPath, directory, state, currentGraphicsState,
                             worldBegin, true);
                });
            }
            break;
            case "Include"_sid:
            {
                string importedFilename;
                b32 result = GetBetweenPair(importedFilename, &tokenizer, '"');
                Assert(result);
                string importedFullPath = StrConcat(temp.arena, directory, importedFilename);
                Assert(numTokenizers < ArrayLength(oldTokenizers));
                oldTokenizers[numTokenizers++] = tokenizer;

                tokenizer.input  = OS_MapFileRead(importedFullPath);
                tokenizer.cursor = tokenizer.input.str;
            }
            break;
            case "LookAt"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
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
                ScenePacket *packet = &lights.AddBack();
                CreateScenePacket(tempArena, word, packet, &tokenizer, &stringCache,
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

                ScenePacket *packet = &materials.AddBack();
                packet->type        = stringCache.GetOrCreate(tempArena, materialNameOrType);
                u32 materialIndex   = materials.Length();
                if (IsEndOfLine(&tokenizer))
                {
                    SkipToNextLine(&tokenizer);
                }
                else
                {
                    SkipToNextChar(&tokenizer);
                }
                ReadParameters(tempArena, packet, &tokenizer, &stringCache,
                               MemoryType_Material);

                if (isNamedMaterial)
                {
                    currentGraphicsState.materialId =
                        stringCache.GetOrCreate(tempArena, materialNameOrType);
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

                currentGraphicsState.materialId =
                    stringCache.GetOrCreate(tempArena, materialName);
                currentGraphicsState.materialIndex = -1;
            }
            break;
            case "ObjectBegin"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                Error(currentObject == 0, "ObjectBegin cannot be called recursively.");
                Error(currentGraphicsState.areaLightIndex == -1,
                      "Area lights instancing not supported.");
                string objectName;

                b32 result = GetBetweenPair(objectName, &tokenizer, '"');
                Assert(result);

                currentObject                 = &instanceTypes.AddBack();
                currentObject->name           = stringCache.GetOrCreate(tempArena, objectName);
                currentObject->transformIndex = currentGraphicsState.transformIndex;
                currentObject->shapeIndexStart = shapes.Length();

                AddTransform();
            }
            break;
            case "ObjectEnd"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);

                currentObject->shapeIndexEnd = shapes.Length();
                state->totalNumInstTypes += PopCount(currentObject->shapeTypeFlags);
                Assert(currentObject->shapeIndexEnd >= currentObject->shapeIndexStart);
                Error(currentObject != 0, "ObjectEnd must occur after ObjectBegin");
                currentObject = 0;
            }
            break;
            case "ObjectInstance"_sid:
            {
                Error(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                string objectName;
                b32 result = GetBetweenPair(objectName, &tokenizer, '"');
                Assert(result);

                SceneInstance &instance = instances.AddBack();
                instance.name           = stringCache.GetOrCreate(tempArena, objectName);
                instance.transformIndex = (i32)transforms.Length();

                AddTransform();
                Assert(IsEndOfLine(&tokenizer));
                SkipToNextLine(&tokenizer);
            }
            break;
            case "PixelFilter"_sid:
            {
                SkipToNextLine(&tokenizer);
            }
            break;
            case "Rotate"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
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
                SceneLoadState::Type type = SceneLoadState::Type::Sampler;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
                continue;
            }
            break;
            case "Scale"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
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
                ScenePacket *packet;
                if (currentObject)
                {
                    packet = &instanceShapes.AddBack();
                }
                else
                {
                    packet = &shapes.AddBack();
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
                            state->numQuadMeshes[threadIndex]++;
                        else state->numTriMeshes[threadIndex]++;
                    }
                }
                if (packet->type == "trianglemesh"_sid && numVertices && numIndices &&
                    numVertices / 2 == numIndices / 3)
                {
                    packet->type = stringCache.GetOrCreate(tempArena, "quadmesh");
                    state->numQuadMeshes[GetThreadIndex()]++;
                    if (currentObject)
                    {
                        u32 index = (u32)(GeometryType::QuadMesh);
                        currentObject->shapeTypeFlags |= (1 << index);
                        currentObject->shapeTypeCount[index]++;
                    }
                    else
                    {
                        u32 index = (u32)(GeometryType::QuadMesh);
                        state->shapeTypeCount[GetThreadIndex()][index]++;
                    }
                }
                else if (packet->type == "trianglemesh"_sid)
                {
                    state->numTriMeshes[GetThreadIndex()]++;
                    if (currentObject)
                    {
                        u32 index = (u32)(GeometryType::TriangleMesh);
                        currentObject->shapeTypeFlags |= (1 << index);
                        currentObject->shapeTypeCount[index]++;
                    }
                    else
                    {
                        u32 index = (u32)(GeometryType::TriangleMesh);
                        state->shapeTypeCount[GetThreadIndex()][index]++;
                    }
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

                u32 currentParameter = packet->parameterCount++;
                packet->parameterNames[currentParameter] =
                    stringCache.GetOrCreate(tempArena, "Indices");
                packet->bytes[currentParameter] = (u8 *)indices;
                packet->sizes[currentParameter] = sizeof(i32) * 4;

                AddTransform();
            }
            break;
            case "Translate"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
                f32 t0                              = ReadFloat(&tokenizer);
                f32 t1                              = ReadFloat(&tokenizer);
                f32 t2                              = ReadFloat(&tokenizer);

                AffineSpace t                  = AffineSpace::Translate(Vec3f(t0, t1, t2));
                currentGraphicsState.transform = currentGraphicsState.transform * t;
            }
            break;
            case "Transform"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
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

                SkipToNextChar(&tokenizer);
                bool result = Advance(&tokenizer, "]\n");
                Assert(result);
            }
            break;
            case "Texture"_sid:
            {
                string textureName;
                b32 result = GetBetweenPair(textureName, &tokenizer, '"');

                Assert(result);
                string textureType;
                result = GetBetweenPair(textureType, &tokenizer, '"');
                Assert(result);
                string textureClass;
                result = GetBetweenPair(textureClass, &tokenizer, '"');
                Assert(result);

                ScenePacket *packet = &textures.AddBack();
                packet->type        = stringCache.GetOrCreate(
                    tempArena, StrConcat(tempArena, textureType, textureClass));

                if (IsEndOfLine(&tokenizer))
                {
                    SkipToNextLine(&tokenizer);
                }
                else
                {
                    SkipToNextChar(&tokenizer);
                }
                ReadParameters(tempArena, packet, &tokenizer, &stringCache,
                               MemoryType_Texture);
            }
            break;
            case "WorldBegin"_sid:
            {
                // NOTE: this assumes "WorldBegin" only occurs in one
                // file
                worldBegin = true;

                const ScenePacket *filmPacket = &state->packets[SceneLoadState::Type::Film];
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
                    &state->packets[SceneLoadState::Type::Sampler];
                // state->scene->sampler =
                //     Sampler::Create(state->mainArena, samplerPacket,
                //     fullResolution);

                AddTransform();
                // TODO: instantiate the camera with the current
                // transform
                currentGraphicsState.transform = AffineSpace::Identity();
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
    scheduler.Wait(&counter);
    // When everything is done...

    // TODO: see if this is the "beginning file" if so...
    // if (main)
    // {
    // }
    // StringBuilder builder = {};

    ScratchEnd(temp);
}

// struct PBRTMetaInfo
// {
//     u32 numMaterials[(u32)MaterialType::Max];
// };

struct MaterialHashNode
{
    u64 hash;
    ScenePacket *packet;
    MaterialHashNode *next;
};

void WriteMeta(StringBuilder *builder, string filename, SceneLoadState *state)
{
    TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();

    u32 tentativeCount = 0;
    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        auto *list = &state->materials[pIndex];
        tentativeCount += list->totalCount;
    }

    u32 hashTableSize     = Max(1024, NextPowerOfTwo(tentativeCount));
    u32 hashMask          = hashTableSize - 1;
    MaterialHashNode *map = PushArray(temp.arena, MaterialHashNode, hashTableSize);
    u32 materialCount     = 0;
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
                        u64 popPos    = ArenaPos(temp.arena);
                        u32 typeSize  = CalculateCStringLength("diffuse");
                        u32 totalSize = typeSize;

                        u8 *reflectanceBytes = 0;
                        for (u32 i = 0; i < packet->parameterCount; i++)
                        {
                            if (packet->parameterNames[i] == "reflectance"_sid)
                            {
                                totalSize += packet->sizes[i];
                                reflectanceBytes = packet->bytes[i];
                            }
                        }
                        Assert(reflectanceBytes);
                        u8 *buffer = PushArrayNoZero(temp.arena, u8, totalSize);
                        MemoryCopy(buffer, "diffuse", typeSize);
                        u32 offset = typeSize;
                        for (u32 i = 0; i < packet->parameterCount; i++)
                        {
                            if (packet->parameterNames[i] == "reflectance"_sid)
                            {
                                MemoryCopy(buffer + offset, packet->bytes, packet->sizes[i]);
                                offset += packet->sizes[i];
                            }
                        }
                        u64 hash = MurmurHash64A(buffer, totalSize, 0);
                        ArenaPopTo(temp.arena, popPos);

                        MaterialHashNode *node = &map[hash & hashMask];
                        MaterialHashNode *prev;
                        while (node)
                        {
                            if (node->hash == hash)
                            {
                                bool done = false;
                                for (u32 i = 0; i < node->packet->parameterCount; i++)
                                {
                                    if (node->packet->parameterNames[i] == "reflectance"_sid)
                                    {
                                        if (memcmp(node->packet->bytes[i], reflectanceBytes,
                                                   node->packet->sizes[i]) == 0)
                                        {
                                            done = true;
                                            break;
                                        }
                                    }
                                }
                                if (done) break;
                            }
                            prev = node;
                            node = node->next;
                        }
                        if (!node)
                        {
                            prev->hash   = hash;
                            prev->packet = packet;
                            prev->next   = PushStruct(temp.arena, MaterialHashNode);
                            materialCount++;
                        }
                    }
                    break;
                    case "diffusetransmission"_sid:
                    {
                    }
                    break;
                    case "dielectric"_sid:
                    {
                    }
                    break;
                    case "coateddiffuse"_sid:
                    {
                    }
                    break;
                    default: Error(0, "Material type string is invalid. Aborting...\n");
                }
            }
        }
    }
    printf("Total # unique materials: %u\n", materialCount);
}

#if 0
void WriteRTSceneFile(string filename, string directory, SceneLoadState *state)
{
    u32 numProcessors     = OS_NumProcessors();
    TempArena temp        = ScratchStart(0, 0);
    StringBuilder builder = {};
    builder.arena         = temp.arena;

    // Magic
    Put(&builder, "RTF_START");
    FileOffsets offsets = {};
    offsets.metaOffset  = builder.totalSize + sizeof(FileOffsets);
    PutPointerValue(&builder, &offsets);

    // Materials
    Put(&builder, "MATERIALS_START");
    u32 materialTotal = 0;
    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        auto *list = &state->materials[pIndex];
        for (auto *node = list->first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ScenePacket *packet = &node->values[i];
                switch (packet->type)
                {
                    case "diffuse"_id:
                    {
                    }
                    break;
                    case "diffusetransmission"_id:
                    {
                    }
                    break;
                    case "dielectric"_id:
                    {
                    }
                    break;
                    case "coateddiffuse"_id:
                    {
                    }
                    break;
                    default: Error(0, "Material type string is invalid. Aborting...\n");
                }
            }
        }
    }
    // Put(&builder, "#DIFFUSE_MATERIAL$");
    // Put(&builder, "MATERIALS_END");
    // Lights
}
#endif

void LoadPBRT(Arena *arena, string filename)
{
#define COMMA ,
    SceneLoadState state;
    u32 numProcessors       = OS_NumProcessors();
    state.numTriMeshes      = PushArray(arena, u32, numProcessors);
    state.numQuadMeshes     = PushArray(arena, u32, numProcessors);
    state.totalNumInstTypes = PushArray(arena, u32, numProcessors);
    state.shapeTypeCount    = PushArray(arena, u32 *, numProcessors);
    state.shapes =
        PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Shape>,
                  numProcessors);
    state.instanceShapes =
        PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Shape>,
                  numProcessors);
    state.materials =
        PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Material>,
                  numProcessors);
    state.textures =
        PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Texture>,
                  numProcessors);
    state.lights =
        PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Light>,
                  numProcessors);
    state.instanceTypes = PushArray(
        arena, ChunkedLinkedList<ObjectInstanceType COMMA 512 COMMA MemoryType_Instance>,
        numProcessors);
    state.instances =
        PushArray(arena, ChunkedLinkedList<SceneInstance COMMA 1024 COMMA MemoryType_Instance>,
                  numProcessors);
    state.transforms = PushArray(
        arena, ChunkedLinkedList<const AffineSpace * COMMA 16384 COMMA MemoryType_Transform>,
        numProcessors);
    state.tempArenas = PushArray(arena, Arena *, numProcessors);
#undef COMMA

    for (u32 i = 0; i < numProcessors; i++)
    {
        state.tempArenas[i]     = ArenaAlloc(16);
        Arena *threadArena      = state.tempArenas[i];
        state.shapeTypeCount[i] = PushArray(threadArena, u32, (u32)(GeometryType::Max));
        state.shapes[i] = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape>(threadArena);
        state.instanceShapes[i] =
            ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape>(threadArena);
        state.materials[i] =
            ChunkedLinkedList<ScenePacket, 1024, MemoryType_Material>(threadArena);
        state.textures[i] =
            ChunkedLinkedList<ScenePacket, 1024, MemoryType_Texture>(threadArena);
        state.lights[i] = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Light>(threadArena);
        state.instanceTypes[i] =
            ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance>(threadArena);
        state.instances[i] =
            ChunkedLinkedList<SceneInstance, 1024, MemoryType_Instance>(threadArena);
        state.transforms[i] =
            ChunkedLinkedList<const AffineSpace *, 16384, MemoryType_Transform>(threadArena);
    }
    state.mainArena      = arena;
    state.stringCache    = InternedStringCache<16384, 8, 64>(arena);
    state.transformCache = HashSet<AffineSpace, 1048576, 8, 1024, MemoryType_Transform>(arena);

    string baseDirectory = Str8PathChopPastLastSlash(filename);
    LoadPBRT(filename, baseDirectory, &state);

    scheduler.Wait(&state.counter);

    u64 totalNumShapes        = 0;
    u64 totalNumMaterials     = 0;
    u64 totalNumTextures      = 0;
    u64 totalNumLights        = 0;
    u64 totalNumInstanceTypes = 0;
    u64 totalNumInstances     = 0;
    u64 totalNumTransforms    = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalNumShapes += state.shapes[i].totalCount;
        totalNumMaterials += state.materials[i].totalCount;
        totalNumTextures += state.textures[i].totalCount;
        totalNumLights += state.lights[i].totalCount;
        totalNumInstanceTypes += state.instanceTypes[i].totalCount;
        totalNumInstances += state.instances[i].totalCount;
        totalNumTransforms += state.transforms[i].totalCount;
    }

    printf("Total num shapes: %lld\n", totalNumShapes);
    printf("Total num materials: %lld\n", totalNumMaterials);
    printf("Total num textures: %lld\n", totalNumTextures);
    printf("Total num lights: %lld\n", totalNumLights);
    printf("Total num instance types: %lld\n", totalNumInstanceTypes);
    printf("Total num instances: %lld\n", totalNumInstances);
    printf("Total num transforms: %lld\n", totalNumTransforms);

    // WriteRTSceneFile(filename, &state);

    StringBuilder builder = {};
    WriteMeta(&builder, filename, &state);

    for (u32 i = 0; i < numProcessors; i++)
    {
        ArenaClear(state.tempArenas[i]);
    }
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
