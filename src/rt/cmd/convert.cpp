#include "../base.h"
#include "../thread_statistics.h"
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
#include "../math/eigen.h"
#include "../math/sphere.h"

#include "../platform.h"
#include "../memory.h"
#include "../string.h"
#include "../containers.h"
#include "../thread_context.h"
#include "../hash.h"
#include <functional>
#include "../radix_sort.h"
#include "../random.h"
#include "../parallel.h"
#include "../graphics/ptex.h"
#include "../graphics/vulkan.h"
#include "../handles.h"
#include "../scene_load.h"
#include "../mesh.h"
#ifdef USE_GPU
#include "../virtual_geometry/mesh_simplification.h"
#endif

namespace rt
{

struct DisneyMaterial
{
    string name;
    string colorMap;
    float diffTrans;
    Vec4f baseColor;
    float specTrans;
    float clearcoatGloss;
    Vec3f scatterDistance;
    float clearcoat;
    float specularTint;
    float ior;
    float metallic;
    float flatness;
    float sheen;
    float sheenTint;
    float anisotropic;
    float alpha;
    float roughness;

    bool thin;

    Array<string> assignments;
};

struct DiskDisneyMaterial
{
    float diffTrans;
    Vec4f baseColor;
    float specTrans;
    float clearcoatGloss;
    Vec3f scatterDistance;
    float clearcoat;
    float specularTint;
    float ior;
    float metallic;
    float flatness;
    float sheen;
    float sheenTint;
    float anisotropic;
    float alpha;
    float roughness;
    bool thin;

    DiskDisneyMaterial(DisneyMaterial &material)
    {
        diffTrans       = material.diffTrans;
        baseColor       = material.baseColor;
        specTrans       = material.specTrans;
        clearcoatGloss  = material.clearcoatGloss;
        scatterDistance = material.scatterDistance;
        clearcoat       = material.clearcoat;
        specularTint    = material.specularTint;
        ior             = material.ior;
        metallic        = material.metallic;
        flatness        = material.flatness;
        sheen           = material.sheen;
        sheenTint       = material.sheenTint;
        anisotropic     = material.anisotropic;
        alpha           = material.alpha;
        roughness       = material.roughness;
        thin            = material.thin;
    }
};

struct Instance
{
    u32 id;
    u32 transformIndex;
};

struct ShapeType
{
    ScenePacket packet;

    ScenePacket *areaLight;
    string materialName;
    string groupName;
    int transformIndex;

    Mesh mesh;
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
    ChunkedLinkedList<ShapeType, MemoryType_Shape> shapes;
    ChunkedLinkedList<InstanceType, MemoryType_Instance> fileInstances;
    u32 numInstances;

    ChunkedLinkedList<AffineSpace, MemoryType_Instance> transforms;

    string virtualGeoFilename;
    bool base;
    bool differentGeometry;

    PBRTFileInfo *imports[32];
    u32 numImports;
    Scheduler::Counter counter = {};

    void Init(string inFilename)
    {
        arena    = ArenaAlloc(8);
        filename = PushStr8Copy(arena, inFilename);
        shapes   = ChunkedLinkedList<ShapeType, MemoryType_Shape>(arena, 1024);

        fileInstances     = ChunkedLinkedList<InstanceType, MemoryType_Instance>(arena, 1024);
        transforms        = ChunkedLinkedList<AffineSpace, MemoryType_Instance>(arena, 16384);
        numInstances      = 0;
        base              = false;
        differentGeometry = false;
    }

    void Merge(PBRTFileInfo *import)
    {
        numInstances += import->numInstances;

        u32 shapeOffset = shapes.totalCount;

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
        node = &map[index];
        while (node)
        {
            if (node->filename == filename)
            {
                EndWMutex(&mutexes[index]);
                return true;
            }
            prev = node;
            node = node->next;
        }

        prev->filename = PushStr8Copy(arena, filename);
        prev->next     = PushStruct(arena, IncludeHashNode);
        EndWMutex(&mutexes[index]);
        return false;
    }
};

struct MaterialHashNode
{
    u32 hash;
    string buffer;
    u32 index;
    MaterialHashNode *next;
};

struct MaterialMap
{
    MaterialHashNode *map;
    Mutex *mutexes;
    u32 count;

    u32 FindOrAdd(Arena *arena, string buffer, std::atomic<u32> &materialCount)
    {
        u32 hash  = Hash(buffer);
        u32 index = hash & (count - 1);
        BeginRMutex(&mutexes[index]);
        MaterialHashNode *node = &map[index];
        MaterialHashNode *prev;
        while (node)
        {
            if (hash == node->hash && node->buffer == buffer)
            {
                EndRMutex(&mutexes[index]);
                return true;
            }
            Assert(hash != node->hash);
            prev = node;
            node = node->next;
        }
        Assert(!node);
        EndRMutex(&mutexes[index]);

        BeginWMutex(&mutexes[index]);
        node = &map[index];
        while (node)
        {
            if (hash == node->hash && node->buffer == buffer)
            {
                EndWMutex(&mutexes[index]);
                return true;
            }
            Assert(hash != node->hash);
            prev = node;
            node = node->next;
        }

        prev->hash   = hash;
        prev->buffer = PushStr8Copy(arena, buffer);
        prev->next   = PushStruct(arena, MaterialHashNode);
        prev->index  = materialCount.fetch_add(1, std::memory_order_relaxed);
        EndWMutex(&mutexes[index]);

        return false;
    }
    MaterialHashNode *Find(string buffer)
    {
        u32 hash  = Hash(buffer);
        u32 index = hash & (count - 1);
        BeginRMutex(&mutexes[index]);
        MaterialHashNode *node = &map[index];
        MaterialHashNode *prev;
        while (node)
        {
            if (hash == node->hash && node->buffer == buffer)
            {
                EndRMutex(&mutexes[index]);
                return node;
            }
            Assert(hash != node->hash);
            prev = node;
            node = node->next;
        }
        return node;
    }
};

struct MeshHashNode
{
    Mesh *meshes;
    u32 numMeshes;

    // InstanceType *instances;
    // u32 numInstances;
    // AffineSpace *transforms;

    string idFilename;

    string filename;
    u64 hash;
    MeshHashNode *next;
};

struct MeshHashMap
{
    MeshHashNode *map;
    Mutex *mutexes;
    u32 count;

    MeshHashNode **filenameMap;

    u32 FindOrAdd(Arena *arena, Mesh *meshes, u32 numMeshes, u64 hash, string &filename,
                  string idFilename)
    {
        u32 index = hash & (count - 1);
        BeginRMutex(&mutexes[index]);
        MeshHashNode *node = &map[index];
        MeshHashNode *prev;
        while (node)
        {
            if (hash == node->hash && numMeshes == node->numMeshes)
            {
                bool same = true;
                for (u32 i = 0; i < numMeshes; i++)
                {
                    Mesh &otherMesh = node->meshes[i];
                    Mesh &mesh      = meshes[i];
                    if (mesh.numVertices == otherMesh.numVertices &&
                        mesh.numIndices == otherMesh.numIndices)
                    {
                        same &= memcmp(mesh.p, otherMesh.p,
                                       sizeof(Vec3f) * mesh.numVertices) == 0 &&
                                memcmp(mesh.indices, otherMesh.indices,
                                       sizeof(u32) * mesh.numIndices) == 0;
                        same &= bool(mesh.uv) == bool(otherMesh.uv);
                        if (mesh.uv && otherMesh.uv)
                        {
                            same &= memcmp(mesh.uv, otherMesh.uv,
                                           sizeof(Vec2f) * mesh.numVertices) == 0;
                        }
                        same &= bool(mesh.n) == bool(otherMesh.n);
                        if (mesh.n && otherMesh.n)
                        {
                            same &= memcmp(mesh.n, otherMesh.n,
                                           sizeof(Vec3f) * mesh.numVertices) == 0;
                        }
                    }
                    else
                    {
                        same = false;
                        break;
                    }
                    if (!same) break;
                }
                if (same)
                {
                    filename = node->filename;
                    EndRMutex(&mutexes[index]);
                    return true;
                }
            }
            Assert(hash != node->hash);
            prev = node;
            node = node->next;
        }
        Assert(!node);
        EndRMutex(&mutexes[index]);

        BeginWMutex(&mutexes[index]);
        node = &map[index];
        while (node)
        {
            if (hash == node->hash)
            {
                bool same = true;
                for (u32 i = 0; i < numMeshes; i++)
                {
                    Mesh &otherMesh = node->meshes[i];
                    Mesh &mesh      = meshes[i];
                    if (mesh.numVertices == otherMesh.numVertices &&
                        mesh.numIndices == otherMesh.numIndices)
                    {
                        same &= memcmp(mesh.p, otherMesh.p,
                                       sizeof(Vec3f) * mesh.numVertices) == 0 &&
                                memcmp(mesh.indices, otherMesh.indices,
                                       sizeof(u32) * mesh.numIndices) == 0;
                        same &= bool(mesh.uv) == bool(otherMesh.uv);
                        if (mesh.uv && otherMesh.uv)
                        {
                            same &= memcmp(mesh.uv, otherMesh.uv,
                                           sizeof(Vec2f) * mesh.numVertices) == 0;
                        }
                        same &= bool(mesh.n) == bool(otherMesh.n);
                        if (mesh.n && otherMesh.n)
                        {
                            same &= memcmp(mesh.n, otherMesh.n,
                                           sizeof(Vec3f) * mesh.numVertices) == 0;
                        }
                    }
                    else
                    {
                        same = false;
                        break;
                    }
                    if (!same) break;
                }
                if (same)
                {
                    filename = node->filename;
                    EndWMutex(&mutexes[index]);
                    return true;
                }
            }
            Assert(hash != node->hash);
            prev = node;
            node = node->next;
        }

        prev->hash = hash;

        Mesh *storedMeshes = PushArrayNoZero(arena, Mesh, numMeshes);
        for (u32 i = 0; i < numMeshes; i++)
        {
            Mesh &mesh       = meshes[i];
            Mesh copy        = {};
            copy.numVertices = mesh.numVertices;
            copy.numIndices  = mesh.numIndices;
            copy.p           = PushArrayNoZero(arena, Vec3f, mesh.numVertices);
            MemoryCopy(copy.p, mesh.p, sizeof(Vec3f) * mesh.numVertices);
            copy.indices = PushArrayNoZero(arena, u32, mesh.numIndices);
            MemoryCopy(copy.indices, mesh.indices, sizeof(u32) * mesh.numIndices);
            if (mesh.uv)
            {
                copy.uv = PushArrayNoZero(arena, Vec2f, mesh.numVertices);
                MemoryCopy(copy.uv, mesh.uv, sizeof(Vec2f) * mesh.numVertices);
            }
            if (mesh.n)
            {
                copy.n = PushArrayNoZero(arena, Vec3f, mesh.numVertices);
                MemoryCopy(copy.n, mesh.n, sizeof(Vec3f) * mesh.numVertices);
            }
            storedMeshes[i] = copy;
        }
        prev->meshes     = storedMeshes;
        prev->numMeshes  = numMeshes;
        prev->idFilename = PushStr8Copy(arena, idFilename);

        prev->filename = PushStr8Copy(arena, filename);
        prev->next     = PushStruct(arena, MeshHashNode);
        EndWMutex(&mutexes[index]);

        u32 fileHash            = Hash(idFilename);
        u32 fileIndex           = fileHash & (count - 1);
        MeshHashNode **fileNode = &filenameMap[fileIndex];
        BeginWMutex(&mutexes[fileIndex]);
        while (*fileNode)
        {
            fileNode = &(*fileNode)->next;
        }
        *fileNode = prev;
        EndWMutex(&mutexes[fileIndex]);

        return false;
    }
    // u32 AddInstances(Arena *arena, InstanceType *instances, u32 numInstances,
    //                  AffineSpace *transforms, string idFilename)
    // {
    //     u32 fileHash  = Hash(idFilename);
    //     u32 fileIndex = fileHash & (count - 1);
    //     BeginWMutex(&mutexes[fileIndex]);
    //     MeshHashNode **fileNode = &filenameMap[fileIndex];
    //
    //     while (*fileNode)
    //     {
    //         fileNode = &(*fileNode)->next;
    //     }
    //     *fileNode                 = PushStruct(arena, MeshHashNode);
    //     (*fileNode)->idFilename   = PushStr8Copy(arena, idFilename);
    //     (*fileNode)->transforms   = transforms;
    //     (*fileNode)->instances    = instances;
    //     (*fileNode)->numInstances = numInstances;
    //     EndWMutex(&mutexes[fileIndex]);
    //
    //     return false;
    // }

    MeshHashNode *Find(string filename)
    {
        u32 fileHash  = Hash(filename);
        u32 fileIndex = fileHash & (count - 1);
        BeginRMutex(&mutexes[fileIndex]);
        MeshHashNode *fileNode = filenameMap[fileIndex];
        while (fileNode)
        {
            if (fileNode->idFilename == filename)
            {
                return fileNode;
            }
            fileNode = fileNode->next;
        }
        EndRMutex(&mutexes[fileIndex]);
        return 0;
    }
};

struct SceneLoadState
{
    Arena **arenas;
    u32 numProcessors;
    ChunkedLinkedList<NamedPacket, MemoryType_Material> *materials;
    ChunkedLinkedList<NamedPacket, MemoryType_Light> *lights;

    std::atomic<u32> materialCounter;

    SceneHashMap *textureHashMaps;
    const u32 hashMapSize = 8192;

    IncludeMap includeMap;

    MaterialMap materialMap;
    MeshHashMap meshMap;

    void Init(Arena *arena)
    {
        u32 threadIndex = GetThreadIndex();
        numProcessors   = OS_NumProcessors();
        arenas          = PushArray(arena, Arena *, numProcessors);
        materials = PushArray(arena, ChunkedLinkedList<NamedPacket COMMA MemoryType_Material>,
                              numProcessors);
        textureHashMaps = PushArray(arena, SceneHashMap, numProcessors);
        lights = PushArray(arena, ChunkedLinkedList<NamedPacket COMMA MemoryType_Light>,
                           numProcessors);

        for (u32 i = 0; i < numProcessors; i++)
        {
            arenas[i]    = ArenaAlloc(16);
            materials[i] = ChunkedLinkedList<NamedPacket, MemoryType_Material>(arena, 1024);
            lights[i]    = ChunkedLinkedList<NamedPacket, MemoryType_Light>(arena, 1024);
            textureHashMaps[i] = SceneHashMap(arena, hashMapSize);
        }

        includeMap.count   = 1024;
        includeMap.map     = PushArray(arena, IncludeHashNode, includeMap.count);
        includeMap.mutexes = PushArray(arena, Mutex, includeMap.count);

        materialMap.count   = 16384;
        materialMap.map     = PushArray(arena, MaterialHashNode, materialMap.count);
        materialMap.mutexes = PushArray(arena, Mutex, materialMap.count);

        meshMap.count   = 16384;
        meshMap.map     = PushArray(arena, MeshHashNode, meshMap.count);
        meshMap.mutexes = PushArray(arena, Mutex, meshMap.count);

        meshMap.filenameMap = PushArray(arena, MeshHashNode *, meshMap.count);

        materialCounter = 1;
    }
};

string GetMaterialBuffer(Arena *arena, ScenePacket *packet, string materialType)
{
    u32 typeSize  = (u32)materialType.size;
    u32 totalSize = typeSize;

    u32 type                       = (u32)ConvertStringToMaterialType(materialType);
    u32 count                      = materialParameterCounts[type];
    const StringId *parameterNames = materialParameterIDs[type];

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

    u8 *buffer = PushArrayNoZero(arena, u8, totalSize);
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
    return Str8(buffer, offset);
}

struct GraphicsState
{
    string materialName   = {};
    AffineSpace transform = AffineSpace::Identity();

    i32 transformIndex = -1;
    // i32 areaLightIndex = -1;
    ScenePacket *areaLightPacket;
    i32 mediaIndex = -1;
};

int CheckForID(ScenePacket *packet, StringId id)
{
    for (u32 p = 0; p < packet->parameterCount; p++)
    {
        if (packet->parameterNames[p] == id) return p;
    }
    return -1;
}

void PBRTSkipToNextChar(Tokenizer *tokenizer) { SkipToNextChar(tokenizer, '#'); }

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
            dt = {};
            ErrorExit(0, "Invalid data type: %S\n", dataType);
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

void WriteFile(string directory, PBRTFileInfo *info, SceneLoadState *state = 0,
               Array<DisneyMaterial> *disneyMaterials = 0);

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

static string WriteNanite(PBRTFileInfo *state, SceneLoadState *sls, string directory,
                          string currentFilename)
{

#if 1
    if (state->shapes.Length())
    {
        ScratchArena scratch;
        Print("%S, num: %i\n", currentFilename, state->shapes.Length());
        // Remove duplicate meshes
        ShapeType *shapes =
            PushArrayNoZero(scratch.temp.arena, ShapeType, state->shapes.Length());
        state->shapes.Flatten(shapes);

        StaticArray<Mesh> meshes(scratch.temp.arena, state->shapes.Length(),
                                 state->shapes.Length());
        StaticArray<u32> materialIndices(scratch.temp.arena, meshes.Length(), meshes.Length());

        StaticArray<int> hashes(scratch.temp.arena, meshes.Length(), meshes.Length());
        BitVector shapeCulled(scratch.temp.arena, meshes.Length());

        u32 hashSize = NextPowerOfTwo(meshes.Length());
        HashIndex meshHashMap(scratch.temp.arena, hashSize, hashSize);

        Arena **arenas = GetArenaArray(scratch.temp.arena);

        ParallelFor(0, state->shapes.Length(), 32, 32, [&](int jobID, int start, int count) {
            Arena *arena = arenas[GetThreadIndex()];
            for (int meshIndex = start; meshIndex < start + count; meshIndex++)
            {
                ShapeType &shape = shapes[meshIndex];
                MaterialHashNode *node =
                    shape.materialName.size ? sls->materialMap.Find(shape.materialName) : 0;
                u32 materialIndex = node ? node->index : 0;

                materialIndices[meshIndex] = materialIndex;
                int posParameterIndex      = CheckForID(&shape.packet, "P"_sid);
                Mesh mesh                  = {};
                if (shape.mesh.numVertices)
                {
                    mesh = shape.mesh;
                }
                else if (posParameterIndex != -1)
                {
                    mesh.numVertices = shape.packet.sizes[posParameterIndex] / sizeof(Vec3f);
                    mesh.p           = (Vec3f *)shape.packet.bytes[posParameterIndex];

                    int indexParameterIndex = CheckForID(&shape.packet, "indices"_sid);
                    Assert(indexParameterIndex != -1);
                    mesh.numIndices = shape.packet.sizes[indexParameterIndex] / sizeof(u32);
                    mesh.indices    = (u32 *)shape.packet.bytes[indexParameterIndex];

                    int normParameterIndex = CheckForID(&shape.packet, "N"_sid);
                    Assert(normParameterIndex != -1);
                    Assert(shape.packet.sizes[normParameterIndex] / sizeof(Vec3f) ==
                           mesh.numVertices);
                    if (normParameterIndex != -1)
                        mesh.n = (Vec3f *)shape.packet.bytes[normParameterIndex];

                    int uvParameterIndex = CheckForID(&shape.packet, "uv"_sid);
                    Assert(uvParameterIndex != -1);
                    Assert(shape.packet.sizes[uvParameterIndex] / sizeof(Vec2f) ==
                           mesh.numVertices);
                    if (uvParameterIndex != -1)
                        mesh.uv = (Vec2f *)shape.packet.bytes[uvParameterIndex];
                }
                else
                {
                    int fileParameterIndex = CheckForID(&shape.packet, "filename"_sid);
                    Assert(fileParameterIndex != -1);
                    string filename   = StrConcat(arena, directory,
                                                  Str8(shape.packet.bytes[fileParameterIndex],
                                                       shape.packet.sizes[fileParameterIndex]));
                    GeometryType type = ConvertStringIDToGeometryType(shape.packet.type);
                    mesh              = LoadPLY(arena, filename, type);
                }
                meshes[meshIndex] = mesh;

                int hash =
                    MurmurHash32((const char *)mesh.p, sizeof(Vec3f) * mesh.numVertices, 0);
                meshHashMap.AddConcurrent(hash, meshIndex);
                hashes[meshIndex] = hash;
            }
        });

        u32 testVert = 0;
        u32 testInd  = 0;
        for (int meshIndex = 0; meshIndex < meshes.Length(); meshIndex++)
        {
            testVert += meshes[meshIndex].numVertices;
            testInd += meshes[meshIndex].numIndices;
        }
        Print("stats for %S pre cull: vert %u ind %u num %u\n", currentFilename, testVert,
              testInd, meshes.Length());

        ParallelFor(0, meshes.Length(), 32, 32, [&](int jobID, int start, int count) {
            for (int meshIndex = start; meshIndex < start + count; meshIndex++)
            {
                Mesh &mesh = meshes[meshIndex];
                int hash =
                    MurmurHash32((const char *)mesh.p, sizeof(Vec3f) * mesh.numVertices, 0);
                for (int hashIndex = meshHashMap.FirstInHash(hash); hashIndex != -1;
                     hashIndex     = meshHashMap.NextInHash(hashIndex))
                {
                    if (meshIndex != hashIndex && hash == hashes[hashIndex])
                    {
                        Mesh &otherMesh = meshes[hashIndex];
                        if (mesh.numVertices != otherMesh.numVertices) continue;
                        if (memcmp(mesh.p, otherMesh.p, sizeof(Vec3f) * mesh.numVertices) == 0)
                        {
                            if (materialIndices[meshIndex] != 0 &&
                                materialIndices[hashIndex] != 0 && meshIndex > hashIndex)
                            {
                                shapeCulled.SetBit(meshIndex);
                                break;
                            }
                            else if (materialIndices[meshIndex] == 0 &&
                                     materialIndices[hashIndex] == 0 && meshIndex > hashIndex)
                            {
                                shapeCulled.SetBit(meshIndex);
                                break;
                            }
                            else if (materialIndices[meshIndex] == 0 &&
                                     materialIndices[hashIndex] != 0)
                            {
                                shapeCulled.SetBit(meshIndex);
                                break;
                            }
                        }
                    }
                }
            }
        });

        int newNumMeshes = 0;
        for (int meshIndex = 0; meshIndex < meshes.Length(); meshIndex++)
        {
            if (!shapeCulled.GetBit(meshIndex))
            {
                meshes[newNumMeshes++] = meshes[meshIndex];
            }
        }

        ParallelFor(0, newNumMeshes, 32, 32, [&](int jobID, int start, int count) {
            for (int meshIndex = start; meshIndex < start + count; meshIndex++)
            {
                ScratchArena scratch;
                Mesh &mesh      = meshes[meshIndex];
                u32 numVertices = mesh.numVertices;
                u32 hashSize    = NextPowerOfTwo(numVertices);
                HashIndex hashMap(scratch.temp.arena, hashSize, hashSize);
                StaticArray<u32> remap(scratch.temp.arena, numVertices, numVertices);

                Vec3f *vertices = mesh.p;
                Vec3f *normals  = mesh.n;
                // Vec2f *uvs      = mesh.uv;

                int newVertexCount = 0;
                for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++)
                {
                    const Vec3f &vertex = vertices[vertexIndex];
                    int hash            = Hash(vertex);
                    int newVertexIndex  = -1;
                    for (int hashIndex = hashMap.FirstInHash(hash); hashIndex != -1;
                         hashIndex     = hashMap.NextInHash(hashIndex))
                    {
                        if (vertices[hashIndex] == vertex)
                        {
                            if (normals && normals[hashIndex] != normals[vertexIndex])
                                continue;
                            // if (uvs && uvs[hashIndex] != uvs[vertexIndex]) continue;
                            newVertexIndex = hashIndex;
                            break;
                        }
                    }

                    if (newVertexIndex == -1)
                    {
                        newVertexIndex = newVertexCount++;
                        hashMap.AddInHash(hash, newVertexIndex);
                        vertices[newVertexIndex] = vertex;
                        if (normals) normals[newVertexIndex] = normals[vertexIndex];
                        // if (uvs) uvs[newVertexIndex] = uvs[vertexIndex];
                    }
                    Assert(newVertexIndex != -1);
                    remap[vertexIndex] = (u32)newVertexIndex;
                }

                u32 numIndices = mesh.numIndices;
                u32 *indices   = mesh.indices;

                for (u32 indexIndex = 0; indexIndex < numIndices; indexIndex++)
                {
                    indices[indexIndex] = remap[indices[indexIndex]];
                }

                Mesh newMesh        = {};
                newMesh.numVertices = newVertexCount;
                newMesh.numIndices  = numIndices;
                newMesh.p           = vertices;
                newMesh.indices     = indices;
                newMesh.n           = normals;
                newMesh.faceIDs     = mesh.faceIDs;

                meshes[meshIndex] = newMesh;
            }
        });
        // Remove duplicated triangles
        ParallelFor(0, newNumMeshes, 32, 32, [&](int jobID, int start, int count) {
            for (int meshIndex = start; meshIndex < start + count; meshIndex++)
            {
                ScratchArena scratch;
                Mesh &mesh = meshes[meshIndex];
                HashIndex hashMap(scratch.temp.arena, NextPowerOfTwo(mesh.numIndices / 3),
                                  NextPowerOfTwo(mesh.numIndices / 3));
                u32 numTriangles = 0;

                int newVertexCount = 0;
                for (int tri = 0; tri < mesh.numIndices / 3; tri++)
                {
                    u32 index0 = mesh.indices[3 * tri + 0];
                    u32 index1 = mesh.indices[3 * tri + 1];
                    u32 index2 = mesh.indices[3 * tri + 2];

                    int hash           = MixBits(index0);
                    int newVertexIndex = -1;
                    for (int hashIndex = hashMap.FirstInHash(hash); hashIndex != -1;
                         hashIndex     = hashMap.NextInHash(hashIndex))
                    {
                        if (mesh.indices[3 * hashIndex + 0] == index0 &&
                            mesh.indices[3 * hashIndex + 1] == index1 &&
                            mesh.indices[3 * hashIndex + 2] == index2)
                        {
                            newVertexIndex = hashIndex;
                            break;
                        }
                    }

                    if (newVertexIndex == -1)
                    {
                        u32 newTri = numTriangles++;
                        hashMap.AddInHash(hash, newTri);
                        mesh.indices[3 * newTri + 0] = index0;
                        mesh.indices[3 * newTri + 1] = index1;
                        mesh.indices[3 * newTri + 2] = index2;
                        if (mesh.faceIDs)
                        {
                            mesh.faceIDs[newTri] = mesh.faceIDs[tri];
                        }
                    }
                }

                mesh.numIndices = 3 * numTriangles;
            }
        });

        string virtualGeoFilename = PushStr8F(state->arena, "%S%S.geo", directory,
                                              RemoveFileExtension(currentFilename));
        string geoFilename        = virtualGeoFilename;

        Arena *arena = sls->arenas[GetThreadIndex()];
        u32 size     = 0;
        for (u32 i = 0; i < newNumMeshes; i++)
        {
            Mesh &mesh = meshes[i];
            size += sizeof(Vec3f) * mesh.numVertices;
            size += sizeof(u32) * mesh.numIndices;
            if (mesh.n)
            {
                size += sizeof(Vec3f) * mesh.numVertices;
            }
            if (mesh.uv)
            {
                size += sizeof(Vec2f) * mesh.numVertices;
            }
        }
        u8 *buffer = PushArrayNoZero(scratch.temp.arena, u8, size);

        u32 offset           = 0;
        u32 totalNumVertices = 0;
        u32 totalNumIndices  = 0;
        for (u32 i = 0; i < newNumMeshes; i++)
        {
            Mesh &mesh = meshes[i];
            MemoryCopy(buffer + offset, mesh.p, sizeof(Vec3f) * mesh.numVertices);
            offset += sizeof(Vec3f) * mesh.numVertices;
            MemoryCopy(buffer + offset, mesh.indices, sizeof(u32) * mesh.numIndices);
            totalNumVertices += mesh.numVertices;
            totalNumIndices += mesh.numIndices;
            if (mesh.n)
            {
                MemoryCopy(buffer + offset, mesh.n, sizeof(Vec3f) * mesh.numVertices);
                offset += sizeof(Vec3f) * mesh.numVertices;
            }
            if (mesh.uv)
            {
                MemoryCopy(buffer + offset, mesh.uv, sizeof(Vec2f) * mesh.numVertices);
                offset += sizeof(Vec2f) * mesh.numVertices;
            }
        }
        Print("stats for %S: vert %u ind %u num %u\n", currentFilename, totalNumVertices,
              totalNumIndices, newNumMeshes);

        string str = {buffer, size};
        u64 hash   = MurmurHash64A(str.str, size, 0);

        u32 result = sls->meshMap.FindOrAdd(arena, meshes.data, newNumMeshes, hash,
                                            virtualGeoFilename, currentFilename);

        if (!result)
        {
            CreateClusters2(meshes.data, newNumMeshes, materialIndices, virtualGeoFilename);
        }
        else
        {
            Print("%S is a duplicate of %S\n", geoFilename, virtualGeoFilename);
        }
        ReleaseArenaArray(arenas);
        return virtualGeoFilename;
    }
#endif
    return {};
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
        if (currentGraphicsState.transformIndex == (i32)transforms->Length())
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
                string geoFilename = state->filename;
                if (originFile && state->shapes.Length() && state->fileInstances.Length())
                {
                    geoFilename = PushStr8F(tempArena, "%S_rtshape_tri.rtscene",
                                            RemoveFileExtension(state->filename));
                }
                state->virtualGeoFilename = WriteNanite(state, sls, directory, geoFilename);
                WriteFile(directory, state, originFile ? sls : 0);

                // if (state->fileInstances.totalCount)
                // {
                //     InstanceType instance;
                //     AffineSpace *transforms = PushArrayNoZero(threadArena, AffineSpace,
                //                                               state->transforms.totalCount);
                //     state->transforms.Flatten(transforms);
                //     InstanceType *instances = PushArrayNoZero(threadArena, InstanceType,
                //                                               state->fileInstances.totalCount);
                //     state->fileInstances.Flatten(instances);
                //
                //     for (u32 i = 0; i < state->fileInstances.totalCount; i++)
                //     {
                //         InstanceType &instance = instances[i];
                //         instance.filename      = PushStr8Copy(threadArena,
                //         instance.filename);
                //     }
                //     sls->meshMap.AddInstances(threadArena, instances,
                //                               state->fileInstances.totalCount, transforms,
                //                               state->filename);
                // }
#if 0
                if (state->fileInstances.totalCount)
                {
                    ScratchArena scratch;
                    Instance *instances =
                        PushArrayNoZero(scratch.temp.arena, Instance, state->numInstances);
                    for (auto *instNode = state->fileInstances.first; instNode != 0;
                         instNode       = instNode->next)
                    {
                        for (u32 i = 0; i < instNode->count; i++)
                        {
                            InstanceType *fileInst = &instNode->values[i];
                            u32 instanceCount      = 0;

                            for (u32 transformIndex = fileInst->transformIndexStart;
                                 transformIndex <= fileInst->transformIndexEnd;
                                 transformIndex++)
                            {
                                Instance &instance      = instances[instanceCount++];
                                instance.id             = 0;
                                instance.transformIndex = transformIndex;
                            }

                            Mesh *meshes;
                            u32 numMeshes;
                            sls->meshMap.Find(meshes, numMeshes, fileInst->filename);
                            AffineSpace *transforms = PushArrayNoZero(
                                scratch.temp.arena, AffineSpace, state->transforms.totalCount);
                            state->transforms.Flatten(transforms);
                            SimplifyInstances(directory, instances, instanceCount, transforms,
                                              state->transforms.totalCount, meshes[0]);
                        }
                    }
                }
#endif

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
                ErrorExit(!worldBegin,
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
                ErrorExit(worldBegin,
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
                ErrorExit(worldBegin,
                          "%S cannot be specified before WorldBegin "
                          "statement\n",
                          word);
                ErrorExit(scopeCount, "Unmatched AttributeEnd statement.\n");
                ScopeType type = scope[--scopeCount];
                ErrorExit(type == ScopeType::Attribute,
                          "Unmatched AttributeEnd statement. Aborting...\n");
                Assert(graphicsStateCount > 0);

                // Pop stack
                currentGraphicsState = graphicsStateStack[--graphicsStateCount];
            }
            break;
            case "AreaLightSource"_sid:
            {
                ErrorExit(worldBegin,
                          "%S cannot be specified before WorldBegin "
                          "statement\n",
                          word);
                // currentGraphicsState.areaLightIndex = lights->Length();
                // NamedPacket *packet                 = &lights->AddBack();

                // TODO: make sure this is the right arena
                ScenePacket *packet = PushStruct(tempArena, ScenePacket);
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Light);
                currentGraphicsState.areaLightPacket = packet;
            }
            break;
            case "Attribute"_sid:
            {
                ErrorExit(0, "Not implemented Attribute");
            }
            break;
            case "Camera"_sid:
            {
                ErrorExit(!worldBegin,
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
                AdvanceToNextParameter(&tokenizer);
            }
            break;
            case "CoordinateSystem"_sid:
            case "CoordSysTransform"_sid:
            {
                ErrorExit(0, "Not implemented %S\n", word);
            }
            break;
            case "Film"_sid:
            {
                ErrorExit(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
                PBRTFileInfo::Type type = PBRTFileInfo::Type::Film;
                ScenePacket *packet     = &state->packets[type];
                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Other);
                continue;
            }
            case "Integrator"_sid:
            {
                ErrorExit(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
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
                ErrorExit(worldBegin, "Tried to specify %S before WorldBegin\n", word);
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

                NamedPacket nPacket;
                ScenePacket *packet = &nPacket.packet;
                *packet             = {};
                packet->type        = "material"_sid;

                PBRTSkipToNextChar(&tokenizer);
                ReadParameters(threadArena, packet, &tokenizer, MemoryType_Material);

                if (isNamedMaterial)
                {
                    bool found = false;
                    for (u32 i = 0; i < packet->parameterCount; i++)
                    {
                        if (packet->parameterNames[i] == "type"_sid)
                        {
                            nPacket.type = PushStr8Copy(
                                threadArena, Str8(packet->bytes[i], packet->sizes[i]));
                            found = true;
                            break;
                        }
                    }
                    ErrorExit(found, "Named material must have a type\n");
                }
                else
                {
                    nPacket.type = PushStr8Copy(threadArena, materialNameOrType);
                }

                string materialName;
                if (isNamedMaterial)
                {
                    materialName = PushStr8Copy(threadArena, materialNameOrType);
                }
                else
                {
                    materialName = PushStr8Copy(
                        threadArena, GetMaterialBuffer(temp.arena, packet, nPacket.type));
                }

                nPacket.name = materialName;

                // NOTE: this changes the material name if a duplicate is found
                if (!sls->materialMap.FindOrAdd(threadArena, materialName,
                                                sls->materialCounter))
                {
                    materials->AddBack() = nPacket;
                }
                else
                {
                    threadLocalStatistics[threadIndex].misc++;
                }

                currentGraphicsState.materialName = materialName;
            }
            break;
            case "MakeNamedMedium"_sid:
            case "MediumInterface"_sid:
            {
                // not implemented yet
                ErrorExit(0, "Not implemented %S\n", word);
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
                ErrorExit(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                ErrorExit(!scopeCount || scope[scopeCount - 1] != ScopeType::Object,
                          "ObjectBegin cannot be called recursively.");
                scope[scopeCount++] = ScopeType::Object;

                string objectName;

                b32 result = GetBetweenPair(objectName, &tokenizer, '"');
                Assert(result);

                PBRTFileInfo *newState = PushStruct(threadArena, PBRTFileInfo);

                string objectFileName = PushStr8F(threadArena, "objects/%S_obj.rtscene",
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
                ErrorExit(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                ErrorExit(scopeCount, "Unmatched AttributeEnd statement. Aborting...\n");
                ScopeType type = scope[--scopeCount];
                ErrorExit(type == ScopeType::Object,
                          "Unmatched AttributeEnd statement. Aborting...\n");

                scheduler.Wait(&state->counter);
                for (u32 i = 0; i < state->numImports; i++)
                {
                    state->Merge(state->imports[i]);
                }
                state->virtualGeoFilename =
                    WriteNanite(state, sls, directory, state->filename);
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
                // TODO: remove duplicate shapes

                ErrorExit(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                ErrorExit(!scopeCount || scope[scopeCount - 1] != ScopeType::Object,
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
                ErrorExit(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
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
                ErrorExit(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                ShapeType *shape    = &shapes->AddBack();
                *shape              = {};
                ScenePacket *packet = &shape->packet;

                CreateScenePacket(tempArena, word, packet, &tokenizer, MemoryType_Shape);

                // TODO: temp
                if (packet->type == "curve"_sid)
                {
                    shapes->Last() = {};
                    shapes->last->count--;
                    shapes->totalCount--;
                    continue;
                }

                u32 numVertices   = 0;
                u32 numIndices    = 0;
                int positionIndex = -1;
                for (u32 i = 0; i < packet->parameterCount; i++)
                {
                    if (packet->parameterNames[i] == "P"_sid)
                    {
                        numVertices   = packet->sizes[i] / sizeof(Vec3f);
                        positionIndex = i;
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

                        // TODO: this is hardcoded for the moana island scene
#ifndef USE_GPU
                        if (GetFileExtension(plyMeshFile) == "obj" ||
                            CheckQuadPLY(StrConcat(temp.arena, directory, plyMeshFile)))
                            packet->type = "catclark"_sid;
                        else packet->type = "trianglemesh"_sid;
#else
                        packet->type = "trianglemesh"_sid;
#endif
                    }
                }
#ifndef USE_GPU
                if (packet->type == "trianglemesh"_sid && numVertices && numIndices &&
                    numVertices / 2 == numIndices / 3)
                {
                    packet->type = "catclark"_sid;
                }
#endif

                shape->materialName   = currentGraphicsState.materialName;
                shape->areaLight      = currentGraphicsState.areaLightPacket
                                            ? currentGraphicsState.areaLightPacket
                                            : 0;
                shape->transformIndex = currentGraphicsState.transformIndex;

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
                packet->type        = "texture"_sid;

                PBRTSkipToNextChar(&tokenizer);

                ReadParameters(threadArena, packet, &tokenizer, MemoryType_Texture);

#ifdef USE_GPU
                // if (textureClass == "ptex")
                // {
                //     for (int i = 0; i < packet->parameterCount; i++)
                //     {
                //         if (packet->parameterNames[i] == "filename"_sid)
                //         {
                //             string textureFilename =
                //                 StrConcat(threadArena, directory,
                //                           Str8(packet->bytes[i], packet->sizes[i]));
                //             scheduler.Schedule(&state->counter,
                //                                [=](u32 jobID) { Convert(textureFilename);
                //                                });
                //         }
                //     }
                // }
#endif

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
                ErrorExit(0, "ErrorExit while parsing scene. Buffer: %S", line);
            }
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
                        PutData(builder, scenePacket->bytes[j], scenePacket->sizes[j]);
                        Put(builder, " ");
                    }
                }
            }
        }
        break;
        default: Assert(0);
    }
}

void WriteDataType(StringBuilder *builder, ScenePacket *scenePacket, int p,
                   SceneHashMap *textureHashMap = 0)
{
    switch (scenePacket->types[p])
    {
        case DataType::Float:
        {
            u32 count = scenePacket->sizes[p] / sizeof(f32);
            Assert(count == 1);

            PutData(builder, scenePacket->bytes[p], scenePacket->sizes[p]);
            Put(builder, " ");
        }
        break;
        case DataType::Vec3:
        {
            PutData(builder, scenePacket->bytes[p], scenePacket->sizes[p]);
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
            Assert(textureHashMap);
            string textureName         = Str8(scenePacket->bytes[p], scenePacket->sizes[p]);
            const NamedPacket *nPacket = textureHashMap->Get(textureName);
            WriteTexture(builder, nPacket);
        }
        break;
        default: ErrorExit(0, "not supported yet\n");
    }
}

void WriteNameTypeAndData(StringBuilder *builder, ScenePacket *packet, string name, int p,
                          SceneHashMap *textureHashMap = 0)
{
    if (p >= 0 && p < packet->parameterCount)
    {
        Put(builder, "%S ", name);
        Put(builder, "%u ", packet->types[p]);
        WriteDataType(builder, packet, p, textureHashMap);
    }
}

void WriteMaterials(StringBuilder *builder, SceneHashMap *textureHashMap, NamedPacket &packet,
                    u32 hashMask)
{
    MaterialTypes type = ConvertStringToMaterialType(packet.type);
    u32 index          = (u32)type;
    Put(builder, "m %S ", packet.name);
    Put(builder, "%S ", packet.type);
    const string *names = materialParameterNames[index];
    u32 count           = materialParameterCounts[index];
    const StringId *ids = materialParameterIDs[index];
    for (u32 i = 0; i < count; i++)
    {
        ScenePacket *scenePacket = &packet.packet;
        int p                    = CheckForID(scenePacket, ids[i]);
        WriteNameTypeAndData(builder, scenePacket, names[i], p, textureHashMap);
    }
}

void WriteData(StringBuilder *builder, StringBuilderMapped *dataBuilder, void *ptr, u64 size,
               string out, u64 *builderOffset = 0, u64 cap = 0)
{
    Assert(ptr);
    if (builderOffset)
    {
        ErrorExit(*builderOffset + size <= cap, "offset: %llu, size, %llu, cap: %llu\n",
                  *builderOffset, size, cap);
        Put(dataBuilder, ptr, size, *builderOffset);
        Put(builder, "%S %llu ", out, *builderOffset);
        *builderOffset += size;
    }
    else
    {
        u64 offset = dataBuilder->totalSize;
        PutData(dataBuilder, ptr, size);
        Put(builder, "%S %llu ", out, offset);
    }
}

void WriteMesh(Mesh &mesh, StringBuilder &builder, StringBuilderMapped &dataBuilder,
               u64 *builderOffset = 0, u64 cap = 0)
{
    WriteData(&builder, &dataBuilder, mesh.p, mesh.numVertices * sizeof(Vec3f), "p",
              builderOffset, cap);

    if (mesh.n)
        WriteData(&builder, &dataBuilder, mesh.n, mesh.numVertices * sizeof(Vec3f), "n",
                  builderOffset, cap);

    if (mesh.uv)
        WriteData(&builder, &dataBuilder, mesh.uv, mesh.numVertices * sizeof(Vec2f), "uv",
                  builderOffset, cap);
    if (mesh.indices)
        WriteData(&builder, &dataBuilder, mesh.indices, mesh.numIndices * sizeof(u32),
                  "indices", builderOffset, cap);
    Put(&builder, "v %u ", mesh.numVertices);
    if (mesh.indices) Put(&builder, "i %u ", mesh.numIndices);
}

int ComputeShapeSize(Arena *arena, ShapeType *shape, string directory)
{
    // if (shape->cancelled) return 0;
    if (shape->mesh.numVertices)
    {
        Mesh &mesh = shape->mesh;
        int total  = mesh.numVertices * sizeof(Vec3f);
        total += mesh.n ? mesh.numVertices * sizeof(Vec3f) : 0;
        total += mesh.uv ? mesh.numVertices * sizeof(Vec2f) : 0;
        total += mesh.indices ? mesh.numIndices * sizeof(int) : 0;
        return total;
    }
    else
    {
        for (int i = 0; i < shape->packet.parameterCount; i++)
        {
            if (shape->packet.parameterNames[i] == "filename"_sid)
            {
                string filename = StrConcat(
                    arena, directory, Str8(shape->packet.bytes[i], shape->packet.sizes[i]));

                Assert(GetFileExtension(filename) == "ply");
                GeometryType type = ConvertStringIDToGeometryType(shape->packet.type);
                Assert(type == GeometryType::TriangleMesh);

                Mesh mesh   = LoadPLY(arena, filename, type);
                shape->mesh = mesh;

                return ComputeShapeSize(arena, shape, directory);
            }
        }

        const StringId table[] = {"P"_sid, "N"_sid, "uv"_sid, "indices"_sid};
        int total              = 0;
        for (int i = 0; i < ArrayLength(table); i++)
        {
            for (int j = 0; j < shape->packet.parameterCount; j++)
            {
                if (shape->packet.parameterNames[j] == table[i])
                {
                    total += shape->packet.sizes[j];
                    break;
                }
            }
        }
        return total;
    }
}

i32 WriteData(ScenePacket *packet, StringBuilder *builder, StringBuilderMapped *dataBuilder,
              StringId matchId, string out, u64 *builderOffset, u64 cap)
{
    for (u32 c = 0; c < packet->parameterCount; c++)
    {
        if (packet->parameterNames[c] == matchId)
        {
            WriteData(builder, dataBuilder, packet->bytes[c], packet->sizes[c], out,
                      builderOffset, cap);
            return c;
        }
    }
    return -1;
}

void WriteMesh(StringBuilder &builder, StringBuilderMapped &dataBuilder, ShapeType *shape,
               string directory, GeometryType type, u64 *builderOffset = 0, u64 cap = 0)
{
    if (shape->mesh.numVertices)
    {
        WriteMesh(shape->mesh, builder, dataBuilder, builderOffset, cap);
        return;
    }

    TempArena temp      = ScratchStart(&builder.arena, 1);
    ScenePacket *packet = &shape->packet;
    bool fileMesh       = false;
    for (u32 c = 0; c < packet->parameterCount; c++)
    {
        if (packet->parameterNames[c] == "filename"_sid)
        {
            string filename =
                StrConcat(temp.arena, directory, Str8(packet->bytes[c], packet->sizes[c]));
            Mesh mesh = LoadPLY(temp.arena, filename, type);
            Assert(GetFileExtension(filename) == "ply");

            WriteMesh(mesh, builder, dataBuilder, builderOffset, cap);
            fileMesh = true;
            break;
        }
    }

    if (!fileMesh)
    {
        u32 numVertices = 0;
        u32 numIndices  = 0;
        i32 c           = -1;
        c = WriteData(packet, &builder, &dataBuilder, "P"_sid, "p", builderOffset, cap);
        Assert(c != -1);
        numVertices = packet->sizes[c] / sizeof(Vec3f);
        Assert(numVertices);
        WriteData(packet, &builder, &dataBuilder, "N"_sid, "n", builderOffset, cap);
        WriteData(packet, &builder, &dataBuilder, "uv"_sid, "uv", builderOffset, cap);
        c = WriteData(packet, &builder, &dataBuilder, "indices"_sid, "indices", builderOffset,
                      cap);
        if (c != -1) numIndices = packet->sizes[c] / sizeof(u32);
        Put(&builder, "v %u ", numVertices);
        if (numIndices) Put(&builder, "i %u ", numIndices);
    }
    ScratchEnd(temp);
}

void WriteAreaLight(StringBuilder *builder, ScenePacket *light)
{
    Put(builder, "a ");
    const string areaLightNames[] = {
        "filename",
        "L",
        "twosided",
    };

    const StringId areaLightIDs[] = {
        "filename"_sid,
        "L"_sid,
        "twosided"_sid,
    };

    for (int i = 0; i < ArrayLength(areaLightNames); i++)
    {
        for (int j = 0; j < light->parameterCount; j++)
        {
            if (light->parameterNames[j] == areaLightIDs[i])
            {
                Put(builder, "%S ", areaLightNames[i]);
                Put(builder, "%u ", light->types[j]);
                WriteDataType(builder, light, j);
                break;
            }
        }
    }
}

void WriteShape(PBRTFileInfo *info, ShapeType *shapeType, StringBuilder &builder,
                StringBuilderMapped &dataBuilder, string directory, u64 *builderOffset = 0,
                u64 cap = 0)
{
    // if (shapeType->cancelled) return;
    ScenePacket *packet = &shapeType->packet;
    switch (packet->type)
    {
        case "catclark"_sid:
        {
            Put(&builder, "Catclark ");
            WriteMesh(builder, dataBuilder, shapeType, directory, GeometryType::CatmullClark,
                      builderOffset, cap);
        }
        break;
        case "quadmesh"_sid:
        {
            Put(&builder, "Quad ");
            WriteMesh(builder, dataBuilder, shapeType, directory, GeometryType::QuadMesh,
                      builderOffset, cap);
        }
        break;
        case "trianglemesh"_sid:
        {
            Put(&builder, "Tri ");
            WriteMesh(builder, dataBuilder, shapeType, directory, GeometryType::TriangleMesh,
                      builderOffset, cap);
        }
        break;
        // case "curve"_sid:
        // {
        //     Put(&builder, "Curve ");
        // }
        // break;
        default: Assert(0);
    }
    if (shapeType->materialName.size) Put(&builder, "m %S ", shapeType->materialName);
    if (shapeType->transformIndex != -1)
    {
        Put(&builder, "transform %i ", shapeType->transformIndex);
    }
    if (shapeType->areaLight) WriteAreaLight(&builder, shapeType->areaLight);

    int alphaID = CheckForID(packet, "alpha"_sid);
    WriteNameTypeAndData(&builder, packet, "alpha", alphaID);
}

void SeparateShapeTypes(Arena *arena, PBRTFileInfo *info, string directory)
{
    u32 transformIndex         = info->transforms.Length();
    info->transforms.AddBack() = AffineSpace::Identity();
    for (int type = 0; type < (int)GeometryType::Max; type++)
    {
        GeometryType ty = (GeometryType)type;
        if (ty == GeometryType::Instance) continue;

        PBRTFileInfo *shapeInfo = PushStruct(arena, PBRTFileInfo);
        shapeInfo->shapes       = decltype(shapeInfo->shapes)(arena);
        for (auto *node = info->shapes.first; node != 0; node = node->next)
        {
            for (int i = 0; i < node->count; i++)
            {
                ShapeType *shape = node->values + i;
                if (ty == ConvertStringIDToGeometryType(shape->packet.type))
                {
                    shapeInfo->shapes.AddBack() = *shape;
                }
            }
        }

        if (shapeInfo->shapes.totalCount)
        {
            shapeInfo->filename =
                PushStr8F(arena, "%S_rtshape_%S.rtscene", RemoveFileExtension(info->filename),
                          ConvertGeometryTypeToString(ty));
            WriteFile(directory, shapeInfo);

            info->numInstances++;
            info->fileInstances.AddBack() = {shapeInfo->filename, transformIndex,
                                             transformIndex};
        }
    }
}

void WriteTransforms(PBRTFileInfo *info, StringBuilderMapped &dataBuilder)
{
    if (info->transforms.totalCount)
    {
        Put(&dataBuilder, "TRANSFORM_START ");
        Put(&dataBuilder, "Count %u ", info->transforms.totalCount);

        u32 runningCount = 0;
        for (auto *node = info->transforms.first; node != 0; node = node->next)
        {
            runningCount += node->count;
            PutData(&dataBuilder, node->values, sizeof(node->values[0]) * node->count);
        }
        ErrorExit(runningCount == info->transforms.totalCount, "running: %i, total: %i\n",
                  runningCount, info->transforms.totalCount);
        Put(&dataBuilder, "TRANSFORM_END");
    }
}

void WriteFile(string directory, PBRTFileInfo *info, SceneLoadState *state,
               Array<DisneyMaterial> *disneyMaterials)
{
    TempArena temp = ScratchStart(0, 0);
    Assert(GetFileExtension(info->filename) == "rtscene");
    string outFile = StrConcat(temp.arena, directory, info->filename);

    StringBuilder builder = {};
    builder.arena         = temp.arena;

    u32 totalMaterialCount = 0;
    Put(&builder, "RTSCENE_START ");

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

    Arena **arenas = PushArray(temp.arena, Arena *, 32);
    for (u32 i = 0; i < 32; i++)
    {
        arenas[i] = ArenaAlloc(16);
    }

    if (state)
    {
        SceneHashMap *textureHashMap = &state->textureHashMaps[0];
        for (u32 i = 1; i < state->numProcessors; i++)
        {
            textureHashMap->Merge(state->textureHashMaps[i]);
        }

        Put(&builder, "MATERIALS_START ");

        u32 totalNumMaterials = 0;
        for (u32 i = 0; i < state->numProcessors; i++)
        {
            auto &list = state->materials[i];
            totalNumMaterials += list.Length();
        }

        struct Handle
        {
            u32 sortKey;
            u32 index;
        };

        NamedPacket *materials = PushArrayNoZero(temp.arena, NamedPacket, totalNumMaterials);
        Print("total num materials: %u\n", totalNumMaterials);
        Handle *handles    = PushArrayNoZero(temp.arena, Handle, totalNumMaterials);
        u32 materialOffset = 0;
        for (u32 i = 0; i < state->numProcessors; i++)
        {
            auto &list = state->materials[i];
            list.Flatten(materials + materialOffset);

            for (u32 materialIndex = materialOffset;
                 materialIndex < materialOffset + list.Length(); materialIndex++)
            {
                NamedPacket &material          = materials[materialIndex];
                handles[materialIndex].sortKey = state->materialMap.Find(material.name)->index;
                handles[materialIndex].index   = materialIndex;
            }
            materialOffset += list.Length();
        }

        SortHandles(handles, totalNumMaterials);

        for (u32 materialIndex = 0; materialIndex < totalNumMaterials; materialIndex++)
        {
            const Handle &handle = handles[materialIndex];
            WriteMaterials(&builder, textureHashMap, materials[handle.index],
                           state->hashMapSize - 1);
        }
        Put(&builder, "MATERIALS_END ");
    }
    else if (disneyMaterials)
    {
        Put(&builder, "MATERIALS_START ");

        for (DisneyMaterial &material : *disneyMaterials)
        {
            Put(&builder, "m %S ", material.name);
            Put(&builder, "disney ");
            if (material.colorMap.size)
            {
                Put(&builder, "t %S ", material.colorMap);
            }
            else
            {
                Put(&builder, "t none ");
            }
            DiskDisneyMaterial writeMaterial(material);
            PutPointerValue(&builder, &writeMaterial);
            Put(&builder, " ");
        }

        Put(&builder, "MATERIALS_END ");
    }

    if (info->shapes.totalCount == 0 && info->fileInstances.totalCount == 0)
    {
        Assert(info->virtualGeoFilename.size);

        Put(&builder, "Geo Filename ");
        Put(&builder, info->virtualGeoFilename);
        Put(&builder, " ");
    }
    if (info->shapes.totalCount && info->fileInstances.totalCount == 0)
    {
        // First, loop to see if all the types are the same
        GeometryType type   = GeometryType::Max;
        bool differentTypes = false;
        for (auto *node = info->shapes.first; node != 0; node = node->next)
        {
            for (int i = 0; i < node->count; i++)
            {
                GeometryType ty = ConvertStringIDToGeometryType(node->values[i].packet.type);
                if (type == GeometryType::Max)
                {
                    type = ty;
                }
                else if (type != ty)
                {
                    differentTypes = true;
                    break;
                }
            }
            if (differentTypes) break;
        }
        if (differentTypes)
        {
            Assert(info->filename == "test.rtscene");
            SeparateShapeTypes(temp.arena, info, directory);
            WriteTransforms(info, dataBuilder);
        }
        else
        {
            WriteTransforms(info, dataBuilder);
            Put(&builder, "SHAPE_START ");

            u32 numMeshes;
            // ShapeType *shapes = PruneDuplicateMeshesAndVertices(info, temp.arena,
            // numMeshes);
            // StringBuilder *builders = PushArray(temp.arena, StringBuilder, taskCount);
            // ParallelForLoop(0, numMeshes, 32, 32, [&](u32 jobID, u32 index) {
            //     ShapeType &shape       = shapes[index];
            //     StringBuilder &builder = builders[GetThreadIndex()];
            //     WriteShape(PBRTFileInfo * info, ShapeType * shapeType, StringBuilder &
            //     builder,
            //                StringBuilderMapped & dataBuilder, string directory);
            // });

            // TODO: need to handle duplicates as well
            for (auto *node = info->shapes.first; node != 0; node = node->next)
            {
                const int groupSize = 32;
                if (node->count < groupSize)
                {
                    for (int i = 0; i < node->count; i++)
                    {
                        ShapeType *shapeType = node->values + i;
                        WriteShape(info, shapeType, builder, dataBuilder, directory);
                    }
                    continue;
                }
                int num       = node->count;
                int taskCount = (num + groupSize - 1) / groupSize;

                StringBuilder *builders = PushArray(temp.arena, StringBuilder, taskCount);
                Assert(taskCount <= 32);
                for (int i = 0; i < taskCount; i++)
                {
                    builders[i].arena = arenas[i];
                }

                // Precalculate offsets into mapped buffer
                u64 *offsets = PushArray(temp.arena, u64, taskCount);
                for (int taskIndex = 0; taskIndex < taskCount; taskIndex++)
                {
                    int total = 0;
                    int start = taskIndex * groupSize;
                    int end   = Min((taskIndex + 1) * groupSize, num);
                    for (int i = start; i < end; i++)
                    {
                        int packetIndex      = i;
                        ShapeType *shapeType = node->values + packetIndex;
                        total += ComputeShapeSize(temp.arena, shapeType, directory);
                    }
                    offsets[taskIndex] = total;
                }

                u64 tempOffset = dataBuilder.totalSize;
                for (int taskIndex = 0; taskIndex < taskCount; taskIndex++)
                {
                    u64 tempTotal      = offsets[taskIndex];
                    offsets[taskIndex] = tempOffset;
                    tempOffset += tempTotal;
                }

                // Preallocate write buffer
                Expand(&dataBuilder, tempOffset - dataBuilder.totalSize);
                dataBuilder.totalSize = tempOffset;
                dataBuilder.writePtr  = dataBuilder.ptr + dataBuilder.totalSize;

                ParallelFor(0, taskCount, 1, 1, [&](int jobID, int id, int count) {
                    Assert(count == 1);
                    Assert(jobID < taskCount);
                    StringBuilder &builder = builders[jobID];
                    int start              = jobID * groupSize;
                    int end                = Min((jobID + 1) * groupSize, num);

                    u64 builderOffset = offsets[jobID];
                    u64 cap = jobID == taskCount - 1 ? tempOffset : offsets[jobID + 1];
                    for (int i = start; i < end; i++)
                    {
                        int packetIndex = i;
                        Assert(packetIndex < node->count);
                        ShapeType *shapeType = node->values + packetIndex;

                        WriteShape(info, shapeType, builder, dataBuilder, directory,
                                   &builderOffset, cap);
                    }
                });

                for (int i = 0; i < taskCount; i++)
                {
                    builder = ConcatBuilders(&builder, &builders[i]);
                }
            }
            Put(&builder, "SHAPE_END ");
        }
    }
    else if (info->shapes.totalCount)
    {
        SeparateShapeTypes(temp.arena, info, directory);
        WriteTransforms(info, dataBuilder);
    }
    else
    {
        WriteTransforms(info, dataBuilder);
    }

    if (info->fileInstances.totalCount)
    {
        Put(&builder, "INCLUDE_START ");
        Assert(info->numInstances);
        Put(&builder, "Count: %u ", info->numInstances);
        threadLocalStatistics[GetThreadIndex()].misc4 += info->numInstances;
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

    Put(&builder, "RTSCENE_END");
    WriteFileMapped(&builder, outFile);
    OS_UnmapFile(dataBuilder.ptr);
    OS_ResizeFile(dataBuilder.filename, dataBuilder.totalSize);

    for (int i = 0; i < 32; i++)
    {
        ArenaClear(arenas[i]);
    }
    ScratchEnd(temp);
}

static Array<DisneyMaterial> LoadMaterialJSON(SceneLoadState *sls, string directory,
                                              string filename)
{
    ScratchArena scratch;
    Tokenizer tokenizer;
    tokenizer.input =
        OS_ReadFile(scratch.temp.arena, StrConcat(scratch.temp.arena, directory, filename));
    tokenizer.cursor = tokenizer.input.str;

    Arena *arena = sls->arenas[GetThreadIndex()];

    Array<DisneyMaterial> disneyMaterials(arena, 8);

    for (;;)
    {
        SkipToNextChar2(&tokenizer, '"');
        if (EndOfBuffer(&tokenizer))
        {
            break;
        }
        string materialName;
        bool result = GetBetweenPair(materialName, &tokenizer, '"');
        Assert(result);

        materialName = PushStr8Copy(arena, materialName);

        DisneyMaterial material = {};
        material.name           = materialName;

        for (;;)
        {
            SkipToNextChar(&tokenizer);
            if (EndOfBuffer(&tokenizer) || *tokenizer.cursor == '}') break;

            SkipToNextChar2(&tokenizer, '"');
            string parameterType;
            result = GetBetweenPair(parameterType, &tokenizer, '"');

            if (parameterType == "diffTrans")
            {
                SkipToNextDigit(&tokenizer);
                material.diffTrans = ReadFloat(&tokenizer);
            }
            else if (parameterType == "baseColor")
            {
                Vec4f baseColor;
                baseColor.w = 1.f;

                for (u32 i = 0; i < 4; i++)
                {
                    if (*tokenizer.cursor == ']') break;
                    SkipToNextDigit(&tokenizer);
                    float f      = ReadFloat(&tokenizer);
                    baseColor[i] = Pow(f, 2.2f);
                    SkipToNextChar(&tokenizer, ',');
                }
                material.baseColor = baseColor;
            }
            else if (parameterType == "specTrans")
            {
                SkipToNextDigit(&tokenizer);
                material.specTrans = ReadFloat(&tokenizer);
            }
            else if (parameterType == "colorMap")
            {
                string colorMapFilename;
                SkipToNextChar2(&tokenizer, '"');
                bool success = GetBetweenPair(colorMapFilename, &tokenizer, '"');
                Assert(success);

                material.colorMap = PushStr8Copy(arena, colorMapFilename);
            }
            else if (parameterType == "clearcoatGloss")
            {
                SkipToNextDigit(&tokenizer);
                material.clearcoatGloss = ReadFloat(&tokenizer);
            }
            else if (parameterType == "scatterDistance")
            {
                Vec3f scatterDistance;
                for (u32 i = 0; i < 3; i++)
                {
                    SkipToNextDigit(&tokenizer);
                    scatterDistance[i] = ReadFloat(&tokenizer);
                }
                material.scatterDistance = scatterDistance;
            }
            else if (parameterType == "assignment")
            {
                SkipToNextChar2(&tokenizer, ']');
                bool success = Advance(&tokenizer, "]");
                Assert(success);
            }
            else if (parameterType == "clearcoat")
            {
                SkipToNextDigit(&tokenizer);
                material.clearcoat = ReadFloat(&tokenizer);
            }
            else if (parameterType == "specularTint")
            {
                SkipToNextDigit(&tokenizer);
                material.specularTint = ReadFloat(&tokenizer);
            }
            else if (parameterType == "ior")
            {
                SkipToNextDigit(&tokenizer);
                material.ior = ReadFloat(&tokenizer);
            }
            else if (parameterType == "metallic")
            {
                SkipToNextDigit(&tokenizer);
                material.metallic = ReadFloat(&tokenizer);
            }
            else if (parameterType == "flatness")
            {
                SkipToNextDigit(&tokenizer);
                material.flatness = ReadFloat(&tokenizer);
            }
            else if (parameterType == "sheen")
            {
                SkipToNextDigit(&tokenizer);
                material.sheen = ReadFloat(&tokenizer);
            }
            else if (parameterType == "sheenTint")
            {
                SkipToNextDigit(&tokenizer);
                material.sheenTint = ReadFloat(&tokenizer);
            }
            else if (parameterType == "anisotropic")
            {
                SkipToNextDigit(&tokenizer);
                material.anisotropic = ReadFloat(&tokenizer);
            }
            else if (parameterType == "alpha")
            {
                SkipToNextDigit(&tokenizer);
                material.alpha = ReadFloat(&tokenizer);
            }
            else if (parameterType == "roughness")
            {
                SkipToNextDigit(&tokenizer);
                material.roughness = ReadFloat(&tokenizer);
            }
            else if (parameterType == "refractive")
            {
            }
            else if (parameterType == "mask")
            {
                string blank;
                SkipToNextChar2(&tokenizer, '"');
                bool success = Advance(&tokenizer, "\"\"");
                Assert(success);
            }
            else if (parameterType == "type")
            {
                string type;
                SkipToNextChar2(&tokenizer, '"');
                bool success = GetBetweenPair(type, &tokenizer, '"');
                Assert(success);
                bool thin = false;
                if (type == "thin")
                {
                    thin = true;
                }
                material.thin = thin;
            }
            else if (parameterType == "displacementMap")
            {
                string displacementMapFilename;
                SkipToNextChar2(&tokenizer, '"');
                GetBetweenPair(displacementMapFilename, &tokenizer, '"');
                Advance(&tokenizer, ",");
            }
            else
            {
                Assert(0);
            }
        }
        disneyMaterials.Push(material);
    }
    return disneyMaterials;
}

static Mesh *LoadObj(Arena *arena, string filename, string *&outMaterialNames,
                     string *&outGroupNames, int &numMeshes)
{
    TempArena temp = ScratchStart(0, 0);
    string buffer  = OS_MapFileRead(filename);
    Tokenizer tokenizer;
    tokenizer.input  = buffer;
    tokenizer.cursor = buffer.str;

    std::vector<Vec3f> vertices;
    vertices.reserve(128);
    std::vector<Vec3f> normals;
    normals.reserve(128);
    std::vector<Vec3i> indices;
    indices.reserve(128);

    std::vector<Mesh> meshes;
    int vertexOffset = 1;
    int normalOffset = 1;

    std::unordered_set<string> meshHashSet;
    bool repeatedMesh   = false;
    bool processingMesh = false;

    std::vector<string> materialNames;
    std::vector<string> groupNames;
    string materialName  = {};
    string meshGroupName = {};

    auto Skip = [&]() { SkipToNextChar(&tokenizer, '#'); };

    for (;;)
    {
        Skip();

        string word = ReadWord(&tokenizer);

        bool isEndOfBuffer = EndOfBuffer(&tokenizer);
        if (word == "g" || isEndOfBuffer)
        {
            Skip();
            string groupName = ReadWord(&tokenizer);
            if (groupName == "default" || isEndOfBuffer)
            {
                if (processingMesh)
                {
                    materialNames.push_back(materialName);
                    groupNames.push_back(meshGroupName);
                    materialName  = {};
                    meshGroupName = {};

                    u32 normalsSize = (u32)normals.size();
                    u32 numVertices = Max(normalsSize, (u32)vertices.size());

                    struct Index
                    {
                        int normalIndex;
                        int vertexIndex;
                        int next;
                    };

                    Index *tempIndices = PushArray(temp.arena, Index, numVertices);
                    MemorySet(tempIndices, 0xff, sizeof(Index) * numVertices);
                    u32 currentVertexCount = vertices.size();

                    Mesh mesh        = {};
                    mesh.numVertices = numVertices;
                    mesh.numIndices  = (u32)indices.size();
                    mesh.p           = PushArrayNoZero(arena, Vec3f, numVertices);
                    MemoryCopy(mesh.p, vertices.data(), sizeof(Vec3f) * vertices.size());
                    mesh.n       = PushArrayNoZero(arena, Vec3f, numVertices);
                    mesh.indices = PushArrayNoZero(arena, u32, indices.size());

                    threadLocalStatistics[GetThreadIndex()].misc2 += mesh.numVertices;
                    threadLocalStatistics[GetThreadIndex()].misc3 += mesh.numIndices;
                    // Ensure that vertex index and normal index pairings are
                    // consistent
                    for (u32 i = 0; i < mesh.numIndices; i++)
                    {
                        i32 vertexIndex = indices[i][0];
                        i32 normalIndex = indices[i][2];

                        if (tempIndices[vertexIndex].normalIndex != -1 &&
                            tempIndices[vertexIndex].normalIndex != normalIndex)
                        {
                            int vertexIndexIndex = vertexIndex;
                            while (vertexIndexIndex != -1)
                            {
                                Index index = tempIndices[vertexIndexIndex];
                                if (index.normalIndex == normalIndex)
                                {
                                    vertexIndex = index.vertexIndex;
                                    break;
                                }
                                vertexIndexIndex = index.next;
                            }

                            if (vertexIndexIndex == -1)
                            {
                                int prevVertexIndex = vertexIndex;
                                vertexIndex         = currentVertexCount++;
                                Assert(vertexIndex < numVertices);

                                mesh.p[vertexIndex] = mesh.p[prevVertexIndex];
                                mesh.n[vertexIndex] = normals[normalIndex];
                                Index index;
                                index.vertexIndex = vertexIndex;
                                index.normalIndex = normalIndex;
                                index.next        = -1;

                                tempIndices[vertexIndex] = index;

                                Index *changedIndex = &tempIndices[prevVertexIndex];
                                while (changedIndex->next != -1)
                                {
                                    changedIndex = &tempIndices[changedIndex->next];
                                }
                                changedIndex->next = vertexIndex;
                            }
                        }
                        else if (tempIndices[vertexIndex].normalIndex == normalIndex)
                        {
                        }
                        else
                        {
                            Index index;
                            index.vertexIndex        = vertexIndex;
                            index.normalIndex        = normalIndex;
                            index.next               = -1;
                            tempIndices[vertexIndex] = index;
                            mesh.n[vertexIndex]      = normals[normalIndex];
                        }

                        Assert(normalIndex < (int)normals.size());
                        Assert(vertexIndex < numVertices);
                        mesh.indices[i] = vertexIndex;
                    }
                    Assert(currentVertexCount == numVertices);

                    meshes.push_back(mesh);

                    vertexOffset += (int)vertices.size();
                    normalOffset += (int)normals.size();
                    vertices.clear();
                    normals.clear();
                    indices.clear();
                }
                processingMesh = false;
            }
            else
            {
                meshGroupName  = PushStr8Copy(arena, groupName);
                processingMesh = true;
            }
            if (isEndOfBuffer) break;
        }
        else if (word == "mtllib")
        {
            SkipToNextLine(&tokenizer);
        }
        else if (word == "usemtl")
        {
            while (!EndOfBuffer(&tokenizer) && CharIsWhitespace(*tokenizer.cursor))
            {
                tokenizer.cursor++;
            }
            if (IsBlank(&tokenizer))
            {
                materialName = {};
            }
            else
            {
                string s     = ReadWord(&tokenizer);
                materialName = PushStr8Copy(arena, s);
            }
        }
        else if (word == "s")
        {
            SkipToNextLine(&tokenizer);
        }
        else if (word == "v")
        {
            Skip();
            f32 x = ReadFloat(&tokenizer);
            f32 y = ReadFloat(&tokenizer);
            f32 z = ReadFloat(&tokenizer);
            vertices.push_back(Vec3f(x, y, z));
        }
        else if (word == "vn")
        {
            Skip();
            f32 x = ReadFloat(&tokenizer);
            f32 y = ReadFloat(&tokenizer);
            f32 z = ReadFloat(&tokenizer);
            normals.push_back(Vec3f(x, y, z));
        }
        else if (word == "f")
        {
            Skip();
            u32 faceVertexCount = 0;
            while (CharIsDigit(tokenizer.cursor[0]))
            {
                faceVertexCount++;

                u32 vertexIndex = ReadUint(&tokenizer);
                bool result     = Advance(&tokenizer, "/");
                Assert(result);
                u32 texIndex = ReadUint(&tokenizer);
                result       = Advance(&tokenizer, "/");
                Assert(result);
                u32 normalIndex = ReadUint(&tokenizer);
                Assert(vertexIndex - vertexOffset >= 0 && normalIndex - normalOffset >= 0);
                indices.push_back(Vec3i(vertexIndex - vertexOffset, texIndex - 1,
                                        normalIndex - normalOffset));
                Skip();
            }
            // Assert(faceVertexCount == 4)
        }
        else
        {
            Assert(0);
        }
    }

    ScratchEnd(temp);
    Assert(materialNames.size() == meshes.size());
    outMaterialNames = PushArrayNoZero(arena, string, materialNames.size());
    MemoryCopy(outMaterialNames, materialNames.data(), sizeof(string) * materialNames.size());
    outGroupNames = PushArrayNoZero(arena, string, groupNames.size());
    MemoryCopy(outGroupNames, groupNames.data(), sizeof(string) * groupNames.size());
    numMeshes          = (int)meshes.size();
    Mesh *outputMeshes = PushArrayNoZero(arena, Mesh, numMeshes);
    MemoryCopy(outputMeshes, meshes.data(), sizeof(Mesh) * numMeshes);
    return outputMeshes;
}

static void LoadMoanaOBJ(PBRTFileInfo *info, string filePath)
{
    int numMeshes;
    string *materials, *groupNames;
    Mesh *meshes = LoadObj(info->arena, filePath, materials, groupNames, numMeshes);

    for (int i = 0; i < numMeshes; i++)
    {
        ShapeType &shape   = info->shapes.AddBack();
        shape.packet.type  = "trianglemesh"_sid;
        meshes[i].numFaces = meshes[i].numIndices / 4;
        if (Contains(info->filename, "osOcean"))
        {
            shape.mesh = meshes[i];
        }
        else
        {
            shape.mesh = ConvertQuadToTriangleMesh(info->arena, meshes[i]);
        }
        shape.materialName = materials[i];
        shape.groupName    = groupNames[i];
    }
}

static void LoadMoanaTransforms(PBRTFileInfo *info, string directory, string filePath)
{
    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(filePath);
    tokenizer.cursor = tokenizer.input.str;

    for (;;)
    {
        SkipToNextChar2(&tokenizer, '"');
        if (EndOfBuffer(&tokenizer)) break;

        string objFile;
        bool success = GetBetweenPair(objFile, &tokenizer, '"');
        Assert(success);

        string objectFileName =
            PushStr8F(info->arena, "%S.rtscene", RemoveFileExtension(objFile));

        InstanceType &inst       = info->fileInstances.AddBack();
        inst.filename            = objectFileName;
        inst.transformIndexStart = info->transforms.Length();

        for (;;)
        {
            SkipToNextChar(&tokenizer);
            if (*tokenizer.cursor == '}') break;

            info->numInstances++;
            SkipToNextChar2(&tokenizer, '[');
            AffineSpace transform;
            for (u32 c = 0; c < 4; c++)
            {
                for (u32 r = 0; r < 4; r++)
                {
                    SkipToNextDigit(&tokenizer);
                    float value = ReadFloat(&tokenizer);
                    if (r < 3)
                    {
                        transform[c][r] = value;
                    }
                }
            }
            SkipToNextChar2(&tokenizer, ']');
            Advance(&tokenizer, "]");
            info->transforms.Push(transform);
        }
        inst.transformIndexEnd = info->transforms.Length() - 1;
    }
}

static void ProcessInstancedPrimitiveJson(Arena *arena, string directory, Tokenizer &tokenizer,
                                          ChunkedLinkedList<PBRTFileInfo> &fileInfos,
                                          Scheduler::Counter &counter,
                                          std::atomic<u32> *totalMeshCount,
                                          Array<string> &archiveFilenames)
{
    SkipToNextChar2(&tokenizer, '"');
    for (;;)
    {
        string jsonFile = {};
        if (*tokenizer.cursor == '}')
        {
            tokenizer.cursor++;

            SkipToNextChar(&tokenizer);
            if (*tokenizer.cursor == '}') break;
        }
        SkipToNextChar(&tokenizer, ',');

        string name;
        bool success = GetBetweenPair(name, &tokenizer, '"');
        Assert(success);
        SkipToNextChar2(&tokenizer, '"');
        for (;;)
        {
            SkipToNextChar(&tokenizer, ',');
            if (*tokenizer.cursor == '}') break;
            string field;
            success = GetBetweenPair(field, &tokenizer, '"');
            Assert(success);

            if (field == "jsonFile")
            {
                SkipToNextChar2(&tokenizer, '"');
                success = GetBetweenPair(jsonFile, &tokenizer, '"');
                Assert(success);
            }
            else if (field == "archives")
            {
                Assert(jsonFile.size != 0);
                PBRTFileInfo *transformsInfo = &fileInfos.AddBack();
                string transformsFilename    = StrConcat(arena, directory, jsonFile);
                transformsInfo->Init(
                    PushStr8F(arena, "%S.rtscene", RemoveFileExtension(jsonFile)));

                scheduler.Schedule(&counter, [=](u32 jobID) {
                    LoadMoanaTransforms(transformsInfo, directory, transformsFilename);
                });

                for (;;)
                {
                    if (*tokenizer.cursor == ']')
                    {
                        tokenizer.cursor++;
                        SkipToNextChar(&tokenizer, ',');
                        break;
                    }

                    SkipToNextChar2(&tokenizer, '"');
                    string archiveFilename;
                    success = GetBetweenPair(archiveFilename, &tokenizer, '"');
                    Assert(success);

                    string fileInfoName =
                        PushStr8F(arena, "%S.rtscene", RemoveFileExtension(archiveFilename));

                    bool found = false;
                    for (string &archiveName : archiveFilenames)
                    {
                        if (archiveName == fileInfoName)
                        {
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        archiveFilenames.Push(fileInfoName);
                        PBRTFileInfo *archiveInfo = &fileInfos.AddBack();

                        archiveInfo->Init(fileInfoName);
                        archiveInfo->base = false;

                        string archiveFilepath = StrConcat(arena, directory, archiveFilename);

                        scheduler.Schedule(&counter, [=](u32 jobID) {
                            LoadMoanaOBJ(archiveInfo, archiveFilepath);
                            totalMeshCount->fetch_add(archiveInfo->shapes.Length());
                        });
                    }

                    SkipToNextChar(&tokenizer);
                }
            }
            else if (field == "type")
            {
                SkipToNextChar2(&tokenizer, '"');
                string type;
                success = GetBetweenPair(type, &tokenizer, '"');
                Assert(success);
                if (type == "curve")
                {
                    SkipToNextChar2(&tokenizer, '}');
                }
                else if (type == "archive")
                {
                }
                else
                {
                    Assert(0);
                }
            }
            else if (field == "widthTip" || field == "widthRoot" || field == "degrees" ||
                     field == "faceCamera")
            {
                SkipToNextChar2(&tokenizer, '}');
            }
            else
            {
                Assert(0);
            }
        }
    }
}

static void LoadMoanaJSON(SceneLoadState *sls, PBRTFileInfo *base, string directory,
                          string filePath, Array<DisneyMaterial> &allDisneyMaterials)
{
    Arena *arena = sls->arenas[GetThreadIndex()];
    ScratchArena scratch;
    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(filePath);
    tokenizer.cursor = tokenizer.input.str;

    ChunkedLinkedList<PBRTFileInfo> fileInfos(scratch.temp.arena, 16);

    Scheduler::Counter counter = {};

    ChunkedLinkedList<AffineSpace> instanceTransforms(scratch.temp.arena, 1024);
    Array<DisneyMaterial> disneyMaterials;
    Array<string> archiveFilenames(scratch.temp.arena, 8);
    string name;
    std::atomic<u32> totalMeshCount = 0;
    for (;;)
    {
        SkipToNextChar2(&tokenizer, '"');
        if (EndOfBuffer(&tokenizer)) break;
        string parameterType;
        bool success = GetBetweenPair(parameterType, &tokenizer, '"');
        Assert(success);

        if (parameterType == "transformMatrix")
        {
            AffineSpace baseTransform;
            for (u32 c = 0; c < 4; c++)
            {
                for (u32 r = 0; r < 4; r++)
                {
                    SkipToNextDigit(&tokenizer);
                    float value = ReadFloat(&tokenizer);
                    if (r < 3)
                    {
                        baseTransform[c][r] = value;
                    }
                }
            }
            instanceTransforms.Push(baseTransform);
        }
        else if (parameterType == "geomObjFile")
        {
            SkipToNextChar2(&tokenizer, '"');
            string objFileName;
            success = GetBetweenPair(objFileName, &tokenizer, '"');
            Assert(success);

            string fullObjFilename = StrConcat(scratch.temp.arena, directory, objFileName);

            string objectFileName =
                PushStr8F(scratch.temp.arena, "%S.rtscene", RemoveFileExtension(objFileName));
            PBRTFileInfo *objFileInfo = &fileInfos.AddBack();
            objFileInfo->Init(objectFileName);
            objFileInfo->base = true;

            LoadMoanaOBJ(objFileInfo, fullObjFilename);
            totalMeshCount.fetch_add(objFileInfo->shapes.Length());
            // WriteFile(directory, objFileInfo);
        }
        else if (parameterType == "matFile")
        {
            SkipToNextChar2(&tokenizer, '"');
            string matFilename;
            success = GetBetweenPair(matFilename, &tokenizer, '"');
            Assert(success);
            disneyMaterials = LoadMaterialJSON(sls, directory, matFilename);
        }
        else if (parameterType == "name")
        {
            SkipToNextChar2(&tokenizer, '"');
            success = GetBetweenPair(name, &tokenizer, '"');
            Assert(success);
        }
        else if (parameterType == "animated")
        {
        }
        else if (parameterType == "instancedPrimitiveJsonFiles")
        {
            ProcessInstancedPrimitiveJson(scratch.temp.arena, directory, tokenizer, fileInfos,
                                          counter, &totalMeshCount, archiveFilenames);
        }
        else if (parameterType == "instancedCopies")
        {
            SkipToNextChar2(&tokenizer, '"');

            for (;;)
            {
                string name;
                SkipToNextChar(&tokenizer, ',');
                if (*tokenizer.cursor == '}') break;
                bool success = GetBetweenPair(name, &tokenizer, '"');
                Assert(success);
                SkipToNextChar2(&tokenizer, '"');

                bool differentGeometry = false;
                AffineSpace baseTransform;

                auto *startInfo    = fileInfos.last;
                u32 startInfoIndex = startInfo->count;

                for (;;)
                {
                    SkipToNextChar(&tokenizer, ',');
                    if (*tokenizer.cursor == '}')
                    {
                        bool success = Advance(&tokenizer, "}");
                        Assert(success);
                        break;
                    }

                    string field;
                    success = GetBetweenPair(field, &tokenizer, '"');
                    Assert(success);

                    if (field == "name")
                    {
                        SkipToNextChar2(&tokenizer, '"');
                        string instanceName;
                        success = GetBetweenPair(instanceName, &tokenizer, '"');
                        Assert(success);
                    }
                    else if (field == "transformMatrix")
                    {
                        for (u32 c = 0; c < 4; c++)
                        {
                            for (u32 r = 0; r < 4; r++)
                            {
                                SkipToNextDigit(&tokenizer);
                                float value = ReadFloat(&tokenizer);
                                if (r < 3)
                                {
                                    baseTransform[c][r] = value;
                                }
                            }
                        }
                        SkipToNextChar2(&tokenizer, ']');
                        success = Advance(&tokenizer, "]");
                        Assert(success);
                    }
                    else if (field == "geomObjFile")
                    {
                        SkipToNextChar2(&tokenizer, '"');
                        string objFileName;
                        success = GetBetweenPair(objFileName, &tokenizer, '"');
                        Assert(success);

                        string fullObjFilename =
                            StrConcat(scratch.temp.arena, directory, objFileName);

                        string objectFileName     = PushStr8F(scratch.temp.arena, "%S.rtscene",
                                                              RemoveFileExtension(objFileName));
                        PBRTFileInfo *objFileInfo = &fileInfos.AddBack();
                        objFileInfo->Init(objectFileName);
                        objFileInfo->base = true;

                        LoadMoanaOBJ(objFileInfo, fullObjFilename);
                        totalMeshCount.fetch_add(objFileInfo->shapes.Length());
                    }
                    else if (field == "instancedPrimitiveJsonFiles")
                    {
                        ProcessInstancedPrimitiveJson(scratch.temp.arena, directory, tokenizer,
                                                      fileInfos, counter, &totalMeshCount,
                                                      archiveFilenames);
                        differentGeometry = true;
                        bool success      = Advance(&tokenizer, "}");
                        Assert(success);
                    }
                    else
                    {
                        // TODO
                        Assert(0);
                    }
                }

                if (!differentGeometry)
                {
                    instanceTransforms.Push(baseTransform);
                }
                else
                {
                    scheduler.Wait(&counter);
                    u32 infoIndex = startInfoIndex;
                    for (auto *node = startInfo; node != 0; node = node->next)
                    {
                        for (; infoIndex < node->count; infoIndex++)
                        {
                            PBRTFileInfo *info      = &node->values[infoIndex];
                            info->differentGeometry = true;

                            if (info->transforms.Length())
                            {
                                AffineSpace &transform = baseTransform;
                                if (transform[0] == Vec3f(1, 0, 0) &&
                                    transform[1] == Vec3f(0, 1, 0) &&
                                    transform[2] == Vec3f(0, 0, 1) && transform[3] == Vec3f(0))
                                {
                                    Assert(info->numInstances);
                                    base->Merge(info);
                                }
                                else
                                {
                                    InstanceType instance;
                                    instance.filename =
                                        PushStr8Copy(base->arena, info->filename);
                                    instance.transformIndexStart = base->transforms.Length();
                                    instance.transformIndexEnd   = base->transforms.Length();
                                    base->fileInstances.Push(instance);
                                    base->numInstances += 1;
                                    WriteFile(directory, info);
                                    ArenaRelease(info->arena);
                                }
                            }
                            else if (info->base)
                            {
                                InstanceType instance;
                                instance.filename = PushStr8Copy(base->arena, info->filename);
                                instance.transformIndexStart = base->transforms.Length();
                                instance.transformIndexEnd   = base->transforms.Length();
                                base->fileInstances.Push(instance);
                                base->numInstances += 1;
                            }
                        }
                        infoIndex = 0;
                    }
                    base->transforms.Push(baseTransform);
                }
            }
        }
        else if (parameterType == "variants")
        {
            Assert(0);
        }
    }

    // PBRTFileInfo base;
    // base.Init("base.rtscene");
    u32 transformIndexStart = base->transforms.Length();
    for (auto *node = instanceTransforms.first; node != 0; node = node->next)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            base->transforms.Push(node->values[i]);
        }
    }
    scheduler.Wait(&counter);

    PBRTFileInfo *infos =
        PushArrayNoZero(scratch.temp.arena, PBRTFileInfo, fileInfos.Length());
    fileInfos.Flatten(infos);

    u32 hashSize = NextPowerOfTwo(disneyMaterials.Length());
    HashIndex materialHash(scratch.temp.arena, hashSize, hashSize);
    StaticArray<DisneyMaterial> newDisneyMaterials(scratch.temp.arena, totalMeshCount);

    StaticArray<string> colorMaps(scratch.temp.arena, totalMeshCount);
    HashIndex newMaterialHash(scratch.temp.arena, NextPowerOfTwo(totalMeshCount),
                              NextPowerOfTwo(totalMeshCount));

    for (u32 materialIndex = 0; materialIndex < disneyMaterials.Length(); materialIndex++)
    {
        DisneyMaterial &material = disneyMaterials[materialIndex];
        string newMaterialName   = StrConcat(arena, name, material.name);
        material.name            = newMaterialName;
        u32 hash                 = Hash(material.name);

        materialHash.AddInHash(hash, materialIndex);

        if (material.colorMap.size == 0)
        {
            bool result =
                sls->materialMap.FindOrAdd(arena, newMaterialName, sls->materialCounter);
            Assert(!result);
            newDisneyMaterials.Push(material);
        }
    }

    for (u32 fileInfoIndex = 0; fileInfoIndex < fileInfos.Length(); fileInfoIndex++)
    {
        PBRTFileInfo *info = &infos[fileInfoIndex];
        if (info->transforms.Length() && !info->differentGeometry)
        {
            if (instanceTransforms.Length() == 1)
            {
                AffineSpace &transform = instanceTransforms.first->values[0];
                if (transform[0] == Vec3f(1, 0, 0) && transform[1] == Vec3f(0, 1, 0) &&
                    transform[2] == Vec3f(0, 0, 1) && transform[3] == Vec3f(0))
                {
                    Assert(info->numInstances);
                    base->Merge(info);
                }
                else
                {
                    InstanceType instance;
                    instance.filename            = PushStr8Copy(base->arena, info->filename);
                    instance.transformIndexStart = transformIndexStart;
                    instance.transformIndexEnd =
                        transformIndexStart + instanceTransforms.Length() - 1;
                    base->fileInstances.Push(instance);
                    base->numInstances += instanceTransforms.Length();
                    WriteFile(directory, info);
                }
            }
            else
            {
                InstanceType instance;
                instance.filename            = PushStr8Copy(base->arena, info->filename);
                instance.transformIndexStart = transformIndexStart;
                instance.transformIndexEnd =
                    transformIndexStart + instanceTransforms.Length() - 1;
                base->fileInstances.Push(instance);
                base->numInstances += instanceTransforms.Length();
                WriteFile(directory, info);
            }
        }
        else if (info->shapes.Length())
        {
            for (auto *shapeNode = info->shapes.first; shapeNode != 0;
                 shapeNode       = shapeNode->next)
            {
                for (u32 i = 0; i < shapeNode->count; i++)
                {
                    ShapeType &shape = shapeNode->values[i];
                    // TODO handle this :)
                    // if (shape.materialName == "hidden")
                    // {
                    //     ArenaRelease(info->arena);
                    // }
                    shape.materialName = StrConcat(arena, name, shape.materialName);
                    u32 hash           = Hash(shape.materialName);
                    bool found         = false;
                    bool hasColorMap   = false;
                    DisneyMaterial material;
                    for (int hashIndex = materialHash.FirstInHash(hash); hashIndex != -1;
                         hashIndex     = materialHash.NextInHash(hashIndex))
                    {
                        if (disneyMaterials[hashIndex].name == shape.materialName)
                        {
                            found       = true;
                            hasColorMap = disneyMaterials[hashIndex].colorMap.size != 0;
                            material    = disneyMaterials[hashIndex];
                            break;
                        }
                    }
                    // ErrorExit(found, "material not found\n");
                    if (hasColorMap)
                    {
                        string newMaterialName =
                            StrConcat(arena, shape.materialName, shape.groupName);
                        string colorMapName;
                        if (material.colorMap.str[material.colorMap.size - 1] != '/')
                        {
                            colorMapName = PushStr8F(arena, "%S/%S.ptx", material.colorMap,
                                                     shape.groupName);
                        }
                        else
                        {
                            colorMapName = PushStr8F(arena, "%S%S.ptx", material.colorMap,
                                                     shape.groupName);
                        }

                        u32 hash   = Hash(newMaterialName);
                        bool found = false;
                        for (int hashIndex = newMaterialHash.FirstInHash(hash);
                             hashIndex != -1;
                             hashIndex = newMaterialHash.NextInHash(hashIndex))
                        {
                            if (newMaterialName == newDisneyMaterials[hashIndex].name)
                            {
                                found = true;
                                break;
                            }
                        }

                        if (!found)
                        {
                            newMaterialHash.AddInHash(hash, newDisneyMaterials.Length());
                            bool result = sls->materialMap.FindOrAdd(arena, newMaterialName,
                                                                     sls->materialCounter);
                            Assert(!result);
                            material.name     = newMaterialName;
                            material.colorMap = colorMapName;
                            newDisneyMaterials.Push(material);
                            colorMaps.Push(colorMapName);
                        }

                        shape.materialName = newMaterialName;
                        shape.mesh.faceIDs = PushArrayNoZero(scratch.temp.arena, u32,
                                                             shape.mesh.numIndices / 3);
                        for (u32 i = 0; i < shape.mesh.numIndices / 3; i++)
                        {
                            shape.mesh.faceIDs[i] = i / 2;
                        }
                    }
                }
            }

            if (info->base && !info->differentGeometry)
            {
                InstanceType instance;
                instance.filename            = PushStr8Copy(base->arena, info->filename);
                instance.transformIndexStart = transformIndexStart;
                instance.transformIndexEnd =
                    transformIndexStart + instanceTransforms.Length() - 1;
                base->fileInstances.Push(instance);

                base->numInstances += instanceTransforms.Length();
            }

#ifdef USE_GPU
            info->virtualGeoFilename = WriteNanite(info, sls, directory, info->filename);
            info->shapes.Clear();
#endif
            WriteFile(directory, info);
            ArenaRelease(info->arena);
        }
    }

    for (DisneyMaterial &material : newDisneyMaterials)
    {
        allDisneyMaterials.Push(material);
    }

#ifdef USE_GPU
    for (string colorMap : colorMaps)
    {
        string textureFilename = StrConcat(scratch.temp.arena, directory, colorMap);
        Convert(textureFilename);
    }
#endif

    scheduler.Wait(&counter);
}

static void LoadMoanaJSON(Arena *arena, string directory)
{
    ScratchArena scratch(&arena, 1);
    SceneLoadState sls;
    sls.Init(arena);

    PBRTFileInfo baseInfo;
    baseInfo.Init("base.rtscene");

    Array<DisneyMaterial> disneyMaterials(arena, 100);

    // TODO: don't hardcode this
    string testFilename = "../../data/island/pbrt-v4/json/isMountainB/isMountainB.json";
    LoadMoanaJSON(&sls, &baseInfo, directory, testFilename, disneyMaterials);

    testFilename = "../../data/island/pbrt-v4/json/osOcean/osOcean.json";
    LoadMoanaJSON(&sls, &baseInfo, directory, testFilename, disneyMaterials);

    testFilename = "../../data/island/pbrt-v4/json/isMountainA/isMountainA.json";
    LoadMoanaJSON(&sls, &baseInfo, directory, testFilename, disneyMaterials);

    testFilename = "../../data/island/pbrt-v4/json/isNaupakaA/isNaupakaA.json";
    LoadMoanaJSON(&sls, &baseInfo, directory, testFilename, disneyMaterials);

    testFilename = "../../data/island/pbrt-v4/json/isCoral/isCoral.json";
    LoadMoanaJSON(&sls, &baseInfo, directory, testFilename, disneyMaterials);

    WriteFile(directory, &baseInfo, 0, &disneyMaterials);
}

void LoadPBRT(Arena *arena, string filename)
{
    TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();

    SceneLoadState sls;
    sls.Init(arena);

    string directory = Str8PathChopPastLastSlash(filename);
    string baseFile  = PathSkipLastSlash(filename);

    OS_CreateDirectory(StrConcat(temp.arena, directory, "objects"));

    PerformanceCounter counter = OS_StartCounter();
    LoadPBRT(&sls, directory, baseFile);

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
    u32 numProcessors     = OS_NumProcessors();
    threadLocalStatistics = PushArray(arena, ThreadStatistics, numProcessors);
    scheduler.Init(numProcessors);

    threadLocalStatistics  = PushArray(arena, ThreadStatistics, numProcessors);
    threadMemoryStatistics = PushArray(arena, ThreadMemoryStatistics, numProcessors);

    TempArena temp        = ScratchStart(0, 0);
    StringBuilder builder = {};
    builder.arena         = arena;

    InitializePtex(1, gigabytes(1));

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

    ValidationMode mode = ValidationMode::Verbose;
    Vulkan *v           = PushStructConstruct(arena, Vulkan)(mode);
    device              = v;

#if 0
    LoadPBRT(arena, filename);
    // string directory = Str8PathChopPastLastSlash(filename);
    // string baseFile  = PathSkipLastSlash(filename);
    //
    // string rtSceneFilename = PushStr8F(arena, "%S.rtscene", RemoveFileExtension(filename));
    // SimplifyScene(arena, directory, rtSceneFilename);
#else
    string testFilename = "../../data/island/pbrt-v4/json/isMountainB/isMountainB.json";
    string directory    = "../../data/island/pbrt-v4/";
    string baseFile     = PathSkipLastSlash(testFilename);
    LoadMoanaJSON(arena, directory);

    // int numMeshes, actualNumMeshes;
    // // string testFilename = "../../data/island/pbrt-v4/obj/isMountainB/archives/"
    // //                       "xgFoliageA_treeMadronaBaked_canopyOnly_lo.obj";
    //
    // string testFilename =
    //     "../../data/island/pbrt-v4/obj/isBeach/archives/xgPebbles_archiveRock0002_geo.obj";
    // // string testFilename = "../../data/island/pbrt-v4/obj/osOcean/osOcean.obj";
    //
    // Mesh *meshes = LoadObj(arena, testFilename, numMeshes, actualNumMeshes);
    // // Mesh *meshes = LoadObjWithWedges(arena, testFilename, numMeshes); //,
    // actualNumMeshes);
    //
    // meshes[0].numFaces = meshes[0].numIndices / 4;
    // Mesh mesh          = ConvertQuadToTriangleMesh(arena, meshes[0]);
    //
    // PerformanceCounter counter = OS_StartCounter();
    // StaticArray<u32> materialIndices(arena, 1);
    // materialIndices.Push(0);
    // CreateClusters(&mesh, 1, materialIndices, testFilename);
    //
    // f32 t = OS_GetMilliseconds(counter);
    // Print("%f\n", t);
#endif

    u64 count        = 0;
    f64 time         = 0;
    u32 verts        = 0;
    u32 inds         = 0;
    u32 numInstances = 0;
    for (int i = 0; i < numProcessors; i++)
    {
        count += threadLocalStatistics[i].misc;
        verts += threadLocalStatistics[i].misc2;
        numInstances += threadLocalStatistics[i].misc4;

        inds += threadLocalStatistics[i].misc3;
        time += threadLocalStatistics[i].miscF;
    }
    printf("num materials pruned: %llu\n", count);
    printf("total gpu time: %f\n", time);
    printf("verts: %u indices: %u\n", verts, inds);
    printf("num instances: %u\n", numInstances);

    // read pbrt as i've done before, getting scene packets
    // list of things to do
    // 1. reduce material count by hashing?
    return 0;
}
