#ifndef SCENE_LOAD_H
#define SCENE_LOAD_H

#include "hash.h"
#include "thread_statistics.h"
#include "platform.h"
#include "thread_context.h"
#include "mesh.h"
#include <unordered_set>

namespace rt
{

static const u32 MAX_PARAMETER_COUNT = 16;

inline GeometryType ConvertStringIDToGeometryType(StringId id)
{
    switch (id)
    {
        case "quadmesh"_sid: return GeometryType::QuadMesh;
        case "trianglemesh"_sid: return GeometryType::TriangleMesh;
        case "catclark"_sid: return GeometryType::CatmullClark;
        default: Assert(0); return GeometryType::Max;
    }
}

enum class DataType
{
    Float,
    Floats,
    Vec2,
    Vec3,
    Int,
    Bool,
    String,
    Texture,
    Blackbody,
    Spectrum,
};

inline MaterialTypes ConvertStringToMaterialType(string type)
{
    if (type == "diffuse") return MaterialTypes::Diffuse;
    else if (type == "diffusetransmission") return MaterialTypes::DiffuseTransmission;
    else if (type == "coateddiffuse") return MaterialTypes::CoatedDiffuse;
    else if (type == "dielectric") return MaterialTypes::Dielectric;
    else if (type == "interface") return MaterialTypes::Interface;
    else
    {
        ErrorExit(0, "Material type not supported or valid.\n");
        return MaterialTypes(0);
    }
}

enum class TextureType
{
    bilerp,
    checkerboard,
    constant,
    directionmix,
    dots,
    fbm,
    imagemap,
    marble,
    mix,
    ptex,
    scale,
    windy,
    wrinkled,
    Max,
};
ENUM_CLASS_FLAGS(TextureType)

inline TextureType ConvertStringToTextureType(string type)
{
    if (type == "ptex") return TextureType::ptex;
    // else if (type == "imagemap") return TextureType::imagemap;
    else ErrorExit(0, "Texture type not supported or valid\n");
    return TextureType(0);
}

static const string materialTypeNames[] = {
    "interface", "diffuse", "diffusetransmission", "coateddiffuse", "dielectric",
};

static const StringId materialTypeIDs[] = {
    "interface"_sid,     "diffuse"_sid,    "diffusetransmission"_sid,
    "coateddiffuse"_sid, "dielectric"_sid,
};

static const StringId diffuseParameterIds[] = {
    "reflectance"_sid,
    "displacement"_sid,
};

static const string diffuseParameterNames[] = {
    "reflectance",
    "displacement",
};

static const StringId diffuseTransmissionIds[] = {
    "reflectance"_sid,
    "transmittance"_sid,
    "scale"_sid,
    "displacement"_sid,
};

static const string diffuseTransmissionNames[] = {
    "reflectance",
    "transmittance",
    "scale",
    "displacement",
};

static const StringId dielectricIds[] = {
    "roughness"_sid,      "uroughness"_sid, "vroughness"_sid,
    "remaproughness"_sid, "eta"_sid,        "displacement"_sid,
};

static const string dielectricNames[] = {
    "roughness", "uroughness", "vroughness", "remaproughness", "eta", "displacement",
};

static const StringId coatedDiffuseIds[] = {
    "roughness"_sid, "uroughness"_sid,  "vroughness"_sid, "remaproughness"_sid,
    "eta"_sid,       "reflectance"_sid, "albedo"_sid,     "g"_sid,
    "maxdepth"_sid,  "nsamples"_sid,    "thickness"_sid,  "displacement"_sid,
};

static const string coatedDiffuseNames[] = {
    "roughness", "uroughness", "vroughness", "remaproughness", "eta",       "reflectance",
    "albedo",    "g",          "maxdepth",   "nsamples",       "thickness", "displacement",
};

static const StringId *materialParameterIDs[] = {
    0, diffuseParameterIds, diffuseTransmissionIds, coatedDiffuseIds, dielectricIds};

static const StringId materialParameterCounts[] = {
    0,
    ArrayLength(diffuseParameterIds),
    ArrayLength(diffuseTransmissionIds),
    ArrayLength(coatedDiffuseIds),
    ArrayLength(dielectricIds),
};

static const string *materialParameterNames[] = {
    0, diffuseParameterNames, diffuseTransmissionNames, coatedDiffuseNames, dielectricNames,
};

static const string textureTypeNames[] = {
    "bilerp", "checkerboard", "constant", "directionmix", "dots",  "fbm",      "imagemap",
    "marble", "mix",          "ptex",     "scale",        "windy", "wrinkled",
};

struct FileOffsets
{
    u64 metaOffset;
    u64 infoOffset;
    u64 dataOffset;
};

struct ScenePacket
{
    StringId *parameterNames;
    u8 **bytes;
    u32 *sizes;
    DataType *types;

    StringId type;

    // const string **parameterNames;
    // SceneByteType *types;
    u32 parameterCount;

    void Initialize(Arena *arena, u32 count)
    {
        // parameterCount = count;
        // parameterCount = 0;
        parameterNames = PushArray(arena, StringId, count);
        bytes          = PushArray(arena, u8 *, count);
        sizes          = PushArray(arena, u32, count);
        types          = PushArray(arena, DataType, count);
    }

    inline i32 GetInt(i32 i) const { return *(i32 *)(bytes[i]); }
    inline bool GetBool(i32 i) const { return *(bool *)(bytes[i]); }
    inline f32 GetFloat(i32 i) const { return *(f32 *)(bytes[i]); }
    inline i32 FindKey(StringId parameterName)
    {
        for (u32 i = 0; i < parameterCount; i++)
        {
            if (parameterNames[i] == parameterName) return i;
        }
        return -1;
    }
};

inline bool CheckQuadPLY(string filename)
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
                ErrorExit(0, "elementType: %s\n", (char *)elementType.str);
            }
        }
        else if (word == "end_header") break;
    }

    // 2 triangles/1 quad for every 4 vertices. If this condition isn't met, it isn't a quad
    // mesh
    return numFaces == numVertices / 2;
}

inline u32 GetTypeStride(string word)
{
    if (word == "uint8" || word == "char" || word == "uchar") return 1;
    else if (word == "short" || word == "ushort") return 2;
    else if (word == "int" || word == "uint" || word == "float") return 4;
    else if (word == "double") return 8;
    ErrorExit(0, "Invalid type: %s\n", (char *)word.str);
    return 0;
}

inline Mesh LoadPLY(Arena *arena, string filename, GeometryType type)
{
    string buffer = OS_MapFileRead(filename);
    Tokenizer tokenizer;
    tokenizer.input  = buffer;
    tokenizer.cursor = buffer.str;

    string line = ReadLine(&tokenizer);
    Assert(line == "ply");
    line = ReadLine(&tokenizer);
    Assert(line == "format binary_little_endian 1.0");

    u32 numVertices = 0;
    u32 numFaces    = 0;

    u32 totalVertexStride = 0;

    // Face
    u32 countStride       = 0;
    u32 faceIndicesStride = 0;
    u32 otherStride       = 0;

    bool hasVertices = 0;
    bool hasNormals  = 0;
    bool hasUv       = 0;
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

                    if (word == "x")
                    {
                        hasVertices = 1;
                        totalVertexStride += 12;
                    }
                    if (word == "nx")
                    {
                        hasNormals = 1;
                        totalVertexStride += 12;
                    }
                    if (word == "u")
                    {
                        hasUv = 1;
                        totalVertexStride += 8;
                    }
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
                        string countType    = ReadWord(&tokenizer);
                        string listType     = ReadWord(&tokenizer);
                        string propertyType = ReadWord(&tokenizer);
                        if (propertyType == "vertex_indices")
                        {
                            countStride       = GetTypeStride(countType);
                            faceIndicesStride = GetTypeStride(listType);
                        }
                        else
                        {
                            Assert(0);
                        }
                    }
                    else
                    {
                        string propertyType = ReadWord(&tokenizer);
                        if (propertyType == "face_indices")
                        {
                            otherStride = GetTypeStride(word);
                        }
                        else
                        {
                            Assert(0);
                        }
                    }
                }
            }
            else
            {
                ErrorExit(0, "elementType: %s\n", (char *)elementType.str);
            }
        }
        else if (word == "end_header") break;
    }

    bool isQuad = type == GeometryType::QuadMesh;

    // Read binary data
    Mesh mesh        = {};
    mesh.numVertices = numVertices;
    mesh.numIndices  = numFaces * 3;
    mesh.numFaces    = isQuad ? mesh.numFaces / 2 : numFaces;

    if (hasVertices) mesh.p = PushArray(arena, Vec3f, numVertices);
    if (hasNormals) mesh.n = PushArray(arena, Vec3f, numVertices);
    if (hasUv && !isQuad) mesh.uv = PushArray(arena, Vec2f, numVertices);
    if (!isQuad) mesh.indices = PushArray(arena, u32, numFaces * 3);

    for (u32 i = 0; i < numVertices; i++)
    {
        string bytes = ReadBytes(&tokenizer, totalVertexStride);
        if (hasVertices)
        {
            Assert(totalVertexStride >= 12);
            f32 *pos  = (f32 *)bytes.str;
            mesh.p[i] = Vec3f(pos[0], pos[1], pos[2]);
        }
        if (hasNormals)
        {
            Assert(totalVertexStride >= 24);
            f32 *norm = (f32 *)bytes.str + 3;
            mesh.n[i] = Vec3f(norm[0], norm[1], norm[2]);
        }
        if (hasUv && !isQuad)
        {
            Assert(totalVertexStride >= 32);
            f32 *uv    = (f32 *)bytes.str + 6;
            mesh.uv[i] = Vec2f(uv[0], uv[1]);
        }
    }

    Assert(countStride == 1);
    Assert(faceIndicesStride == 4);
    // Assert(otherStride == 4);
    // u32 *faceIndices = PushArray(arena, u32, numFaces);
    if (!isQuad)
    {
        for (u32 i = 0; i < numFaces; i++)
        {
            u8 *bytes    = tokenizer.cursor;
            u8 count     = bytes[0];
            u32 *indices = (u32 *)(bytes + 1);
            for (u32 j = 0; j < count; j++)
            {
                mesh.indices[count * i + j] = indices[j];
            }

            u32 faceIndex = indices[3];
            // faceIndices[i] = faceIndex;
            Advance(&tokenizer, countStride + count * faceIndicesStride + otherStride);
        }
        Assert(EndOfBuffer(&tokenizer));
    }
    OS_UnmapFile(buffer.str);
    return mesh;
}

inline Mesh LoadQuadPLY(Arena *arena, string filename)
{
    return LoadPLY(arena, filename, GeometryType::QuadMesh);
}

inline Mesh LoadTrianglePLY(Arena *arena, string filename)
{
    return LoadPLY(arena, filename, GeometryType::TriangleMesh);
}

inline Mesh *LoadObj(Arena *arena, string filename, int &numMeshes, int &actualNumMeshes)
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
    int totalNumMeshes  = 0;

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
                    if (!repeatedMesh)
                    {
                        Mesh mesh;
                        mesh.numVertices = (u32)vertices.size();
                        mesh.numIndices  = (u32)indices.size();

                        mesh.p = PushArrayNoZero(arena, Vec3f, vertices.size());
                        MemoryCopy(mesh.p, vertices.data(), sizeof(Vec3f) * vertices.size());

                        threadLocalStatistics[GetThreadIndex()].misc2 += mesh.numVertices;
                        threadLocalStatistics[GetThreadIndex()].misc3 += mesh.numIndices;

                        int *normalIndices = PushArray(temp.arena, int, mesh.numVertices);
                        MemorySet(normalIndices, 0xff, sizeof(int) * mesh.numVertices);

                        mesh.n       = PushArrayNoZero(arena, Vec3f, vertices.size());
                        mesh.indices = PushArrayNoZero(arena, u32, indices.size());
                        // Ensure that vertex index and normal index pairings are
                        // consistent
                        for (u32 i = 0; i < mesh.numIndices; i++)
                        {
                            i32 vertexIndex = indices[i][0];
                            i32 normalIndex = indices[i][2];

                            if (normalIndices[vertexIndex] != 0xffffffff)
                            {
                                ErrorExit(normalIndices[vertexIndex] == normalIndex,
                                          "Face-varying normals currently unsupported. "
                                          "file: %S\n",
                                          filename);
                            }

                            Assert(normalIndex < (int)normals.size());
                            Assert(vertexIndex < (int)vertices.size());
                            normalIndices[vertexIndex] = normalIndex;
                            mesh.n[vertexIndex]        = normals[normalIndex];
                            mesh.indices[i]            = vertexIndex;
                        }

                        meshes.push_back(mesh);

                        vertexOffset += (int)vertices.size();
                        normalOffset += (int)normals.size();
                        vertices.clear();
                        normals.clear();
                        indices.clear();
                    }
                    repeatedMesh = false;
                    totalNumMeshes++;
                }
                processingMesh = false;
            }
            else
            {
                processingMesh     = true;
                const auto &result = meshHashSet.find(groupName);
                if (result != meshHashSet.end()) repeatedMesh = true;
                else meshHashSet.insert(groupName);
            }
            if (isEndOfBuffer) break;
        }
        else if (word == "mtllib")
        {
            SkipToNextLine(&tokenizer);
        }
        else if (word == "usemtl")
        {
            SkipToNextLine(&tokenizer);
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
    numMeshes          = totalNumMeshes;
    actualNumMeshes    = (int)meshes.size();
    Mesh *outputMeshes = PushArrayNoZero(arena, Mesh, actualNumMeshes);
    MemoryCopy(outputMeshes, meshes.data(), sizeof(Mesh) * actualNumMeshes);
    return outputMeshes;
}

inline Mesh *LoadObjWithWedges(Arena *arena, string filename, int &numMeshes,
                               u32 *outFaceVertexCount = 0)
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
    std::vector<Vec2f> uvs;
    uvs.reserve(128);

    std::vector<Mesh> meshes;
    int vertexOffset = 1;
    int normalOffset = 1;
    int texOffset    = 1;

    int totalNumMeshes = 0;

    auto Skip = [&]() { SkipToNextChar(&tokenizer, '#'); };

    bool hasUvs     = false;
    bool hasNormals = false;

    auto GetUV = [&](int uvIndex) {
        Vec2f uv = uvIndex == -1 ? Vec2f(-1e8, -1e8) : uvs[uvIndex];
        return uv;
    };

    auto GetNorm = [&](int normIndex) {
        Vec3f n = normIndex == -1 ? Vec3f(-1e8, -1e8, -1e8) : normals[normIndex];
        return n;
    };

    bool processingMesh = false;

    u32 meshVertexCount = 0;

    for (;;)
    {
        Skip();

        string word = ReadWord(&tokenizer);

        bool isEndOfBuffer = EndOfBuffer(&tokenizer);
        if (word == "g" || isEndOfBuffer)
        {
            Skip();
            string groupName = ReadWord(&tokenizer);
            if ((groupName == "default" && processingMesh) || isEndOfBuffer)
            {
                std::vector<Vec3f> newPositions;
                newPositions.reserve(vertices.size());
                std::vector<Vec2f> newUvs;
                if (hasUvs) newUvs.reserve(uvs.size());
                std::vector<Vec3f> newNormals;
                newNormals.reserve(normals.size());

                int numIndices  = indices.size();
                int numVertices = vertices.size();

                int *remap   = PushArray(temp.arena, int, numIndices);
                int hashSize = NextPowerOfTwo(numIndices);
                HashIndex hashTable(temp.arena, hashSize, hashSize);

                Mesh mesh       = {};
                mesh.numIndices = (u32)indices.size();
                mesh.indices    = PushArrayNoZero(arena, u32, indices.size());

                for (u32 i = 0; i < numIndices; i++)
                {
                    i32 vertexIndex = indices[i][0];
                    int uvIndex     = indices[i][1];
                    i32 normalIndex = indices[i][2];
                    Vec3f p         = vertices[vertexIndex];
                    Vec3f n         = GetNorm(normalIndex);
                    Vec2f uv        = GetUV(uvIndex);

                    int hash   = Hash(p, uv, n);
                    bool found = false;
                    for (int j = hashTable.FirstInHash(hash); j != -1;
                         j     = hashTable.NextInHash(j))
                    {
                        Vec3i testIndices = indices[j];
                        Vec2f testUv      = GetUV(testIndices[1]);
                        Vec3f testNorm    = GetNorm(testIndices[2]);
                        if (p == vertices[testIndices[0]] && uv == testUv && n == testNorm)
                        {
                            mesh.indices[i] = remap[j];
                            found           = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        int currentIndex = newPositions.size();
                        remap[i]         = currentIndex;

                        newPositions.push_back(p);
                        if (hasNormals)
                        {
                            newNormals.push_back(n);
                        }

                        if (hasUvs)
                        {
                            newUvs.push_back(uv);
                        }

                        mesh.indices[i] = currentIndex;

                        int hash = Hash(p, uv, n);
                        hashTable.AddInHash(hash, i);
                    }
                }

                // u32 newSize = newPositions.size();
                // u32 oldSize = vertices.size();
                // Assert(newSize == oldSize);

                mesh.numVertices = (u32)newPositions.size();

                mesh.p = PushArrayNoZero(arena, Vec3f, mesh.numVertices);
                if (hasUvs) mesh.uv = PushArrayNoZero(arena, Vec2f, newUvs.size());
                if (hasNormals) mesh.n = PushArrayNoZero(arena, Vec3f, mesh.numVertices);

                MemoryCopy(mesh.p, newPositions.data(), sizeof(Vec3f) * newPositions.size());
                if (hasUvs) MemoryCopy(mesh.uv, newUvs.data(), sizeof(Vec2f) * newUvs.size());
                if (hasNormals)
                    MemoryCopy(mesh.n, newNormals.data(), sizeof(Vec3f) * newNormals.size());

                threadLocalStatistics[GetThreadIndex()].misc2 += mesh.numVertices;
                threadLocalStatistics[GetThreadIndex()].misc3 += mesh.numIndices;
                meshes.push_back(mesh);

                vertexOffset += (int)vertices.size();
                normalOffset += (int)normals.size();
                vertices.clear();
                normals.clear();
                indices.clear();
                totalNumMeshes++;

                processingMesh = false;
            }
            else
            {
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
            SkipToNextLine(&tokenizer);
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
            hasNormals = true;
            Skip();
            f32 x = ReadFloat(&tokenizer);
            f32 y = ReadFloat(&tokenizer);
            f32 z = ReadFloat(&tokenizer);
            normals.push_back(Vec3f(x, y, z));
        }
        else if (word == "vt")
        {
            hasUvs = true;
            Skip();
            f32 u = ReadFloat(&tokenizer);
            f32 v = ReadFloat(&tokenizer);
            uvs.push_back(Vec2f(u, v));
        }
        else if (word == "f")
        {
            Skip();
            u32 faceVertexCount = 0;
            while (CharIsDigit(tokenizer.cursor[0]))
            {
                faceVertexCount++;

                int vertexIndex = (int)ReadUint(&tokenizer);
                bool result     = Advance(&tokenizer, "/");
                Assert(result);
                int texIndex = (int)ReadUint(&tokenizer);
                result       = Advance(&tokenizer, "/");
                Assert(result);
                int normalIndex = ReadUint(&tokenizer);
                indices.push_back(Vec3i(vertexIndex - vertexOffset, texIndex - texOffset,
                                        normalIndex - normalOffset));
                Skip();
            }

            if (meshVertexCount == 0)
            {
                meshVertexCount = faceVertexCount;
            }
            else if (meshVertexCount != faceVertexCount)
            {
                Print("Mesh has multiple types of polygons\n");
            }
        }
        else
        {
            Assert(0);
        }
    }

    ScratchEnd(temp);

    if (outFaceVertexCount) *outFaceVertexCount = meshVertexCount;

    numMeshes          = totalNumMeshes;
    Mesh *outputMeshes = PushArrayNoZero(arena, Mesh, numMeshes);
    MemoryCopy(outputMeshes, meshes.data(), sizeof(Mesh) * numMeshes);
    return outputMeshes;
}

inline void WriteQuadOBJ(Mesh &mesh, string filename)
{
    StringBuilder builder = {};
    ScratchArena scratch;
    builder.arena = scratch.temp.arena;
    for (int i = 0; i < mesh.numVertices; i++)
    {
        const Vec3f &v = mesh.p[i];
        Put(&builder, "v %f %f %f\n", v.x, v.y, v.z);
    }
    if (mesh.uv)
    {
        for (int i = 0; i < mesh.numVertices; i++)
        {
            const Vec2f &uv = mesh.uv[i];
            Put(&builder, "vt %f %f \n", uv.x, uv.y);
        }
    }
    if (mesh.n)
    {
        for (int i = 0; i < mesh.numVertices; i++)
        {
            const Vec3f &n = mesh.n[i];
            Put(&builder, "vn %f %f %f\n", n.x, n.y, n.z);
        }
    }
    for (int i = 0; i < mesh.numIndices; i += 4)
    {
        Put(&builder, "f ");
        for (int j = 0; j < 4; j++)
        {
            int idx = mesh.indices[i + j] + 1;
            Put(&builder, "%u/", idx);
            if (mesh.uv) Put(&builder, "%u/", idx);
            else Put(&builder, "/");
            if (mesh.n) Put(&builder, "%u ", idx);
            else Put(&builder, " ");
        }
        Put(&builder, "\n");
    }
    WriteFileMapped(&builder, filename);
}

inline void WriteTriOBJ(Mesh &mesh, string filename)
{
    StringBuilder builder = {};
    ScratchArena scratch;
    builder.arena = scratch.temp.arena;
    Put(&builder, "g default\n");
    for (int i = 0; i < mesh.numVertices; i++)
    {
        const Vec3f &v = mesh.p[i];
        Put(&builder, "v %f %f %f\n", v.x, v.y, v.z);
    }
    if (mesh.uv)
    {
        for (int i = 0; i < mesh.numVertices; i++)
        {
            const Vec2f &uv = mesh.uv[i];
            Put(&builder, "vt %f %f \n", uv.x, uv.y);
        }
    }
    if (mesh.n)
    {
        for (int i = 0; i < mesh.numVertices; i++)
        {
            const Vec3f &n = mesh.n[i];
            Put(&builder, "vn %f %f %f\n", n.x, n.y, n.z);
        }
    }
    Put(&builder, "g geo\n");
    for (int i = 0; i < mesh.numIndices; i += 3)
    {
        Put(&builder, "f ");
        for (int j = 0; j < 3; j++)
        {
            int idx = mesh.indices[i + j] + 1;
            Put(&builder, "%u/", idx);
            if (mesh.uv) Put(&builder, "%u/", idx);
            else Put(&builder, "/");
            if (mesh.n) Put(&builder, "%u ", idx);
            else Put(&builder, " ");
        }
        Put(&builder, "\n");
    }
    WriteFileMapped(&builder, filename);
}

inline Mesh ConvertQuadToTriangleMesh(Arena *arena, Mesh mesh)
{
    Assert(mesh.numFaces * 4 == mesh.numIndices);
    u32 newNumIndices = mesh.numIndices / 4 * 6;
    u32 *newIndices   = PushArrayNoZero(arena, u32, mesh.numIndices / 4 * 6);
    u32 *faceIDs      = PushArrayNoZero(arena, u32, mesh.numFaces * 2);
    for (int faceIndex = 0; faceIndex < mesh.numFaces; faceIndex++)
    {
        u32 id0                    = mesh.indices[4 * faceIndex + 0];
        u32 id1                    = mesh.indices[4 * faceIndex + 1];
        u32 id2                    = mesh.indices[4 * faceIndex + 2];
        u32 id3                    = mesh.indices[4 * faceIndex + 3];
        u32 *writeIndices          = newIndices + 6 * faceIndex;
        writeIndices[0]            = id0;
        writeIndices[1]            = id1;
        writeIndices[2]            = id2;
        writeIndices[3]            = id0;
        writeIndices[4]            = id2;
        writeIndices[5]            = id3;
        faceIDs[2 * faceIndex + 0] = faceIndex;
        faceIDs[2 * faceIndex + 1] = faceIndex;
    }

    Mesh result       = mesh;
    result.indices    = newIndices;
    result.numIndices = newNumIndices;
    result.numFaces   = mesh.numFaces * 2;
    result.faceIDs    = faceIDs;
    return result;
}

} // namespace rt

#endif
