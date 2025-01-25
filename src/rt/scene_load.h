#ifndef SCENE_LOAD_H
#define SCENE_LOAD_H

namespace rt
{

static const u32 MAX_PARAMETER_COUNT = 16;

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

MaterialTypes ConvertStringToMaterialType(string type)
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

TextureType ConvertStringToTextureType(string type)
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

bool GetSectionOffsets(Tokenizer *tokenizer, FileOffsets *offsets)
{
    bool result = Advance(tokenizer, "META");
    if (!result) return false;
    GetPointerValue(tokenizer, &offsets->metaOffset);
    result = Advance(tokenizer, "INFO");
    if (!result) return false;
    GetPointerValue(tokenizer, &offsets->infoOffset);
    result = Advance(tokenizer, "DATA");
    if (!result) return false;
    GetPointerValue(tokenizer, &offsets->dataOffset);
    return true;
}

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

struct Options
{
    string filename;
    i32 pixelX = -1;
    i32 pixelY = -1;
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

Mesh LoadPLY(Arena *arena, string filename, GeometryType type)
{
    string buffer = OS_MapFileRead(filename); // OS_ReadFile(arena, filename);
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

Mesh LoadQuadPLY(Arena *arena, string filename)
{
    return LoadPLY(arena, filename, GeometryType::QuadMesh);
}

Mesh LoadTrianglePLY(Arena *arena, string filename)
{
    return LoadPLY(arena, filename, GeometryType::TriangleMesh);
}

Mesh *LoadQuadObj(Arena *arena, string filename, int &numMeshes)
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

    bool processingMesh = false;

    auto Skip = [&]() { SkipToNextChar(&tokenizer, '#'); };

    for (;;)
    {
        Skip();

        string word = ReadWord(&tokenizer);
        if (word == "f")
        {
            if (!processingMesh)
            {
                processingMesh = true;
            }
        }
        else
        {
            if (processingMesh)
            {
                Mesh mesh;
                mesh.numVertices = (u32)vertices.size();
                mesh.numIndices  = (u32)indices.size();

                mesh.p = PushArrayNoZero(arena, Vec3f, vertices.size());
                vertexOffset += (int)vertices.size();
                MemoryCopy(mesh.p, vertices.data(), sizeof(Vec3f) * vertices.size());

                Assert(normals.size() == vertices.size());

                int *normalIndices = PushArray(temp.arena, int, mesh.numVertices);
                MemorySet(normalIndices, 0xff, sizeof(int) * mesh.numVertices);

                mesh.n = PushArrayNoZero(arena, Vec3f, normals.size());
                normalOffset += (int)normals.size();
                mesh.indices = PushArrayNoZero(arena, u32, indices.size());
                // Ensure that vertex index and normal index pairings are consistent
                for (u32 i = 0; i < mesh.numIndices; i++)
                {
                    i32 vertexIndex = indices[i][0];
                    i32 normalIndex = indices[i][2];

                    if (normalIndices[vertexIndex] != 0xffffffff)
                    {
                        ErrorExit(normalIndices[vertexIndex] == normalIndex,
                                  "Face-varying normals currently unsupported.\n");
                    }
                    normalIndices[vertexIndex] = normalIndex;
                    mesh.n[vertexIndex]        = normals[normalIndex];
                    mesh.indices[i]            = vertexIndex;
                }

                meshes.push_back(mesh);

                vertices.clear();
                normals.clear();
                indices.clear();
                processingMesh = false;
            }
        }
        if (EndOfBuffer(&tokenizer)) break;

        if (word == "g")
        {
            SkipToNextLine(&tokenizer);
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
            Assert(faceVertexCount == 4)
        }
        else
        {
            Assert(0);
        }
    }

    ScratchEnd(temp);
    numMeshes          = (int)meshes.size();
    Mesh *outputMeshes = PushArrayNoZero(arena, Mesh, numMeshes);
    MemoryCopy(outputMeshes, meshes.data(), sizeof(Mesh) * numMeshes);
    return outputMeshes;
}

} // namespace rt

#endif
