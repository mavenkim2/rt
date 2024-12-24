#ifndef SCENE_LOAD_H
#define SCENE_LOAD_H

namespace rt
{
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

enum class DataType
{
    Float,
    Vec2,
    Vec3,
    Int,
    Bool,
    String,
    Blackbody,
    Spectrum,
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
        // types          = PushArray(arena, SceneByteType, count);
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
    // inline u8 *GetByParamName(const string &name) const
    // {
    //     for (u32 i = 0; i < parameterCount; i++)
    //     {
    //         if (*parameterNames[i] == name)
    //         {
    //             return bytes[i];
    //         }
    //     }
    //     return 0;
    // }
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

inline u32 GetTypeStride(string word)
{
    if (word == "uint8" || word == "char" || word == "uchar") return 1;
    else if (word == "short" || word == "ushort") return 2;
    else if (word == "int" || word == "uint" || word == "float") return 4;
    else if (word == "double") return 8;
    Error(0, "Invalid type: %s\n", (char *)word.str);
    return 0;
}

QuadMesh LoadQuadPLY(Arena *arena, string filename)
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
                Error(0, "elementType: %s\n", (char *)elementType.str);
            }
        }
        else if (word == "end_header") break;
    }

    // Read binary data
    QuadMesh mesh    = {};
    mesh.numVertices = numVertices;

    // 2 triangles/1 quad for every 4 vertices. If this condition isn't met, it isn't a quad
    // mesh
    if (numFaces != numVertices / 2) return mesh;

    // mesh.numQuads    = numFaces / 2;
    if (hasVertices) mesh.p = PushArrayNoZero(arena, Vec3f, numVertices);
    if (hasNormals) mesh.n = PushArrayNoZero(arena, Vec3f, numVertices);

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
    }

    Assert(countStride == 1);
    Assert(faceIndicesStride == 4);
    // Assert(otherStride == 4);
    OS_UnmapFile(buffer.str);
    return mesh;
}
TriangleMesh LoadPLY(Arena *arena, string filename)
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
                Error(0, "elementType: %s\n", (char *)elementType.str);
            }
        }
        else if (word == "end_header") break;
    }

    // Read binary data
    TriangleMesh mesh = {};
    mesh.numVertices  = numVertices;
    mesh.numIndices   = numFaces * 3;
    if (hasVertices) mesh.p = PushArray(arena, Vec3f, numVertices);
    if (hasNormals) mesh.n = PushArray(arena, Vec3f, numVertices);
    if (hasUv) mesh.uv = PushArray(arena, Vec2f, numVertices);
    mesh.indices = PushArray(arena, u32, numFaces * 3);

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
        if (hasUv)
        {
            Assert(totalVertexStride >= 32);
            f32 *uv    = (f32 *)bytes.str + 6;
            mesh.uv[i] = Vec2f(uv[0], uv[1]);
        }
    }

    Assert(countStride == 1);
    Assert(faceIndicesStride == 4);
    // Assert(otherStride == 4);
    for (u32 i = 0; i < numFaces; i++)
    {
        u8 *bytes = tokenizer.cursor;
        u8 count  = bytes[0];
        Assert(count == 3);
        u32 *indices            = (u32 *)(bytes + 1);
        mesh.indices[3 * i + 0] = indices[0];
        mesh.indices[3 * i + 1] = indices[1];
        mesh.indices[3 * i + 2] = indices[2];

        // what is this value? I think it's for ptex or something??? where are the materials?
        u32 faceIndex = indices[3];
        Advance(&tokenizer, countStride + count * faceIndicesStride + otherStride);
    }
    Assert(EndOfBuffer(&tokenizer));
    OS_UnmapFile(buffer.str);
    return mesh;
}

} // namespace rt

#endif
