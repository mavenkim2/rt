AABB Transform(const HomogeneousTransform &transform, const AABB &aabb)
{
    AABB result;
    vec3 vecs[] = {
        vec3(aabb.minX, aabb.minY, aabb.minZ),
        vec3(aabb.maxX, aabb.minY, aabb.minZ),
        vec3(aabb.maxX, aabb.maxY, aabb.minZ),
        vec3(aabb.minX, aabb.maxY, aabb.minZ),
        vec3(aabb.minX, aabb.minY, aabb.maxZ),
        vec3(aabb.maxX, aabb.minY, aabb.maxZ),
        vec3(aabb.maxX, aabb.maxY, aabb.maxZ),
        vec3(aabb.minX, aabb.maxY, aabb.maxZ),
    };
    f32 cosTheta = cos(transform.rotateAngleY);
    f32 sinTheta = sin(transform.rotateAngleY);
    for (u32 i = 0; i < ArrayLength(vecs); i++)
    {
        vec3 &vec = vecs[i];
        vec.x     = cosTheta * vec.x + sinTheta * vec.z;
        vec.z     = -sinTheta * vec.x + cosTheta * vec.z;
        vec += transform.translation;

        result.minX = result.minX < vec.x ? result.minX : vec.x;
        result.minY = result.minY < vec.y ? result.minY : vec.y;
        result.minZ = result.minZ < vec.z ? result.minZ : vec.z;

        result.maxX = result.maxX > vec.x ? result.maxX : vec.x;
        result.maxY = result.maxY > vec.y ? result.maxY : vec.y;
        result.maxZ = result.maxZ > vec.z ? result.maxZ : vec.z;
    }
    return result;
}

//////////////////////////////
// Basis
//
Basis GenerateBasis(vec3 n)
{
    Basis result;
    n        = Normalize(n);
    vec3 up  = fabs(n.x) > 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 t   = Normalize(Cross(n, up));
    vec3 b   = Cross(n, t);
    result.t = t;
    result.b = b;
    result.n = n;
    return result;
}

// TODO: I'm pretty sure this is converting to world space. not really sure about this
vec3 ConvertToLocal(Basis *basis, vec3 vec)
{
    // vec3 cosDirection     = RandomCosineDirection();
    vec3 result = basis->t * vec.x + basis->b * vec.y + basis->n * vec.z;
    return result;
}

//////////////////////////////
// Sphere
//
bool Sphere::Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
{
    // (C - P) Dot (C - P) = r^2
    // (C - (O + Dt)) Dot (C - (O + Dt)) - r^2 = 0
    // (-Dt + C - O) Dot (-Dt + C - O) - r^2 = 0
    // t^2(D Dot D) - 2t(D Dot (C - O)) + (C - O Dot C - O) - r^2 = 0
    vec3 oc = Center(r.time()) - r.origin();
    f32 a   = Dot(r.direction(), r.direction());
    f32 h   = Dot(r.direction(), oc);
    f32 c   = Dot(oc, oc) - radius * radius;

    f32 discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    f32 result = (h - sqrt(discriminant)) / a;
    if (result <= tMin || result >= tMax)
    {
        result = (h + sqrt(discriminant)) / a;
        if (result <= tMin || result >= tMax)
            return false;
    }

    record.t    = result;
    record.p    = r.at(record.t);
    vec3 normal = (record.p - center) / radius;
    record.SetNormal(r, normal);
    record.material = material;
    Sphere::GetUV(record.u, record.v, normal);

    return true;
}
vec3 Sphere::Center(f32 time) const
{
    return center + centerVec * time;
}
AABB Sphere::GetAABB() const
{
    vec3 boxRadius = vec3(radius, radius, radius);
    vec3 center2   = center + centerVec;
    AABB box1      = AABB(center - boxRadius, center + boxRadius);
    AABB box2      = AABB(center2 - boxRadius, center2 + boxRadius);
    AABB aabb      = AABB(box1, box2);
    return aabb;
}
f32 Sphere::PdfValue(const vec3 &origin, const vec3 &direction) const
{
    HitRecord rec;
    if (!this->Hit(Ray(origin, direction), 0.001f, infinity, rec))
        return 0;
    f32 cosThetaMax = sqrt(1 - radius * radius / (center - origin).lengthSquared());
    f32 solidAngle  = 2 * PI * (1 - cosThetaMax);
    return 1 / solidAngle;
}
vec3 Sphere::Random(const vec3 &origin, vec2 u) const
{
    vec3 dir            = center - origin;
    f32 distanceSquared = dir.lengthSquared();
    Basis basis         = GenerateBasis(dir);

    f32 r1 = u.x;
    f32 r2 = u.y;
    f32 z  = 1 + r2 * (sqrt(1 - radius * radius / distanceSquared) - 1);

    f32 phi     = 2 * PI * r1;
    f32 x       = cos(phi) * sqrt(1 - z * z);
    f32 y       = sin(phi) * sqrt(1 - z * z);
    vec3 result = ConvertToLocal(&basis, vec3(x, y, z));
    return result;
}

//////////////////////////////
// Scene
//
void Scene::FinalizePrimitives()
{
    totalPrimitiveCount = sphereCount + quadCount + boxCount;
    primitiveIndices    = (PrimitiveIndices *)malloc(sizeof(PrimitiveIndices) * totalPrimitiveCount);
    for (u32 i = 0; i < totalPrimitiveCount; i++)
    {
        primitiveIndices[i].transformIndex     = -1;
        primitiveIndices[i].constantMediaIndex = -1;
    }
}

inline i32 Scene::GetIndex(PrimitiveType type, i32 primIndex) const
{
    i32 index = -1;
    switch (type)
    {
        case PrimitiveType::Sphere:
        {
            index = primIndex;
        }
        break;
        case PrimitiveType::Quad:
        {
            index = primIndex + sphereCount;
        }
        break;
        case PrimitiveType::Box:
        {
            index = primIndex + sphereCount + quadCount;
        }
        break;
    }
    return index;
}
inline void Scene::GetTypeAndLocalindex(const u32 totalIndex, PrimitiveType *type, u32 *localIndex) const
{
    if (totalIndex < sphereCount)
    {
        *type       = PrimitiveType::Sphere;
        *localIndex = totalIndex;
    }
    else if (totalIndex < quadCount + sphereCount)
    {
        *type       = PrimitiveType::Quad;
        *localIndex = totalIndex - sphereCount;
    }
    else if (totalIndex < quadCount + sphereCount + boxCount)
    {
        *type       = PrimitiveType::Box;
        *localIndex = totalIndex - sphereCount - quadCount;
    }
    else
    {
        assert(0);
    }
}

void Scene::AddConstantMedium(PrimitiveType type, i32 primIndex, i32 constantMediumIndex)
{
    i32 index                                  = GetIndex(type, primIndex);
    primitiveIndices[index].constantMediaIndex = constantMediumIndex;
}

void Scene::AddTransform(PrimitiveType type, i32 primIndex, i32 transformIndex)
{
    i32 index                              = GetIndex(type, primIndex);
    primitiveIndices[index].transformIndex = transformIndex;
}
void Scene::GetAABBs(AABB *aabbs)
{
    for (u32 i = 0; i < sphereCount; i++)
    {
        Sphere &sphere = spheres[i];
        u32 index      = GetIndex(PrimitiveType::Sphere, i);
        aabbs[index]   = sphere.GetAABB();
    }
    for (u32 i = 0; i < quadCount; i++)
    {
        Quad &quad   = quads[i];
        u32 index    = GetIndex(PrimitiveType::Quad, i);
        aabbs[index] = quad.GetAABB();
    }
    for (u32 i = 0; i < boxCount; i++)
    {
        Box &box     = boxes[i];
        u32 index    = GetIndex(PrimitiveType::Box, i);
        aabbs[index] = box.GetAABB();
    }
    for (u32 i = 0; i < totalPrimitiveCount; i++)
    {
        if (primitiveIndices[i].transformIndex != -1)
        {
            aabbs[i] = Transform(transforms[primitiveIndices[i].transformIndex], aabbs[i]);
        }
    }
}

bool Scene::Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &temp, u32 index)
{
    bool result = false;

    Ray ray;
    HomogeneousTransform *transform = 0;
    f32 cosTheta;
    f32 sinTheta;

    if (primitiveIndices[index].transformIndex != -1)
    {
        transform             = &transforms[primitiveIndices[index].transformIndex];
        vec3 translatedOrigin = r.origin() - transform->translation;
        cosTheta              = cos(transform->rotateAngleY);
        sinTheta              = sin(transform->rotateAngleY);

        vec3 origin;
        origin.x = cosTheta * translatedOrigin.x - sinTheta * translatedOrigin.z;
        origin.y = translatedOrigin.y;
        origin.z = sinTheta * translatedOrigin.x + cosTheta * translatedOrigin.z;
        vec3 dir;
        dir.x = cosTheta * r.direction().x - sinTheta * r.direction().z;
        dir.y = r.direction().y;
        dir.z = sinTheta * r.direction().x + cosTheta * r.direction().z;
        // convert ray to object space
        ray = Ray(origin, dir, r.time());
    }
    else
    {
        ray = r;
    }

    PrimitiveType type;
    u32 localIndex;
    GetTypeAndLocalindex(index, &type, &localIndex);
    if (primitiveIndices[index].constantMediaIndex != -1)
    {
        ConstantMedium &medium = media[primitiveIndices[index].constantMediaIndex];
        switch (type)
        {
            case PrimitiveType::Sphere:
            {
                Sphere &sphere = spheres[localIndex];
                result         = medium.Hit(sphere, ray, tMin, tMax, temp);
            }
            break;
            case PrimitiveType::Quad:
            {
                Quad &quad = quads[localIndex];
                result     = medium.Hit(quad, ray, tMin, tMax, temp);
            }
            break;
            case PrimitiveType::Box:
            {
                Box &box = boxes[localIndex];
                result   = medium.Hit(box, ray, tMin, tMax, temp);
            }
            break;
        }
    }
    else
    {
        switch (type)
        {
            case PrimitiveType::Sphere:
            {
                Sphere &sphere = spheres[localIndex];
                result         = sphere.Hit(ray, tMin, tMax, temp);
            }
            break;
            case PrimitiveType::Quad:
            {
                Quad &quad = quads[localIndex];
                result     = quad.Hit(ray, tMin, tMax, temp);
            }
            break;
            case PrimitiveType::Box:
            {
                Box &box = boxes[localIndex];
                result   = box.Hit(ray, tMin, tMax, temp);
            }
            break;
        }
    }

    if (result && primitiveIndices[index].transformIndex != -1)
    {
        assert(transform);
        vec3 p;
        p.x = cosTheta * temp.p.x + sinTheta * temp.p.z;
        p.y = temp.p.y;
        p.z = -sinTheta * temp.p.x + cosTheta * temp.p.z;
        p += transform->translation;
        temp.p = p;

        vec3 normal;
        normal.x    = cosTheta * temp.normal.x + sinTheta * temp.normal.z;
        normal.y    = temp.normal.y;
        normal.z    = -sinTheta * temp.normal.x + cosTheta * temp.normal.z;
        temp.normal = normal;
    }
    return result;
}

struct TriangleMesh
{
    vec3 *p;
    vec3 *n;
    vec2 *uv;
    u32 *indices;
    u32 numVertices;
    u32 numIndices;
};

inline u32 GetTypeStride(string word)
{
    if (word == "uint8" || word == "char" || word == "uchar") return 1;
    else if (word == "short" || word == "ushort") return 2;
    else if (word == "int" || word == "uint" || word == "float") return 4;
    else if (word == "double") return 8;
    Error(0, "Invalid type: %s\n", (char *)word.str);
    return 0;
}

TriangleMesh LoadPLY(Arena *arena, string filename)
{
    string buffer = OS_ReadFile(arena, filename);
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

    bool vertexProperty = 0;

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
                    Assert(ReadWord(&tokenizer) == "float");
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
    if (hasVertices) mesh.p = PushArray(arena, vec3, numVertices);
    if (hasNormals) mesh.n = PushArray(arena, vec3, numVertices);
    if (hasUv) mesh.uv = PushArray(arena, vec2, numVertices);
    mesh.indices = PushArray(arena, u32, numFaces * 3);

    for (u32 i = 0; i < numVertices; i++)
    {
        string bytes = ReadBytes(&tokenizer, totalVertexStride);
        if (hasVertices)
        {
            Assert(totalVertexStride >= 12);
            f32 *pos  = (f32 *)bytes.str;
            mesh.p[i] = vec3(pos[0], pos[1], pos[2]);
        }
        if (hasNormals)
        {
            Assert(totalVertexStride >= 24);
            f32 *norm = (f32 *)bytes.str + 3;
            mesh.n[i] = vec3(norm[0], norm[1], norm[2]);
        }
        if (hasUv)
        {
            Assert(totalVertexStride >= 32);
            f32 *uv    = (f32 *)bytes.str + 6;
            mesh.uv[i] = vec2(uv[0], uv[1]);
        }
    }

    Assert(countStride == 1);
    Assert(faceIndicesStride == 4);
    Assert(otherStride == 4);
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
    return mesh;
}

enum class SceneByteType
{
    Float,
    Int,
    Vec3,
    Point3 = Vec3,
};

template <typename T, i32 numNodes = 1024, i32 chunkSize = 8>
struct Map
{
    StaticAssert(IsPow2(numNodes), CachePow2N);
    struct ChunkNode
    {
        T values[chunkSize];
        u32 count;
        ChunkNode *next;
    };
    ChunkNode *nodes;

    Map(Arena *arena)
    {
        nodes = PushArray(arena, ChunkNode, numNodes);
    }
    const T *GetOrCreate(Arena *arena, T value);

    // const string *GetOrCreate(Arena *arena, char *fmt, ...);

    T *Add(Arena *arena, string key);
};

template <typename T, i32 numNodes, i32 chunkSize>
const T *Map<T, numNodes, chunkSize>::GetOrCreate(Arena *arena, T value)
{
    u32 hash        = (u32)Hash<T>(value);
    ChunkNode *node = &nodes[hash & (numNodes - 1)];
    ChunkNode *prev = 0;
    while (node)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (value == node->values[i]) return &node->values[i];
        }
        prev = node;
        node = node->next;
    }

    if (prev->count == ArrayLength(prev->values))
    {
        node       = PushStruct(arena, ChunkNode);
        prev->next = node;
        prev       = node;
    }
    prev->values[prev->count++] = value; // Copy<T>(arena, value);
    return &prev->values[prev->count - 1];
};

// NOTE: allows for nodes with the same key to be used
template <typename T, i32 numNodes, i32 chunkSize>
T *Map<T, numNodes, chunkSize>::Add(Arena *arena, string key)
{
    u32 hash        = (u32)Hash<string>(key);
    ChunkNode *node = &nodes[hash & (numNodes - 1)];
    if (node->count == chunkSize)
    {
        ChunkNode *newNode = PushStruct(arena, ChunkNode);
        node->next         = newNode;
        node               = newNode;
    }
    return &node->values[node->count++];
}

template <i32 numNodes, i32 chunkSize>
struct StringCache
{
    StaticAssert(IsPow2(numNodes), CachePow2N);
    struct ChunkNode
    {
        string values[chunkSize];
        u32 count;
        ChunkNode *next;
    };
    ChunkNode *nodes;

    StringCache(Arena *arena)
    {
        nodes = PushArray(arena, ChunkNode, numNodes);
    }
    const string *GetOrCreate(Arena *arena, string value);
    const string *GetOrCreate(Arena *arena, char *fmt, ...);
    string *Add(Arena *arena, string key);
};

template <i32 numNodes, i32 chunkSize>
const string *StringCache<numNodes, chunkSize>::GetOrCreate(Arena *arena, string value)
{
    u32 hash        = (u32)Hash<string>(value);
    ChunkNode *node = &nodes[hash & (numNodes - 1)];
    ChunkNode *prev = 0;
    while (node)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (value == node->values[i]) return &node->values[i];
        }
        prev = node;
        node = node->next;
    }

    if (prev->count == ArrayLength(prev->values))
    {
        node       = PushStruct(arena, ChunkNode);
        prev->next = node;
        prev       = node;
    }
    prev->values[prev->count++] = PushStr8Copy(arena, value);
    return &prev->values[prev->count - 1];
}

template <i32 numNodes, i32 chunkSize>
string *StringCache<numNodes, chunkSize>::Add(Arena *arena, string key)
{
    u32 hash        = (u32)Hash<string>(key);
    ChunkNode *node = &nodes[hash & (numNodes - 1)];
    if (node->count == chunkSize)
    {
        ChunkNode *newNode = PushStruct(arena, ChunkNode);
        node->next         = newNode;
        node               = newNode;
    }
    return &node->values[node->count++];
}

template <i32 numNodes, i32 chunkSize>
const string *StringCache<numNodes, chunkSize>::GetOrCreate(Arena *arena, char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    TempArena temp       = ScratchStart(0, 0);
    string str           = PushStr8FV(temp.arena, fmt, args);
    const string *result = GetOrCreate(arena, str);
    va_end(args);
    ScratchEnd(temp);
    return result;
}

// I'm thinking an array of names (for parameters), an array of arrays of bytes (representing data for a parameter),
// an array of sizes of each byte array
struct ScenePacket
{
    const string *name;

    const string **parameterNames;
    u8 **bytes;
    u32 *sizes;
    // SceneByteType *types;
    u32 parameterCount;

    void Initialize(Arena *arena, u32 count)
    {
        // parameterCount = count;
        parameterNames = PushArray(arena, const string *, count);
        bytes          = PushArray(arena, u8 *, count);
        sizes          = PushArray(arena, u32, count);
        // types          = PushArray(arena, SceneByteType, count);
    }
};

struct ScenePacketChunkNode
{
    ScenePacket packet[256];
    u32 count;
    ScenePacketChunkNode *next;
};

struct ScenePacketMap
{
    ScenePacketChunkNode *nodes;
    u32 numNodes;
};

void ReadMat3(ScenePacket *packet, Tokenizer *tokenizer)
{
}

void ReadFloat(ScenePacket *packet, Tokenizer *tokenizer)
{
}

void LoadPBRT(Arena *arena, string filename)
{
    TempArena temp = ScratchStart(0, 0);

    Tokenizer tokenizer;
    tokenizer.input  = OS_ReadFile(temp.arena, filename);
    tokenizer.cursor = tokenizer.input.str;

    Map<ScenePacket, 1024, 256> scenePacketCache(arena);

    StringCache<1024, 8> stringCache(arena);

    for (;;)
    {
        if (EndOfBuffer(&tokenizer)) break;
        string word = CheckWord(&tokenizer);
        // Comments/Blank lines
        if (word.size == 0 || word.str[0] == '#')
        {
            SkipToNextLine(&tokenizer);
            continue;
        }
        ScenePacket *packet = scenePacketCache.Add(arena, word);

        // if (word == "Film")
        // {
        // }
        if (word == "LookAt")
        {
            ReadWord(&tokenizer);
            packet->Initialize(arena, 1);
            f32 r0c0 = ReadFloat(&tokenizer);
            f32 r0c1 = ReadFloat(&tokenizer);
            f32 r0c2 = ReadFloat(&tokenizer);
            SkipToNextDigit(&tokenizer);
            f32 r1c0 = ReadFloat(&tokenizer);
            f32 r1c1 = ReadFloat(&tokenizer);
            f32 r1c2 = ReadFloat(&tokenizer);
            SkipToNextDigit(&tokenizer);
            f32 r2c0 = ReadFloat(&tokenizer);
            f32 r2c1 = ReadFloat(&tokenizer);
            f32 r2c2 = ReadFloat(&tokenizer);

            f32 *bytes       = PushArray(arena, f32, 9);
            bytes[0]         = r0c0;
            bytes[1]         = r0c1;
            bytes[2]         = r0c2;
            bytes[3]         = r1c0;
            bytes[4]         = r1c1;
            bytes[5]         = r1c2;
            bytes[6]         = r2c0;
            bytes[7]         = r2c1;
            bytes[8]         = r2c2;
            packet->name     = stringCache.GetOrCreate(arena, word);
            packet->bytes[0] = (u8 *)bytes;
            packet->sizes[0] = sizeof(f32) * 9;
        }
        else if (word == "Camera")
        {
            ReadWord(&tokenizer);
            string cameraType;
            GetBetweenPair(cameraType, &tokenizer, '"');

            if (cameraType == "perspective")
            {
                packet->name = stringCache.GetOrCreate(arena, "%S_camera", cameraType);
            }
            else
            {
                Error(0, "Camera type not supported");
            }
            SkipToNextLine(&tokenizer);

            // Assert(GetBetweenPair(cameraType, option, '"'));
            u32 numParameters = CountLinesStartWith(&tokenizer, '"');
            packet->Initialize(arena, numParameters);

            string infoType;
            while (GetBetweenPair(infoType, &tokenizer, '"'))
            {
                string dataType = GetFirstWord(infoType);
                if (dataType == "float")
                {
                    u32 currentParam     = packet->parameterCount++;
                    string parameterName = GetNthWord(infoType, 2);

                    SkipToNextWord(&tokenizer);

                    u32 numFloats = CountBetweenPair(&tokenizer, '[');

                    f32 *floats = PushArray(arena, f32, numFloats);

                    Assert(Advance(&tokenizer, "["));
                    SkipToNextDigit(&tokenizer);
                    for (u32 i = 0; i < numFloats; i++)
                    {
                        floats[i] = ReadFloat(&tokenizer);
                    }
                    packet->parameterNames[currentParam] = stringCache.GetOrCreate(arena, parameterName);
                    packet->bytes[currentParam]          = (u8 *)floats;
                    packet->sizes[currentParam]          = sizeof(f32) * numFloats;

                    SkipToNextLine(&tokenizer);
                }
                else
                {
                    Assert(0);
                }
            }
        }
        else
        {
            SkipToNextLine(&tokenizer);
        }
    }
}
