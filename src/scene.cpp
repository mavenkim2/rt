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

template <typename T, i32 numSlots = 1024, i32 chunkSize = 8>
struct Map
{
    StaticAssert(IsPow2(numSlots), CachePow2N);
    struct ChunkNode
    {
        T values[chunkSize];
        u32 count;
        ChunkNode *next;
    };
    struct Slot
    {
        ChunkNode *first;
        ChunkNode *last;
    };
    Slot *slots;

    Map() {}
    Map(Arena *arena)
    {
        slots = PushArray(arena, Slot, numSlots);
    }
    const T *GetOrCreate(Arena *arena, T value);
    T *Add(Arena *arena, string key);
    const T *Get(string key) const;
};

template <typename T, i32 numNodes, i32 chunkSize>
const T *Map<T, numNodes, chunkSize>::GetOrCreate(Arena *arena, T value)
{
    u32 hash        = (u32)Hash<T>(value);
    Slot *slot      = &slots[hash & (numNodes - 1)];
    ChunkNode *prev = 0;
    for (ChunkNode *node = slot->first, prev = 0; node != 0; prev = node, node = node->next)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (node->values[i] == value) return &node->values[i];
        }
    }

    if (prev->count == ArrayLength(prev->values))
    {
        node = PushStruct(arena, ChunkNode);
        QueuePush(slot->first, slot->last, node);
        prev = node;
    }
    prev->values[prev->count++] = value; // Copy<T>(arena, value);
    return &prev->values[prev->count - 1];
};

// NOTE: allows for nodes with the same key to be used
template <typename T, i32 numNodes, i32 chunkSize>
T *Map<T, numNodes, chunkSize>::Add(Arena *arena, string key)
{
    u32 hash   = (u32)Hash<string>(key);
    Slot *slot = &slots[hash & (numNodes - 1)];
    if (!slot->last || slot->last->count == chunkSize)
    {
        ChunkNode *newNode = PushStruct(arena, ChunkNode);
        QueuePush(slot->first, slot->last, newNode);
    }
    return &slot->last->values[slot->last->count++];
}

// TODO: what's even the point of having this templated
template <typename T, i32 numNodes, i32 chunkSize>
const T *Map<T, numNodes, chunkSize>::Get(string key) const
{
    u32 hash   = (u32)Hash<string>(key);
    Slot *slot = &slots[hash & (numNodes - 1)];
    for (ChunkNode *node = slot->first; node != 0; node = node->next)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (key == *node->values[i].name) return &node->values[i];
        }
    }

    return 0;
};

template <i32 numNodes, i32 chunkSize, i32 numStripes>
struct StringCache
{
    StaticAssert(IsPow2(numNodes), CachePow2N);
    StaticAssert(IsPow2(numStripes), CachePow2Stripes);
    struct ChunkNode
    {
        string values[chunkSize];
        u32 count;
        ChunkNode *next;
    };
    ChunkNode *nodes;
    Mutex *mutexes;

    StringCache(Arena *arena)
    {
        nodes   = PushArray(arena, ChunkNode, numNodes);
        mutexes = PushArray(arena, Mutex, numStripes);
    }
    const string *GetOrCreate(Arena *arena, string value);
    const string *GetOrCreate(Arena *arena, char *fmt, ...);
};

template <i32 numNodes, i32 chunkSize, i32 numStripes>
const string *StringCache<numNodes, chunkSize, numStripes>::GetOrCreate(Arena *arena, string value)
{
    u32 hash        = (u32)Hash<string>(value);
    ChunkNode *node = &nodes[hash & (numNodes - 1)];
    ChunkNode *prev = 0;

    u32 stripe = hash & (numStripes - 1);
    BeginRMutex(&mutexes[stripe]);
    while (node)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (value == node->values[i])
            {
                EndRMutex(&mutexes[stripe]);
                return &node->values[i];
            }
        }
        prev = node;
        node = node->next;
    }
    EndRMutex(&mutexes[stripe]);

    // TODO: there's a very rare case where the thread gets unscheduled here, another thread writes the same
    // string value to the cache. and this thread writes that value again. this doesn't impact correctness
    // but some values will be duplicated, wasting some memory

    string *out = 0;
    BeginWMutex(&mutexes[stripe]);
    if (prev->count == ArrayLength(prev->values))
    {
        node       = PushStruct(arena, ChunkNode);
        prev->next = node;
        prev       = node;
    }
    prev->values[prev->count++] = PushStr8Copy(arena, value);
    out                         = &prev->values[prev->count - 1];
    EndWMutex(&mutexes[stripe]);

    return out;
}

template <i32 numNodes, i32 chunkSize, i32 numStripes>
const string *StringCache<numNodes, chunkSize, numStripes>::GetOrCreate(Arena *arena, char *fmt, ...)
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

// NOTE: sets the camera, film, sampler, etc.
template <i32 numNodes, i32 chunkSize, i32 numStripes>
void CreateScenePacket(Arena *arena,
                       string word,
                       ScenePacket *packet,
                       Tokenizer *tokenizer,
                       StringCache<numNodes, chunkSize, numStripes> *stringCache,
                       u32 additionalParameters = 0)
{
    ReadWord(tokenizer);
    string type;
    Assert(GetBetweenPair(type, tokenizer, '"'));
    packet->type = stringCache->GetOrCreate(arena, type);
    if (IsEndOfLine(tokenizer))
    {
        SkipToNextLine(tokenizer);
    }
    else
    {
        SkipToNextChar(tokenizer);
    }

    ReadParameters(arena, packet, tokenizer, stringCache, additionalParameters);
}

inline void SkipToNextDigitArray(Tokenizer *tokenizer)
{
    while (!EndOfBuffer(tokenizer) && (!IsDigit(tokenizer) && *tokenizer->cursor != '-' &&
                                       *tokenizer->cursor != ']')) tokenizer->cursor++;
}

template <i32 numNodes, i32 chunkSize, i32 numStripes>
void ReadParameters(Arena *arena, ScenePacket *packet, Tokenizer *tokenizer,
                    StringCache<numNodes, chunkSize, numStripes> *stringCache, u32 additionalParameters = 0)
{
    u32 numParameters = CountLinesStartWith(tokenizer, '"') + additionalParameters;
    Assert(numParameters);
    packet->Initialize(arena, numParameters);

    string infoType;
    b8 result;
    for (;;)
    {
        result = GetBetweenPair(infoType, tokenizer, '"');
        if (result == 0) break;
        else if (result == 2)
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
            f32 *floats = PushArray(arena, f32, numValues);

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
        }
        else if (dataType == "point2" || dataType == "vector2")
        {
            Assert((numValues & 1) == 0);
            vec2 *vectors = PushArray(arena, vec2, numValues / 2);

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
        }
        else if (dataType == "rgb" || dataType == "point3" || dataType == "vector3" || dataType == "normal3" ||
                 dataType == "normal" || dataType == "vector")
        {
            Assert(numValues % 3 == 0);
            vec3 *vectors = PushArray(arena, vec3, numValues / 3);

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
        }
        else if (dataType == "integer")
        {
            i32 *ints    = PushArray(arena, i32, numValues);
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
        }
        else if (dataType == "bool")
        {
            out  = PushStruct(arena, u8);
            size = sizeof(u8);
            Advance(tokenizer, "[");
            SkipToNextChar(tokenizer);
            // NOTE: this assumes that the bool is true or false (and not garbage and not capitalized)
            if (*tokenizer->cursor == 'f')
            {
                *out = 0;
            }
            else
            {
                *out = 1;
            }
        }
        else if (dataType == "string" || dataType == "texture")
        {
            Assert(numValues == 1);
            Advance(tokenizer, "[");
            SkipToNextChar(tokenizer);

            string str;
            Assert(GetBetweenPair(str, tokenizer, '"'));

            out  = str.str;
            size = (u32)str.size;
        }
        else if (dataType == "blackbody")
        {
            Assert(numValues == 1);
            SkipToNextDigit(tokenizer);
            i32 val = ReadInt(tokenizer);
            tokenizer->cursor++;

            i32 *ints = PushArray(arena, i32, 1);
            ints[0]   = val;
            out       = (u8 *)ints;
            size      = (u32)sizeof(i32);
        }

        // NOTE: either a series of wavelength value pairs or the name of a file with wavelength value pairs
        // TODO: handle cases where brackets are on new line
        else if (dataType == "spectrum")
        {
            if (numValues > 1)
            {
                string str;
                Assert(GetBetweenPair(str, tokenizer, '"'));

                out  = str.str;
                size = (u32)str.size;
            }
            else
            {
                Advance(tokenizer, "[");
                Assert((numValues & 1) == 0);
                out = PushArray(arena, u8, sizeof(f32) * numValues);
                for (u32 i = 0; i < numValues / 2; i++)
                {
                    *((i32 *)out + 2 * i)     = ReadInt(tokenizer);
                    *((f32 *)out + 2 * i + 1) = ReadFloat(tokenizer);
                }
                size = sizeof(f32) * numValues;
            }
        }
        else
        {
            Error(0, "Invalid data type: %S\n", dataType);
        }
        packet->parameterNames[currentParam] = stringCache->GetOrCreate(arena, parameterName);
        packet->bytes[currentParam]          = out;
        packet->sizes[currentParam]          = size;
        SkipToNextLine(tokenizer);
    }
}

// struct Instance
// {
//     i32 shapeIndices;
//     i32 transformIndex;
// };

struct MaterialIndex
{
    u32 threadIndex;
    i32 materialIndex;
};

struct ObjectIndex
{
    u32 threadIndex;
    i32 objectIndex;
};

struct ObjectInstanceType
{
    const string *name;
    i32 transformIndex;
    Array<i32> shapeIndices;
};

struct Instance
{
    const string *name;
    i32 transformIndex;
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

    Array<ScenePacket> *shapes;
    Array<ScenePacket> *materials;
    Array<ScenePacket> *textures;
    Array<ScenePacket> *lights;
    Array<ObjectInstanceType> *instanceTypes;
    Array<Instance> *instances;
    Array<mat4> *transforms;
    Arena **threadArenas;

    Arena *mainArena;
    Scene *scene;
    jobsystem::Counter counter = {};
};

struct SceneThreadState
{
    const string **materialNames;
    MaterialIndex *materialIndices;
    const string **objectNames;
    ObjectIndex *objectIndices;

    u32 materialCount;
    u32 objectCount;
    // Array<ObjectInstance> instances;
};

void LoadPBRT(string filename, string directory, SceneLoadState *state, bool inWorldBegin = false);

Scene *LoadPBRT(Arena *arena, string filename)
{
    TempArena temp = ScratchStart(0, 0);
    Scene *scene   = PushStruct(arena, Scene);
    SceneLoadState state;
    u32 numProcessors   = OS_NumProcessors();
    state.shapes        = PushArray(arena, Array<ScenePacket>, numProcessors);
    state.materials     = PushArray(arena, Array<ScenePacket>, numProcessors);
    state.textures      = PushArray(arena, Array<ScenePacket>, numProcessors);
    state.lights        = PushArray(arena, Array<ScenePacket>, numProcessors);
    state.instanceTypes = PushArray(arena, Array<ObjectInstanceType>, numProcessors);
    state.instances     = PushArray(arena, Array<Instance>, numProcessors);
    state.transforms    = PushArray(arena, Array<mat4>, numProcessors);
    state.threadArenas  = PushArray(arena, Arena *, numProcessors);

    for (u32 i = 0; i < numProcessors; i++)
    {
        state.threadArenas[i]  = ArenaAlloc();
        state.shapes[i]        = Array<ScenePacket>(state.threadArenas[i], 32);
        state.materials[i]     = Array<ScenePacket>(state.threadArenas[i], 32);
        state.textures[i]      = Array<ScenePacket>(state.threadArenas[i], 32);
        state.lights[i]        = Array<ScenePacket>(state.threadArenas[i], 32);
        state.instanceTypes[i] = Array<ObjectInstanceType>(state.threadArenas[i], 32);
        state.instances[i]     = Array<Instance>(state.threadArenas[i], 32);
        state.transforms[i]    = Array<mat4>(state.threadArenas[i], 32);
    }
    state.mainArena = arena;
    state.scene     = scene;

    LoadPBRT(filename, Str8PathChopPastLastSlash(filename), &state);

    // TODO: combine the arrays
    jobsystem::WaitJobs(&state.counter);

    for (u32 i = 0; i < ArrayLength(state.threadArenas); i++)
    {
        ArenaClear(state.threadArenas[i]);
    }
    ScratchEnd(temp);
    return scene;
}

struct GraphicsState
{
    const string *materialName;
    i32 materialIndex;
    mat4 transform;
    i32 transformIndex;
};

// template <i32 numSlots, i32 chunkSize>
void LoadPBRT(string filename, string directory, SceneLoadState *state, bool inWorldBegin)
{
    TempArena temp  = ScratchStart(0, 0);
    u32 threadIndex = GetThreadIndex();
    Arena *arena    = state->threadArenas[threadIndex];

    // if (filename == "data/island/pbrt-v4/isBeach/isBeach.pbrt")
    // {
    //     int stop = 5;
    // }

    Tokenizer tokenizer;
    tokenizer.input  = OS_ReadFile(temp.arena, filename);
    tokenizer.cursor = tokenizer.input.str;

    StringCache<1024, 8, 64> stringCache(arena);
    Array<ScenePacket> &shapes               = state->shapes[threadIndex];
    Array<ScenePacket> &materials            = state->materials[threadIndex];
    Array<ScenePacket> &textures             = state->textures[threadIndex];
    Array<ScenePacket> &lights               = state->lights[threadIndex];
    Array<ObjectInstanceType> &instanceTypes = state->instanceTypes[threadIndex];
    Array<Instance> &instances               = state->instances[threadIndex];

    const string *currentInstanceTypeName = 0;
    Array<mat4> &transforms               = state->transforms[threadIndex];

    bool worldBegin = inWorldBegin;

    // Stack variables
    Tokenizer oldTokenizers[32];
    u32 numTokenizers = 0;

    string currentFilename = filename;

    GraphicsState graphicsStateStack[64];
    u32 graphicsStateCount = 0;

    const string *currentMaterialName = 0;
    ObjectInstanceType *currentObject = 0;

    i32 currentMaterialIndex  = -1;
    i32 currentAreaLightIndex = -1;
    // TODO: media
    i32 currentMediaIndex = -1;

    mat4 currentTransform     = mat4::Identity();
    i32 currentTransformIndex = 0;

    for (;;)
    {
        // if (currentFilename == "data/island/pbrt-v4/isBeach/xgHibiscus/xgHibiscus_archiveHibiscusFlower0009_mod_geometry.pbrt")
        // {
        //     int stop = 5;
        // }
        if (EndOfBuffer(&tokenizer))
        {
            if (numTokenizers == 0) break;
            tokenizer = oldTokenizers[--numTokenizers];
            continue;
        }
        if (graphicsStateCount != 0) SkipToNextChar(&tokenizer);

        string word = CheckWord(&tokenizer);
        // Comments/Blank lines
        if (word.size == 0 || word.str[0] == '#')
        {
            SkipToNextLine(&tokenizer);
            continue;
        }

        if (word == "Film" || word == "Camera" || word == "Sampler" || word == "Integrator" || word == "Accelerator")
        {
            if (!worldBegin)
            {
                SceneLoadState::Type type;
                if (word == "Film")
                {
                    type = SceneLoadState::Type::Film;
                }
                else if (word == "Camera")
                {
                    type = SceneLoadState::Type::Camera;
                }
                else if (word == "Sampler")
                {
                    type = SceneLoadState::Type::Sampler;
                }
                else if (word == "Integrator")
                {
                    type = SceneLoadState::Type::Integrator;
                }
                else if (word == "Accelerator")
                {
                    type = SceneLoadState::Type::Accelerator;
                }
                ScenePacket *packet = &state->packets[type];
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache);
                continue;
            }
            else
            {
                Error(0, "Tried to specify %S after WorldBegin\n", word);
            }
        }
        else if (word == "WorldBegin")
        {
            ReadWord(&tokenizer);
            // NOTE: this assumes "WorldBegin" only occurs in one file
            worldBegin = true;

            const ScenePacket *filmPacket = &state->packets[SceneLoadState::Type::Film];
            vec2i fullResolution;
            for (u32 i = 0; i < filmPacket->parameterCount; i++)
            {
                if (*filmPacket->parameterNames[i] == "xresolution")
                {
                    fullResolution.x = filmPacket->GetInt(i);
                }
                else if (*filmPacket->parameterNames[i] == "yresolution")
                {
                    fullResolution.y = filmPacket->GetInt(i);
                }
            }

            const ScenePacket *samplerPacket = &state->packets[SceneLoadState::Type::Sampler];
            state->scene->sampler            = Sampler::Create(state->mainArena, samplerPacket, fullResolution);

            if (currentTransform != mat4::Identity())
            {
                transforms.Push(std::move(currentTransform));
            }
            // TODO: instantiate the camera with the current transform

            currentTransform = mat4::Identity();
        }
        else if (word == "Identity")
        {
            ReadWord(&tokenizer);
            currentTransform = mat4::Identity();
        }
        else if (word == "Translate")
        {
            currentTransformIndex = transforms.Length();
            ReadWord(&tokenizer);
            f32 t0 = ReadFloat(&tokenizer);
            f32 t1 = ReadFloat(&tokenizer);
            f32 t2 = ReadFloat(&tokenizer);
            mat4 translationMatrix(0.f, 0.f, 0.f, 0.f,
                                   0.f, 0.f, 0.f, 0.f,
                                   0.f, 0.f, 0.f, 0.f,
                                   t0, t1, t2, 1.f);
            currentTransform = mul(currentTransform, translationMatrix);
        }
        else if (word == "Rotate")
        {
            currentTransformIndex = transforms.Length();
            ReadWord(&tokenizer);
            f32 angle           = ReadFloat(&tokenizer);
            f32 axisX           = ReadFloat(&tokenizer);
            f32 axisY           = ReadFloat(&tokenizer);
            f32 axisZ           = ReadFloat(&tokenizer);
            mat4 rotationMatrix = mat4::Rotate(vec3(axisX, axisY, axisZ), angle);
            currentTransform    = mul(currentTransform, rotationMatrix);
        }
        else if (word == "Scale")
        {
            currentTransformIndex = transforms.Length();
            ReadWord(&tokenizer);
            f32 s0 = ReadFloat(&tokenizer);
            f32 s1 = ReadFloat(&tokenizer);
            f32 s2 = ReadFloat(&tokenizer);
            mat4 scaleMatrix(s0, 0.f, 0.f, 0.f,
                             0.f, s1, 0.f, 0.f,
                             0.f, 0.f, s2, 0.f,
                             0.f, 0.f, 0.f, 1.f);
            currentTransform = mul(currentTransform, scaleMatrix);
        }
        else if (word == "LookAt")
        {
            currentTransformIndex = transforms.Length();
            ReadWord(&tokenizer);
            f32 posX = ReadFloat(&tokenizer);
            f32 posY = ReadFloat(&tokenizer);
            f32 posZ = ReadFloat(&tokenizer);
            SkipToNextDigit(&tokenizer);
            f32 lookX = ReadFloat(&tokenizer);
            f32 lookY = ReadFloat(&tokenizer);
            f32 lookZ = ReadFloat(&tokenizer);
            SkipToNextDigit(&tokenizer);
            f32 upX = ReadFloat(&tokenizer);
            f32 upY = ReadFloat(&tokenizer);
            f32 upZ = ReadFloat(&tokenizer);

            currentTransform = LookAt(vec3(posX, posY, posZ), vec3(lookX, lookY, lookZ), Normalize(vec3(upX, upY, upZ)));
        }
        else if (word == "Transform")
        {
            // if (filename == "data/island/pbrt-v4/isLavaRocks/isLavaRocks.pbrt")
            // {
            //     int stop = 5;
            // }
            currentTransformIndex = transforms.Length();
            ReadWord(&tokenizer);
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
            currentTransform = mat4(r0c0, r0c1, r0c2, r0c3,
                                    r1c0, r1c1, r1c2, r1c3,
                                    r2c0, r2c1, r2c2, r2c3,
                                    r3c0, r3c1, r3c2, r3c3);

            SkipToNextLine(&tokenizer);
        }
        else if (word == "ConcatTransform")
        {
            currentTransformIndex = transforms.Length();
            ReadWord(&tokenizer);
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
            currentTransform = mul(currentTransform, mat4(r0c0, r0c1, r0c2, r0c3,
                                                          r1c0, r1c1, r1c2, r1c3,
                                                          r2c0, r2c1, r2c2, r2c3,
                                                          r3c0, r3c1, r3c2, r3c3));
            SkipToNextLine(&tokenizer);
        }
        else if (worldBegin)
        {
            if (word == "AttributeBegin")
            {
                Assert(graphicsStateCount < ArrayLength(graphicsStateStack));
                GraphicsState *graphicsState  = &graphicsStateStack[graphicsStateCount++];
                graphicsState->transform      = currentTransform;
                graphicsState->materialIndex  = currentMaterialIndex;
                graphicsState->materialName   = currentMaterialName;
                graphicsState->transformIndex = currentTransformIndex;

                currentMaterialIndex  = -1;
                currentTransform      = mat4::Identity();
                currentTransformIndex = 0;
                SkipToNextLine(&tokenizer);
            }
            else if (word == "AttributeEnd")
            {
                Assert(graphicsStateCount > 0);

                // Add transform to cache
                transforms.Push(std::move(currentTransform));

                // Pop stack
                GraphicsState *graphicsState = &graphicsStateStack[--graphicsStateCount];
                currentMaterialIndex         = graphicsState->materialIndex;
                currentMaterialName          = graphicsState->materialName;
                currentTransform             = graphicsState->transform;
                currentTransformIndex        = graphicsState->transformIndex;

                currentAreaLightIndex = -1;

                SkipToNextLine(&tokenizer);
            }
            else if (word == "AreaLightSource")
            {
                currentAreaLightIndex = lights.Length();
                ScenePacket *packet   = &lights.AddBack();
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache);
            }
            else if (word == "LightSource")
            {
                // ReadWord(&tokenizer);
                ScenePacket *packet = &lights.AddBack();
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache);
            }
            else if (word == "ObjectBegin")
            {
                Error(currentObject == 0, "ObjectBegin cannot be called recursively.");
                ReadWord(&tokenizer);
                string objectName;

                Assert(GetBetweenPair(objectName, &tokenizer, '"'));

                currentObject                 = &instanceTypes.AddBack();
                currentObject->name           = stringCache.GetOrCreate(arena, objectName);
                currentObject->transformIndex = currentTransformIndex;
                currentObject->shapeIndices   = Array<i32>(arena);
            }
            else if (word == "ObjectEnd")
            {
                ReadWord(&tokenizer);
                Error(currentObject != 0, "ObjectEnd must occur after ObjectBegin");
                currentObject = 0;
            }
            else if (word == "ObjectInstance")
            {
                ReadWord(&tokenizer);
                string objectName;
                Assert(GetBetweenPair(objectName, &tokenizer, '"'));

                Instance &instance      = instances.AddBack();
                instance.name           = stringCache.GetOrCreate(arena, objectName);
                instance.transformIndex = (i32)transforms.Length();

                transforms.Push(currentTransform);
            }
            // TODO IMPORTANT: the indices are clockwise since PBRT uses a left-handed coordinate system. either need to
            // revert the winding or use a left handed system as well
            else if (word == "Shape")
            {
                ScenePacket *packet = &shapes.AddBack();
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, currentMaterialName ? 2 : 1);
                i32 *indices = PushArray(arena, i32, 4);
                // ORDER: Light, Medium, Transform, Material
                indices[0] = currentAreaLightIndex;
                indices[1] = currentMediaIndex;
                indices[2] = currentTransformIndex;
                indices[3] = currentMaterialIndex;

                if (currentObject)
                {
                    currentObject->shapeIndices.Push(shapes.Length() - 1);
                }

                u32 currentParameter                     = packet->parameterCount++;
                packet->parameterNames[currentParameter] = stringCache.GetOrCreate(arena, "Indices");
                packet->bytes[currentParameter]          = (u8 *)indices;
                packet->sizes[currentParameter]          = sizeof(i32) * 4;

                if (currentMaterialName)
                {
                    currentParameter                         = packet->parameterCount++;
                    packet->parameterNames[currentParameter] = stringCache.GetOrCreate(arena, "MaterialName");
                    packet->bytes[currentParameter]          = currentMaterialName->str;
                    packet->sizes[currentParameter]          = (u32)currentMaterialName->size;
                }
                transforms.Push(currentTransform);
            }
            else if (word == "NamedMaterial")
            {
                ReadWord(&tokenizer);
                string materialName;
                Assert(GetBetweenPair(materialName, &tokenizer, '"'));

                currentMaterialName  = stringCache.GetOrCreate(arena, materialName);
                currentMaterialIndex = -1;
            }
            else if (word == "Material" || word == "MakeNamedMaterial")
            {
                bool isNamedMaterial = (word == "MakeNamedMaterial");
                // scenePacketCache
                ReadWord(&tokenizer);
                string materialNameOrType;
                Assert(GetBetweenPair(materialNameOrType, &tokenizer, '"'));

                ScenePacket *packet = &materials.AddBack();
                packet->type        = stringCache.GetOrCreate(arena, materialNameOrType);
                u32 materialIndex   = materials.Length() - 1; //++materialIndex;
                if (IsEndOfLine(&tokenizer))
                {
                    SkipToNextLine(&tokenizer);
                }
                else
                {
                    SkipToNextChar(&tokenizer);
                }
                ReadParameters(arena, packet, &tokenizer, &stringCache);

                if (isNamedMaterial)
                {
                    currentMaterialName  = stringCache.GetOrCreate(arena, materialNameOrType);
                    currentMaterialIndex = -1;
                }
                else
                {
                    currentMaterialIndex = materialIndex;
                    currentMaterialName  = 0;
                }
            }
            else if (word == "Texture")
            {
                ReadWord(&tokenizer);
                string textureName;
                Assert(GetBetweenPair(textureName, &tokenizer, '"'));
                string textureType;
                Assert(GetBetweenPair(textureType, &tokenizer, '"'));
                string textureClass;
                Assert(GetBetweenPair(textureClass, &tokenizer, '"'));

                ScenePacket *packet = &textures.AddBack();
                packet->type        = stringCache.GetOrCreate(arena, StrConcat(arena, textureType, textureClass));

                if (IsEndOfLine(&tokenizer))
                {
                    SkipToNextLine(&tokenizer);
                }
                else
                {
                    SkipToNextChar(&tokenizer);
                }
                ReadParameters(arena, packet, &tokenizer, &stringCache);
            }
            else if (word == "Import")
            {
                ReadWord(&tokenizer);
                string importedFilename;
                Assert(GetBetweenPair(importedFilename, &tokenizer, '"'));
                string importedFullPath = StrConcat(arena, directory, importedFilename);

                jobsystem::KickJob(&state->counter, [&](jobsystem::JobArgs args) {
                    LoadPBRT(importedFullPath, directory, state, worldBegin);
                });
            }
            else if (word == "Include")
            {
                ReadWord(&tokenizer);
                string importedFilename;
                Assert(GetBetweenPair(importedFilename, &tokenizer, '"'));
                string importedFullPath = StrConcat(temp.arena, directory, importedFilename);
                Assert(numTokenizers < ArrayLength(oldTokenizers));
                oldTokenizers[numTokenizers++] = tokenizer;
                tokenizer.input                = OS_ReadFile(temp.arena, importedFullPath);
                tokenizer.cursor               = tokenizer.input.str;

                currentFilename = importedFullPath;
            }
            else if (word == "Attribute" || word == "MakeNamedMedium" || word == "MediumInterface" ||
                     word == "CoordinateSystem" || word == "CoordSysTransform")
            {
                // not implemented yet
                Assert(0);
            }
            else
            {
                string line = ReadLine(&tokenizer);
                Error(0, "Error while parsing scene. Buffer: %S", line);
            }
        }
        else
        {
            string line = ReadLine(&tokenizer);
            Error(0, "Error while parsing scene. Buffer: %S, WorldBegin: %b\n", line, worldBegin);
            // SkipToNextLine(&tokenizer);
        }
    }
    ScratchEnd(temp);
}
