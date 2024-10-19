namespace rt
{
AABB Transform(const HomogeneousTransform &transform, const AABB &aabb)
{
    AABB result;
    Vec3f vecs[] = {
        Vec3f(aabb.minX, aabb.minY, aabb.minZ),
        Vec3f(aabb.maxX, aabb.minY, aabb.minZ),
        Vec3f(aabb.maxX, aabb.maxY, aabb.minZ),
        Vec3f(aabb.minX, aabb.maxY, aabb.minZ),
        Vec3f(aabb.minX, aabb.minY, aabb.maxZ),
        Vec3f(aabb.maxX, aabb.minY, aabb.maxZ),
        Vec3f(aabb.maxX, aabb.maxY, aabb.maxZ),
        Vec3f(aabb.minX, aabb.maxY, aabb.maxZ),
    };
    f32 cosTheta = cos(transform.rotateAngleY);
    f32 sinTheta = sin(transform.rotateAngleY);
    for (u32 i = 0; i < ArrayLength(vecs); i++)
    {
        Vec3f &vec = vecs[i];
        vec.x      = cosTheta * vec.x + sinTheta * vec.z;
        vec.z      = -sinTheta * vec.x + cosTheta * vec.z;
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
Basis GenerateBasis(Vec3f n)
{
    Basis result;
    n        = Normalize(n);
    Vec3f up = fabs(n.x) > 0.9 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0);
    Vec3f t  = Normalize(Cross(n, up));
    Vec3f b  = Cross(n, t);
    result.t = t;
    result.b = b;
    result.n = n;
    return result;
}

// TODO: I'm pretty sure this is converting to world space. not really sure about this
Vec3f ConvertToLocal(Basis *basis, Vec3f vec)
{
    // Vec3f cosDirection     = RandomCosin.d;
    Vec3f result = basis->t * vec.x + basis->b * vec.y + basis->n * vec.z;
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
    Vec3f oc = Center(r.t) - r.o;
    f32 a    = Dot(r.d, r.d);
    f32 h    = Dot(r.d, oc);
    f32 c    = Dot(oc, oc) - radius * radius;

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

    record.t     = result;
    record.p     = r.at(record.t);
    Vec3f normal = (record.p - center) / radius;
    record.SetNormal(r, normal);
    record.material = material;
    Sphere::GetUV(record.u, record.v, normal);

    return true;
}
Vec3f Sphere::Center(f32 time) const
{
    return center + centerVec * time;
}
AABB Sphere::GetAABB() const
{
    Vec3f boxRadius = Vec3f(radius, radius, radius);
    Vec3f center2   = center + centerVec;
    AABB box1       = AABB(center - boxRadius, center + boxRadius);
    AABB box2       = AABB(center2 - boxRadius, center2 + boxRadius);
    AABB aabb       = AABB(box1, box2);
    return aabb;
}
f32 Sphere::PdfValue(const Vec3f &origin, const Vec3f &direction) const
{
    HitRecord rec;
    if (!this->Hit(Ray(origin, direction), 0.001f, infinity, rec))
        return 0;
    f32 cosThetaMax = Sqrt(1 - radius * radius / LengthSquared(center - origin));
    f32 solidAngle  = 2 * PI * (1 - cosThetaMax);
    return 1 / solidAngle;
}
Vec3f Sphere::Random(const Vec3f &origin, Vec2f u) const
{
    Vec3f dir           = center - origin;
    f32 distanceSquared = LengthSquared(dir);
    Basis basis         = GenerateBasis(dir);

    f32 r1 = u.x;
    f32 r2 = u.y;
    f32 z  = 1 + r2 * (sqrt(1 - radius * radius / distanceSquared) - 1);

    f32 phi      = 2 * PI * r1;
    f32 x        = Cos(phi) * Sqrt(1 - z * z);
    f32 y        = Sin(phi) * Sqrt(1 - z * z);
    Vec3f result = ConvertToLocal(&basis, Vec3f(x, y, z));
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
        case PrimitiveType_Sphere:
        {
            index = primIndex;
        }
        break;
        case PrimitiveType_Quad:
        {
            index = primIndex + sphereCount;
        }
        break;
        case PrimitiveType_Box:
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
        *type       = PrimitiveType_Sphere;
        *localIndex = totalIndex;
    }
    else if (totalIndex < quadCount + sphereCount)
    {
        *type       = PrimitiveType_Quad;
        *localIndex = totalIndex - sphereCount;
    }
    else if (totalIndex < quadCount + sphereCount + boxCount)
    {
        *type       = PrimitiveType_Box;
        *localIndex = totalIndex - sphereCount - quadCount;
    }
    else
    {
        Assert(0);
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
        u32 index      = GetIndex(PrimitiveType_Sphere, i);
        aabbs[index]   = sphere.GetAABB();
    }
    for (u32 i = 0; i < quadCount; i++)
    {
        Quad &quad   = quads[i];
        u32 index    = GetIndex(PrimitiveType_Quad, i);
        aabbs[index] = quad.GetAABB();
    }
    for (u32 i = 0; i < boxCount; i++)
    {
        Box &box     = boxes[i];
        u32 index    = GetIndex(PrimitiveType_Box, i);
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
        transform              = &transforms[primitiveIndices[index].transformIndex];
        Vec3f translatedOrigin = r.o - transform->translation;
        cosTheta               = cos(transform->rotateAngleY);
        sinTheta               = sin(transform->rotateAngleY);

        Vec3f origin;
        origin.x = cosTheta * translatedOrigin.x - sinTheta * translatedOrigin.z;
        origin.y = translatedOrigin.y;
        origin.z = sinTheta * translatedOrigin.x + cosTheta * translatedOrigin.z;
        Vec3f dir;
        dir.x = cosTheta * r.d.x - sinTheta * r.d.z;
        dir.y = r.d.y;
        dir.z = sinTheta * r.d.x + cosTheta * r.d.z;
        // convert ray to object space
        ray = Ray(origin, dir, r.t);
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
            case PrimitiveType_Sphere:
            {
                Sphere &sphere = spheres[localIndex];
                result         = medium.Hit(sphere, ray, tMin, tMax, temp);
            }
            break;
            case PrimitiveType_Quad:
            {
                Quad &quad = quads[localIndex];
                result     = medium.Hit(quad, ray, tMin, tMax, temp);
            }
            break;
            case PrimitiveType_Box:
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
            case PrimitiveType_Sphere:
            {
                Sphere &sphere = spheres[localIndex];
                result         = sphere.Hit(ray, tMin, tMax, temp);
            }
            break;
            case PrimitiveType_Quad:
            {
                Quad &quad = quads[localIndex];
                result     = quad.Hit(ray, tMin, tMax, temp);
            }
            break;
            case PrimitiveType_Box:
            {
                Box &box = boxes[localIndex];
                result   = box.Hit(ray, tMin, tMax, temp);
            }
            break;
        }
    }

    if (result && primitiveIndices[index].transformIndex != -1)
    {
        Assert(transform);
        Vec3f p;
        p.x = cosTheta * temp.p.x + sinTheta * temp.p.z;
        p.y = temp.p.y;
        p.z = -sinTheta * temp.p.x + cosTheta * temp.p.z;
        p += transform->translation;
        temp.p = p;

        Vec3f normal;
        normal.x    = cosTheta * temp.normal.x + sinTheta * temp.normal.z;
        normal.y    = temp.normal.y;
        normal.z    = -sinTheta * temp.normal.x + cosTheta * temp.normal.z;
        temp.normal = normal;
    }
    return result;
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

// NOTE: specifically for the moana island scene
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

    // 2 triangles/1 quad for every 4 vertices. If this condition isn't met, it isn't a quad mesh
    Assert(numFaces == numVertices / 2);
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

template <typename T, i32 numSlots, i32 chunkSize, i32 numStripes, i32 tag>
struct Map
{
    StaticAssert(IsPow2(numSlots), CachePow2N);
    struct ChunkNode
    {
        T values[chunkSize];
        u32 count;
        ChunkNode *next;
    };
    ChunkNode *nodes;
    Mutex *mutexes;

    Map() {}
    Map(Arena *arena)
    {
        nodes   = PushArrayTagged(arena, ChunkNode, numSlots, tag);
        mutexes = PushArrayTagged(arena, Mutex, numStripes, tag);
    }
    const T *GetOrCreate(Arena *arena, T value);
};

template <typename T, i32 numSlots, i32 chunkSize, i32 numStripes, i32 tag>
const T *Map<T, numSlots, chunkSize, numStripes, tag>::GetOrCreate(Arena *arena, T value)
{
    u64 hash        = Hash<T>(value);
    ChunkNode *node = &nodes[hash & (numSlots - 1)];
    ChunkNode *prev = 0;

    u32 stripe = hash & (numStripes - 1);
    BeginRMutex(&mutexes[stripe]);
    while (node)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (node->values[i] == value)
            {
                EndRMutex(&mutexes[stripe]);
                return &node->values[i];
            }
        }
        prev = node;
        node = node->next;
    }
    EndRMutex(&mutexes[stripe]);

    T *out = 0;
    BeginWMutex(&mutexes[stripe]);
    if (prev->count == ArrayLength(prev->values))
    {
        node       = PushStructTagged(arena, ChunkNode, tag);
        prev->next = node;
        prev       = node;
    }
    prev->values[prev->count] = value;
    out                       = &prev->values[prev->count++];
    EndWMutex(&mutexes[stripe]);
    return out;
}

template <i32 numNodes, i32 chunkSize, i32 numStripes>
struct InternedStringCache
{
    StaticAssert(IsPow2(numNodes), CachePow2N);
    StaticAssert(IsPow2(numStripes), CachePow2Stripes);
    struct ChunkNode
    {
        StringId stringIds[chunkSize];
#if DEBUG
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
#if DEBUG
    string Get(StringId id);
#endif
};

template <i32 numNodes, i32 chunkSize, i32 numStripes>
StringId InternedStringCache<numNodes, chunkSize, numStripes>::GetOrCreate(Arena *arena, string value)
{
    StringId sid    = Hash(value);
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
#if DEBUG
                Error(node->str[i] == value, "Hash collision between %S and %S\n", value, node->str[i]);
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
#if DEBUG
    prev->str[prev->count] = PushStr8Copy(arena, value);
#endif
    prev->count++;
    EndWMutex(&mutexes[stripe]);

    return sid;
}

#if DEBUG
template <i32 numNodes, i32 chunkSize, i32 numStripes>
string InternedStringCache<numNodes, chunkSize, numStripes>::Get(StringId id)
{
    ChunkNode *node = &nodes[id & (numNodes - 1)];
    ChunkNode *prev = 0;

    u32 stripe = id & (numStripes - 1);
    BeginRMutex(&mutexes[stripe]);
    while (node)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (id == node->stringIds[i])
            {
                EndRMutex(&mutexes[stripe]);
                return node->str[i];
            }
        }
        prev = node;
        node = node->next;
    }
    EndRMutex(&mutexes[stripe]);
    return {};
}
#endif

//////////////////////////////

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

    StringCache() {}
    StringCache(Arena *arena)
    {
        nodes   = PushArrayTagged(arena, ChunkNode, numNodes, MemoryType_String);
        mutexes = PushArray(arena, Mutex, numStripes);
    }
    const string *GetOrCreate(Arena *arena, string value);
};

template <i32 numNodes, i32 chunkSize, i32 numStripes>
const string *StringCache<numNodes, chunkSize, numStripes>::GetOrCreate(Arena *arena, string value)
{
    u32 hash        = Hash(value);
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
        node       = PushStructTagged(arena, ChunkNode, MemoryType_String);
        prev->next = node;
        prev       = node;
    }
    prev->values[prev->count++] = PushStr8Copy(arena, value);
    out                         = &prev->values[prev->count - 1];
    EndWMutex(&mutexes[stripe]);

    return out;
}

// NOTE: sets the camera, film, sampler, etc.
template <i32 numNodes, i32 chunkSize, i32 numStripes>
void CreateScenePacket(Arena *arena,
                       string word,
                       ScenePacket *packet,
                       Tokenizer *tokenizer,
                       InternedStringCache<numNodes, chunkSize, numStripes> *stringCache,
                       MemoryType memoryType,
                       u32 additionalParameters = 0)
{
    ReadWord(tokenizer);
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
    while (!EndOfBuffer(tokenizer) && (!IsDigit(tokenizer) && *tokenizer->cursor != '-' &&
                                       *tokenizer->cursor != ']')) tokenizer->cursor++;
}

template <i32 numNodes, i32 chunkSize, i32 numStripes>
void ReadParameters(Arena *arena, ScenePacket *packet, Tokenizer *tokenizer,
                    InternedStringCache<numNodes, chunkSize, numStripes> *stringCache,
                    MemoryType memoryType, u32 additionalParameters = 0)
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
        }
        else if (dataType == "rgb" || dataType == "point3" || dataType == "vector3" || dataType == "normal3" ||
                 dataType == "normal" || dataType == "vector")
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
        }
        else if (dataType == "bool")
        {
            out  = PushStructNoZeroTagged(arena, u8, memoryType);
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
            b32 pairResult = GetBetweenPair(str, tokenizer, '"');
            Assert(pairResult);

            out  = str.str;
            size = (u32)str.size;
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

        // NOTE: either a series of wavelength value pairs or the name of a file with wavelength value pairs
        // TODO: handle cases where brackets are on new line
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
    inline void Push(T &val)
    {
        AddBack() = val;
    }
    inline void Push(T &&val)
    {
        AddBack() = std::move(val);
    }
    inline void AddNode()
    {
        ChunkNode *newNode = PushStructTagged(arena, ChunkNode, memoryTag);
        QueuePush(first, last, newNode);
    }
    inline u32 Length() const
    {
        return totalCount;
    }
};

struct ObjectInstanceType
{
    StringId name;
    // string name;
    i32 transformIndex;
    Array<i32, MemoryType_Instance> shapeIndices;
};

struct SceneInstance
{
    StringId name;
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

    // TODO: these numbers are really ad hoc. honestly the best solution would be to just serialize the scene
    // so that the number of transforms, shapes, instances, etc. is known from the very start. that way you
    // don't even need hash tables. you would just need an index into a global array that contains everything
    // and it would be really simple and easy to use.
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape> *shapes;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Material> *materials;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Texture> *textures;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Light> *lights;
    ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance> *instanceTypes;
    ChunkedLinkedList<SceneInstance, 1024, MemoryType_Instance> *instances;

    ChunkedLinkedList<const Mat4 *, 16384, MemoryType_Transform> *transforms;

    // TODO: other shapes?
    u32 *numQuadMeshes;
    u32 *numTriMeshes;
    u32 *numCurves;

    InternedStringCache<16384, 8, 64> stringCache;
    Map<Mat4, 1048576, 8, 1024, MemoryType_Transform> transformCache;

    Arena **threadArenas;

    Arena *mainArena;
    Scene *scene;

    Scheduler::Counter counter = {};
};

struct GraphicsState
{
    StringId materialId = 0;
    i32 materialIndex   = -1;
    Mat4 transform      = Mat4::Identity();
    i32 transformIndex  = -1;

    i32 areaLightIndex = -1;
    i32 mediaIndex     = -1;
};

void LoadPBRT(string filename, string directory, SceneLoadState *state, GraphicsState graphicsState = {}, bool inWorldBegin = false);

// TODO IMPORTANT: for some god forsaken reason with clang this generate a vmovaps unaligned error?? i don't want to deal
// with that rn so im putting this here.
Scene *LoadPBRT(Arena *arena, string filename)
{
#define COMMA ,
    Scene *scene = PushStruct(arena, Scene);
    SceneLoadState state;
    u32 numProcessors   = OS_NumProcessors();
    state.shapes        = PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Shape>, numProcessors);
    state.materials     = PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Material>, numProcessors);
    state.textures      = PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Texture>, numProcessors);
    state.lights        = PushArray(arena, ChunkedLinkedList<ScenePacket COMMA 1024 COMMA MemoryType_Light>, numProcessors);
    state.instanceTypes = PushArray(arena, ChunkedLinkedList<ObjectInstanceType COMMA 512 COMMA MemoryType_Instance>, numProcessors);
    state.instances     = PushArray(arena, ChunkedLinkedList<SceneInstance COMMA 1024 COMMA MemoryType_Instance>, numProcessors);
    state.transforms    = PushArray(arena, ChunkedLinkedList<const Mat4 * COMMA 16384 COMMA MemoryType_Transform>, numProcessors);
    state.threadArenas  = PushArray(arena, Arena *, numProcessors);
#undef COMMA

    for (u32 i = 0; i < numProcessors; i++)
    {
        state.threadArenas[i]  = ArenaAlloc();
        state.shapes[i]        = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape>(state.threadArenas[i]);
        state.materials[i]     = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Material>(state.threadArenas[i]);
        state.textures[i]      = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Texture>(state.threadArenas[i]);
        state.lights[i]        = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Light>(state.threadArenas[i]);
        state.instanceTypes[i] = ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance>(state.threadArenas[i]);
        state.instances[i]     = ChunkedLinkedList<SceneInstance, 1024, MemoryType_Instance>(state.threadArenas[i]);
        state.transforms[i]    = ChunkedLinkedList<const Mat4 *, 16384, MemoryType_Transform>(state.threadArenas[i]);
    }
    state.mainArena      = arena;
    state.scene          = scene;
    state.stringCache    = InternedStringCache<16384, 8, 64>(arena);
    state.transformCache = Map<Mat4, 1048576, 8, 1024, MemoryType_Transform>(arena);

    LoadPBRT(filename, Str8PathChopPastLastSlash(filename), &state);

    // TODO: combine the arrays
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

    for (u32 i = 0; i < numProcessors; i++)
    {
        ArenaClear(state.threadArenas[i]);
    }
    return scene;
}

void LoadPBRT(string filename, string directory, SceneLoadState *state, GraphicsState graphicsState, bool inWorldBegin)
{
    TempArena temp  = ScratchStart(0, 0);
    u32 threadIndex = GetThreadIndex();
    Arena *arena    = state->threadArenas[threadIndex];

    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(filename);
    tokenizer.cursor = tokenizer.input.str;

    // TODO: run through the file really fast to find the total number of shapes/materials/etc, and then allocate that amount.
    // also stop reading files when they're already on disk (for the current thread)
    auto &shapes        = state->shapes[threadIndex];
    auto &materials     = state->materials[threadIndex];
    auto &textures      = state->textures[threadIndex];
    auto &lights        = state->lights[threadIndex];
    auto &instanceTypes = state->instanceTypes[threadIndex];
    auto &instances     = state->instances[threadIndex];

    const string *currentInstanceTypeName = 0;
    auto &transforms                      = state->transforms[threadIndex];
    auto &stringCache                     = state->stringCache;

    bool worldBegin = inWorldBegin;

    // Stack variables
    Tokenizer oldTokenizers[32];
    u32 numTokenizers = 0;

    GraphicsState graphicsStateStack[64];
    u32 graphicsStateCount = 0;

    GraphicsState currentGraphicsState = graphicsState;

    auto AddTransform = [&]() {
        if (currentGraphicsState.transformIndex != -1)
        {
            const Mat4 *transform = state->transformCache.GetOrCreate(arena, currentGraphicsState.transform);
            transforms.Push(transform);
        }
    };

    ObjectInstanceType *currentObject = 0;

    // TODO: media

    for (;;)
    {
        if (EndOfBuffer(&tokenizer))
        {
            OS_UnmapFile(tokenizer.input.str);
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

        StringId sid = stringCache.GetOrCreate(arena, word);
        switch (sid)
        {
            case "Accelerator"_sid:
            {
                Error(!worldBegin, "%S cannot be specified after WorldBegin statement\n", word);
                SceneLoadState::Type type = SceneLoadState::Type::Accelerator;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, MemoryType_Other);
                continue;
            }
            break;
            case "AttributeBegin"_sid:
            {
                Error(worldBegin, "%S cannot be specified before WorldBegin statement\n", word);
                Assert(graphicsStateCount < ArrayLength(graphicsStateStack));
                GraphicsState *gs    = &graphicsStateStack[graphicsStateCount++];
                *gs                  = currentGraphicsState;
                currentGraphicsState = {};

                SkipToNextLine(&tokenizer);
            }
            break;
            case "AttributeEnd"_sid:
            {
                Error(worldBegin, "%S cannot be specified before WorldBegin statement\n", word);
                Assert(graphicsStateCount > 0);

                AddTransform();

                // Pop stack
                currentGraphicsState = graphicsStateStack[--graphicsStateCount];

                SkipToNextLine(&tokenizer);
            }
            break;
            case "AreaLightSource"_sid:
            {
                Error(worldBegin, "%S cannot be specified before WorldBegin statement\n", word);
                currentGraphicsState.areaLightIndex = lights.Length();
                ScenePacket *packet                 = &lights.AddBack();
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, MemoryType_Light);
            }
            break;
            case "Attribute"_sid:
            {
                Error(0, "Not implemented Attribute");
            }
            break;
            case "Camera"_sid:
            {
                Error(!worldBegin, "%S cannot be specified after WorldBegin statement\n", word);
                SceneLoadState::Type type = SceneLoadState::Type::Camera;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, MemoryType_Other);
                continue;
            }
            case "ConcatTransform"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
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
                currentGraphicsState.transform = Mul(currentGraphicsState.transform, Mat4(r0c0, r0c1, r0c2, r0c3,
                                                                                          r1c0, r1c1, r1c2, r1c3,
                                                                                          r2c0, r2c1, r2c2, r2c3,
                                                                                          r3c0, r3c1, r3c2, r3c3));
                SkipToNextLine(&tokenizer);
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
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, MemoryType_Other);
                continue;
            }
            case "Integrator"_sid:
            {
                Error(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
                SceneLoadState::Type type = SceneLoadState::Type::Integrator;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, MemoryType_Other);
                continue;
            }
            case "Identity"_sid:
            {
                ReadWord(&tokenizer);
                currentGraphicsState.transform = Mat4::Identity();
            }
            break;
            case "Import"_sid:
            {
                ReadWord(&tokenizer);
                string importedFilename;

                b32 result = GetBetweenPair(importedFilename, &tokenizer, '"');
                Assert(result);
                string importedFullPath = StrConcat(arena, directory, importedFilename);

                scheduler.Schedule(&state->counter, [importedFullPath, directory, state,
                                                     currentGraphicsState, worldBegin](u32 jobID) {
                    LoadPBRT(importedFullPath, directory, state, currentGraphicsState, worldBegin);
                });
            }
            break;
            case "Include"_sid:
            {
                ReadWord(&tokenizer);
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

                currentGraphicsState.transform = Mul(currentGraphicsState.transform,
                                                     LookAt(Vec3f(posX, posY, posZ), Vec3f(lookX, lookY, lookZ), Normalize(Vec3f(upX, upY, upZ))));
            }
            break;
            case "LightSource"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                ScenePacket *packet = &lights.AddBack();
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, MemoryType_Light);
            }
            break;
            case "Material"_sid:
            case "MakeNamedMaterial"_sid:
            {
                bool isNamedMaterial = (sid == "MakeNamedMaterial"_sid);
                // scenePacketCache
                ReadWord(&tokenizer);
                string materialNameOrType;
                b32 result = GetBetweenPair(materialNameOrType, &tokenizer, '"');
                Assert(result);

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
                ReadParameters(arena, packet, &tokenizer, &stringCache, MemoryType_Material);

                if (isNamedMaterial)
                {
                    currentGraphicsState.materialId    = stringCache.GetOrCreate(arena, materialNameOrType);
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
                ReadWord(&tokenizer);
                string materialName;
                b32 result = GetBetweenPair(materialName, &tokenizer, '"');
                Assert(result);

                currentGraphicsState.materialId    = stringCache.GetOrCreate(arena, materialName);
                currentGraphicsState.materialIndex = -1;
            }
            break;
            case "ObjectBegin"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                Error(currentObject == 0, "ObjectBegin cannot be called recursively.");
                Error(currentGraphicsState.areaLightIndex == -1, "Area lights instancing not supported.");
                ReadWord(&tokenizer);
                string objectName;

                b32 result = GetBetweenPair(objectName, &tokenizer, '"');
                Assert(result);

                currentObject = &instanceTypes.AddBack();
                // StringId id                   = stringCache.GetOrCreate(arena, objectName);
                // currentObject->name           = stringCache.Get(id);
                currentObject->name           = stringCache.GetOrCreate(arena, objectName);
                currentObject->transformIndex = currentGraphicsState.transformIndex;
                currentObject->shapeIndices   = Array<i32, MemoryType_Instance>(arena);
            }
            break;
            case "ObjectEnd"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                ReadWord(&tokenizer);
                Error(currentObject != 0, "ObjectEnd must occur after ObjectBegin");
                currentObject = 0;
            }
            break;
            case "ObjectInstance"_sid:
            {
                Error(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                ReadWord(&tokenizer);
                string objectName;
                b32 result = GetBetweenPair(objectName, &tokenizer, '"');
                Assert(result);

                SceneInstance &instance = instances.AddBack();
                instance.name           = stringCache.GetOrCreate(arena, objectName);
                instance.transformIndex = (i32)transforms.Length();

                AddTransform();
                SkipToNextLine(&tokenizer);
            }
            break;
            case "Rotate"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
                ReadWord(&tokenizer);
                f32 angle                      = ReadFloat(&tokenizer);
                f32 axisX                      = ReadFloat(&tokenizer);
                f32 axisY                      = ReadFloat(&tokenizer);
                f32 axisZ                      = ReadFloat(&tokenizer);
                Mat4 rotationMatrix            = Mat4::Rotate(Vec3f(axisX, axisY, axisZ), angle);
                currentGraphicsState.transform = Mul(currentGraphicsState.transform, rotationMatrix);
            }
            break;
            case "Sampler"_sid:
            {
                Error(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
                SceneLoadState::Type type = SceneLoadState::Type::Sampler;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, MemoryType_Other);
                continue;
            }
            break;
            case "Scale"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
                ReadWord(&tokenizer);
                f32 s0 = ReadFloat(&tokenizer);
                f32 s1 = ReadFloat(&tokenizer);
                f32 s2 = ReadFloat(&tokenizer);
                Mat4 scaleMatrix(s0, 0.f, 0.f, 0.f,
                                 0.f, s1, 0.f, 0.f,
                                 0.f, 0.f, s2, 0.f,
                                 0.f, 0.f, 0.f, 1.f);
                currentGraphicsState.transform = Mul(currentGraphicsState.transform, scaleMatrix);
            }
            break;
            case "Shape"_sid:
            {
                Error(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                ScenePacket *packet = &shapes.AddBack();
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache, MemoryType_Shape, 1);
                i32 *indices = PushArray(arena, i32, 4);
                // ORDER: Light, Medium, Transform, Material Index, Material StringID (if present)
                indices[0] = currentGraphicsState.areaLightIndex;
                indices[1] = currentGraphicsState.mediaIndex;
                indices[2] = currentGraphicsState.transformIndex;
                // NOTE: the highest bit is set if it's an index
                indices[3] = currentGraphicsState.materialIndex == -1 ? currentGraphicsState.materialId
                                                                      : (u32)currentGraphicsState.materialIndex | 0x80000000;

                if (currentObject)
                {
                    currentObject->shapeIndices.Push(shapes.Length() - 1);
                }

                u32 currentParameter                     = packet->parameterCount++;
                packet->parameterNames[currentParameter] = stringCache.GetOrCreate(arena, "Indices");
                packet->bytes[currentParameter]          = (u8 *)indices;
                packet->sizes[currentParameter]          = sizeof(i32) * 4;

                AddTransform();
            }
            break;
            case "Translate"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
                ReadWord(&tokenizer);
                f32 t0 = ReadFloat(&tokenizer);
                f32 t1 = ReadFloat(&tokenizer);
                f32 t2 = ReadFloat(&tokenizer);
                Mat4 translationMatrix(0.f, 0.f, 0.f, 0.f,
                                       0.f, 0.f, 0.f, 0.f,
                                       0.f, 0.f, 0.f, 0.f,
                                       t0, t1, t2, 1.f);
                currentGraphicsState.transform = Mul(currentGraphicsState.transform, translationMatrix);
            }
            break;
            case "Transform"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
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
                currentGraphicsState.transform = Mat4(r0c0, r0c1, r0c2, r0c3,
                                                      r1c0, r1c1, r1c2, r1c3,
                                                      r2c0, r2c1, r2c2, r2c3,
                                                      r3c0, r3c1, r3c2, r3c3);

                SkipToNextLine(&tokenizer);
            }
            break;
            case "Texture"_sid:
            {
                ReadWord(&tokenizer);
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
                packet->type        = stringCache.GetOrCreate(arena, StrConcat(arena, textureType, textureClass));

                if (IsEndOfLine(&tokenizer))
                {
                    SkipToNextLine(&tokenizer);
                }
                else
                {
                    SkipToNextChar(&tokenizer);
                }
                ReadParameters(arena, packet, &tokenizer, &stringCache, MemoryType_Texture);
            }
            break;
            case "WorldBegin"_sid:
            {
                ReadWord(&tokenizer);
                // NOTE: this assumes "WorldBegin" only occurs in one file
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

                const ScenePacket *samplerPacket = &state->packets[SceneLoadState::Type::Sampler];
                state->scene->sampler            = Sampler::Create(state->mainArena, samplerPacket, fullResolution);

                AddTransform();
                // TODO: instantiate the camera with the current transform

                currentGraphicsState.transform = Mat4::Identity();
            }
            break;
            default:
            {
                string line = ReadLine(&tokenizer);
                Error(0, "Error while parsing scene. Buffer: %S", line);
            }
                // TODO IMPORTANT: the indices are clockwise since PBRT uses a left-handed coordinate system. either need to
                // revert the winding or use a left handed system as well
        }
    }
    ScratchEnd(temp);
}

void SerializeShapes(Arena *arena, SceneLoadState *state) // ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape> *list)
{
    u32 numProcessors = OS_NumProcessors();

    u32 totalNumQuadMeshes = 0;
    u32 totalNumTriMeshes  = 0;
    u32 totalNumCurves     = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalNumQuadMeshes += state->numQuadMeshes[i];
        totalNumTriMeshes += state->numTriMeshes[i];
        totalNumCurves += state->numCurves[i];
    }

    enum PrimitiveTy
    {
        P_NoneTy,
        P_TriMesh,
        P_QuadMesh,
        P_Curve,
    };

    QuadMesh *qMeshes       = PushArray(arena, QuadMesh, totalNumQuadMeshes);
    u32 *qMeshBaseOffsets   = PushArrayNoZero(arena, u32, numProcessors);
    TriangleMesh *triMeshes = PushArray(arena, TriangleMesh, totalNumTriMeshes);
    u32 *triMeshBaseOffsets = PushArrayNoZero(arena, u32, numProcessors);

    PrimitiveTy **types = PushArray(arena, PrimitiveTy *, numProcessors);
    u32 **offsets       = PushArray(arena, u32 *, numProcessors);

    using InstanceTypeList = ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance>;

    using ShapeTypeList = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape>;

    u32 quadOffset = 0;
    u32 triOffset  = 0;
    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        ShapeTypeList *list = &state->shapes[pIndex];
        types[pIndex]       = PushArray(arena, PrimitiveTy, list->totalCount);
        offsets[pIndex]     = PushArray(arena, u32, list->totalCount);

        u32 *pOffsets       = offsets[pIndex];
        PrimitiveTy *pTypes = types[pIndex];
        for (ShapeTypeList::ChunkNode *node = list->first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ScenePacket *packet = &node->values[i];
                switch (packet->type)
                {
                    case "quadmesh"_sid:
                    {
                        pTypes[i]      = P_QuadMesh;
                        pOffsets[i]    = quadOffset;
                        QuadMesh *mesh = &qMeshes[quadOffset++];
                        for (u32 parameterIndex = 0; parameterIndex < packet->parameterCount; parameterIndex++)
                        {
                            switch (packet->parameterNames[parameterIndex])
                            {
                                case "P"_sid:
                                {
                                    mesh->p           = (Vec3f *)packet->bytes[parameterIndex];
                                    mesh->numVertices = packet->sizes[parameterIndex] / sizeof(Vec3f);
                                }
                                break;
                                case "N"_sid:
                                {
                                    mesh->n = (Vec3f *)packet->bytes[parameterIndex];
                                    Assert(mesh->numVertices == packet->sizes[parameterIndex] / sizeof(Vec3f));
                                }
                                break;
                                // NOTE: this is specific to the moana island data set (not needing the indices or uvs)
                                default: continue;
                            }
                        }
                    }
                    break;
                    case "trianglemesh"_sid:
                    {
                        pTypes[i]          = P_TriMesh;
                        pOffsets[i]        = triOffset;
                        TriangleMesh *mesh = &triMeshes[triOffset++];
                        for (u32 parameterIndex = 0; parameterIndex < packet->parameterCount; parameterIndex++)
                        {
                            switch (packet->parameterNames[parameterIndex])
                            {
                                case "P"_sid:
                                {
                                    mesh->p           = (Vec3f *)packet->bytes[parameterIndex];
                                    mesh->numVertices = packet->sizes[parameterIndex] / sizeof(Vec3f);
                                }
                                break;
                                case "N"_sid:
                                {
                                    mesh->n = (Vec3f *)packet->bytes[parameterIndex];
                                    Assert(mesh->numVertices == packet->sizes[parameterIndex] / sizeof(Vec3f));
                                }
                                break;
                                case "indices"_sid:
                                {
                                    mesh->indices    = (u32 *)packet->bytes[parameterIndex];
                                    mesh->numIndices = packet->sizes[parameterIndex] / sizeof(u32);
                                }
                                break;
                                // TODO: need to make sure that this is always just 0, 1, 2, 3... etc
                                case "faceIndices"_sid: continue;
                                case "uv"_sid:
                                {
                                    mesh->uv = (Vec2f *)packet->bytes[parameterIndex];
                                    Assert(mesh->numVertices == packet->sizes[parameterIndex] / sizeof(Vec2f));
                                }
                                break;
                            }
                        }
                    }
                    break;
                    // TODO: curves
                    case "curve"_sid: continue;
                }
            }
        }
    }

    // Merge meshes belonging to the same object instance type
    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        InstanceTypeList *list   = &state->instanceTypes[pIndex];
        ShapeTypeList *shapeList = &state->shapes[pIndex];
        PrimitiveTy *pTypes      = types[pIndex];
        u32 *pOffsets            = offsets[pIndex];
        for (InstanceTypeList::ChunkNode *node = list->first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ObjectInstanceType *instType = &node->values[i];
                bool allQuads                = true;
                for (u32 shapeIndexIndex = 0; shapeIndexIndex < instType->shapeIndices.Length(); shapeIndexIndex++)
                {
                    u32 shapeIndex = instType->shapeIndices[shapeIndexIndex];
                    Assert(shapeIndex < shapeList->totalCount);
                    if (pTypes[shapeIndex] != P_QuadMesh)
                    {
                        allQuads = false;
                        break;
                    }
                }
                if (allQuads)
                {
                    QuadMesh *accumMesh  = &qMeshes[pOffsets[instType->shapeIndices[0]]];
                    u32 totalVertexCount = 0;
                    for (u32 shapeIndexIndex = 0; shapeIndexIndex < instType->shapeIndices.Length(); shapeIndexIndex++)
                    {
                        QuadMesh *mesh = &qMeshes[pOffsets[instType->shapeIndices[shapeIndexIndex]]];
                        totalVertexCount += mesh->numVertices;
                    }
                    Vec3f *p         = PushArrayNoZero(arena, Vec3f, totalVertexCount);
                    Vec3f *n         = PushArrayNoZero(arena, Vec3f, totalVertexCount);
                    totalVertexCount = 0;
                    for (u32 shapeIndexIndex = 0; shapeIndexIndex < instType->shapeIndices.Length(); shapeIndexIndex++)
                    {
                        QuadMesh *mesh = &qMeshes[pOffsets[instType->shapeIndices[shapeIndexIndex]]];
                        MemoryCopy(accumMesh->p + totalVertexCount, mesh->p, sizeof(Vec3f) * mesh->numVertices);
                        MemoryCopy(accumMesh->n + totalVertexCount, mesh->n, sizeof(Vec3f) * mesh->numVertices);
                        totalVertexCount += mesh->numVertices;
                    }
                }
            }
        }
    }
}

} // namespace rt
