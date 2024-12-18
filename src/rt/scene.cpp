#include "scene.h"
#include "bvh/bvh_types.h"
namespace rt
{
#if 0
AABB Transform(const HomogeneousTransform &transform, const AABB &aabb)
{
    AABB result;
    Vec3f vecs[] = {
        Vec3f(aabb.minX, aabb.minY, aabb.minZ), Vec3f(aabb.maxX, aabb.minY, aabb.minZ),
        Vec3f(aabb.maxX, aabb.maxY, aabb.minZ), Vec3f(aabb.minX, aabb.maxY, aabb.minZ),
        Vec3f(aabb.minX, aabb.minY, aabb.maxZ), Vec3f(aabb.maxX, aabb.minY, aabb.maxZ),
        Vec3f(aabb.maxX, aabb.maxY, aabb.maxZ), Vec3f(aabb.minX, aabb.maxY, aabb.maxZ),
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
    if (discriminant < 0) return false;

    f32 result = (h - sqrt(discriminant)) / a;
    if (result <= tMin || result >= tMax)
    {
        result = (h + sqrt(discriminant)) / a;
        if (result <= tMin || result >= tMax) return false;
    }

    record.t     = result;
    record.p     = r.at(record.t);
    Vec3f normal = (record.p - center) / radius;
    record.SetNormal(r, normal);
    record.material = material;
    Sphere::GetUV(record.u, record.v, normal);

    return true;
}
Vec3f Sphere::Center(f32 time) const { return center + centerVec * time; }
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
    if (!this->Hit(Ray(origin, direction), 0.001f, infinity, rec)) return 0;
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
#endif

//////////////////////////////
// Scene
//
#if 0
void Scene::FinalizePrimitives()
{
    totalPrimitiveCount = sphereCount + quadCount + boxCount;
    primitiveIndices =
        (PrimitiveIndices *)malloc(sizeof(PrimitiveIndices) * totalPrimitiveCount);
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
inline void Scene::GetTypeAndLocalindex(const u32 totalIndex, PrimitiveType *type,
                                        u32 *localIndex) const
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
#endif

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

#ifdef DEBUG
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
const string *StringCache<numNodes, chunkSize, numStripes>::GetOrCreate(Arena *arena,
                                                                        string value)
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

    // NOTE: there's a very rare case where the thread gets unscheduled here, another thread
    // writes the same string value to the cache. and this thread writes that value again. this
    // doesn't impact correctness but some values will be duplicated, wasting some memory

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

struct ObjectInstanceType
{
    StringId name;
    // string name;
    u32 transformIndex  = 0;
    u32 shapeIndexStart = 0xffffffff;
    u32 shapeIndexEnd   = 0xffffffff;

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
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Material> *materials;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Texture> *textures;
    ChunkedLinkedList<ScenePacket, 1024, MemoryType_Light> *lights;
    ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance> *instanceTypes;
    ChunkedLinkedList<SceneInstance, 1024, MemoryType_Instance> *instances;

    ChunkedLinkedList<const AffineSpace *, 16384, MemoryType_Transform> *transforms;

    // TODO: other shapes?
    u32 *numQuadMeshes;
    u32 *numTriMeshes;
    // u32 *numCurves;

    InternedStringCache<16384, 8, 64> stringCache;
    HashSet<AffineSpace, 1048576, 8, 1024, MemoryType_Transform> transformCache;

    Arena **threadArenas;

    Arena *mainArena;
    Scene *scene;

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
    while (!EndOfBuffer(tokenizer) &&
           (!IsDigit(tokenizer) && *tokenizer->cursor != '-' && *tokenizer->cursor != ']'))
        tokenizer->cursor++;
}

inline void AdvanceToNextLine(Tokenizer *tokenizer)
{
    if (Advance(tokenizer, " ]\n")) return;
    if (Advance(tokenizer, "]\n")) return;
    if (Advance(tokenizer, "\n")) return;

    // if it's not on the next line, make sure to still advance
    if (Advance(tokenizer, "]")) return;
    if (Advance(tokenizer, " ]")) return;
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
              bool imported = false);
void Serialize(Arena *arena, string directory, SceneLoadState *state);

Scene *LoadPBRT(Arena *arena, string filename)
{
#define COMMA ,
    Scene *scene = PushStruct(arena, Scene);
    SceneLoadState state;
    u32 numProcessors   = OS_NumProcessors();
    state.numTriMeshes  = PushArray(arena, u32, numProcessors);
    state.numQuadMeshes = PushArray(arena, u32, numProcessors);
    state.shapes =
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
    state.threadArenas = PushArray(arena, Arena *, numProcessors);
#undef COMMA

    for (u32 i = 0; i < numProcessors; i++)
    {
        state.threadArenas[i] = ArenaAlloc(16);
        state.shapes[i] =
            ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape>(state.threadArenas[i]);
        state.materials[i] =
            ChunkedLinkedList<ScenePacket, 1024, MemoryType_Material>(state.threadArenas[i]);
        state.textures[i] =
            ChunkedLinkedList<ScenePacket, 1024, MemoryType_Texture>(state.threadArenas[i]);
        state.lights[i] =
            ChunkedLinkedList<ScenePacket, 1024, MemoryType_Light>(state.threadArenas[i]);
        state.instanceTypes[i] =
            ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance>(
                state.threadArenas[i]);
        state.instances[i] =
            ChunkedLinkedList<SceneInstance, 1024, MemoryType_Instance>(state.threadArenas[i]);
        state.transforms[i] =
            ChunkedLinkedList<const AffineSpace *, 16384, MemoryType_Transform>(
                state.threadArenas[i]);
    }
    state.mainArena      = arena;
    state.scene          = scene;
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

    Serialize(arena, baseDirectory, &state);

    for (u32 i = 0; i < numProcessors; i++)
    {
        ArenaClear(state.threadArenas[i]);
    }
    return scene;
}

void LoadPBRT(string filename, string directory, SceneLoadState *state,
              GraphicsState graphicsState, bool inWorldBegin, bool imported)
{
    TempArena temp  = ScratchStart(0, 0);
    u32 threadIndex = GetThreadIndex();
    Arena *arena    = state->threadArenas[threadIndex];

    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(filename);
    tokenizer.cursor = tokenizer.input.str;

    auto &shapes        = state->shapes[threadIndex];
    auto &materials     = state->materials[threadIndex];
    auto &textures      = state->textures[threadIndex];
    auto &lights        = state->lights[threadIndex];
    auto &instanceTypes = state->instanceTypes[threadIndex];
    auto &instances     = state->instances[threadIndex];

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
        transforms.Push(state->transformCache.GetOrCreate(arena, AffineSpace::Identity()));
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
                // Assert(currentGraphicsState.transformIndex == transforms.Length());
                const AffineSpace *transform =
                    state->transformCache.GetOrCreate(arena, currentGraphicsState.transform);
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
                Error(!worldBegin, "%S cannot be specified after WorldBegin statement\n",
                      word);
                SceneLoadState::Type type = SceneLoadState::Type::Accelerator;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
                continue;
            }
            break;
            case "AttributeBegin"_sid:
            {
                Error(worldBegin, "%S cannot be specified before WorldBegin statement\n",
                      word);
                Assert(graphicsStateCount < ArrayLength(graphicsStateStack));
                GraphicsState *gs = &graphicsStateStack[graphicsStateCount++];
                *gs               = currentGraphicsState;

                SkipToNextLine(&tokenizer);
            }
            break;
            case "AttributeEnd"_sid:
            {
                Error(worldBegin, "%S cannot be specified before WorldBegin statement\n",
                      word);
                Assert(graphicsStateCount > 0);

                // AddTransform();

                // Pop stack
                currentGraphicsState = graphicsStateStack[--graphicsStateCount];

                SkipToNextLine(&tokenizer);
            }
            break;
            // TODO: area light count is reported as 23 when there's 22
            case "AreaLightSource"_sid:
            {
                Error(worldBegin, "%S cannot be specified before WorldBegin statement\n",
                      word);
                currentGraphicsState.areaLightIndex = lights.Length();
                ScenePacket *packet                 = &lights.AddBack();
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache,
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
                Error(!worldBegin, "%S cannot be specified after WorldBegin statement\n",
                      word);
                SceneLoadState::Type type = SceneLoadState::Type::Camera;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
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
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
                continue;
            }
            case "Integrator"_sid:
            {
                Error(!worldBegin, "Tried to specify %S after WorldBegin\n", word);
                SceneLoadState::Type type = SceneLoadState::Type::Integrator;
                ScenePacket *packet       = &state->packets[type];
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
                continue;
            }
            case "Identity"_sid:
            {
                ReadWord(&tokenizer);
                currentGraphicsState.transform = AffineSpace::Identity();
            }
            break;
            case "Import"_sid:
            {
                ReadWord(&tokenizer);
                string importedFilename;

                b32 result = GetBetweenPair(importedFilename, &tokenizer, '"');
                Assert(result);
                string importedFullPath = StrConcat(arena, directory, importedFilename);

                scheduler.Schedule(&state->counter,
                                   [importedFullPath, directory, state, currentGraphicsState,
                                    worldBegin](u32 jobID) {
                                       LoadPBRT(importedFullPath, directory, state,
                                                currentGraphicsState, worldBegin, true);
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
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Light);
            }
            break;
            case "Material"_sid:
            case "MakeNamedMaterial"_sid:
            {
                bool isNamedMaterial = (sid == "MakeNamedMaterial"_sid);
                ReadWord(&tokenizer);
                string materialNameOrType;
                b32 result = GetBetweenPair(materialNameOrType, &tokenizer, '"');
                Assert(result);

                ScenePacket *packet = &materials.AddBack();
                packet->type        = stringCache.GetOrCreate(arena, materialNameOrType);
                u32 materialIndex   = materials.Length();
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
                    currentGraphicsState.materialId =
                        stringCache.GetOrCreate(arena, materialNameOrType);
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

                currentGraphicsState.materialId = stringCache.GetOrCreate(arena, materialName);
                currentGraphicsState.materialIndex = -1;
            }
            break;
            case "ObjectBegin"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                Error(currentObject == 0, "ObjectBegin cannot be called recursively.");
                Error(currentGraphicsState.areaLightIndex == -1,
                      "Area lights instancing not supported.");
                ReadWord(&tokenizer);
                string objectName;

                b32 result = GetBetweenPair(objectName, &tokenizer, '"');
                Assert(result);

                currentObject                  = &instanceTypes.AddBack();
                currentObject->name            = stringCache.GetOrCreate(arena, objectName);
                currentObject->transformIndex  = currentGraphicsState.transformIndex;
                currentObject->shapeIndexStart = shapes.Length();

                AddTransform();
            }
            break;
            case "ObjectEnd"_sid:
            {
                Error(worldBegin, "Tried to specify %S before WorldBegin\n", word);
                ReadWord(&tokenizer);

                currentObject->shapeIndexEnd = shapes.Length();
                Assert(currentObject->shapeIndexEnd >= currentObject->shapeIndexStart);
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
                Assert(IsEndOfLine(&tokenizer));
                SkipToNextLine(&tokenizer);
            }
            break;
            case "Rotate"_sid:
            {
                currentGraphicsState.transformIndex = transforms.Length();
                ReadWord(&tokenizer);
                f32 angle = ReadFloat(&tokenizer);
                f32 axisX = ReadFloat(&tokenizer);
                f32 axisY = ReadFloat(&tokenizer);
                f32 axisZ = ReadFloat(&tokenizer);
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
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Other);
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

                AffineSpace scale              = AffineSpace::Scale(Vec3f(s0, s1, s2));
                currentGraphicsState.transform = currentGraphicsState.transform * scale;
            }
            break;
            case "Shape"_sid:
            {
                Error(worldBegin, "Tried to specify %S after WorldBegin\n", word);
                ScenePacket *packet = &shapes.AddBack();
                CreateScenePacket(arena, word, packet, &tokenizer, &stringCache,
                                  MemoryType_Shape, 1);

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
                    packet->type = stringCache.GetOrCreate(arena, "quadmesh");
                    state->numQuadMeshes[GetThreadIndex()]++;
                }
                else if (packet->type == "trianglemesh"_sid)
                {
                    state->numTriMeshes[GetThreadIndex()]++;
                }

                i32 *indices = PushArray(arena, i32, 4);
                // ORDER: Light, Medium, Transform, Material Index, Material StringID (if
                // present)
                indices[0] = currentGraphicsState.areaLightIndex;
                indices[1] = currentGraphicsState.mediaIndex;
                indices[2] = currentGraphicsState.transformIndex;
                // NOTE: the highest bit is set if it's an index
                indices[3] = currentGraphicsState.materialIndex == -1
                                 ? i32(currentGraphicsState.materialId)
                                 : (u32)currentGraphicsState.materialIndex | 0x80000000;

                u32 currentParameter = packet->parameterCount++;
                packet->parameterNames[currentParameter] =
                    stringCache.GetOrCreate(arena, "Indices");
                packet->bytes[currentParameter] = (u8 *)indices;
                packet->sizes[currentParameter] = sizeof(i32) * 4;

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

                AffineSpace t                  = AffineSpace::Translate(Vec3f(t0, t1, t2));
                currentGraphicsState.transform = currentGraphicsState.transform * t;
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
                currentGraphicsState.transform = AffineSpace(
                    r0c0, r1c0, r2c0, r3c0, r0c1, r1c1, r2c1, r3c1, r0c2, r1c2, r2c2, r3c2);

                SkipToNextChar(&tokenizer);
                bool result = Advance(&tokenizer, "]\n");
                Assert(result);
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
                packet->type        = stringCache.GetOrCreate(
                    arena, StrConcat(arena, textureType, textureClass));

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

                const ScenePacket *samplerPacket =
                    &state->packets[SceneLoadState::Type::Sampler];
                // state->scene->sampler =
                //     Sampler::Create(state->mainArena, samplerPacket, fullResolution);

                AddTransform();
                // TODO: instantiate the camera with the current transform
                currentGraphicsState.transform = AffineSpace::Identity();
            }
            break;
            default:
            {
                string line = ReadLine(&tokenizer);
                Error(0, "Error while parsing scene. Buffer: %S", line);
            }
                // TODO IMPORTANT: the indices are clockwise since PBRT uses a left-handed
                // coordinate system. either need to revert the winding or use a left handed
                // system as well
        }
    }
    ScratchEnd(temp);
}

struct LookupEntry
{
    // u64 offset;
    u32 name;
    u32 quadMeshCount;
};

u64 GetOffset(u32 name, u32 offset) { return (u64(offset) << 32ull) | u64(name); }

#define SERIALIZE_SHAPES    0
#define SERIALIZE_INSTANCES 1

struct HashTable
{
    struct Node
    {
        Vec3f *v;
        u32 numVertices;
        u64 ptr;
        Node *next;
    };
    Node *slots;
    Arena *arena;
    u32 tableSize;

    HashTable() {}
    HashTable(Arena *arena, u32 tableSize) : arena(arena), tableSize(tableSize)
    {
        Assert(IsPow2(tableSize));
        slots = PushArray(arena, Node, tableSize);
    }
    Node *FindOrCreate(u64 hash, Vec3f *data, u32 numVertices, u32 name, u32 offset)
    {
        u64 key    = hash & (tableSize - 1);
        Node *node = &slots[key];
        Node *prev = 0;
        while (node)
        {
            if (numVertices == node->numVertices)
            {
                bool equal = true;
                for (u32 i = 0; i < numVertices; i++)
                {
                    if (data[i] != node->v[i])
                    {
                        equal = false;
                        break;
                    }
                }
                if (equal)
                {
                    return node;
                }
            }
            prev = node;
            node = node->next;
        }
        Assert(node == 0);
        node              = PushStruct(arena, Node);
        node->v           = data;
        node->numVertices = numVertices;
        node->ptr         = GetOffset(name, offset);
        prev->next        = node;
        return node;
    }
};

void SerializeMeshes(Vec3f *&dataPtr, u8 *mappedPtr, QuadMesh *mesh, QuadMesh *newMesh,
                     u32 entryIndex, HashTable &pTable, HashTable &nTable)
{
    newMesh->numVertices = mesh->numVertices;
    u64 pOffset          = u64((u8 *)dataPtr - mappedPtr);
    Assert(pOffset <= 0xffffffff);

    u64 pHash = MurmurHash64A((const u8 *)mesh->p, sizeof(Vec3f) * mesh->numVertices, 0);
    HashTable::Node *pNode =
        pTable.FindOrCreate(pHash, mesh->p, mesh->numVertices, entryIndex, u32(pOffset));

    if (pNode->v == mesh->p)
    {
        MemoryCopy(dataPtr, mesh->p, sizeof(Vec3f) * mesh->numVertices);
        pNode->v = dataPtr;
        dataPtr += mesh->numVertices;
    }
    u64 offset            = pNode->ptr;
    *(u64 *)(&newMesh->p) = offset;

    HashTable::Node *nNode = 0;
    if (mesh->n)
    {
        u64 nOffset = u64((u8 *)dataPtr - mappedPtr);
        u64 nHash   = MurmurHash64A((const u8 *)mesh->n, sizeof(Vec3f) * mesh->numVertices, 0);
        nNode =
            nTable.FindOrCreate(nHash, mesh->n, mesh->numVertices, entryIndex, u32(nOffset));
        if (nNode->v == mesh->n)
        {
            MemoryCopy(dataPtr, mesh->n, sizeof(Vec3f) * mesh->numVertices);
            nNode->v = dataPtr;
            dataPtr += mesh->numVertices;
        }
        offset                = nNode->ptr;
        *(u64 *)(&newMesh->n) = offset;
    }
    else
    {
        newMesh->n = 0;
    }
}

void Serialize(Arena *arena, string directory, SceneLoadState *state)
{
    TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();

    u32 *quadOffsets = PushArrayNoZero(temp.arena, u32, numProcessors + 1);
    u32 *triOffsets  = PushArrayNoZero(temp.arena, u32, numProcessors + 1);

    u32 totalNumQuadMeshes = 0;
    u32 totalNumTriMeshes  = 0;
    u32 totalNumInstTypes  = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        quadOffsets[i] = totalNumQuadMeshes;
        triOffsets[i]  = totalNumTriMeshes;

        totalNumQuadMeshes += state->numQuadMeshes[i];
        totalNumTriMeshes += state->numTriMeshes[i];
        totalNumInstTypes += state->instanceTypes[i].totalCount;
        // totalNumCurves += state->numCurves[i];
    }
    quadOffsets[numProcessors] = totalNumQuadMeshes;
    triOffsets[numProcessors]  = totalNumTriMeshes;

    enum PrimitiveTy
    {
        P_NoneTy,
        P_RemovedTy,
        P_TriMesh,
        P_QuadMesh,
        P_Curve,
    };

    QuadMesh *qMeshes       = PushArray(temp.arena, QuadMesh, totalNumQuadMeshes);
    TriangleMesh *triMeshes = PushArray(temp.arena, TriangleMesh, totalNumTriMeshes);

    PrimitiveTy **types = PushArray(temp.arena, PrimitiveTy *, numProcessors);
    u32 **offsets       = PushArray(temp.arena, u32 *, numProcessors);

    u32 *totalNumVertices = PushArray(temp.arena, u32, numProcessors);

    using InstanceTypeList = ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance>;

    using ShapeTypeList = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape>;

    scheduler.ScheduleAndWait(numProcessors, 1, [&](u32 jobID) {
        TempArena temp      = ScratchStart(0, 0);
        u32 quadOffset      = quadOffsets[jobID];
        u32 quadLimit       = quadOffsets[jobID + 1];
        u32 triOffset       = triOffsets[jobID];
        u32 triLimit        = triOffsets[jobID + 1];
        u32 pIndex          = jobID;
        Arena *arena        = state->threadArenas[pIndex];
        ShapeTypeList *list = &state->shapes[pIndex];
        types[pIndex]       = PushArray(arena, PrimitiveTy, list->totalCount);
        offsets[pIndex]     = PushArray(arena, u32, list->totalCount);

        u32 totalNumQuadVertices = 0;

        u32 *pOffsets       = offsets[pIndex];
        PrimitiveTy *pTypes = types[pIndex];
        u32 currentOffset   = 0;
        for (ShapeTypeList::ChunkNode *node = list->first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ScenePacket *packet = &node->values[i];
                switch (packet->type)
                {
                    case "quadmesh"_sid:
                    {
                        pTypes[currentOffset + i] = P_QuadMesh;
                        u32 quadIndex             = quadOffset++;
                        Assert(quadIndex < quadLimit);
                        pOffsets[currentOffset + i] = quadIndex;
                        QuadMesh *mesh              = &qMeshes[quadIndex];
                        // quadCounts[pIndex]++;
                        for (u32 parameterIndex = 0; parameterIndex < packet->parameterCount;
                             parameterIndex++)
                        {
                            switch (packet->parameterNames[parameterIndex])
                            {
                                case "P"_sid:
                                {
                                    mesh->p = (Vec3f *)packet->bytes[parameterIndex];
                                    mesh->numVertices =
                                        packet->sizes[parameterIndex] / sizeof(Vec3f);
                                    totalNumQuadVertices += mesh->numVertices;
                                }
                                break;
                                case "N"_sid:
                                {
                                    mesh->n = (Vec3f *)packet->bytes[parameterIndex];
                                    Assert(mesh->numVertices ==
                                           packet->sizes[parameterIndex] / sizeof(Vec3f));
                                }
                                break;
                                // NOTE: this is specific to the moana island data set (not
                                // needing the indices or uvs)
                                default: continue;
                            }
                        }
                    }
                    break;
                    case "trianglemesh"_sid:
                    {
                        pTypes[currentOffset + i] = P_TriMesh;
                        u32 triIndex              = triOffset++;
                        Assert(triIndex < triLimit);
                        pOffsets[currentOffset + i] = triIndex;
                        TriangleMesh *mesh          = &triMeshes[triIndex++];
                        for (u32 parameterIndex = 0; parameterIndex < packet->parameterCount;
                             parameterIndex++)
                        {
                            switch (packet->parameterNames[parameterIndex])
                            {
                                case "P"_sid:
                                {
                                    mesh->p = (Vec3f *)packet->bytes[parameterIndex];
                                    mesh->numVertices =
                                        packet->sizes[parameterIndex] / sizeof(Vec3f);
                                }
                                break;
                                case "N"_sid:
                                {
                                    mesh->n = (Vec3f *)packet->bytes[parameterIndex];
                                    Assert(mesh->numVertices ==
                                           packet->sizes[parameterIndex] / sizeof(Vec3f));
                                }
                                break;
                                case "indices"_sid:
                                {
                                    mesh->indices = (u32 *)packet->bytes[parameterIndex];
                                    mesh->numIndices =
                                        packet->sizes[parameterIndex] / sizeof(u32);
                                }
                                break;
                                case "faceIndices"_sid: continue;
                                case "uv"_sid:
                                {
                                    mesh->uv = (Vec2f *)packet->bytes[parameterIndex];
                                    Assert(mesh->numVertices ==
                                           packet->sizes[parameterIndex] / sizeof(Vec2f));
                                }
                                break;
                                default: Assert(0);
                            }
                        }
                    }
                    break;
                    case "plymesh"_sid:
                    {
                        for (u32 parameterIndex = 0; parameterIndex < packet->parameterCount;
                             parameterIndex++)
                        {
                            if (packet->parameterNames[parameterIndex] == "filename"_sid)
                            {
                                string filename;
                                filename.str  = packet->bytes[parameterIndex];
                                filename.size = packet->sizes[parameterIndex];
                                string fullFilePath =
                                    StrConcat(temp.arena, directory, filename);
                                QuadMesh mesh = LoadQuadPLY(arena, fullFilePath);
                                // NOTE: should only happen for the ocean geometry (twice)
                                if (mesh.p == 0)
                                {
                                    TriangleMesh triMesh      = LoadPLY(arena, fullFilePath);
                                    pTypes[currentOffset + i] = P_TriMesh;
                                    u32 triIndex              = triOffset++;
                                    Assert(triIndex < triLimit);
                                    pOffsets[currentOffset + i] = triIndex;
                                    triMeshes[triIndex]         = triMesh;
                                }
                                else
                                {
                                    pTypes[currentOffset + i] = P_QuadMesh;
                                    u32 quadIndex             = quadOffset++;
                                    Assert(quadIndex < quadLimit);
                                    pOffsets[currentOffset + i] = quadIndex;
                                    qMeshes[quadIndex]          = mesh;
                                    totalNumQuadVertices += mesh.numVertices;
                                    // quadCounts[pIndex]++;
                                }
                                break;
                            }
                        }
                    }
                    break;
                    // TODO: curves
                    case "curve"_sid:
                    {
                        pTypes[currentOffset + i] = P_Curve;
                        // pOffsets[currentOffset + i] =
                    }
                    break;
                    default: Assert(!"not parsed");
                }
            }
            currentOffset += node->count;
        }
        Assert(quadOffset == quadLimit);
        totalNumVertices[jobID] = totalNumQuadVertices;
        ScratchEnd(temp);
    });

    u32 totalVertexCount = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalVertexCount += totalNumVertices[i];
    }

#if SERIALIZE_SHAPES
    LookupEntry *lookUpStart =
        PushArrayNoZero(temp.arena, LookupEntry, totalNumQuadMeshes + totalNumTriMeshes);

    u8 **mappedPtrs   = PushArray(temp.arena, u8 *, totalNumInstTypes);
    u32 *sizes        = PushArray(temp.arena, u32, totalNumInstTypes);
    string *filenames = PushArray(temp.arena, string, totalNumInstTypes);

#endif
    u32 lookUpOffset       = 0;
    u32 quadInstancedCount = 0;

    ObjectInstanceType *currentObject[64] = {};
    u32 pIndices[64]                      = {};
    u32 currentInstanceCount              = 0;

    // NOTE: externally chained hash table that cannot grow

    HashTable pTable(temp.arena, 524288);
    HashTable nTable(temp.arena, 524288);

    HashExt<StringId, u32> instanceTypeTable(arena, totalNumQuadMeshes + totalNumTriMeshes,
                                             StringId::Invalid);

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
                // NOTE: have to multiply by the object type transform in general case (but for
                // the moana scene desc, none of the object types have a transform)
                StringId name = instType->name;
                if (!instType->Invalid())
                {
                    Assert(currentInstanceCount == 0);

                    currentObject[currentInstanceCount] = instType;
                    pIndices[currentInstanceCount++]    = pIndex;
                    u32 restIndex                       = i + 1;
                    for (InstanceTypeList::ChunkNode *remainingNode = node; remainingNode != 0;
                         remainingNode                              = remainingNode->next)
                    {
                        for (; restIndex < remainingNode->count; restIndex++)
                        {
                            instType = &remainingNode->values[restIndex];
                            if (instType->name == name)
                            {
                                currentObject[currentInstanceCount] = instType;
                                pIndices[currentInstanceCount++]    = pIndex;
                            }
                        }
                        restIndex = 0;
                    }
                    for (u32 pIndex2 = pIndex + 1; pIndex2 < numProcessors; pIndex2++)
                    {
                        InstanceTypeList *list2 = &state->instanceTypes[pIndex2];
                        for (InstanceTypeList::ChunkNode *node2 = list2->first; node2 != 0;
                             node2                              = node2->next)
                        {
                            for (u32 j = 0; j < node2->count; j++)
                            {
                                instType = &node2->values[j];
                                if (instType->name == name)
                                {
                                    currentObject[currentInstanceCount] = instType;
                                    pIndices[currentInstanceCount++]    = pIndex2;
                                }
                            }
                        }
                    }

                    bool someQuads    = false;
                    bool hasNormals   = false;
                    u32 quadMeshCount = 0;
                    u32 vertexCount   = 0;
                    for (u32 instanceIndex = 0; instanceIndex < currentInstanceCount;
                         instanceIndex++)
                    {
                        ObjectInstanceType *currentObj = currentObject[instanceIndex];

                        u32 pIdx                    = pIndices[instanceIndex];
                        u32 *currentOffsetGroup     = offsets[pIdx];
                        PrimitiveTy *currentTyGroup = types[pIdx];
                        Assert(currentObj->shapeIndexStart <= currentObj->shapeIndexEnd);
                        for (u32 shapeIndex = currentObj->shapeIndexStart;
                             shapeIndex < currentObj->shapeIndexEnd; shapeIndex++)
                        {
                            if (currentTyGroup[shapeIndex] == P_QuadMesh)
                            {
                                quadMeshCount++;
                                someQuads      = true;
                                QuadMesh *mesh = &qMeshes[currentOffsetGroup[shapeIndex]];
                                if (!hasNormals && mesh->n) hasNormals = true;
                                quadInstancedCount++;
                                vertexCount += mesh->numVertices;
#if SERIALIZE_SHAPES == 0
                                currentTyGroup[shapeIndex] = P_RemovedTy;
#endif
                            }
                        }
#if SERIALIZE_SHAPES == 0
                        currentObj->Invalidate();
#endif
                    }
                    if (!someQuads)
                    {
                        printf("no quads\n");
                        currentInstanceCount = 0;
                        continue;
                    }

                    Assert(lookUpOffset < totalNumInstTypes);
                    u32 entryIndex = lookUpOffset++;
#if SERIALIZE_INSTANCES
                    instanceTypeTable.Create(name, entryIndex);
#endif
#if SERIALIZE_SHAPES
                    // For each object instance type containing quads, write the position data
                    // to a memory mapped file
                    // TODO: need to keep track of the material used for each
                    string filename = PushStr8F(temp.arena, "%Smeshes/%u.mesh", directory,
                                                entryIndex); // name);
                    filenames[entryIndex] = filename;
                    u8 *mappedPtr =
                        OS_MapFileWrite(filename, vertexCount * sizeof(Vec3f) * 2 +
                                                      quadMeshCount * sizeof(QuadMesh));
                    mappedPtrs[entryIndex] = mappedPtr;

                    QuadMesh *meshes   = (QuadMesh *)mappedPtr;
                    Vec3f *dataPtr     = (Vec3f *)(meshes + quadMeshCount);
                    u32 quadMeshOffset = 0;

                    const u32 flushSize = megabytes(64);
                    u8 *flushBase       = (u8 *)dataPtr;

                    LookupEntry *lookUpPtr   = &lookUpStart[entryIndex];
                    lookUpPtr->name          = name;
                    lookUpPtr->quadMeshCount = quadMeshCount;
                    for (u32 instanceIndex = 0; instanceIndex < currentInstanceCount;
                         instanceIndex++)
                    {
                        ObjectInstanceType *currentObj = currentObject[instanceIndex];

                        u32 pIdx                    = pIndices[instanceIndex];
                        u32 *currentOffsetGroup     = offsets[pIdx];
                        PrimitiveTy *currentTyGroup = types[pIdx];
                        u32 shapeStart              = currentObj->shapeIndexStart;
                        currentObj->Invalidate();
                        for (u32 shapeIndex = shapeStart;
                             shapeIndex < currentObj->shapeIndexEnd; shapeIndex++)
                        {
                            if (currentTyGroup[shapeIndex] == P_QuadMesh)
                            {
                                currentTyGroup[shapeIndex] = P_RemovedTy;
                                QuadMesh *mesh = &qMeshes[currentOffsetGroup[shapeIndex]];
                                Assert(quadMeshOffset < quadMeshCount);
                                QuadMesh *newMesh = &meshes[quadMeshOffset++];

                                SerializeMeshes(dataPtr, mappedPtr, mesh, newMesh, entryIndex,
                                                pTable, nTable);
                                u32 size = u32((u8 *)dataPtr - flushBase);
                                if (size >= flushSize)
                                {
                                    OS_FlushMappedFile(flushBase, size);
                                    flushBase = (u8 *)dataPtr;
                                }
                            }
                        }
                    }

                    sizes[entryIndex] = u32((u8 *)dataPtr - mappedPtr);
#endif
                    currentInstanceCount = 0;
                }
            }
        }
    }

    printf("quad instanced total: %u\n", quadInstancedCount);
    printf("total num instance types: %u\n", lookUpOffset);
#if SERIALIZE_SHAPES
    for (u32 i = 0; i < lookUpOffset; i++)
    {
        OS_UnmapFile(mappedPtrs[i]);
        OS_ResizeFile(filenames[i], sizes[i]);
    }
#endif

    using InstanceList = ChunkedLinkedList<SceneInstance, 1024, MemoryType_Instance>;
    // Next, write the uninstanced meshes
    // NOTE: this is just the area lights
    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        ShapeTypeList *list    = &state->shapes[pIndex];
        InstanceList *instList = &state->instances[pIndex];
        PrimitiveTy *pTypes    = types[pIndex];
        u32 *pOffsets          = offsets[pIndex];
        u32 currentOffset      = 0;
        for (ShapeTypeList::ChunkNode *node = list->first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ScenePacket *packet = &node->values[i];
                if (pTypes[currentOffset] == P_QuadMesh)
                {
                    pTypes[currentOffset] = P_RemovedTy;

                    u32 entryIndex = lookUpOffset++;

#if SERIALIZE_INSTANCES
                    for (u32 parameterIndex = 0; parameterIndex < packet->parameterCount;
                         parameterIndex++)
                    {
                        if (packet->parameterNames[parameterIndex] == "Indices"_sid)
                        {
                            i32 *indices            = (i32 *)(packet->bytes[parameterIndex]);
                            i32 transformIndex      = indices[2];
                            SceneInstance &instance = instList->AddBack();
                            instance.name           = entryIndex;
                            instance.transformIndex = transformIndex;
                            instanceTypeTable.Create(instance.name, entryIndex);
                            break;
                        }
                    }
#endif

#if SERIALIZE_SHAPES
                    LookupEntry *lookUpPtr   = &lookUpStart[entryIndex];
                    lookUpPtr->name          = 0;
                    lookUpPtr->quadMeshCount = 1;
                    string filename =
                        PushStr8F(temp.arena, "%Smeshes/%u.mesh", directory, entryIndex);
                    QuadMesh *mesh = &qMeshes[pOffsets[currentOffset]];
                    u8 *mappedPtr  = OS_MapFileWrite(
                        filename, mesh->numVertices * sizeof(Vec3f) * 2 + sizeof(QuadMesh));

                    QuadMesh *newMesh = (QuadMesh *)mappedPtr;
                    Vec3f *dataPtr    = (Vec3f *)(newMesh + 1);
                    SerializeMeshes(dataPtr, mappedPtr, mesh, newMesh, entryIndex, pTable,
                                    nTable);

                    OS_UnmapFile(mappedPtr);
#endif
                }
                currentOffset++;
            }
        }
    }

#if SERIALIZE_SHAPES
    u8 *mappedPtr     = OS_MapFileWrite(PushStr8F(temp.arena, "%Smeshes/lut.mesh", directory),
                                        sizeof(LookupEntry) * lookUpOffset + sizeof(u64));
    *(u64 *)mappedPtr = lookUpOffset;
    LookupEntry *entries = (LookupEntry *)(mappedPtr + sizeof(u64));
    MemoryCopy(entries, lookUpStart, sizeof(LookupEntry) * lookUpOffset);
    OS_UnmapFile(mappedPtr);

    // convert pointers to offsets (for pointer fix ups) and write to file
    // StringBuilder builder = {};
    // builder.arena         = temp.arena;
    //
    // u64 triMeshFileOffset = AppendArray(&builder, triMeshes, totalNumTriMeshes);
    // Assert(triMeshFileOffset == 0);
    //
    // u64 *positionWrites = PushArrayNoZero(temp.arena, u64, totalNumTriMeshes);
    // u64 *normalWrites   = PushArray(temp.arena, u64, totalNumTriMeshes);
    // u64 *uvWrites       = PushArray(temp.arena, u64, totalNumTriMeshes);
    // u64 *indexWrites    = PushArray(temp.arena, u64, totalNumTriMeshes);
    // for (u32 i = 0; i < totalNumTriMeshes; i++)
    // {
    //     TriangleMesh *mesh = &triMeshes[i];
    //     positionWrites[i]  = AppendArray(&builder, mesh->p, mesh->numVertices);
    //     if (mesh->n)
    //     {
    //         normalWrites[i] = AppendArray(&builder, mesh->n, mesh->numVertices);
    //     }
    //     if (mesh->uv)
    //     {
    //         uvWrites[i] = AppendArray(&builder, mesh->uv, mesh->numVertices);
    //     }
    //     if (mesh->indices)
    //     {
    //         indexWrites[i] = AppendArray(&builder, mesh->indices, mesh->numIndices);
    //     }
    // }
    // string result = CombineBuilderNodes(&builder);
    // for (u32 i = 0; i < totalNumTriMeshes; i++)
    // {
    //     ConvertPointerToOffset(result.str, triMeshFileOffset + OffsetOf(TriangleMesh, p),
    //     positionWrites[i]); ConvertPointerToOffset(result.str, triMeshFileOffset +
    //     OffsetOf(TriangleMesh, n), normalWrites[i]); ConvertPointerToOffset(result.str,
    //     triMeshFileOffset + OffsetOf(TriangleMesh, uv), uvWrites[i]);
    //     ConvertPointerToOffset(result.str, triMeshFileOffset + OffsetOf(TriangleMesh,
    //     indices), indexWrites[i]); triMeshFileOffset += sizeof(TriangleMesh);
    // }
    // string triFilename = StrConcat(temp.arena, directory, "tris.mesh");
    // b32 success        = OS_WriteFile(triFilename, result.str, (u32)result.size);
    // if (!success)
    // {
    //     printf("Failed to write tri file");
    //     Assert(0);
    // }
#endif

#if SERIALIZE_INSTANCES
    ScratchEnd(temp);
    temp = ScratchStart(0, 0);

    u64 numInstances  = 0;
    u32 numTransforms = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        numInstances += state->instances[i].totalCount;
        numTransforms += state->transforms[i].totalCount;
    }

    size_t instFileSize      = sizeof(Instance) * numInstances + sizeof(numInstances);
    size_t transformFileSize = sizeof(AffineSpace) * numTransforms + sizeof(u32);

    string instFilename      = PushStr8F(temp.arena, "%Sinstances.data", directory);
    string transformFilename = PushStr8F(temp.arena, "%Stransforms.data", directory);

    const u32 flushSize  = megabytes(64);
    u8 *instanceFilePtr  = OS_MapFileWrite(instFilename, instFileSize);
    u8 *transformFilePtr = OS_MapFileWrite(transformFilename, transformFileSize);

    Instance *instances     = (Instance *)(instanceFilePtr + sizeof(numInstances));
    AffineSpace *transforms = (AffineSpace *)(transformFilePtr);

    Instance *instanceFlushBase     = instances;
    AffineSpace *transformFlushBase = transforms;

    u32 instanceOffset  = 0;
    u32 transformOffset = 0;
    using TransformList = ChunkedLinkedList<const AffineSpace *, 16384, MemoryType_Transform>;

    HashExt<const AffineSpace *, u32> transformHashTable(temp.arena, numTransforms, 0);

    for (u32 pIndex = 0; pIndex < numProcessors; pIndex++)
    {
        auto &list          = state->instances[pIndex];
        auto &transformList = state->transforms[pIndex];
        for (InstanceList::ChunkNode *node = list.first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                SceneInstance &instance = node->values[i];
                u32 index;
                bool found = instanceTypeTable.Find(instance.name, index);
                if (!found)
                {
                    printf("instance not found\n");
                    continue;
                }
                const AffineSpace *t = transformList[instance.transformIndex];

                Assert(instanceOffset < numInstances);
                Instance *outInstance = &instances[instanceOffset];
                outInstance->id       = index;

                if (transformHashTable.Find(t, index))
                {
                    outInstance->transformIndex = index;
                }
                else
                {
                    u32 transformIndex         = transformOffset++;
                    transforms[transformIndex] = *t;
                    transformHashTable.Create(t, transformIndex);
                    outInstance->transformIndex = transformIndex;
                    u64 size =
                        u64((u8 *)(&transforms[transformIndex]) - (u8 *)transformFlushBase);
                    if (size >= flushSize)
                    {
                        OS_FlushMappedFile(transformFlushBase, size);
                        transformFlushBase = &transforms[transformIndex];
                    }
                }
                instanceOffset++;

                u64 size = u64((u8 *)outInstance - (u8 *)instanceFlushBase);
                if (size >= flushSize)
                {
                    OS_FlushMappedFile(instanceFlushBase, size);
                    instanceFlushBase = outInstance;
                }
                size = u64((u8 *)(transforms + transformOffset) - (u8 *)transformFlushBase);
                if (size >= flushSize)
                {
                    OS_FlushMappedFile(transformFlushBase, size);
                    transformFlushBase = transforms + transformOffset;
                }
            }
        }
    }

    // NOTE: padded to support unaligned loads
    *(u64 *)instanceFilePtr = instanceOffset;
    u64 finalSize           = u64((u8 *)(instances + instanceOffset) - instanceFilePtr);
    OS_UnmapFile(instanceFilePtr);
    OS_ResizeFile(instFilename, finalSize);

    finalSize = u64((u8 *)(transforms + transformOffset) + sizeof(u32) - transformFilePtr);
    OS_UnmapFile(transformFilePtr);
    OS_ResizeFile(transformFilename, finalSize);
#endif
    ScratchEnd(temp);
}

void BuildTLASBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)
{
    TempArena temp = ScratchStart(0, 0);
    // for (u32 i = 0; i < numScenes; i++)
    // {
    //     Scene2 *childScene = childScenes[i];
    //     if (childScene->nodePtr.data == 0)
    //     {
    //         childScene->BuildBVH(arenas, settings, &numPrims[i]);
    //     }
    // }
    // build tlas
    RecordAOSSplits record(neg_inf);
    BRef *refs            = GenerateBuildRefs(scene, temp.arena, record);
    scene->nodePtr        = BuildTLASQuantized(settings, arenas, scene, refs, record);
    using IntersectorType = typename IntersectorHelper<Instance, BRef>::IntersectorType;
    scene->intersectFunc  = &IntersectorType::Intersect;
    scene->occludedFunc   = &IntersectorType::Occluded;

    scene->SetBounds(Bounds(record.geomBounds));
    ScratchEnd(temp);
}

template <typename Mesh>
void BuildBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)

{
    TempArena temp = ScratchStart(0, 0);
    Mesh *meshes   = (Mesh *)scene->primitives;
    RecordAOSSplits record(neg_inf);

    u32 totalNumFaces = 0;
    if (scene->numPrimitives > 1)
    {
        PrimRef *refs;
        u32 extEnd;
        if (scene->numPrimitives > PARALLEL_THRESHOLD)
        {
            ParallelForOutput output =
                ParallelFor<u32>(temp, 0, scene->numPrimitives, PARALLEL_THRESHOLD,
                                 [&](u32 &faceCount, u32 jobID, u32 start, u32 count) {
                                     u32 outCount = 0;

                                     for (u32 i = start; i < start + count; i++)
                                     {
                                         Mesh &mesh = meshes[i];
                                         outCount += mesh.GetNumFaces();
                                     }
                                     faceCount = outCount;
                                 });
            Reduce(totalNumFaces, output, [&](u32 &l, const u32 &r) { l += r; });

            u32 offset   = 0;
            u32 *offsets = (u32 *)output.out;
            for (u32 i = 0; i < output.num; i++)
            {
                u32 numFaces = offsets[i];
                offsets[i]   = offset;
                offset += numFaces;
            }
            Assert(totalNumFaces == offset);
            extEnd = u32(totalNumFaces * GROW_AMOUNT);

            // Generate PrimRefs
            refs = PushArrayNoZero(temp.arena, PrimRef, extEnd);

            ParallelReduce<RecordAOSSplits>(
                &record, 0, scene->numPrimitives, PARALLEL_THRESHOLD,
                [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                    GenerateMeshRefs(meshes, refs, offsets[jobID],
                                     jobID == output.num - 1 ? totalNumFaces
                                                             : offsets[jobID + 1],
                                     start, count, record);
                },
                [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
        }
        else
        {
            for (u32 i = 0; i < scene->numPrimitives; i++)
            {
                Mesh &mesh = meshes[i];
                totalNumFaces += mesh.GetNumFaces();
            }
            extEnd = u32(totalNumFaces * GROW_AMOUNT);
            refs   = PushArrayNoZero(temp.arena, PrimRef, extEnd);
            GenerateMeshRefs(meshes, refs, 0, totalNumFaces, 0, scene->numPrimitives, record);
        }
        record.SetRange(0, totalNumFaces, extEnd);
        scene->nodePtr = BuildQuantizedSBVH<Mesh>(settings, arenas, scene, refs, record);
        using IntersectorType = typename IntersectorHelper<Mesh, PrimRef>::IntersectorType;
        scene->intersectFunc  = &IntersectorType::Intersect;
        scene->occludedFunc   = &IntersectorType::Occluded;
    }
    else
    {
        totalNumFaces           = meshes->GetNumFaces();
        u32 extEnd              = u32(totalNumFaces * GROW_AMOUNT);
        PrimRefCompressed *refs = PushArrayNoZero(temp.arena, PrimRefCompressed, extEnd);
        GenerateMeshRefs<PrimRefCompressed>(meshes, refs, 0, totalNumFaces, 0, 1, record);
        record.SetRange(0, totalNumFaces, extEnd);
        scene->nodePtr = BuildQuantizedSBVH<Mesh>(settings, arenas, scene, refs, record);
        using IntersectorType =
            typename IntersectorHelper<Mesh, PrimRefCompressed>::IntersectorType;
        scene->intersectFunc = &IntersectorType::Intersect;
        scene->occludedFunc  = &IntersectorType::Occluded;
    }
    scene->SetBounds(Bounds(record.geomBounds));
    scene->numFaces = totalNumFaces;
    ScratchEnd(temp);
}

void BuildTriangleBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)
{
    BuildBVH<TriangleMesh>(arenas, settings, scene);
}

void BuildQuadBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)
{
    BuildBVH<QuadMesh>(arenas, settings, scene);
}

template <typename PrimRef, typename Mesh>
void GenerateMeshRefs(Mesh *meshes, PrimRef *refs, u32 offset, u32 offsetMax, u32 start,
                      u32 count, RecordAOSSplits &record)
{
    RecordAOSSplits r(neg_inf);
    for (u32 i = start; i < start + count; i++)
    {
        Mesh *mesh = &meshes[i];

        u32 numFaces = mesh->GetNumFaces();
        RecordAOSSplits tempRecord(neg_inf);
        if (numFaces > PARALLEL_THRESHOLD)
        {
            ParallelReduce<RecordAOSSplits>(
                &tempRecord, 0, numFaces, PARALLEL_THRESHOLD,
                [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                    Assert(offset + start < offsetMax);
                    mesh->GenerateMeshRefs(refs, offset + start, i, start, count, record);
                },
                [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
        }
        else
        {
            Assert(offset < offsetMax);
            mesh->GenerateMeshRefs(refs, offset, i, 0, numFaces, tempRecord);
        }
        r.Merge(tempRecord);
        offset += numFaces;
    }
    Assert(offsetMax == offset);
    record = r;
}

struct alignas(CACHE_LINE_SIZE) FilePtr
{
    u8 *ptr;
};

void LoadFile(Arena *arena, const LookupEntry *entries, const u32 fileID,
              const string meshDirectory, FilePtr *ptrs, Mutex *mutexes)
{
    TempArena temp = ScratchStart(0, 0);
    if (ptrs[fileID].ptr == 0)
    {
        const LookupEntry *entry = &entries[fileID];
        string filename          = PushStr8F(temp.arena, "%S%u.mesh", meshDirectory, fileID);
        BeginMutex(&mutexes[fileID]);
        if (ptrs[fileID].ptr == 0)
        {
            string data      = OS_ReadFile(arena, filename);
            ptrs[fileID].ptr = data.str;
        }
        EndMutex(&mutexes[fileID]);
    }
    ScratchEnd(temp);
}

u32 FixQuadMeshPointers(Arena *arena, QuadMesh *meshes, const LookupEntry *entries,
                        const string meshDirectory, FilePtr *ptrs, Mutex *mutexes,
                        const u32 start, const u32 count)
{
    TempArena temp = ScratchStart(0, 0);
    u32 vertCount  = 0;
    for (u32 i = start; i < start + count; i++)
    {
        QuadMesh *mesh = &meshes[i];
        u64 pOffset    = u64(mesh->p);
        u32 fileID     = pOffset & 0xffffffff;
        LoadFile(arena, entries, fileID, meshDirectory, ptrs, mutexes);

        u32 offset = (pOffset >> 32) & 0xffffffff;
        mesh->p    = (Vec3f *)(ptrs[fileID].ptr + offset);

        u64 nOffset = u64(mesh->n);
        if (nOffset)
        {
            fileID = nOffset & 0xffffffff;
            offset = (nOffset >> 32) & 0xffffffff;
            LoadFile(arena, entries, fileID, meshDirectory, ptrs, mutexes);
            mesh->n = (Vec3f *)(ptrs[fileID].ptr + offset);
        }

        vertCount += mesh->numVertices;
    }
    ScratchEnd(temp);
    return vertCount;
}

#if 0
Scene2 **InitializeScene(Arena **arenas, string meshDirectory, string instanceFile,
                         string transformFile)
{
    TempArena temp = ScratchStart(0, 0);
    Arena *arena   = arenas[GetThreadIndex()];

    // Read the mesh file data
    string lutPath = StrConcat(temp.arena, meshDirectory, "lut.mesh");
    string lutData = OS_ReadFile(temp.arena, lutPath);
    Tokenizer tokenizer(lutData);
    u64 numEntries;
    GetPointerValue(&tokenizer, &numEntries);
    LookupEntry *entries = (LookupEntry *)tokenizer.cursor;

    FilePtr *ptrs  = PushArray(temp.arena, FilePtr, numEntries);
    Mutex *mutexes = PushArray(temp.arena, Mutex, numEntries);

    u32 *sceneNumPrims = PushArray(temp.arena, u32, numEntries);
    Scene2 **out       = PushArray(arena, Scene2 *, numEntries);

    // Fix geometry pointers, start bottom level BVH builds
    // PerformanceCounter counter = OS_StartCounter();

    Scheduler::Counter jobCounter = {};
    scheduler.Schedule(&jobCounter, u32(numEntries), 1, [&](u32 jobID) {
        TempArena temp = ScratchStart(0, 0);
        Arena *arena   = arenas[GetThreadIndex()];
        LoadFile(arena, entries, jobID, meshDirectory, ptrs, mutexes);
        QuadMesh *meshes   = (QuadMesh *)(ptrs[jobID].ptr);
        LookupEntry *entry = &entries[jobID];
        out[jobID]         = PushStruct(arena, Scene2Quad);
        Scene2Quad *group  = (Scene2Quad *)out[jobID];
        // TODO: I changed the quad mesh format so this no longer works. putting this here in
        // case I forget
        Assert(0);
        group->primitives    = meshes;
        group->numPrimitives = entry->quadMeshCount;
        Assert(group->numPrimitives > 0);
        u32 totalVertexCount = 0;

        BuildSettings settings;
        settings.intCost = 1.f;
        if (entry->quadMeshCount > PARALLEL_THRESHOLD)
        {
            ParallelForOutput output =
                ParallelFor<u32>(temp, 0, entry->quadMeshCount, PARALLEL_THRESHOLD,
                                 [&](u32 &vertexCount, u32 jobID, u32 start, u32 count) {
                                     vertexCount = FixQuadMeshPointers(
                                         arenas[GetThreadIndex()], meshes, entries,
                                         meshDirectory, ptrs, mutexes, start, count);
                                 });
            Reduce(totalVertexCount, output, [&](u32 &l, const u32 &r) { l += r; });

            u32 offset   = 0;
            u32 *offsets = (u32 *)output.out;
            for (u32 i = 0; i < output.num; i++)
            {
                u32 numVertices = offsets[i];
                u32 numFaces    = numVertices / 4;
                offsets[i]      = offset;
                offset += numFaces;
            }
            u32 numFaces = totalVertexCount / 4;
            Assert(numFaces == offset);
            u32 extEnd = u32(numFaces * GROW_AMOUNT);
            RecordAOSSplits record(neg_inf);

            // Generate PrimRefs
            PrimRef *refs = PushArrayNoZero(temp.arena, PrimRef, extEnd);

            ParallelReduce<RecordAOSSplits>(
                &record, 0, entry->quadMeshCount, PARALLEL_THRESHOLD,
                [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                    GenerateQuadRefs(meshes, refs, offsets[jobID],
                                     jobID == output.num - 1 ? numFaces : offsets[jobID + 1],
                                     start, count, record);
                },
                [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });

            group->SetBounds(record.geomBounds);
            record.SetRange(0, numFaces, extEnd);
            sceneNumPrims[jobID] = numFaces;

            // printf("start num faces: %u\n", numFaces);
            // PerformanceCounter counter = OS_StartCounter();
            group->nodePtr = BuildQuantizedSBVH(settings, arenas, group, refs, record);
            // printf("build time: %fms\nnum faces: %u\n\n", OS_GetMilliseconds(counter),
            // numFaces);

            // threadLocalStatistics[GetThreadIndex()].misc += numFaces;
        }
        else
        {
            totalVertexCount = FixQuadMeshPointers(arena, meshes, entries, meshDirectory, ptrs,
                                                   mutexes, 0, entry->quadMeshCount);

            u32 numFaces = totalVertexCount / 4;
            u32 extEnd   = u32(numFaces * GROW_AMOUNT);
            RecordAOSSplits record(neg_inf);

            if (entry->quadMeshCount > 1)
            {
                PrimRef *refs = PushArrayNoZero(temp.arena, PrimRef, extEnd);
                GenerateQuadRefs(meshes, refs, 0, numFaces, 0, entry->quadMeshCount, record);
                group->SetBounds(record.geomBounds);
                record.SetRange(0, numFaces, extEnd);
                sceneNumPrims[jobID] = numFaces;

                // printf("start num faces: %u\n", numFaces);
                // PerformanceCounter counter = OS_StartCounter();
                group->nodePtr = BuildQuantizedSBVH(settings, arenas, group, refs, record);
                // printf("build time: %fms\nnum faces: %u\n\n", OS_GetMilliseconds(counter),
                // numFaces);
            }
            else
            {
                PrimRefCompressed *refs =
                    PushArrayNoZero(temp.arena, PrimRefCompressed, extEnd + 1);
                GenerateQuadRefs(refs, 0, &group->primitives[0], 0, 0, numFaces, record);
                group->SetBounds(record.geomBounds);
                record.SetRange(0, numFaces, extEnd);
                sceneNumPrims[jobID] = numFaces;

                // printf("start num faces: %u\n", numFaces);
                // PerformanceCounter counter = OS_StartCounter();
                group->nodePtr = BuildQuantizedSBVH(settings, arenas, group, refs, record);
            }
            // threadLocalStatistics[GetThreadIndex()].misc += numFaces;
        }

        ScratchEnd(temp);
    });

    // printf("blas build time: %fms\n", OS_GetMilliseconds(counter));

    Scene *scene = GetScene();
    Assert(scene);
    scheduler.Schedule(&jobCounter, [&](u32 jobID) {
        string instanceFileData  = OS_ReadFile(arena, instanceFile);
        string transformFileData = OS_ReadFile(arena, transformFile);
        Tokenizer instTokenzier(instanceFileData);
        u64 numInstances;
        GetPointerValue(&instTokenzier, &numInstances);
        scene->instances        = (Instance *)instTokenzier.cursor;
        scene->numInstances     = u32(numInstances);
        scene->affineTransforms = (AffineSpace *)transformFileData.str;
    });
    scheduler.Wait(&jobCounter);

    ScratchEnd(temp);
    scene->numScenes = SafeTruncateU64ToU32(numEntries);
    // numScenes            = numEntries + 1;
    return out;
}
#endif

} // namespace rt
