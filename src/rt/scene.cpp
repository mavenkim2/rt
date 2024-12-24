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

// NOTE: specifically for the moana island scene

#if 0
template <typename Mesh>
void LoadMesh(SceneLoader *loader, ScenePrimitives *scene, Tokenizer *tokenizer)
{
    Mesh *meshes = PushArray(arena, Mesh, numMeshes);
    bool done    = false;

    Mesh *mesh = &meshes[0];
    while (!done)
    {
        switch (tokenizer->cursor[0])
        {
            case 'P':
            {
                tokenizer->cursor++;
                u64 offset;
                u32 numVertices;
                GetPointerValue(tokenizer, &offset);
                GetPointerValue(tokenizer, &numVertices);
                mesh->p           = (Vec3f *)(data.str + offset);
                mesh->numVertices = numVertices;
            }
            break;
            case 'N':
            {
                tokenizer->cursor++;
                u64 offset;
                GetPointerValue(tokenizer, &offset);
                mesh->n = (Vec3f *)(data.str + offset);
            }
            break;
            case 'U':
            {
                tokenizer->cursor++;
                u64 offset;
                GetPointerValue(tokenizer, &offset);
                mesh->uv = (Vec2f *)(data.str + offset);
            }
            break;
            case 'I':
            {
                tokenizer->cursor++;
                u64 offset;
                u32 count;
                GetPointerValue(tokenizer, &offset);
                GetPointerValue(tokenizer, &count);
                mesh->indices    = (u32 *)(data.str + offset);
                mesh->numIndices = count;
            }
            break;
            case 'X':
            {
                tokenizer->cursor++;
                done = true;
            }
            break;
            case '_':
            {
                tokenizer->cursor++;
            }
            break;
            default:
            {
                Error(0, "Invalid token read while parsing Mesh. "
                         "Exiting...\n");
            }
        }
    }
    BuildSettings settings;
    BuildBVH<Mesh>(loader->arenas, settings, scene);
}

void LoadScene(SceneLoader *loader, Arena *arena, string baseDirectory, string filename,
               ScenePrims *scene, bool top = false)
{
    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(filename);
    tokenizer.cursor = tokenizer.input.str;

    bool result = Advance(&tokenizer, "RTF_Start");
    Assert(result);

    // File format:
    // RTF_Start
    // Offset to permanent data:

    u64 permDataOffset;
    GetPointerValue(&tokenizer, &permDataOffset);

    string data = OS_ReadFileOffset(permArena, filename, permDataOffset);

    struct FileData
    {
        TriangleMesh *triangleMeshes;
        QuadMesh *quadMeshes;
        u32 numTriMeshes;
        u32 numQuadMeshes;
    };

    struct Parser
    {
        u32 triOffset  = 0;
        u32 quadOffset = 0;
    };
    FileData fileData;
    Parser parser;

    // NOTE IMPORTANT: OFFSETS ARE FROM THE START OF THE PERMANENT DATA SECTION
    // Handle permanent data
    for (;;)
    {
        switch (tokenizer.cursor[0])
        {
            case 'I':
            {
                if (Advance(&tokenizer, "Import"))
                {
                    u32 size;
                    GetPointerValue(&tokenizer, &size);
                    string relFile;
                    relFile.str         = tokenizer.cursor;
                    relFile.size        = size;
                    string fullFilePath = StrConcat(arena, baseDirectory, relFile);
                    scheduler.
                }
                else if (Advance(&tokenizer, "Instances"))
                {
                    u32 count;
                    GetPointerValue(&tokenizer, &count);
                    Instance *instances = PushArray(arena, Instance, count);
                    for (u32 i = 0; i < count; i++)
                    {
                        Instance *instance = &instances[i];
                        u32 transformIndex, childSceneIndex;
                        GetPointerValue(&tokenizer, &childSceneIndex);
                        GetPointerValue(&tokenizer, &transformIndex);
                        instance->id             = childSceneIndex;
                        instance->transformIndex = transformIndex;
                    }
                    scene->primitives    = instances;
                    scene->numPrimitives = count;
                }
            }
            break;
            case 'M':
            {
                if (Advance(&tokenizer, "Material"))
                {
                    // TODO: maybe materials should be allowed to be specified in multiple
                    // files?
                    if (!top)
                    {
                        Error(0, "Material specifications can only be present in the base "
                                 "input file, not in any included files.\n");
                        exit(0);
                    }
                    LoadMaterial(loader);
                }
            }
            break;
            case 'Q':
            {
                if (Advance(&tokenizer, "QuadMesh"))
                {
                    u32 numMeshes;
                    GetPointerValue(&tokenizer, &numMeshes);
                    LoadMesh(loader, scene, &tokenizer);
                }
            }
            break;
            case 'T':
            {
                if (Advance(&tokenizer, "TriangleMesh"))
                {
                    u32 numMeshes;
                    GetPointerValue(&tokenizer, &numMeshes);
                    LoadMesh(loader, scene, &tokenizer);
                }
                else if (Advance(&tokenizer, "Transforms"))
                {
                    u64 offset;
                    GetPointerValue(&tokenizer, &offset);

                    const AffineSpace *transforms = (const AffineSpace *)(data.str + offset);
                    scene->affineTransforms       = transforms;
                }
            }
            break;
        }
    }
}
#endif

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

#if 0
void CreatePBRTScene(Arena *arena, string directory, SceneLoadState *state)
{
    scene_            = PushStruct(arena, Scene);
    Scene *scene      = GetScene();
    TempArena temp    = ScratchStart(0, 0);
    u32 numProcessors = OS_NumProcessors();

    u32 *quadOffsets = PushArrayNoZero(temp.arena, u32, numProcessors + 1);
    u32 *triOffsets  = PushArrayNoZero(temp.arena, u32, numProcessors + 1);
    u32 *instOffsets = PushArrayNoZero(temp.arena, u32, numProcessors + 1);

    u32 totalNumQuadMeshes = 0;
    u32 totalNumTriMeshes  = 0;
    u32 totalNumInstTypes  = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        quadOffsets[i] = totalNumQuadMeshes;
        triOffsets[i]  = totalNumTriMeshes;
        instOffsets[i] = totalNumInstTypes;

        totalNumQuadMeshes += state->numQuadMeshes[i];
        totalNumTriMeshes += state->numTriMeshes[i];
        totalNumInstTypes += state->totalNumInstTypes[i];
    }
    quadOffsets[numProcessors] = totalNumQuadMeshes;
    triOffsets[numProcessors]  = totalNumTriMeshes;

    // NOTE: no support for multi level instancing with this format
    ScenePrimitives *quadScene = PushStruct(arena, ScenePrimitives);
    ScenePrimitives *triScene  = PushStruct(arena, ScenePrimitives);

    ScenePrimitives *scenePrims =
        PushArray(temp.arena, ScenePrimitives *, totalNumInstTypes + 1);

    QuadMesh *qMeshes       = PushArray(temp.arena, QuadMesh, totalNumQuadMeshes);
    TriangleMesh *triMeshes = PushArray(temp.arena, TriangleMesh, totalNumTriMeshes);
    quadScene->primitives   = qMeshes;
    triScene->primitives    = triMeshes;

    using InstanceTypeList = ChunkedLinkedList<ObjectInstanceType, 512, MemoryType_Instance>;
    using ShapeTypeList    = ChunkedLinkedList<ScenePacket, 1024, MemoryType_Shape>;

    Scheduler::Counter counter = {};
    // the quad meshes don't know what object instance they are a part
    // of, though they probably should

    // what should the end game be?
    // you have the name for the instance type. you need the transform
    // index and scene id could have an atomic for scene id whenever an
    // object instance type is declared

    // map the instance name to an index;
    auto CreateShapes = [](ShapeTypeList *shapeList, QuadMesh *qMeshes,
                           TriangleMesh *triMeshes, u32 start, u32 end, u32 quadOffset = 0,
                           u32 triOffset = 0) {
        u32 quadOffset = 0;
        u32 triOffset  = 0;
        for (auto itr = shapeList->Itr(start, end); !itr.End(); itr.Next())
        {
            ScenePacket *shapePacket = itr.Get();

            switch (shapePacket->types)
            {
                case "quadmesh"_sid:
                {
                    u32 quadIndex  = quadOffset++;
                    QuadMesh *mesh = &qMeshes[quadIndex];
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
                            }
                            break;
                            case "N"_sid:
                            {
                                mesh->n = (Vec3f *)packet->bytes[parameterIndex];
                                Assert(mesh->numVertices ==
                                       packet->sizes[parameterIndex] / sizeof(Vec3f));
                            }
                            break;
                            default: continue;
                        }
                    }
                }
                break;
                case "trianglemesh"_sid:
                {
                    u32 triIndex = triOffset++;
                    Assert(triIndex < triLimit);
                    TriangleMesh *mesh = &triMeshes[triIndex++];
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
                                mesh->indices    = (u32 *)packet->bytes[parameterIndex];
                                mesh->numIndices = packet->sizes[parameterIndex] / sizeof(u32);
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
                            filename.str        = packet->bytes[parameterIndex];
                            filename.size       = packet->sizes[parameterIndex];
                            string fullFilePath = StrConcat(temp.arena, directory, filename);
                            if (CheckQuadPLY(fullFilePath))
                            {
                                qMeshes[quadOffset++] = LoadQuadPLY(permArena, fullFilePath);
                            }
                            else
                            {
                                triMeshes[triOffset++] = LoadPLY(permArena, fullFilePath);
                            }
                            break;
                        }
                    }
                }
                break;
            }
        }
    };

    // Create object types
    scheduler.Schedule(&counter, numProcessors, 1, [&](u32 jobID) {
        u32 pIndex                        = jobID;
        InstanceTypeList *list            = &state->instanceTypes[pIndex];
        ShapeTypeList *shapeList          = &state->instanceShapes[pIndex];
        ObjectInstanceTypeCounts *tCounts = &counts[jobID];
        Arena *permArena                  = state->permArenas[jobID];
        u32 instOffset                    = state->instOffsets[jobID];
        for (InstanceTypeList::ChunkNode *node = list->first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                ObjectInstanceTypeCounts counts = {};
                ObjectInstanceType *instType    = &node->values[i];
                StringId name                   = instType->name;
                u32 quadCount = instType->shapeTypeCount[(u32)GeometryType::QuadMesh];
                u32 triCount  = instType->shapeTypeCount[(u32)GeometryType::TriangleMesh];
                QuadMesh *qMeshes;
                if (quadCount)
                {
                    qMeshes = PushArray(permArena, QuadMesh, quadCount)
                }
                TriangleMesh *triMeshes;
                if (triCount)
                {
                    triMeshes = PushArray(permArena, TriangleMesh, triCount);
                }

                u32 quadOffset = 0;
                u32 triOffset  = 0;

                CreateShapes(shapeList, qMeshes, triMeshes, instType->shapeIndexStart,
                             instType->shapeIndexEnd);
            }
        }
    });

    // Instantiate instances
    scheduler.Schedule(&counter, numProcessors, 1, [&](u32 jobID) {
        u32 instOffset    = instOffsets[jobID];
        auto instanceList = state->instances[jobID];
        for (auto node = instanceList->first; node != 0; node = node->next)
        {
            for (u32 i = 0; i < node->count; i++)
            {
                SceneInstance *sceneInstance = &node->values[i];
                Instance *instance           = &instances[instOffset++];
                instance->id                 = ? ;
                instance->transformIndex     = ? ;
            }
        }
    });

    // Create uninstantiated shapes
    Arena *permArena     = state->permArenas[GetThreadIndex()];
    QuadMesh *quadMeshes = PushArray(permArena, QuadMesh, totalNumQuadMeshes);
    QuadMesh *triMeshes  = PushArray(permArena, TriangleMesh, totalNumTriMeshes);
    scheduler.Schedule(&counter, numProcessors, 1, [&](u32 jobID) {
        u32 quadOffset      = quadOffsets[jobID];
        u32 triOffset       = triOffsets[jobID];
        Arena *arena        = state->threadArenas[jobID];
        ShapeTypeList *list = &state->shapes[jobID];
        CreateShapes(shapeList, quadMeshes, triMeshes, 0, list->totalCount, quadOffset,
                     triOffset);
    });
    scheduler.Wait(&counter);
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
                                // NOTE: this is specific to the moana
                                // island data set (not needing the
                                // indices or uvs)
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
                                // NOTE: should only happen for the
                                // ocean geometry (twice)
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
                // NOTE: have to multiply by the object type transform
                // in general case (but for the moana scene desc, none
                // of the object types have a transform)
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
                    // For each object instance type containing quads,
                    // write the position data to a memory mapped file
                    // TODO: need to keep track of the material used
                    // for each
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

    // convert pointers to offsets (for pointer fix ups) and write to
    // file StringBuilder builder = {}; builder.arena         =
    // temp.arena;
    //
    // u64 triMeshFileOffset = AppendArray(&builder, triMeshes,
    // totalNumTriMeshes); Assert(triMeshFileOffset == 0);
    //
    // u64 *positionWrites = PushArrayNoZero(temp.arena, u64,
    // totalNumTriMeshes); u64 *normalWrites   = PushArray(temp.arena,
    // u64, totalNumTriMeshes); u64 *uvWrites       =
    // PushArray(temp.arena, u64, totalNumTriMeshes); u64 *indexWrites
    // = PushArray(temp.arena, u64, totalNumTriMeshes); for (u32 i = 0;
    // i < totalNumTriMeshes; i++)
    // {
    //     TriangleMesh *mesh = &triMeshes[i];
    //     positionWrites[i]  = AppendArray(&builder, mesh->p,
    //     mesh->numVertices); if (mesh->n)
    //     {
    //         normalWrites[i] = AppendArray(&builder, mesh->n,
    //         mesh->numVertices);
    //     }
    //     if (mesh->uv)
    //     {
    //         uvWrites[i] = AppendArray(&builder, mesh->uv,
    //         mesh->numVertices);
    //     }
    //     if (mesh->indices)
    //     {
    //         indexWrites[i] = AppendArray(&builder, mesh->indices,
    //         mesh->numIndices);
    //     }
    // }
    // string result = CombineBuilderNodes(&builder);
    // for (u32 i = 0; i < totalNumTriMeshes; i++)
    // {
    //     ConvertPointerToOffset(result.str, triMeshFileOffset +
    //     OffsetOf(TriangleMesh, p), positionWrites[i]);
    //     ConvertPointerToOffset(result.str, triMeshFileOffset +
    //     OffsetOf(TriangleMesh, n), normalWrites[i]);
    //     ConvertPointerToOffset(result.str, triMeshFileOffset +
    //     OffsetOf(TriangleMesh, uv), uvWrites[i]);
    //     ConvertPointerToOffset(result.str, triMeshFileOffset +
    //     OffsetOf(TriangleMesh, indices), indexWrites[i]);
    //     triMeshFileOffset += sizeof(TriangleMesh);
    // }
    // string triFilename = StrConcat(temp.arena, directory,
    // "tris.mesh"); b32 success        = OS_WriteFile(triFilename,
    // result.str, (u32)result.size); if (!success)
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
#endif

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
