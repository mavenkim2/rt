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
    n        = normalize(n);
    vec3 up  = fabs(n.x) > 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 t   = normalize(cross(n, up));
    vec3 b   = cross(n, t);
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
    // (C - P) dot (C - P) = r^2
    // (C - (O + Dt)) dot (C - (O + Dt)) - r^2 = 0
    // (-Dt + C - O) dot (-Dt + C - O) - r^2 = 0
    // t^2(D dot D) - 2t(D dot (C - O)) + (C - O dot C - O) - r^2 = 0
    vec3 oc = Center(r.time()) - r.origin();
    f32 a   = dot(r.direction(), r.direction());
    f32 h   = dot(r.direction(), oc);
    f32 c   = dot(oc, oc) - radius * radius;

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
vec3 Sphere::Random(const vec3 &origin) const
{
    vec3 dir            = center - origin;
    f32 distanceSquared = dir.lengthSquared();
    Basis basis         = GenerateBasis(dir);

    f32 r1 = RandomFloat();
    f32 r2 = RandomFloat();
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

    if (primitiveIndices[index].transformIndex != -1)
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
