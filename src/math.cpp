AABB::AABB()
{
    minX = infinity;
    minY = infinity;
    minZ = infinity;
    maxX = -infinity;
    maxY = -infinity;
    maxZ = -infinity;
}
AABB::AABB(vec3 pt1, vec3 pt2)
{
    minX = pt1.x <= pt2.x ? pt1.x : pt2.x;
    minY = pt1.y <= pt2.y ? pt1.y : pt2.y;
    minZ = pt1.z <= pt2.z ? pt1.z : pt2.z;

    maxX = pt1.x >= pt2.x ? pt1.x : pt2.x;
    maxY = pt1.y >= pt2.y ? pt1.y : pt2.y;
    maxZ = pt1.z >= pt2.z ? pt1.z : pt2.z;
    PadToMinimums();
}
AABB::AABB(AABB box1, AABB box2)
{
    minX = box1.minX <= box2.minX ? box1.minX : box2.minX;
    minY = box1.minY <= box2.minY ? box1.minY : box2.minY;
    minZ = box1.minZ <= box2.minZ ? box1.minZ : box2.minZ;

    maxX = box1.maxX >= box2.maxX ? box1.maxX : box2.maxX;
    maxY = box1.maxY >= box2.maxY ? box1.maxY : box2.maxY;
    maxZ = box1.maxZ >= box2.maxZ ? box1.maxZ : box2.maxZ;
    PadToMinimums();
}
bool AABB::Hit(const Ray &r, f32 tMin, f32 tMax)
{
    for (int axis = 0; axis < 3; axis++)
    {
        f32 oneOverDir = 1.f / r.direction().e[axis];
        f32 t0         = (minP[axis] - r.origin()[axis]) * oneOverDir;
        f32 t1         = (maxP[axis] - r.origin()[axis]) * oneOverDir;
        if (t0 > t1)
        {
            f32 temp = t0;
            t0       = t1;
            t1       = temp;
        }
        tMin = t0 > tMin ? t0 : tMin;
        tMax = t1 < tMax ? t1 : tMax;
        if (tMax <= tMin)
            return false;
    }
    return true;
}

bool AABB::Hit(const Ray &r, f32 tMin, f32 tMax, const int dirIsNeg[3]) const
{
    for (int axis = 0; axis < 3; axis++)
    {
        f32 min = (*this)[dirIsNeg[axis]][axis];
        f32 max = (*this)[1 - dirIsNeg[axis]][axis];

        f32 oneOverDir = 1.f / r.direction().e[axis];
        f32 t0         = (min - r.origin()[axis]) * oneOverDir;
        f32 t1         = (max - r.origin()[axis]) * oneOverDir;
        tMin           = t0 > tMin ? t0 : tMin;
        tMax           = t1 < tMax ? t1 : tMax;
        if (tMax <= tMin)
            return false;
    }
    return true;
}
