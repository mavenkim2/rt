#include "../hash.h"
#include "../math/math_include.h"
#include "../math/ray.h"
#include "../math/eigen.h"
#include "../thread_context.h"
#include "../dgfs.h"
#include "../radix_sort.h"
#include <atomic>
#include "../mesh.h"
#include "../sampling.h"
#include <cstring>
#include <type_traits>
#include "../scene/scene.h"
#include "../scene_load.h"
#include "../bvh/bvh_types.h"
#include "../bvh/bvh_aos.h"
#include "../parallel.h"
#include "mesh_simplification.h"
#include "../shader_interop/as_shaderinterop.h"
#include "../../third_party/METIS/include/metis.h"

namespace rt
{

struct ClusterFixup
{
    u32 pageIndex_clusterIndex;
    u32 pageStartIndex_numPages;

    ClusterFixup(u32 pageIndex, u32 clusterIndex, u32 pageStartIndex, u32 numPages)
    {
        pageIndex_clusterIndex  = (pageIndex << MAX_CLUSTERS_PER_PAGE_BITS) | clusterIndex;
        pageStartIndex_numPages = (pageStartIndex << MAX_PARTS_PER_GROUP_BITS) | numPages;
    }
    u32 GetPageIndex() { return pageIndex_clusterIndex >> MAX_CLUSTERS_PER_PAGE_BITS; }
    u32 GetClusterIndex() { return pageIndex_clusterIndex & (MAX_CLUSTERS_PER_PAGE - 1); }
};

// https://en.wikipedia.org/wiki/LU_decomposition
template <typename T>
int LUPDecompose(T *A, int N, T Tol, int *P)
{
    for (int i = 0; i < N; i++) P[i] = i; // Unit permutation matrix, P[N] initialized with N

    for (int i = 0; i < N; i++)
    {
        T maxA   = 0;
        int imax = i;

        for (int k = i; k < N; k++)
        {
            T absA = Abs(A[N * k + i]);
            if (absA > maxA)
            {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0; // failure, matrix is degenerate

        if (imax != i)
        {
            // pivoting P
            Swap(P[i], P[imax]);

            // pivoting rows of A
            for (int j = 0; j < N; j++)
            {
                Swap(A[N * i + j], A[N * imax + j]);
            }
        }

        for (int j = i + 1; j < N; j++)
        {
            A[N * j + i] /= A[N * i + i];

            for (int k = i + 1; k < N; k++) A[N * j + k] -= A[N * j + i] * A[N * i + k];
        }
    }

    return 1; // decomposition done
}

/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
template <typename T>
void LUPSolve(T *A, int *P, T *b, int N, T *x)
{
    for (int i = 0; i < N; i++)
    {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++) x[i] -= A[N * i + k] * x[k];
    }

    for (int i = N - 1; i >= 0; i--)
    {
        for (int k = i + 1; k < N; k++) x[i] -= A[N * i + k] * x[k];

        x[i] /= A[N * i + i];
    }
}

// Due to floating point inaccuracy, use residuals to minimize error
template <typename T>
bool LUPSolveIterate(T *A, T *LU, int *P, T *b, int N, T *x, u32 numIters)
{
    LUPSolve(LU, P, b, N, x);

    ScratchArena scratch;

    T *residual = (T *)PushArrayNoZero(scratch.temp.arena, u8, sizeof(T) * N);
    T *error    = (T *)PushArrayNoZero(scratch.temp.arena, u8, sizeof(T) * N);
    for (int iters = 0; iters < numIters; iters++)
    {
        // Calculate residual
        for (int i = 0; i < N; i++)
        {
            residual[i] = b[i];
            for (int j = 0; j < N; j++)
            {
                residual[i] -= A[N * i + j] * x[j];
            }
        }

        LUPSolve(LU, P, residual, N, error);

        f32 mse = 0.f;
        for (int i = 0; i < N; i++)
        {
            mse += Sqr(error[i]);
            x[i] += error[i];
        }
        if (mse < 1e-4f) return true;
    }

    return false;
}

__forceinline void OuterProduct(const Vec3f &v, f32 &a00, f32 &a01, f32 &a02, f32 &a11,
                                f32 &a12, f32 &a22)
{
    a00 += Sqr(v.x);
    a01 += v.x * v.y;
    a02 += v.x * v.z;

    a11 += Sqr(v.y);
    a12 += v.y * v.z;

    a22 += Sqr(v.z);
}

Quadric::Quadric()
{
    c00 = 0.f;
    c01 = 0.f;
    c02 = 0.f;

    c11 = 0.f;
    c12 = 0.f;

    c22 = 0.f;

    dn = Vec3f(0);

    d2 = 0.f;

    gVol = Vec3f(0);
    dVol = 0.f;

    area = 0.f;
}

Quadric::Quadric(const Vec3f &p0, const Vec3f &p1, const Vec3f &p2)
{
    Vec3f p01 = p1 - p0;
    Vec3f p02 = p2 - p0;

    Vec3f n = Cross(p01, p02);

    gVol = n;
    dVol = -Dot(gVol, p0);

    f32 length = Length(n);
    area       = 0.5f * length;

    if (length < 1e-8f)
    {
        return;
    }

    n /= length;

    c00 = 0.f;
    c01 = 0.f;
    c02 = 0.f;

    c11 = 0.f;
    c12 = 0.f;
    c22 = 0.f;
    OuterProduct(n, c00, c01, c02, c11, c12, c22);

    f32 distToPlane = -Dot(n, p0);
    dn              = distToPlane * n;
    d2              = Sqr(distToPlane);

    c00 *= area;
    c01 *= area;
    c02 *= area;

    c11 *= area;
    c12 *= area;
    c22 *= area;

    dn *= area;
    d2 *= area;
}

void CreateAttributeQuadric(Quadric &q, QuadricGrad *g, const Vec3f &p0, const Vec3f &p1,
                            const Vec3f &p2, f32 *attr0, f32 *attr1, f32 *attr2,
                            f32 *attributeWeights, u32 numAttributes)
{
    Vec3f p01 = p1 - p0;
    Vec3f p02 = p2 - p0;

    Vec3f n = Normalize(Cross(p01, p02));

    // Solve system of equations to find gradient for each attribute
    // (p1 - p0) * g = a1 - a0
    // (p2 - p0) * g = a2 - a0
    // n * g = 0

    f32 M[9] = {
        p01.x, p01.y, p01.z, p02.x, p02.y, p02.z, n.x, n.y, n.z,
    };
    f32 LU[9];
    MemoryCopy(LU, M, sizeof(LU));

    int pivots[3];
    bool isInvertible = LUPDecompose(LU, 3, 1e-12f, pivots);

    for (int i = 0; i < numAttributes; i++)
    {
        // s = Dot(g, p) + d
        f32 a0 = attributeWeights[i] * attr0[i];
        f32 a1 = attributeWeights[i] * attr1[i];
        f32 a2 = attributeWeights[i] * attr2[i];

        Vec3f grad(0);
        f32 b[3] = {
            a1 - a0,
            a2 - a0,
            0.f,
        };

        if (isInvertible) LUPSolveIterate(M, LU, pivots, b, 3, grad.e, 1);

        g[i].g = grad;
        g[i].d = a0 - Dot(grad, p0);

        OuterProduct(grad, q.c00, q.c01, q.c02, q.c11, q.c12, q.c22);

        q.dn += q.area * g[i].d * g[i].g;
        q.d2 += q.area * Sqr(g[i].d);
    }

    // Multiply quadric by area (in preparation to be summed by other faces)

    for (u32 i = 0; i < numAttributes; i++)
    {
        g[i].g *= q.area;
        g[i].d *= q.area;
    }
}

f32 EvaluateQuadric(const Vec3f &p, const Quadric &q)
{
    Vec3f r = q.dn;

    r.x += q.c01 * p.y;
    r.y += q.c12 * p.z;
    r.z += q.c02 * p.x;

    r *= 2.f;

    r.x += q.c00 * p.x;
    r.y += q.c11 * p.y;
    r.z += q.c22 * p.z;

    f32 error = Dot(r, p) + q.d2;
    return error;
}

f32 EvaluateQuadric(const Vec3f &p, const Quadric &q, const QuadricGrad *g, f32 *attributes,
                    const f32 *attributeWeights, u32 numAttributes)
{
    // New matrix Q =
    // [K  B]
    // [Bt a]
    //
    // Where B = g[0 ... numAttributes] (matrix 3xnumAttributes of gradients)
    //
    // Error = pT * Q * p
    //
    // where p = [v]
    //           [s]

    f32 error   = EvaluateQuadric(p, q);
    f32 invArea = 1.f / q.area;

    for (int i = 0; i < numAttributes; i++)
    {
        f32 pgd       = g[i].d + Dot(g[i].g, p);
        f32 s         = pgd * invArea;
        attributes[i] = s / attributeWeights[i];

        // 2s * Dot(-g, p) + -2s * d + dj2 + s^2 * area
        //
        // 1/area(d^2 + 2gp + gp^2 + 2d * -gp - 2gp^2 - 2d^2 - 2gp)
        // 1/area(-d^2 -gp^2 -2dgp)
        // -1/area(pgd^2)
        // -pgd * s

        error -= pgd * s;
    }

    return Abs(error);
}

f32 EvaluateQuadricLocked(const Vec3f &p, const Quadric &q, const QuadricGrad *g,
                          f32 *attributes, const f32 *attributeWeights, u32 numAttributes)
{
    f32 error = EvaluateQuadric(p, q);

    for (int i = 0; i < numAttributes; i++)
    {
        f32 pgd = g[i].d + Dot(g[i].g, p);
        f32 s   = attributes[i] * attributeWeights[i];

        error += s * (q.area * s - 2 * pgd);
    }

    return Abs(error);
}

void Rebase(Quadric &q, QuadricGrad *g, f32 *attributes, f32 *attributeWeights,
            u32 numAttributes, Vec3f &p)
{
    f32 d              = -Dot(q.gVol, p);
    f32 invArea        = 1.f / q.area;
    f32 quarterInvArea = .25f * invArea;

    // gVol is the normalized normal multiplied by twice the area.
    q.dn   = q.gVol * d * quarterInvArea;
    q.d2   = Sqr(d) * quarterInvArea;
    q.dVol = d;

    for (int i = 0; i < numAttributes; i++)
    {
        f32 a0 = attributes[i] * attributeWeights[i];
        f32 gd = a0 - Dot(g[i].g, p) * invArea;

        q.dn += g[i].g * gd;
        g[i].d = gd * q.area;
        q.d2   = g[i].d * gd;
    }
}

void Quadric::InitializeEdge(const Vec3f &p0, const Vec3f &p1, f32 weight)
{
    Vec3f n = Cross(p0, p1);

    gVol = 0.f;
    dVol = 0.f;

    f32 length = Length(n);

    if (length < 1e-8f) return;

    n /= length;

    gVol = n;

    area = weight * length;

    // Multiply quadric by area (in preparation to be summed by other faces)
    c00 = area - area * n.x * n.x;
    c01 *= -area * n.x * n.y;
    c02 *= -area * n.x * n.z;

    c11 *= area - area * n.y * n.y;
    c12 *= -area * n.y * n.z;
    c22 *= area - area * n.z * n.z;
}

void Quadric::Add(Quadric &other)
{
    c00 += other.c00;
    c01 += other.c01;
    c02 += other.c02;

    c11 += other.c11;
    c12 += other.c12;
    c22 += other.c22;

    dn += other.dn;

    d2 += other.d2;

    gVol += other.gVol;
    dVol += other.dVol;

    // Volume optimization
    area = other.area;
}

void Quadric::AddEdgeQuadric(Quadric &edgeQuadric, const Vec3f &p0)
{
    c00 += edgeQuadric.c00;
    c01 += edgeQuadric.c01;
    c02 += edgeQuadric.c02;

    c11 += edgeQuadric.c11;
    c12 += edgeQuadric.c12;
    c22 += edgeQuadric.c22;

    f32 dist = -Dot(p0, edgeQuadric.gVol);
    dn += edgeQuadric.area * (-p0 - edgeQuadric.gVol * dist);

    d2 += edgeQuadric.area * (Dot(p0, p0) - Sqr(dist));
}

void AddQuadric(QuadricGrad *g, const QuadricGrad *other, u32 numAttributes)
{
    for (int i = 0; i < numAttributes; i++)
    {
        g[i].g += other[i].g;
        g[i].d += other[i].d;
    }
}

template <typename T>
struct Heap
{
    u32 *heap;
    u32 *heapIndices;

    u32 heapNum;

    u32 maxSize;

    Heap(Arena *arena, u32 arraySize)
    {
        heap        = PushArrayNoZero(arena, u32, arraySize);
        heapIndices = PushArrayNoZero(arena, u32, arraySize);
        heapNum     = 0;
        maxSize     = arraySize;
    }

    int GetParent(int index) const { return index == 0 ? 0 : (index - 1) >> 1; }
    bool IsPresent(int index) { return (index < maxSize) && heapIndices[index] != -1; }

    void Add(const T *keys, int index)
    {
        heap[heapNum]      = index;
        heapIndices[index] = heapNum;

        UpHeap(keys, heapNum);

        heapNum++;
    }

    int Pop(const T *keys)
    {
        if (heapNum == 0) return -1;

        // Down heap
        int index = heap[0];
        Assert(heapIndices[index] == 0);

        heap[0]              = heap[--heapNum];
        heapIndices[heap[0]] = 0;
        heapIndices[index]   = -1;

        DownHeap(keys, 0);

        return index;
    }

    void Remove(const T *keys, int index)
    {
        int heapIndex = heapIndices[index];
        Assert(heapIndex < heapNum);

        if (heapIndex == -1) return;

        heap[heapIndex]              = heap[--heapNum];
        heapIndices[heap[heapIndex]] = heapIndex;
        heapIndices[index]           = -1;

        if (keys[index] < keys[heap[heapIndex]])
        {
            DownHeap(keys, heapIndex);
        }
        else
        {
            UpHeap(keys, heapIndex);
        }
    }

    void UpHeap(const T *keys, int startIndex)
    {
        int index    = startIndex;
        int parent   = GetParent(startIndex);
        const T &key = keys[heap[startIndex]];
        int m        = heap[startIndex];

        while (index != 0 && key < keys[heap[parent]])
        {
            heap[index]              = heap[parent];
            heapIndices[heap[index]] = index;

            index  = parent;
            parent = GetParent(index);
        }
        if (index != startIndex)
        {
            heap[index]    = m;
            heapIndices[m] = index;
        }
    }

    void DownHeap(const T *keys, int startIndex)
    {
        int index         = heap[startIndex];
        const T &addedVal = keys[index];

        int parent = startIndex;
        while (parent < heapNum)
        {
            int left     = (parent << 1) + 1;
            int right    = left + 1;
            int minIndex = left < heapNum && keys[heap[left]] < addedVal ? left : parent;
            T minVal     = left < heapNum ? Min(keys[heap[left]], addedVal) : addedVal;
            minIndex     = right < heapNum && keys[heap[right]] < minVal ? right : minIndex;

            if (minIndex == parent) break;

            heap[parent]              = heap[minIndex];
            heapIndices[heap[parent]] = parent;

            parent = minIndex;
        }
        if (parent != startIndex)
        {
            heap[parent]              = index;
            heapIndices[heap[parent]] = parent;
        }
    }
};

template <typename Func>
void IterateHashBreak(HashIndex &index, int hash, const Func &func)
{
    for (int i = index.FirstInHash(hash); i != -1; i = index.NextInHash(i))
    {
        if (func(i)) break;
    }
}

template <typename Func>
void IterateHash(HashIndex &index, int hash, const Func &func)
{
    for (int i = index.FirstInHash(hash); i != -1; i = index.NextInHash(i))
    {
        func(i);
    }
}

MeshSimplifier::MeshSimplifier(Arena *arena, f32 *vertexData, u32 numVertices, u32 *indices,
                               u32 numIndices, u32 numAttributes)
    : vertexData(vertexData), indices(indices), numVertices(numVertices),
      numIndices(numIndices), numAttributes(numAttributes),
      cornerHash(arena, NextPowerOfTwo(numIndices), NextPowerOfTwo(numIndices)),
      vertexHash(arena, NextPowerOfTwo(numVertices), NextPowerOfTwo(numVertices)),
      pairHash0(arena, NextPowerOfTwo(numIndices), NextPowerOfTwo(numIndices)),
      pairHash1(arena, NextPowerOfTwo(numIndices), NextPowerOfTwo(numIndices)),
      triangleIsRemoved(arena, numIndices / 3), lockedVertices(arena, numVertices),
      hasEdgeQuadric(arena, numIndices)
{

    for (int vertIndex = 0; vertIndex < numVertices; vertIndex++)
    {
        Vec3f p = GetPosition(vertIndex);
        vertexHash.AddInHash(Hash(p), vertIndex);
    }

    u32 numTriangles      = numIndices / 3;
    remainingNumTriangles = numIndices / 3;
    edgeQuadrics          = PushArrayNoZero(arena, Quadric, numIndices);
    triangleQuadrics      = PushArrayNoZero(arena, Quadric, numTriangles);
    triangleAttrQuadrics  = 0;
    if (numAttributes)
        triangleAttrQuadrics =
            PushArrayNoZero(arena, QuadricGrad, numTriangles * numAttributes);
    attributeWeights = 0;

    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
        Vec3f p0 = GetPosition(indices[3 * triIndex]);
        Vec3f p1 = GetPosition(indices[3 * triIndex + 1]);
        Vec3f p2 = GetPosition(indices[3 * triIndex + 2]);

        if (!(p0 == p1 || p0 == p2 || p1 == p2))
        {
            CalculateTriQuadrics(triIndex);
        }
        else
        {
            triangleIsRemoved.SetBit(triIndex);
            remainingNumTriangles--;
        }
    }

    for (int tri = 0; tri < numTriangles; tri++)
    {
        if (triangleIsRemoved.GetBit(tri)) continue;

        for (int corner = 0; corner < 3; corner++)
        {
            Vec3f position = GetPosition(indices[3 * tri + corner]);
            int hash       = Hash(position);
            cornerHash.AddInHash(hash, 3 * tri + corner);
        }
    }
}

void MeshSimplifier::LockVertex(const Vec3f &p)
{
    int hash = Hash(p);
    IterateHash(vertexHash, hash, [&](int vertexIndex) {
        if (GetPosition(vertexIndex) == p)
        {
            lockedVertices.SetBit(vertexIndex);
        }
    });
}

Vec3f &MeshSimplifier::GetPosition(u32 vertexIndex)
{
    return *(Vec3f *)(vertexData + (3 + numAttributes) * vertexIndex);
}

const Vec3f &MeshSimplifier::GetPosition(u32 vertexIndex) const
{
    return *(Vec3f *)(vertexData + (3 + numAttributes) * vertexIndex);
}

f32 *MeshSimplifier::GetAttributes(u32 vertexIndex)
{
    return vertexData + (3 + numAttributes) * vertexIndex + 3;
}

u32 NextInTriangle(u32 indexIndex, u32 offset)
{
    return indexIndex - indexIndex % 3 + (indexIndex + offset) % 3;
}

bool MeshSimplifier::CheckInversion(const Vec3f &newPosition, u32 *movedCorners,
                                    u32 count) const
{
    for (int i = 0; i < count; i++)
    {
        u32 corner      = movedCorners[i];
        u32 indexIndex0 = corner;
        u32 indexIndex1 = NextInTriangle(indexIndex0, 1);
        u32 indexIndex2 = NextInTriangle(indexIndex0, 2);

        u32 vertexIndex0 = indices[indexIndex0];
        u32 vertexIndex1 = indices[indexIndex1];
        u32 vertexIndex2 = indices[indexIndex2];

        Vec3f p0 = GetPosition(vertexIndex0);
        Vec3f p1 = GetPosition(vertexIndex1);
        Vec3f p2 = GetPosition(vertexIndex2);

        Vec3f p21      = p2 - p1;
        Vec3f p01      = p0 - p1;
        Vec3f pNewEdge = newPosition - p1;

        Vec3f cross0 = Cross(pNewEdge, p21);
        Vec3f cross1 = Cross(p01, p21);
        f32 dot      = Dot(cross0, cross1);

        bool result = dot >= 0.f;
        if (!result) return true;
    }

    return false;
}

static const int next[3] = {1, 2, 0};

template <typename Func>
void MeshSimplifier::IterateCorners(const Vec3f &position, const Func &func)
{
    int hash = Hash(position);
    IterateHash(cornerHash, hash, [&](int cornerIndex) {
        if (position == GetPosition(indices[cornerIndex]))
        {
            func(cornerIndex);
        }
    });
}

void MeshSimplifier::CalculateTriQuadrics(u32 triIndex)
{
    int index0 = 3 * triIndex + 0;
    int index1 = 3 * triIndex + 1;
    int index2 = 3 * triIndex + 2;

    Vec3f p0 = GetPosition(indices[index0]);
    Vec3f p1 = GetPosition(indices[index1]);
    Vec3f p2 = GetPosition(indices[index2]);

    triangleQuadrics[triIndex] = Quadric(p0, p1, p2);
    CreateAttributeQuadric(triangleQuadrics[triIndex],
                           triangleAttrQuadrics + numAttributes * triIndex, p0, p1, p2,
                           GetAttributes(index0), GetAttributes(index1), GetAttributes(index2),
                           attributeWeights, numAttributes);
}

bool MeshSimplifier::AddUniquePair(Pair &pair, int pairIndex)
{
    int p0Hash = Hash(pair.p0);
    int p1Hash = Hash(pair.p1);

    if (p1Hash < p0Hash)
    {
        Swap(pair.p0, pair.p1);
        Swap(p0Hash, p1Hash);
    }

    bool duplicate = false;

    IterateHashBreak(pairHash0, p0Hash, [&](int pairIndex) {
        if (pairs[pairIndex] == pair)
        {
            duplicate = true;
            return true;
        }
        return false;
    });

    if (!duplicate)
    {
        pairHash0.AddInHash(p0Hash, pairIndex);
        pairHash1.AddInHash(p1Hash, pairIndex);
    }

    return !duplicate;
}

void MeshSimplifier::CalculateEdgeQuadric(u32 edgeIndex)
{
    u32 vertexIndex0 = indices[edgeIndex];
    u32 vertexIndex1 = indices[NextInTriangle(edgeIndex, 1)];

    Vec3f pos0 = GetPosition(vertexIndex0);
    Vec3f pos1 = GetPosition(vertexIndex1);

    bool oppositeEdge = false;
    IterateCorners(pos1, [&](int cornerIndex) {
        u32 otherVertexIndex0 = indices[cornerIndex];
        u32 otherVertexIndex1 = indices[NextInTriangle(cornerIndex, 1)];
        if (vertexIndex0 == otherVertexIndex1 && vertexIndex1 == otherVertexIndex0)
        {
            oppositeEdge = true;
        }
    });

    if (!oppositeEdge)
    {
        threadLocalStatistics[GetThreadIndex()].test++;
        Quadric quadric;
        quadric.InitializeEdge(pos0, pos1, 2.f);
        hasEdgeQuadric.SetBit(edgeIndex);
        edgeQuadrics[edgeIndex] = quadric;
    }
}

void MeshSimplifier::EvaluatePair(Pair &pair)
{
    // Find the set of triangles adjacent to the pair
    ScratchArena scratch;

    Array<u32> adjCorners(scratch.temp.arena, 24);

    Vec3f pos[] = {
        pair.p0,
        pair.p1,
    };

    for (int i = 0; i < 2; i++)
    {
        IterateCorners(pos[i], [&](int cornerIndex) { adjCorners.Push(cornerIndex); });
    }

    // Set of all triangles adjacent to the pair
    StaticArray<u32> adjTris(scratch.temp.arena, adjCorners.Length());

    // Find the triangles that are moved and not collapsed
    StaticArray<u32> movedCorners(scratch.temp.arena, adjCorners.Length());

    // if (adjCorners.Length() >= 1000)
    // {
    //     Print("num corners: %u, num remaining tris: %u, num tris: %u\n",
    //     adjCorners.Length(),
    //           remainingNumTriangles, numIndices / 3);
    //     for (int i = 0; i < adjCorners.Length(); i++)
    //     {
    //         Print("%u %u\n", i, adjCorners[i]);
    //         u32 index = indices[adjCorners[i]];
    //         Print("%u\n", index);
    //         Print("%f %f %f\n", GetPosition(index).x, GetPosition(index).y,
    //               GetPosition(index).z);
    //     }
    //     Assert(0);
    // }

    for (u32 corner : adjCorners)
    {
        u32 tri     = corner / 3;
        bool unique = adjTris.PushUnique(tri);

        if (unique)
        {
            u32 vertIndex0 = indices[3 * tri];
            u32 vertIndex1 = indices[3 * tri + 1];
            u32 vertIndex2 = indices[3 * tri + 2];

            Vec3f p0 = GetPosition(vertIndex0);
            Vec3f p1 = GetPosition(vertIndex1);
            Vec3f p2 = GetPosition(vertIndex2);

            bool pos0Found = pair.p0 == p0 || pair.p0 == p1 || pair.p0 == p2;
            bool pos1Found = pair.p1 == p0 || pair.p1 == p1 || pair.p1 == p2;

            if (!(pos0Found && pos1Found))
            {
                movedCorners.Push(corner);
            }
        }
    }

    // Add triangle quadrics
    Vec3f basePosition = pair.p0;
    Quadric quadric;
    QuadricGrad *quadricGrad = 0;
    if (numAttributes)
        quadricGrad = PushArrayNoZero(scratch.temp.arena, QuadricGrad, numAttributes);

    for (int i = 0; i < adjTris.Length(); i++)
    {
        u32 tri                   = adjTris[i];
        u32 vertexIndex           = indices[3 * tri];
        Vec3f rebasedPosition     = GetPosition(vertexIndex) - basePosition;
        QuadricGrad *attrQuadrics = &triangleAttrQuadrics[numAttributes * adjTris[i]];

        Rebase(triangleQuadrics[tri], attrQuadrics, GetAttributes(vertexIndex),
               attributeWeights, numAttributes, rebasedPosition);

        quadric.Add(triangleQuadrics[tri]);
        AddQuadric(quadricGrad, attrQuadrics, numAttributes);
    }

    Vec3f boundsMin(pos_inf);
    Vec3f boundsMax(neg_inf);

    for (u32 corner : adjCorners)
    {
        u32 corner1 = NextInTriangle(corner, 1);
        u32 corner2 = NextInTriangle(corner, 2);

        boundsMin = Min(boundsMin, GetPosition(indices[corner]));
        boundsMin = Min(boundsMin, GetPosition(indices[corner1]));
        boundsMin = Min(boundsMin, GetPosition(indices[corner2]));
        boundsMax = Max(boundsMax, GetPosition(indices[corner]));
        boundsMax = Max(boundsMax, GetPosition(indices[corner1]));
        boundsMax = Max(boundsMax, GetPosition(indices[corner2]));

        if (hasEdgeQuadric.GetBit(corner))
        {
            Vec3f p0 = GetPosition(indices[corner]);
            quadric.AddEdgeQuadric(edgeQuadrics[corner], p0 - basePosition);
        }
        if (hasEdgeQuadric.GetBit(corner2))
        {
            Vec3f p2 = GetPosition(indices[corner2]);
            quadric.AddEdgeQuadric(edgeQuadrics[corner2], p2 - basePosition);
        }
    }

    bool bVertex0IsLocked = false;
    bool bVertex1IsLocked = false;
    IterateHash(vertexHash, Hash(pair.p0), [&](int vertIndex) {
        if (GetPosition(vertIndex) == pair.p0)
        {
            bVertex0IsLocked |= lockedVertices.GetBit(vertIndex);
        }
    });
    IterateHash(vertexHash, Hash(pair.p1), [&](int vertIndex) {
        if (GetPosition(vertIndex) == pair.p1)
        {
            bVertex1IsLocked |= lockedVertices.GetBit(vertIndex);
        }
    });

    f32 error = 0.f;
    Vec3f p;
    bool valid = false;

    auto CheckValidPosition = [&](const Vec3f &p) {
        f32 distSqr = 0.f;
        for (int axis = 0; axis < 3; axis++)
        {
            if (p[axis] < boundsMin[axis])
            {
                distSqr += Sqr(p[axis] - boundsMin[axis]);
            }
            else if (p[axis] > boundsMax[axis])
            {
                distSqr += Sqr(p[axis] - boundsMax[axis]);
            }
        }
        bool valid = distSqr <= (LengthSquared(boundsMax - boundsMin) * 4.f) &&
                     !CheckInversion(p, movedCorners.data, movedCorners.Length());
        return valid;
    };

    if (bVertex0IsLocked && bVertex1IsLocked)
    {
        error = lockedPenaty;
    }

    if (bVertex0IsLocked && !bVertex1IsLocked)
    {
        p     = pair.p0;
        valid = CheckValidPosition(p);
        p -= basePosition;
        pair.strategy = Strategy::Locked;
    }
    else if (bVertex1IsLocked && !bVertex0IsLocked)
    {
        p     = pair.p1;
        valid = CheckValidPosition(p);
        p -= basePosition;
        pair.strategy = Strategy::Locked;
    }
    else
    {
        pair.strategy = Strategy::Optimal;
        // Precalculate optimization information
        if (quadric.area > 1e-12)
        {
            f32 invA = 1.f / quadric.area;

            f32 BBt00 = 0.f;
            f32 BBt01 = 0.f;
            f32 BBt02 = 0.f;
            f32 BBt11 = 0.f;
            f32 BBt12 = 0.f;
            f32 BBt22 = 0.f;

            Vec3f b1 = quadric.dn;
            Vec3f Bb2(0.f);

            for (int i = 0; i < numAttributes; i++)
            {
                OuterProduct(quadricGrad[i].g, BBt00, BBt01, BBt02, BBt11, BBt12, BBt22);
                Bb2 += quadricGrad[i].g * quadricGrad[i].d;
            }

            // A = (C - 1/a * BBt)
            f32 A00 = quadric.c00 - BBt00 * invA;
            f32 A01 = quadric.c01 - BBt01 * invA;
            f32 A02 = quadric.c02 - BBt02 * invA;

            f32 A11 = quadric.c11 - BBt11 * invA;
            f32 A12 = quadric.c12 - BBt12 * invA;
            f32 A22 = quadric.c22 - BBt22 * invA;

            // b = b1 - 1/a * B * b2
            Vec3f bbb2 = b1 - invA * Bb2;

            // Volume
            {
                f32 A[16] = {
                    A00,
                    A01,
                    A02,
                    quadric.gVol.x,
                    A01,
                    A11,
                    A12,
                    quadric.gVol.y,
                    A02,
                    A12,
                    A22,
                    quadric.gVol.z,
                    quadric.gVol.x,
                    quadric.gVol.y,
                    quadric.gVol.z,
                    0,
                };

                f32 LU[16];
                MemoryCopy(LU, A, sizeof(LU));

                f32 b[4] = {-bbb2.x, -bbb2.y, -bbb2.z, -quadric.dVol};

                // Solve the 4x4 linear system
                int pivots[4];
                if (LUPDecompose(LU, 4, 1e-8f, pivots))
                {
                    f32 result[4];
                    if (LUPSolveIterate(A, LU, pivots, b, 4, result, 4))
                    {
                        p.x   = result[0];
                        p.y   = result[1];
                        p.z   = result[2];
                        valid = true;
                    }
                }
                if (valid)
                {
                    valid = CheckValidPosition(p + basePosition);
                }
            }
            if (!valid)
            {
                f32 A[9] = {
                    A00, A01, A02, A01, A11, A12, A02, A12, A22,
                };
                f32 LU[9];
                MemoryCopy(LU, A, sizeof(LU));

                f32 b[3] = {-bbb2.x, -bbb2.y, -bbb2.z};

                // Solve the 4x4 linear system
                int pivots[3];
                if (LUPDecompose(LU, 3, 1e-8f, pivots))
                {
                    f32 result[3];
                    if (LUPSolveIterate(A, LU, pivots, b, 3, result, 4))
                    {
                        p.x   = result[0];
                        p.y   = result[1];
                        p.z   = result[2];
                        valid = true;
                    }
                }
                if (valid)
                {
                    valid = CheckValidPosition(p + basePosition);
                }
            }
        }

        if (!valid)
        {
            p     = (pair.p0 + pair.p1) / 2.f;
            valid = CheckValidPosition(p);
            p -= basePosition;
            pair.strategy = Strategy::Midpoint;
        }

        if (!valid)
        {
            error += inversionPenalty;
        }
    }

    // Evaluate the error for the optimal position
    if (bVertex1IsLocked || bVertex0IsLocked)
    {
        error += EvaluateQuadricLocked(p, quadric, quadricGrad, 0, 0, numAttributes);
    }
    else
    {
        error += EvaluateQuadric(p, quadric, quadricGrad, 0, 0, numAttributes);
    }

    if (error != error)
    {
        error = 0.f;
    }

    pair.error = error;
    pair.newP  = p + basePosition;
}

f32 MeshSimplifier::Simplify(u32 targetNumVerts, u32 targetNumTris, f32 targetError,
                             u32 limitNumVerts, u32 limitNumTris, f32 limitError)
{
    ScratchArena scratch;

    u32 numTriangles = numIndices / 3;

    pairs = StaticArray<Pair>(scratch.temp.arena, 3 * numTriangles);
    Heap<Pair> heap(scratch.temp.arena, 3 * numTriangles);

    for (u32 i = 0; i < numIndices; i++)
    {
        if (triangleIsRemoved.GetBit(i / 3)) continue;
        CalculateEdgeQuadric(i);
    }

    // Add unique pairs to hash
    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
        if (triangleIsRemoved.GetBit(triIndex)) continue;

        int baseIndexIndex = 3 * triIndex;
        for (int vertIndex = 0; vertIndex < 3; vertIndex++)
        {
            int indexIndex0 = baseIndexIndex + vertIndex;
            int indexIndex1 = baseIndexIndex + next[vertIndex];

            int vertexIndex0 = indices[indexIndex0];
            int vertexIndex1 = indices[indexIndex1];

            Vec3f p0 = GetPosition(vertexIndex0);
            Vec3f p1 = GetPosition(vertexIndex1);

            Pair pair;
            pair.p0 = p0;
            pair.p1 = p1;

            if (AddUniquePair(pair, pairs.Length()))
            {
                pairs.Push(pair);
            }
        }
    }

    for (int pairIndex = 0; pairIndex < pairs.size(); pairIndex++)
    {
        EvaluatePair(pairs[pairIndex]);
        heap.Add(pairs.data, pairIndex);
    }

    f32 maxError             = 0.f;
    u32 remainingNumVertices = numVertices;
    for (;;)
    {

        int pairIndex = heap.Pop(pairs.data);
        if (pairIndex == -1) break;

        Pair &pair = pairs[pairIndex];

        Vec3f newPosition = pair.newP;

        // ErrorExit(error == pair.error, "%u %u %f %f %f\n", pairIndex, remainingNumTriangles,
        //           error, pair.error, maxError);

        maxError = Max(maxError, pair.error);

        if (maxError >= targetError && remainingNumVertices <= targetNumVerts &&
            remainingNumTriangles <= targetNumTris)
        {
            break;
        }
        if (maxError >= limitError || remainingNumVertices <= limitNumTris ||
            remainingNumTriangles <= limitNumTris)
        {
            break;
        }

        remainingNumVertices--;

        Array<u32> movedCorners(scratch.temp.arena, 24);
        Array<int> movedPairs(scratch.temp.arena, 24);
        Array<u32> movedVertices(scratch.temp.arena, 24);

        // Change the positions of all corners
        int p0Hash = Hash(pair.p0);
        int p1Hash = Hash(pair.p1);

        int hashes[] = {
            p0Hash,
            p1Hash,
        };

        Vec3f pos[] = {
            pair.p0,
            pair.p1,
        };

        pairHash0.RemoveFromHash(hashes[0], pairIndex);
        pairHash1.RemoveFromHash(hashes[1], pairIndex);

        bool isLocked = false;

        for (int i = 0; i < 2; i++)
        {
            IterateHash(vertexHash, hashes[i], [&](int vertIndex) {
                if (GetPosition(vertIndex) == pos[i])
                {
                    movedVertices.Push(vertIndex);
                    vertexHash.RemoveFromHash(hashes[i], vertIndex);
                    isLocked |= lockedVertices.GetBit(vertIndex);
                }
            });

            IterateCorners(pos[i], [&](int corner) {
                bool unique = movedCorners.PushUnique(corner);
                cornerHash.RemoveFromHash(Hash(GetPosition(indices[corner])), corner);
            });

            IterateHash(pairHash0, hashes[i], [&](int otherPairIndex) {
                if (pairs[otherPairIndex].p0 == pos[i])
                {
                    movedPairs.Push(otherPairIndex);
                    pairHash0.RemoveFromHash(Hash(pairs[otherPairIndex].p0), otherPairIndex);
                    pairHash1.RemoveFromHash(Hash(pairs[otherPairIndex].p1), otherPairIndex);
                }
            });

            IterateHash(pairHash1, hashes[i], [&](int otherPairIndex) {
                if (pairs[otherPairIndex].p1 == pos[i])
                {
                    movedPairs.Push(otherPairIndex);
                    pairHash0.RemoveFromHash(Hash(pairs[otherPairIndex].p0), otherPairIndex);
                    pairHash1.RemoveFromHash(Hash(pairs[otherPairIndex].p1), otherPairIndex);
                }
            });
        }

        StaticArray<u32> movedTriangles(scratch.temp.arena, movedCorners.Length());

        int newHash = Hash(newPosition);
        for (u32 corner : movedCorners)
        {
            GetPosition(indices[corner]) = newPosition;
            movedTriangles.PushUnique(corner / 3);

            cornerHash.AddInHash(newHash, corner);
        }

        for (u32 vertexIndex : movedVertices)
        {
            vertexHash.AddInHash(Hash(GetPosition(vertexIndex)), vertexIndex);
            if (isLocked)
            {
                lockedVertices.SetBit(vertexIndex);
            }
        }

        // Change pairs to have new position
        for (int movedPairIndex : movedPairs)
        {
            Pair &movedPair = pairs[movedPairIndex];
            Assert(movedPairIndex != pairIndex);
            if (movedPair.p0 == pair.p0 || movedPair.p0 == pair.p1)
            {
                movedPair.p0 = newPosition;
            }
            if (movedPair.p1 == pair.p0 || movedPair.p1 == pair.p1)
            {
                movedPair.p1 = newPosition;
            }
        }

        // Remove invalid and duplicate pairs
        for (int movedPairIndex : movedPairs)
        {
            Pair &pair = pairs[movedPairIndex];
            if (pair.p0 == pair.p1 || !AddUniquePair(pair, movedPairIndex))
            {
                heap.Remove(pairs.data, movedPairIndex);
            }
        }

        // Reevaluate all pairs adjacent to all adjacent triangles
        Array<u32> uniqueVerts(scratch.temp.arena, 24);
        for (u32 tri : movedTriangles)
        {
            for (u32 corner = 0; corner < 3; corner++)
            {
                uniqueVerts.PushUnique(indices[3 * tri + corner]);
            }
        }

        Array<u32> changedPairs(scratch.temp.arena, 24);
        for (u32 vert : uniqueVerts)
        {
            Vec3f p  = GetPosition(vert);
            int hash = Hash(p);

            auto GetPairs = [&](int pairIndex) {
                Pair &pair = pairs[pairIndex];
                if (pair.p0 == p || pair.p1 == p)
                {
                    if (heap.IsPresent(pairIndex))
                    {
                        heap.Remove(pairs.data, pairIndex);
                        changedPairs.Push(pairIndex);
                    }
                }
            };

            IterateHash(pairHash0, hash, GetPairs);
            IterateHash(pairHash1, hash, GetPairs);
        }

        // Recalculate quadrics of valid triangles. Remove invalid triangles.
        for (u32 tri : movedTriangles)
        {
            int i0 = indices[3 * tri];
            int i1 = indices[3 * tri + 1];
            int i2 = indices[3 * tri + 2];

            Vec3f p0 = GetPosition(i0);
            Vec3f p1 = GetPosition(i1);
            Vec3f p2 = GetPosition(i2);

            bool removeTri = p0 == p1 || p0 == p2 || p1 == p2;

            if (!removeTri)
            {
                for (int i = 0; i < 3; i++)
                {
                    u32 corner      = 3 * tri + i;
                    u32 vertexIndex = indices[corner];
                    f32 *data       = vertexData + (3 + numAttributes) * vertexIndex;
                    Vec3f p         = GetPosition(vertexIndex);
                    int hash        = Hash(p);
                    IterateHashBreak(vertexHash, hash, [&](int otherVertexIndex) {
                        if (vertexIndex == otherVertexIndex) return true;
                        f32 *otherData = vertexData + (3 + numAttributes) * otherVertexIndex;

                        if (memcmp(data, otherData, sizeof(f32) * (3 + numAttributes)) == 0)
                        {
                            indices[corner] = otherVertexIndex;
                            return true;
                        }
                        return false;
                    });
                }

                int hash = Hash(p0);

                IterateHashBreak(cornerHash, hash, [&](int corner) {
                    u32 otherTri = corner / 3;
                    if (otherTri == tri) return false;

                    u32 triIndices[3];
                    u32 otherIndices[3];
                    triIndices[0] = indices[3 * tri];
                    u32 nextIndex = indices[3 * tri + 1];
                    if (nextIndex < triIndices[0])
                    {
                        triIndices[1] = triIndices[0];
                        triIndices[0] = nextIndex;
                    }
                    else triIndices[1] = nextIndex;
                    nextIndex = indices[3 * tri + 2];
                    if (nextIndex < triIndices[0])
                    {
                        triIndices[2] = triIndices[1];
                        triIndices[1] = triIndices[0];
                        triIndices[0] = nextIndex;
                    }
                    else if (nextIndex < triIndices[1])
                    {
                        triIndices[2] = triIndices[1];
                        triIndices[1] = nextIndex;
                    }
                    else triIndices[2] = nextIndex;

                    otherIndices[0] = indices[3 * otherTri];
                    nextIndex       = indices[3 * otherTri + 1];
                    if (nextIndex < otherIndices[0])
                    {
                        otherIndices[1] = otherIndices[0];
                        otherIndices[0] = nextIndex;
                    }
                    else otherIndices[1] = nextIndex;
                    nextIndex = indices[3 * otherTri + 2];
                    if (nextIndex < otherIndices[0])
                    {
                        otherIndices[2] = otherIndices[1];
                        otherIndices[1] = otherIndices[0];
                        otherIndices[0] = nextIndex;
                    }
                    else if (nextIndex < otherIndices[1])
                    {
                        otherIndices[2] = otherIndices[1];
                        otherIndices[1] = nextIndex;
                    }
                    else otherIndices[2] = nextIndex;

                    for (u32 j = 0; j < 3; j++)
                    {
                        if (triIndices[j] != otherIndices[j])
                        {
                            return false;
                        }
                    }
                    removeTri = true;
                    // Print("dupe tri\n");
                    return true;
                });
            }

            if (!removeTri)
            {
                CalculateTriQuadrics(tri);
            }
            else
            {
                triangleIsRemoved.SetBit(tri);
                remainingNumTriangles--;

                for (int i = 0; i < 3; i++)
                {
                    int corner = 3 * tri + i;
                    cornerHash.RemoveFromHash(Hash(GetPosition(indices[corner])), corner);
                    indices[corner] = ~0u;
                }
            }
        }

        for (u32 changedPair : changedPairs)
        {
            EvaluatePair(pairs[changedPair]);
            heap.Add(pairs.data, changedPair);
        }

        for (u32 tri : movedTriangles)
        {
            if (triangleIsRemoved.GetBit(tri)) continue;
            for (u32 corner = 0; corner < 3; corner++)
            {
                CalculateEdgeQuadric(3 * tri + corner);
            }
        }
    }

    return maxError;
}

#if 0

void MeshSimplifier::ClosestPointTriangleTriangle(Vec3f &p0, Vec3f &p1, u32 tri, u32 otherTri)
{
    // Find closest points on pairs of edges

    // Find closest points on vertex/face
    Vec3f vertices[2][3];

    for (int i = 0; i < 2; i++)
    {
        Vec3f n = Cross(vertices[i][1] - vertices[i][0], vertices[i][2] - vertices[i][0]);

        f32 sqrLen = LengthSquared(n);

        if (sqrLen < 1e-8f) continue;

        f32 invSqrLen = 1.f / sqrLen;

        f32 dots[3] = {Dot(vertices[!i][0] - vertices[i][0], n),
                       Dot(vertices[!i][1] - vertices[i][0], n),
                       Dot(vertices[!i][2] - vertices[i][0], n)};

        bool sameSign = (dots[0] > 0.f && dots[1] > 0.f && dots[2] > 0.f) ||
                        (dots[0] < 0.f && dots[1] < 0.f && dots[2] < 0.f);

        if (sameSign)
        {
            Vec3f candidateVert = vertices[!i][index];

            u32 index  = Abs(dots[0]) < Abs(dots[1]) ? 0 : 1;
            index      = Abs(dots[2]) < Abs(dots[index]) ? 2 : index;
            bool valid = true;
            for (int j = 0; j < 3; j++)
            {
                Vec3f v = candidateVert - vertices[i][j];
                f32 d   = Dot(Cross(n, vertices[i][(j + 1) % 3] - vertices[i][j]), v);
                if (d <= 0.f)
                {
                    valid = false;
                    break;
                }
            }

            if (valid)
            {
                p0 = candidateVert - n * dots[index] * invSqrLen;
                p1 = candidateVert;
                return;
            }
        }
    }
}

void MeshSimplifier::CreateVirtualEdges(f32 maxDist)
{
    // Find the closest
    for (int tri = 0; tri < numIndices / 3; tri++)
    {
        Vec3f center;

        FixedArray<KDTreeNode, 64> stack;

        KDTreeNode node = stack.Pop();
        if (node.IsLeaf())
        {
            for (int otherTri = node.start; otherTri < node.start + node.count; otherTri++)
            {
                // GJK :)
                FixedArray<Vec3f, 4> points;

                Vec3f support;
                Vec3f dir = -support;
                for (;;)
                {
                    Vec3f point = Support(dir, tri) - Support(-dir, otherTri);
                    if (Dot(point, dir) < 0.f) break;

                    points.Push(point);
                    Simplex(points, dir);
                }
            }
        }
        else
        {
            int choice = center[axis] >= node.split;
            stack.Push(choice ? node.left : node.right);
            stack.Push(choice ? node.right : node.left);
        }
    }
}

#endif

void MeshSimplifier::Finalize(u32 &finalNumVertices, u32 &finalNumIndices, u32 *geomIDs)
{
    ScratchArena scratch;

    // Compact the vertex buffer
    u32 *remap = PushArray(scratch.temp.arena, u32, numVertices);

    const u32 attributeLen = sizeof(f32) * (3 + numAttributes);

    u32 vertexCount   = 0;
    u32 triangleCount = 0;

    HashIndex triangleHash(scratch.temp.arena, NextPowerOfTwo(3 * numIndices),
                           NextPowerOfTwo(3 * numIndices));

    auto GetSortedIndices = [&](u32 triIndices[3], u32 tri) {
        triIndices[0] = indices[3 * tri + 0];
        triIndices[1] = indices[3 * tri + 1];
        triIndices[2] = indices[3 * tri + 2];
        if (triIndices[1] < triIndices[0]) Swap(triIndices[0], triIndices[1]);
        if (triIndices[2] < triIndices[1]) Swap(triIndices[1], triIndices[2]);
        if (triIndices[1] < triIndices[0]) Swap(triIndices[0], triIndices[1]);
    };

    // First, reference count every vertex
    for (int i = 0; i < numIndices / 3; i++)
    {
        if (!triangleIsRemoved.GetBit(i))
        {
            // TODO: i don't understand why I have to do this
            u32 triIndices[3];
            GetSortedIndices(triIndices, i);

            int hash = MixBits(triIndices[0] ^ triIndices[1] ^ triIndices[2]);
            triangleHash.AddInHash(hash, i);
            for (int corner = 0; corner < 3; corner++)
            {
                int index = indices[3 * i + corner];
                remap[index]++;
            }
        }
    }

    // If a vertex has a reference, compact it
    for (int i = 0; i < numVertices; i++)
    {
        if (remap[i] > 0)
        {
            f32 *src = vertexData + (3 + numAttributes) * i;
            f32 *dst = vertexData + (3 + numAttributes) * vertexCount;
            MemoryCopy(dst, src, attributeLen);
            remap[i] = vertexCount++;
        }
    }

    // Update index buffer
    for (int i = 0; i < numIndices / 3; i++)
    {
        if (!triangleIsRemoved.GetBit(i))
        {
            geomIDs[triangleCount] = geomIDs[i];

            u32 triIndices[3];
            GetSortedIndices(triIndices, i);

            int hash  = MixBits(triIndices[0] ^ triIndices[1] ^ triIndices[2]);
            bool dupe = false;
            for (int hashIndex = triangleHash.FirstInHash(hash); hashIndex != -1;
                 hashIndex     = triangleHash.NextInHash(hashIndex))
            {
                if (hashIndex == i) break;
                u32 otherIndices[3];
                GetSortedIndices(otherIndices, hashIndex);
                if (triIndices[0] == otherIndices[0] && triIndices[1] == otherIndices[1] &&
                    triIndices[2] == otherIndices[2])
                {
                    dupe = true;
                    // Print("Removed duplicate triangle in final phase\n");
                    break;
                }
            }
            if (dupe) continue;

            for (u32 corner = 0; corner < 3; corner++)
            {
                u32 vertIndex                       = indices[3 * i + corner];
                u32 remappedIndex                   = remap[vertIndex];
                indices[3 * triangleCount + corner] = remappedIndex;
            }
            triangleCount++;
        }
    }

    for (int i = 0; i < triangleCount; i++)
    {
        Vec3f p0 = GetPosition(indices[3 * i + 0]);
        Vec3f p1 = GetPosition(indices[3 * i + 1]);
        Vec3f p2 = GetPosition(indices[3 * i + 2]);
        Assert(p0 != p1 && p1 != p2 && p0 != p2);
    }

    finalNumVertices = vertexCount;
    finalNumIndices  = 3 * triangleCount;

    // Print("%u %u\n", finalNumVertices, finalNumIndices);
}

inline Vec4f ConstructSphereFromPoints(Vec3f *points, u32 numPoints)
{
    u32 min[3] = {};
    u32 max[3] = {};
    for (u32 i = 0; i < numPoints; i++)
    {
        for (u32 axis = 0; axis < 3; axis++)
        {
            min[axis] = points[i][axis] < points[min[axis]][axis] ? i : min[axis];
            max[axis] = points[i][axis] > points[max[axis]][axis] ? i : max[axis];
        }
    }

    f32 largestDistSqr = 0.f;
    u32 chosenAxis     = 0;
    for (u32 axis = 0; axis < 3; axis++)
    {
        f32 distSqr = LengthSquared(points[min[axis]] - points[max[axis]]);
        if (distSqr > largestDistSqr)
        {
            largestDistSqr = distSqr;
            chosenAxis     = axis;
        }
    }

    Vec3f center  = 0.5f * (points[min[chosenAxis]] + points[max[chosenAxis]]);
    f32 radius    = Length(center - points[min[chosenAxis]]);
    f32 radiusSqr = Sqr(radius);

    for (u32 i = 0; i < numPoints; i++)
    {
        f32 distSqr = LengthSquared(center - points[i]);
        if (distSqr > radiusSqr)
        {
            f32 dist = Sqrt(distSqr);
            f32 t    = 0.5f + 0.5f * (radius / dist);
            center   = Lerp(t, points[i], center);
            radius   = 0.5f * (radius + dist);
        }
    }

    return Vec4f(center, radius);
}

inline Vec4f ConstructSphereFromSpheres(Vec4f *spheres, u32 numSpheres)
{
    u32 min[3] = {};
    u32 max[3] = {};
    for (u32 i = 0; i < numSpheres; i++)
    {
        for (u32 axis = 0; axis < 3; axis++)
        {
            min[axis] = spheres[i][axis] < spheres[min[axis]][axis] ? i : min[axis];
            max[axis] = spheres[i][axis] > spheres[max[axis]][axis] ? i : max[axis];
        }
    }

    f32 largestDistSqr = 0.f;
    u32 chosenAxis     = 0;
    for (u32 axis = 0; axis < 3; axis++)
    {
        f32 distSqr = LengthSquared(spheres[min[axis]].xyz - spheres[max[axis]].xyz);
        if (distSqr > largestDistSqr)
        {
            largestDistSqr = distSqr;
            chosenAxis     = axis;
        }
    }

    // Start adding spheres
    auto AddSpheres = [&](const Vec4f &sphere0, const Vec4f &sphere1) {
        Vec3f toOther = sphere1.xyz - sphere0.xyz;
        f32 distSqr   = LengthSquared(toOther);
        if (Sqr(sphere0.w - sphere1.w) >= distSqr)
        {
            return sphere0.w < sphere1.w ? sphere1 : sphere0;
        }
        f32 dist        = Sqrt(distSqr);
        f32 newRadius   = (dist + sphere0.w + sphere1.w) * 0.5f;
        Vec3f newCenter = sphere0.xyz;
        if (dist > 1e-8f) newCenter += toOther * ((newRadius - sphere0.w) / dist);
        f32 tolerance = 1e-4f;

        return Vec4f(newCenter, newRadius);
    };

    Vec4f newSphere = spheres[min[chosenAxis]];
    newSphere       = AddSpheres(newSphere, spheres[max[chosenAxis]]);

    for (u32 i = 0; i < numSpheres; i++)
    {
        newSphere = AddSpheres(newSphere, spheres[i]);
    }

    return newSphere;
}

struct ClusterGroup
{
    Vec4f lodBounds;
    f32 *vertexData;
    u32 *indices;

    u32 buildDataIndex;

    u32 clusterStartIndex;
    u32 clusterCount;

    u32 parentStartIndex;
    u32 parentCount;

    u32 pageStartIndex;
    u32 numPages;

    f32 maxParentError;

    u32 partStartIndex;
    u32 numParts;

    // Debug
    u32 numVertices;
    u32 numIndices;

    u32 mipLevel;

    bool isLeaf;
    bool hasVoxels;
};

struct Cluster
{
    // RecordAOSSplits record;
    Bounds bounds;
    Vec4f lodBounds;
    u32 mipLevel;

    u32 groupIndex;
    u32 childGroupIndex;

    u32 headerIndex;

    StaticArray<int> triangleIndices;
    StaticArray<u32> geomIDs;
    StaticArray<CompressedVoxel> compressedVoxels;
    StaticArray<Vec3i> extraVoxels;

    f32 lodError;
};

int HashEdge(Vec3f &p0, Vec3f &p1)
{
    int hash0 = Hash(p0);
    int hash1 = Hash(p1);

    if (hash1 < hash0)
    {
        Swap(hash0, hash1);
        Swap(p0, p1);
    }
    int hash = Hash(hash0, hash1);
    return hash;
}

struct PartitionRange
{
    u32 begin;
    u32 end;
};

void PartitionGraph(ArrayView<int> clusterIndices, ArrayView<idx_t> clusterOffsets,
                    ArrayView<idx_t> clusterData, ArrayView<idx_t> clusterWeights,
                    ArrayView<int> newClusterIndices, ArrayView<idx_t> newClusterOffsets,
                    ArrayView<idx_t> newClusterData, ArrayView<idx_t> newClusterWeights,
                    int numClusters, PartitionRange &left, PartitionRange &right,
                    u32 &newAdjOffset, u32 numAdjacency, Vec2u &newNumAdjacency,
                    u32 minGroupSize, u32 maxGroupSize)

{
    Assert(numAdjacency == clusterData.Length());
    i32 numConstraints = 1;
    i32 numParts       = 2;

    ScratchArena scratch;

    idx_t *partitionIDs = PushArrayNoZero(scratch.temp.arena, idx_t, numClusters);
    StaticArray<idx_t> tempClusterOffsets(scratch.temp.arena, numClusters + 1);

    clusterOffsets.Copy(tempClusterOffsets);
    tempClusterOffsets.Push(numAdjacency);

    idx_t edgesCut                = 0;
    const u32 maxClustersPerGroup = 32;
    const int targetPartitionSize = (minGroupSize + maxGroupSize) / 2;
    const int targetNumPartitions =
        Max(2, (numClusters + (targetPartitionSize / 2)) / targetPartitionSize);

    real_t partitionWeights[2] = {
        float(targetNumPartitions / 2) / targetNumPartitions,
        1.f - float(targetNumPartitions / 2) / targetNumPartitions,
    };

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    options[METIS_OPTION_UFACTOR] =
        (maxGroupSize / minGroupSize > 1 || targetNumPartitions >= 128) ? 200 : 1;

    static Mutex mutex = {};
    BeginMutex(&mutex);
    int result =
        METIS_PartGraphRecursive(&numClusters, &numConstraints, tempClusterOffsets.data,
                                 clusterData.data, NULL, NULL, clusterWeights.data, &numParts,
                                 partitionWeights, NULL, options, &edgesCut, partitionIDs);
    EndMutex(&mutex);

    ErrorExit(result == METIS_OK, "Metis error\n");

    u32 numClustersLeft     = 0;
    u32 maxNumAdjacencyLeft = 0;

    u32 numPartition[2] = {};

    u32 *remap = PushArrayNoZero(scratch.temp.arena, u32, numClusters);

    for (int clusterIndex = 0; clusterIndex < numClusters; clusterIndex++)
    {
        u32 partitionID = partitionIDs[clusterIndex];
        numClustersLeft += partitionID == 0 ? 1 : 0;
        maxNumAdjacencyLeft += partitionID == 0 ? tempClusterOffsets[clusterIndex + 1] -
                                                      tempClusterOffsets[clusterIndex]
                                                : 0;

        remap[clusterIndex] = numPartition[partitionID]++;
    }

    u32 partitionStart[2];
    partitionStart[0] = 0;
    partitionStart[1] = numClustersLeft;

    u32 partitionOffsets[2];
    partitionOffsets[0] = 0;
    partitionOffsets[1] = numClustersLeft;

    // Global offset into array
    u32 adjacencyOffsets[2];
    adjacencyOffsets[0] = 0;
    adjacencyOffsets[1] = maxNumAdjacencyLeft;

    // Local offset into array
    u32 localAdjacencyOffsets[2] = {};

    for (int clusterIndex = 0; clusterIndex < numClusters; clusterIndex++)
    {
        u32 partitionID     = partitionIDs[clusterIndex];
        u32 partitionOffset = partitionOffsets[partitionID]++;

        newClusterIndices[partitionOffset] = clusterIndices[clusterIndex];

        u32 &adjOffset      = adjacencyOffsets[partitionID];
        u32 &localAdjOffset = localAdjacencyOffsets[partitionID];

        newClusterOffsets[partitionOffset] = localAdjOffset;

        for (int offset = tempClusterOffsets[clusterIndex];
             offset < tempClusterOffsets[clusterIndex + 1]; offset++)
        {
            int neighborClusterIndex = clusterData[offset];

            // If they are in the same cluster, maintain the edge
            if (partitionIDs[neighborClusterIndex] == partitionID)
            {
                newClusterData[adjOffset]    = remap[neighborClusterIndex];
                newClusterWeights[adjOffset] = clusterWeights[offset];
                adjOffset++;
                localAdjOffset++;
            }
        }
    }

    Assert(adjacencyOffsets[0] <= maxNumAdjacencyLeft);
    Assert(adjacencyOffsets[1] <= numAdjacency);
    Assert(partitionOffsets[0] == numClustersLeft);

    left.begin  = 0;
    left.end    = numClustersLeft;
    right.begin = numClustersLeft;
    right.end   = partitionOffsets[1];

    newAdjOffset    = maxNumAdjacencyLeft;
    newNumAdjacency = Vec2u(localAdjacencyOffsets[0], localAdjacencyOffsets[1]);
}

bool RecursivePartitionGraph(ArrayView<int> clusterIndices, ArrayView<idx_t> clusterOffsets,
                             ArrayView<idx_t> clusterData, ArrayView<idx_t> clusterWeights,
                             ArrayView<int> newClusterIndices,
                             ArrayView<idx_t> newClusterOffsets,
                             ArrayView<idx_t> newClusterData,
                             ArrayView<idx_t> newClusterWeights, int numClusters, int numSwaps,
                             int globalClusterOffset, StaticArray<PartitionRange> &ranges,
                             std::atomic<int> &numPartitions, u32 numAdjacency,
                             u32 minGroupSize, u32 maxGroupSize)
{
    if (numClusters <= maxGroupSize)
    {
        int rangeIndex = numPartitions.fetch_add(1, std::memory_order_relaxed);

        PartitionRange range;
        range.begin = globalClusterOffset;
        range.end   = range.begin + numClusters;

        if (rangeIndex == ranges.Length()) return false;

        ranges[rangeIndex] = range;

        if (numSwaps & 1)
        {
            MemoryCopy(newClusterIndices.data, clusterIndices.data, sizeof(int) * numClusters);
        }
        return true;
    }

    u32 newAdjOffset;
    Vec2u newNumAdjacency;
    PartitionRange left, right;

    PartitionGraph(clusterIndices, clusterOffsets, clusterData, clusterWeights,
                   newClusterIndices, newClusterOffsets, newClusterData, newClusterWeights,
                   numClusters, left, right, newAdjOffset, numAdjacency, newNumAdjacency,
                   minGroupSize, maxGroupSize);

    u32 numLeft  = left.end - left.begin;
    u32 numRight = right.end - right.begin;

    auto Recurse = [&](int jobID) {
        u32 clusterOffset  = jobID == 0 ? 0 : right.begin;
        u32 newNumClusters = jobID == 0 ? numLeft : numRight;
        u32 adjOffset      = jobID == 0 ? 0 : newAdjOffset;
        u32 numAdjacency   = newNumAdjacency[jobID];

        return RecursivePartitionGraph(
            ArrayView<idx_t>(newClusterIndices, clusterOffset, newNumClusters),
            ArrayView<idx_t>(newClusterOffsets, clusterOffset, newNumClusters),
            ArrayView<idx_t>(newClusterData, adjOffset, numAdjacency),
            ArrayView<idx_t>(newClusterWeights, adjOffset, numAdjacency),
            ArrayView<idx_t>(clusterIndices, clusterOffset, newNumClusters),
            ArrayView<idx_t>(clusterOffsets, clusterOffset, newNumClusters),
            ArrayView<idx_t>(clusterData, adjOffset, numAdjacency),
            ArrayView<idx_t>(clusterWeights, adjOffset, numAdjacency), newNumClusters,
            numSwaps + 1, globalClusterOffset + clusterOffset, ranges, numPartitions,
            numAdjacency, minGroupSize, maxGroupSize);
    };

    // TODO: for whatever reason multithreading METIS causes inscrutable errors. fix this if
    // speedup needed

    // if (numClusters > 256)
    // {
    //     scheduler.ScheduleAndWait(2, 1, Recurse);
    // }
    // else
    bool result = true;
    {
        result = result && Recurse(0);
        result = result && Recurse(1);
    }
    return result;
}

struct GraphPartitionResult
{
    StaticArray<PartitionRange> ranges;
    StaticArray<int> clusterIndices;
    bool success;
};

GraphPartitionResult RecursivePartitionGraph(Arena *arena, idx_t *clusterOffsets,
                                             idx_t *clusterData, idx_t *clusterWeights,
                                             int numClusters, u32 dataSize, u32 minGroupSize,
                                             u32 maxGroupSize)
{
    ScratchArena scratch(&arena, 1);

    u32 maxNumPartitions = (numClusters + minGroupSize - 1) / minGroupSize;

    StaticArray<int> clusterIndices(arena, numClusters);
    StaticArray<int> newClusterIndices(arena, numClusters, numClusters);
    for (int i = 0; i < numClusters; i++)
    {
        clusterIndices.Push(i);
    }

    StaticArray<idx_t> newClusterOffsets(scratch.temp.arena, numClusters, numClusters);
    StaticArray<idx_t> newClusterData(scratch.temp.arena, dataSize, dataSize);
    StaticArray<idx_t> newClusterWeights(scratch.temp.arena, dataSize, dataSize);

    std::atomic<int> numPartitions(0);

    StaticArray<PartitionRange> ranges(arena, maxNumPartitions, maxNumPartitions);

    bool success = RecursivePartitionGraph(
        ArrayView<int>(clusterIndices), ArrayView<idx_t>(clusterOffsets, (u32)numClusters),
        ArrayView<idx_t>(clusterData, dataSize), ArrayView<idx_t>(clusterWeights, dataSize),
        ArrayView<int>(newClusterIndices), ArrayView<idx_t>(newClusterOffsets),
        ArrayView<idx_t>(newClusterData), ArrayView<idx_t>(newClusterWeights), numClusters, 0,
        0, ranges, numPartitions, dataSize, minGroupSize, maxGroupSize);

    ranges.size() = numPartitions.load();

    GraphPartitionResult result;
    result.ranges         = ranges;
    result.clusterIndices = clusterIndices;
    result.success        = success;

    return result;
}

static_assert(sizeof(PackedDenseGeometryHeader) % 4 == 0, "Header is mult of 4 bytes");

struct HierarchyNode
{
    Bounds bounds[CHILDREN_PER_HIERARCHY_NODE];
    Vec4f lodBounds[CHILDREN_PER_HIERARCHY_NODE];
    f32 maxParentError[CHILDREN_PER_HIERARCHY_NODE];
    HierarchyNode *children;

    u32 partIndices[CHILDREN_PER_HIERARCHY_NODE];
    u32 clusterTotals[CHILDREN_PER_HIERARCHY_NODE];
    u32 numChildren;
};

struct GroupPart
{
    u32 groupIndex;
    u32 clusterStartIndex;
    u32 clusterCount;
    u32 clusterPageStartIndex;
    u32 pageIndex;
};

HierarchyNode BuildHierarchy(Arena *arena, const Array<Cluster> &clusters,
                             const Array<ClusterGroup> &clusterGroups,
                             const StaticArray<GroupPart> &parts, PrimRef *primRefs,
                             RecordAOSSplits &record, u32 &numNodes)
{
    typedef HeuristicObjectBinning<PrimRef> Heuristic;

    HeuristicObjectBinning<PrimRef> heuristic(primRefs, 0); // Log2Int(4));

    Assert(record.count > 0);

    RecordAOSSplits childRecords[CHILDREN_PER_HIERARCHY_NODE];
    u32 numChildren = 0;

    Split split = heuristic.Bin(record);

    if (record.count <= CHILDREN_PER_HIERARCHY_NODE)
    {
        u32 threadIndex = GetThreadIndex();
        heuristic.FlushState(split);

        HierarchyNode node;
        node.children    = 0;
        node.numChildren = record.count;

        for (int i = 0; i < record.count; i++)
        {
            PrimRef &ref              = primRefs[record.start + i];
            u32 partID                = ref.primID;
            u32 groupID               = parts[partID].groupIndex;
            const ClusterGroup &group = clusterGroups[groupID];
            Vec4f lodBounds           = group.lodBounds;

            node.bounds[i]         = Bounds(Lane4F32(-ref.minX, -ref.minY, -ref.minZ, 0.f),
                                            Lane4F32(ref.maxX, ref.maxY, ref.maxZ, 0.f));
            node.lodBounds[i]      = lodBounds;
            node.partIndices[i]    = partID;
            node.clusterTotals[i]  = parts[partID].clusterCount;
            node.maxParentError[i] = group.maxParentError;
        }

        node.children = 0;
        numNodes++;

        return node;
    }
    heuristic.Split(split, record, childRecords[0], childRecords[1]);

    // N - 1 splits produces N children
    for (numChildren = 2; numChildren < CHILDREN_PER_HIERARCHY_NODE; numChildren++)
    {
        i32 bestChild = -1;
        f32 maxArea   = neg_inf;
        for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
        {
            RecordAOSSplits &childRecord = childRecords[recordIndex];
            if (childRecord.count <= CHILDREN_PER_HIERARCHY_NODE) continue;

            f32 childArea = HalfArea(childRecord.geomBounds);
            if (childArea > maxArea)
            {
                bestChild = recordIndex;
                maxArea   = childArea;
            }
        }
        if (bestChild == -1) break;

        split = heuristic.Bin(childRecords[bestChild]);

        RecordAOSSplits out;
        heuristic.Split(split, childRecords[bestChild], out, childRecords[numChildren]);

        childRecords[bestChild] = out;
    }

    HierarchyNode *nodes = PushArrayNoZero(arena, HierarchyNode, numChildren);
    for (int i = 0; i < numChildren; i++)
    {
        nodes[i] = BuildHierarchy(arena, clusters, clusterGroups, parts, primRefs,
                                  childRecords[i], numNodes);
    }

    ScratchArena scratch;

    HierarchyNode node;
    node.children    = nodes;
    node.numChildren = numChildren;

    for (int i = 0; i < numChildren; i++)
    {
        f32 maxParentError       = 0.f;
        HierarchyNode &childNode = nodes[i];
        Vec4f *spheres   = PushArrayNoZero(scratch.temp.arena, Vec4f, childNode.numChildren);
        u32 clusterTotal = 0;
        Bounds bounds;
        for (int j = 0; j < childNode.numChildren; j++)
        {
            bounds.Extend(childNode.bounds[j]);
            spheres[j]     = childNode.lodBounds[j];
            maxParentError = Max(maxParentError, childNode.maxParentError[j]);
            clusterTotal += childNode.clusterTotals[j];
        }

        node.bounds[i]         = bounds;
        node.lodBounds[i]      = ConstructSphereFromSpheres(spheres, childNode.numChildren);
        node.maxParentError[i] = maxParentError;
        node.clusterTotals[i]  = clusterTotal;
    }

    numNodes++;
    return node;
}

HierarchyNode BuildTopLevelHierarchy(Arena *arena,
                                     const StaticArray<HierarchyNode> &hierarchyNodes,
                                     PrimRef *primRefs, RecordAOSSplits &record, u32 &numNodes)
{
    typedef HeuristicObjectBinning<PrimRef> Heuristic;

    HeuristicObjectBinning<PrimRef> heuristic(primRefs, 0); // Log2Int(4));

    Assert(record.count > 0);

    RecordAOSSplits childRecords[CHILDREN_PER_HIERARCHY_NODE];
    u32 numChildren = 0;

    Split split = heuristic.Bin(record);

    if (record.count == 1)
    {
        u32 threadIndex = GetThreadIndex();
        heuristic.FlushState(split);

        return hierarchyNodes[primRefs[record.start].primID];
    }
    heuristic.Split(split, record, childRecords[0], childRecords[1]);

    // N - 1 splits produces N children
    for (numChildren = 2; numChildren < CHILDREN_PER_HIERARCHY_NODE; numChildren++)
    {
        i32 bestChild = -1;
        f32 maxArea   = neg_inf;
        for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
        {
            RecordAOSSplits &childRecord = childRecords[recordIndex];
            if (childRecord.count <= CHILDREN_PER_HIERARCHY_NODE) continue;

            f32 childArea = HalfArea(childRecord.geomBounds);
            if (childArea > maxArea)
            {
                bestChild = recordIndex;
                maxArea   = childArea;
            }
        }
        if (bestChild == -1) break;

        split = heuristic.Bin(childRecords[bestChild]);

        RecordAOSSplits out;
        heuristic.Split(split, childRecords[bestChild], out, childRecords[numChildren]);

        childRecords[bestChild] = out;
    }

    HierarchyNode *nodes = PushArrayNoZero(arena, HierarchyNode, numChildren);
    for (int i = 0; i < numChildren; i++)
    {
        nodes[i] =
            BuildTopLevelHierarchy(arena, hierarchyNodes, primRefs, childRecords[i], numNodes);
    }

    ScratchArena scratch;

    HierarchyNode node;
    node.children    = nodes;
    node.numChildren = numChildren;

    for (int i = 0; i < numChildren; i++)
    {
        f32 maxParentError       = 0.f;
        HierarchyNode &childNode = nodes[i];
        Vec4f *spheres   = PushArrayNoZero(scratch.temp.arena, Vec4f, childNode.numChildren);
        u32 clusterTotal = 0;
        Bounds bounds;
        for (int j = 0; j < childNode.numChildren; j++)
        {
            bounds.Extend(childNode.bounds[j]);
            spheres[j]     = childNode.lodBounds[j];
            maxParentError = Max(maxParentError, childNode.maxParentError[j]);
            clusterTotal += childNode.clusterTotals[j];
        }

        node.bounds[i]         = bounds;
        node.lodBounds[i]      = ConstructSphereFromSpheres(spheres, childNode.numChildren);
        node.maxParentError[i] = maxParentError;
        node.clusterTotals[i]  = clusterTotal;
    }

    numNodes++;
    return node;
}

static_assert((sizeof(PackedDenseGeometryHeader) + 4) % 16 == 0, "bad header size");

static Vec3f &GetPosition(f32 *vertexData, u32 vertexIndex, u32 numAttributes)
{
    return *(Vec3f *)(vertexData + (3 + numAttributes) * vertexIndex);
}

static void AddSpatialLinks(Arena *arena, u32 num, Bounds &bounds,
                            const StaticArray<Vec3f> &centers,
                            StaticArray<Array<int>> &graphNeighbors,
                            StaticArray<Array<int>> &weights)
{
    struct Handle
    {
        u32 sortKey;
        u32 index;
    };

    ScratchArena scratch(&arena, 1);

    Vec3f scale = 1023.f / ToVec3f(bounds.maxP - bounds.minP);
    Vec3f minP  = ToVec3f(bounds.minP);
    StaticArray<Handle> handles(scratch.temp.arena, num, num);

    ParallelFor(0, num, 4096, 4096, [&](u32 jobID, u32 start, u32 count) {
        for (int index = start; index < start + count; index++)
        {
            Vec3f center = centers[index];

            Vec3i quantized = Vec3i((center - minP) * scale);
            u32 morton      = MortonCode3(quantized.x) | (MortonCode3(quantized.y) << 1) |
                         (MortonCode3(quantized.z) << 2);

            Handle handle;
            handle.sortKey = morton;
            handle.index   = index;
            handles[index] = handle;
        }
    });

    SortHandles(handles.data, handles.Length());

    StaticArray<u32> islandIDs(scratch.temp.arena, num, num);
    BitVector visited(scratch.temp.arena, num);
    u32 numIslands = 0;

    StaticArray<u32> stack(scratch.temp.arena, num);
    for (int index = 0; index < num; index++)
    {
        stack.Empty();
        u32 islandIndex = numIslands;
        if (!visited.GetBit(index))
        {
            numIslands++;
            visited.SetBit(index);
            stack.Push(index);
        }
        while (stack.Length())
        {
            u32 neighbor        = stack.Pop();
            islandIDs[neighbor] = islandIndex;

            for (u32 neighborNeighbor : graphNeighbors[neighbor])
            {
                if (!visited.GetBit(neighborNeighbor))
                {
                    visited.SetBit(neighborNeighbor);
                    stack.Push(neighborNeighbor);
                }
            }
        }
    }

    if (numIslands == 1) return;

    StaticArray<PartitionRange> islandRanges(scratch.temp.arena, num, num);
    u32 currentIsland    = islandIDs[handles[0].index];
    int startHandleIndex = 0;
    for (int handleIndex = 1; handleIndex < num; handleIndex++)
    {
        Handle handle = handles[handleIndex];
        u32 islandID  = islandIDs[handle.index];

        if (islandID != currentIsland)
        {
            for (int i = startHandleIndex; i < handleIndex; i++)
            {
                PartitionRange range;
                range.begin     = startHandleIndex;
                range.end       = handleIndex;
                islandRanges[i] = range;
            }
            startHandleIndex = handleIndex;
            currentIsland    = islandID;
        }
    }
    for (int i = startHandleIndex; i < num; i++)
    {
        PartitionRange range;
        range.begin     = startHandleIndex;
        range.end       = num;
        islandRanges[i] = range;
    }

    const u32 numSpatialLinks = 5;
    struct AdjInfo
    {
        FixedArray<u32, numSpatialLinks> neighbors;
        FixedArray<f32, numSpatialLinks> dists;
    };
    StaticArray<AdjInfo> adjInfos(scratch.temp.arena, num, num);

    u32 *offsets = PushArray(scratch.temp.arena, u32, num + 1);

    for (int handleIndex = 0; handleIndex < num; handleIndex++)
    {
        Handle handle       = handles[handleIndex];
        u32 baseIslandID    = islandIDs[handle.index];
        u32 currentIslandID = islandIDs[handle.index];

        if (islandRanges[handleIndex].end - islandRanges[handleIndex].begin >
            MAX_CLUSTER_TRIANGLES)
            continue;

        Vec3f center = centers[handle.index];

        u32 neighbors[numSpatialLinks];
        f32 dists[numSpatialLinks];
        for (int i = 0; i < numSpatialLinks; i++)
        {
            neighbors[i] = ~0u;
            dists[i]     = pos_inf;
        }
        for (int direction = -1; direction <= 1; direction += 2)
        {
            int currentHandleIndex = handleIndex + direction;
            for (int i = 0; i < 16; i++)
            {
                if (currentHandleIndex > num - 1 || currentHandleIndex < 0) break;

                Handle neighborHandle = handles[currentHandleIndex];
                u32 islandID          = islandIDs[neighborHandle.index];

                if (islandID != baseIslandID)
                {
                    Vec3f neighborCenter = centers[neighborHandle.index];

                    f32 distSqr = LengthSquared(center - neighborCenter);
                    u32 index   = neighborHandle.index;
                    for (int i = 0; i < numSpatialLinks; i++)
                    {
                        if (distSqr < dists[i])
                        {
                            Swap(distSqr, dists[i]);
                            Swap(index, neighbors[i]);
                        }
                    }
                    currentHandleIndex += direction;
                }
                else
                {
                    currentHandleIndex = direction == 1
                                             ? islandRanges[currentHandleIndex].end
                                             : islandRanges[currentHandleIndex].begin - 1;
                }
            }
        }

        for (int i = 0; i < numSpatialLinks; i++)
        {
            if (neighbors[i] != ~0u)
            {
                if (graphNeighbors[handle.index].AddUnique(neighbors[i]))
                {
                    weights[handle.index].Push(1);
                }

                if (graphNeighbors[neighbors[i]].AddUnique(handle.index))
                {
                    weights[neighbors[i]].Push(1);
                }
            }
        }
    }
}

static GraphPartitionResult PartitionTriangles(Arena *arena, Mesh &mesh, u32 maxNumTriangles)
{
    const u32 minGroupSize = maxNumTriangles - 4;
    const u32 maxGroupSize = maxNumTriangles;

    ScratchArena scratch(&arena, 1);

    u32 hashSize = NextPowerOfTwo(mesh.numIndices);
    HashIndex cornerHash(scratch.temp.arena, hashSize, hashSize);
    for (u32 index = 0; index < mesh.numIndices; index++)
    {
        Vec3f pos = mesh.p[mesh.indices[index]];
        int hashP = Hash(pos);

        cornerHash.AddInHash(hashP, index);
    }

    Graph<u32> triangleAdjGraph;
    u32 total = triangleAdjGraph.InitializeStatic(
        scratch.temp.arena, mesh.numFaces, [&](u32 tri, u32 *offsets, u32 *data = 0) {
            ScratchArena scratch(&arena, 1);

            Array<u32> tris(scratch.temp.arena, 8);
            u32 num = 0;
            for (int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
            {
                Vec3f &p = mesh.p[mesh.indices[3 * tri + vertexIndex]];
                int hash = Hash(p);
                for (int hashIndex = cornerHash.FirstInHash(hash); hashIndex != -1;
                     hashIndex     = cornerHash.NextInHash(hashIndex))
                {
                    u32 otherTri = (u32)hashIndex / 3;
                    if (otherTri != tri)
                    {
                        Vec3f &otherP = mesh.p[mesh.indices[hashIndex]];
                        if (otherP == p)
                        {
                            if (tris.AddUnique(otherTri))
                            {
                                num++;
                                u32 dataIndex = offsets[tri]++;
                                if (data) data[dataIndex] = otherTri;
                            }
                        }
                    }
                }
            }
            return num;
        });

    Bounds totalBounds;

    StaticArray<Vec3f> centers(scratch.temp.arena, mesh.numFaces);
    StaticArray<Array<int>> graphNeighbors(scratch.temp.arena, mesh.numFaces, mesh.numFaces);
    StaticArray<Array<int>> graphWeights(scratch.temp.arena, mesh.numFaces, mesh.numFaces);

    for (int tri = 0; tri < mesh.numFaces; tri++)
    {
        Vec3f p0 = mesh.p[mesh.indices[3 * tri + 0]];
        Vec3f p1 = mesh.p[mesh.indices[3 * tri + 1]];
        Vec3f p2 = mesh.p[mesh.indices[3 * tri + 2]];

        Vec3f center = (p0 + p1 + p2) / 3.f;
        centers.Push(center);
        totalBounds.Extend(Lane4F32(p0));
        totalBounds.Extend(Lane4F32(p1));
        totalBounds.Extend(Lane4F32(p2));

        u32 count = triangleAdjGraph.offsets[tri + 1] - triangleAdjGraph.offsets[tri];
        graphNeighbors[tri] = Array<int>(scratch.temp.arena, count);
        MemoryCopy(graphNeighbors[tri].data,
                   triangleAdjGraph.data + triangleAdjGraph.offsets[tri], sizeof(u32) * count);
        graphNeighbors[tri].size = count;

        graphWeights[tri] = Array<int>(scratch.temp.arena, count);
        for (int i = 0; i < count; i++)
        {
            graphWeights[tri].Push(4 * 65);
        }
    }

    AddSpatialLinks(scratch.temp.arena, mesh.numFaces, totalBounds, centers, graphNeighbors,
                    graphWeights);

    idx_t *offsets  = PushArray(scratch.temp.arena, idx_t, total + 1);
    idx_t *offsets1 = &offsets[1];

    idx_t totalData = 0;

    for (int tri = 0; tri < mesh.numFaces; tri++)
    {
        idx_t num     = (idx_t)graphNeighbors[tri].Length();
        offsets1[tri] = totalData;
        totalData += num;
    }

    idx_t *weights = PushArrayNoZero(scratch.temp.arena, idx_t, totalData);
    idx_t *data    = PushArrayNoZero(scratch.temp.arena, idx_t, totalData);

    for (int tri = 0; tri < mesh.numFaces; tri++)
    {
        for (int idx = 0; idx < graphNeighbors[tri].Length(); idx++)
        {
            idx_t dataIndex    = offsets1[tri]++;
            data[dataIndex]    = graphNeighbors[tri][idx];
            weights[dataIndex] = graphWeights[tri][idx];
        }
    }

    GraphPartitionResult result = RecursivePartitionGraph(
        arena, offsets, data, weights, mesh.numFaces, totalData, minGroupSize, maxGroupSize);

    return result;
}

static bool GenerateValidTriClusters(Arena *arena, Mesh &mesh, u32 maxNumTriangles,
                                     u32 maxNumClusters, const StaticArray<u32> &geomIDs,
                                     GraphPartitionResult &out)
{
    GraphPartitionResult result = PartitionTriangles(arena, mesh, maxNumTriangles);

    if (!result.success)
    {
        return false;
    }

    ScratchArena scratch(&arena, 1);
    for (u32 cluster = 0; cluster < result.ranges.Length(); cluster++)
    {
        PartitionRange range = result.ranges[cluster];
        if (range.end - range.begin > MAX_CLUSTER_TRIANGLES) return false;
        u32 clusterNumTriangles = range.end - range.begin;

        u32 hashSize = NextPowerOfTwo(clusterNumTriangles * 3);
        HashIndex vertexHashSet(scratch.temp.arena, hashSize, hashSize);

        u32 clusterStart = range.begin;
        u32 vertexCount  = 0;
        for (int i = clusterStart; i < clusterStart + clusterNumTriangles; i++)
        {
            u32 tri    = result.clusterIndices[i];
            u32 geomID = geomIDs[tri];
            for (int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
            {
                u32 index      = mesh.indices[3 * tri + vertexIndex];
                const Vec3f &p = mesh.p[index];
                int hash       = Hash(p);

                bool found = false;
                for (int hashIndex = vertexHashSet.FirstInHash(hash); hashIndex != -1;
                     hashIndex     = vertexHashSet.NextInHash(hashIndex))
                {
                    if (index == hashIndex || mesh.p[hashIndex] == p)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    vertexHashSet.AddInHash(hash, index);
                    vertexCount++;
                }
            }
        }

        if (vertexCount > MAX_CLUSTER_TRIANGLE_VERTICES)
        {
            Print("too many verts\n");
            return false;
        }
    }

    out = result;
    return true;
}

struct ClusterData
{
    Array<int> neighbors;
    Array<int> weights;
    StaticArray<int> externalEdges;
};

static void GetBrickMax(u64 bitMask, Vec3u &maxP)
{
    maxP.z = 4u - (LeadingZeroCount64(bitMask) >> 4u);

    u32 bits = (u32)bitMask | u32(bitMask >> 32u);
    bits |= bits << 16u;
    maxP.y = 4u - (LeadingZeroCount(bits) >> 2u);

    bits |= bits << 8u;
    bits |= bits << 4u;
    maxP.x = 4u - LeadingZeroCount(bits);

    Assert(maxP.x <= 4 && maxP.y <= 4 && maxP.z <= 4);
}

static void GenerateVoxelRefs(PrimRef *out, StaticArray<CompressedVoxel> &voxels,
                              Mesh &voxelMesh, RecordAOSSplits &record, f32 voxelSize)
{
    f32 extent = 0.5f * voxelSize;
    Assert(voxels.Length() == voxelMesh.numFaces);
    for (u32 voxelIndex = 0; voxelIndex < voxels.Length(); voxelIndex++)
    {
        CompressedVoxel &voxel = voxels[voxelIndex];
        Vec3u maxP;
        u64 bitMask = voxel.bitMask;
        GetBrickMax(bitMask, maxP);

        Vec3f voxelMin = voxelMesh.p[voxel.vertexOffset];
        Vec3f voxelMax = voxelMin + Vec3f(maxP) * voxelSize;

        PrimRef ref;
        for (int axis = 0; axis < 3; axis++)
        {
            ref.min[axis] = -voxelMin[axis];
            ref.max[axis] = voxelMax[axis];
            ref.primID    = voxelIndex;

            record.geomMin[axis] = Max(-voxelMin[axis], record.geomMin[axis]);
            record.geomMax[axis] = Max(voxelMax[axis], record.geomMax[axis]);

            f32 centroid         = voxelMin[axis] + voxelMax[axis];
            record.centMin[axis] = Max(record.centMin[axis], -centroid);
            record.centMax[axis] = Max(record.centMax[axis], centroid);
        }
        out[voxelIndex] = ref;
    }
}

static Mesh CompressVoxels(Arena *arena, StaticArray<Voxel> &voxels, StaticArray<u32> &geomIDs,
                           StaticArray<CompressedVoxel> &compressedVoxels, f32 voxelSize,
                           f32 *&coverages, SGGXCompact *&sggx)
{
    struct Handle
    {
        u32 sortKey;
        u32 index;
    };

    u32 numSlots = NextPowerOfTwo(voxels.Length());

    ScratchArena scratch(&arena, 1);
    Handle *handles = PushArrayNoZero(scratch.temp.arena, Handle, voxels.Length());
    HashIndex voxelHash(scratch.temp.arena, numSlots, numSlots);

    Vec3i minLoc(pos_inf);

    for (Voxel &voxel : voxels)
    {
        minLoc = Min(minLoc, voxel.loc);
    }

    for (u32 voxelIndex = 0; voxelIndex < voxels.Length(); voxelIndex++)
    {
        Handle handle;
        Voxel &voxel = voxels[voxelIndex];

        Vec3i loc           = voxel.loc - minLoc;
        u32 key             = (loc.x & 1023) | ((loc.y & 1023) << 10) | ((loc.z & 1023) << 20);
        handle.sortKey      = key;
        handle.index        = voxelIndex;
        handles[voxelIndex] = handle;

        voxelHash.AddInHash(MixBits(key), voxelIndex);
    }

    BitVector visited(scratch.temp.arena, voxels.Length());
    SortHandles(handles, voxels.Length());

    compressedVoxels = StaticArray<CompressedVoxel>(arena, voxels.Length());
    geomIDs          = StaticArray<u32>(arena, voxels.Length());

    Mesh outputMesh    = {};
    outputMesh.p       = PushArrayNoZero(arena, Vec3f, voxels.Length());
    outputMesh.n       = PushArrayNoZero(arena, Vec3f, voxels.Length());
    outputMesh.indices = PushArrayNoZero(arena, u32, 3 * voxels.Length());

    coverages = PushArrayNoZero(arena, f32, voxels.Length());
    sggx      = PushArrayNoZero(arena, SGGXCompact, voxels.Length());

    u32 numBricks = 0;
    u32 numVoxels = 0;
    for (u32 handleIndex = 0; handleIndex < voxels.Length(); handleIndex++)
    {
        u32 voxelIndex = handles[handleIndex].index;
        Voxel &voxel   = voxels[voxelIndex];
        if (visited.GetBit(voxelIndex)) continue;
        visited.SetBit(voxelIndex);
        u32 brickIndex   = numBricks++;
        u32 vertexOffset = numVoxels++;

        u32 key     = handles[handleIndex].sortKey;
        u64 bitMask = 0x1;

        Assert(geomIDs.Length() == vertexOffset);

        geomIDs.Push(voxel.geomID);
        outputMesh.p[vertexOffset] = Vec3f(voxel.loc) * voxelSize;
        outputMesh.n[vertexOffset] = voxel.normal;
        coverages[vertexOffset]    = voxel.coverage;
        sggx[vertexOffset]         = voxel.sggx;

        for (int z = 0; z < 4; z++)
        {
            for (int y = 0; y < 4; y++)
            {
                for (int x = 0; x < 4; x++)
                {
                    u32 bit      = x + y * 4 + z * 16;
                    Vec3i newLoc = voxel.loc + Vec3i(x, y, z);
                    Vec3i keyLoc = newLoc - minLoc;
                    u32 key      = (keyLoc.x & 1023) | ((keyLoc.y & 1023) << 10) |
                              ((keyLoc.z & 1023) << 20);
                    for (int hashIndex = voxelHash.FirstInHash(MixBits(key)); hashIndex != -1;
                         hashIndex     = voxelHash.NextInHash(hashIndex))
                    {
                        Voxel &neighbor = voxels[hashIndex];
                        if (!visited.GetBit(hashIndex) && neighbor.loc == newLoc)
                        {
                            visited.SetBit(hashIndex);
                            bitMask |= (1ull << bit);

                            u32 vertexIndex           = numVoxels++;
                            outputMesh.p[vertexIndex] = Vec3f(neighbor.loc) * voxelSize;
                            outputMesh.n[vertexIndex] = neighbor.normal;
                            coverages[vertexIndex]    = neighbor.coverage;
                            sggx[vertexIndex]         = neighbor.sggx;

                            geomIDs.Push(neighbor.geomID);
                            break;
                        }
                    }
                }
            }
        }

        CompressedVoxel compressedVoxel;
        compressedVoxel.bitMask      = bitMask;
        compressedVoxel.vertexOffset = vertexOffset;
        compressedVoxels.Push(compressedVoxel);
    }
    outputMesh.numVertices = numVoxels;
    outputMesh.numFaces    = numBricks;
    return outputMesh;
}

static void WriteClustersToOBJ(ArrayView<ClusterGroup> &clusterGroups,
                               ArrayView<Cluster> &clusters, string filename, u32 vertexCount,
                               u32 indexCount, u32 voxelCount, u32 numAttributes, u32 depth)
{
    ScratchArena scratch;

    auto GetPosition = [numAttributes](f32 *ptr, u32 index) -> Vec3f & {
        return *(Vec3f *)(ptr + (3 + numAttributes) * index);
    };

    vertexCount += 8 * voxelCount;
    indexCount += 36 * voxelCount;
    Mesh levelMesh    = {};
    u32 vertexOffset  = 0;
    u32 indexOffset   = 0;
    levelMesh.p       = PushArrayNoZero(scratch.temp.arena, Vec3f, vertexCount);
    levelMesh.indices = PushArrayNoZero(scratch.temp.arena, u32, indexCount);
    HashIndex vertexHash(scratch.temp.arena, NextPowerOfTwo(vertexCount),
                         NextPowerOfTwo(vertexCount));

    const u32 cubeTable[] = {
        // Front Face
        0, 1, 2, // Triangle 1
        1, 3, 2, // Triangle 2

        // Back Face
        4, 6, 5, // Triangle 3
        5, 6, 7, // Triangle 4

        // Left Face
        4, 0, 6, // Triangle 5
        0, 2, 6, // Triangle 6

        // Right Face
        1, 5, 3, // Triangle 7
        5, 7, 3, // Triangle 8

        // Top Face
        2, 3, 6, // Triangle 9
        3, 7, 6, // Triangle 10

        // Bottom Face
        4, 5, 0, // Triangle 11
        5, 1, 0  // Triangle 12;
    };

    for (int groupIndex = 0; groupIndex < clusterGroups.Length(); groupIndex++)
    {
        ClusterGroup &clusterGroup = clusterGroups[groupIndex];
        for (int i = 0; i < clusterGroup.numIndices; i++)
        {
            f32 *data =
                clusterGroup.vertexData + (3 + numAttributes) * clusterGroup.indices[i];
            Vec3f p         = *(Vec3f *)data;
            int hash        = Hash(p);
            int vertexIndex = -1;
            for (int hashIndex = vertexHash.FirstInHash(hash); hashIndex != -1;
                 hashIndex     = vertexHash.NextInHash(hashIndex))
            {
                if (levelMesh.p[hashIndex] == p)
                {
                    vertexIndex = hashIndex;
                    break;
                }
            }
            if (vertexIndex == -1)
            {
                vertexIndex              = vertexOffset++;
                levelMesh.p[vertexIndex] = p;
                vertexHash.AddInHash(hash, vertexIndex);
            }
            levelMesh.indices[indexOffset++] = vertexIndex;
        }

        f32 voxelSize = clusterGroup.maxParentError;

        for (int index = clusterGroup.parentStartIndex;
             index < clusterGroup.parentStartIndex + clusterGroup.parentCount; index++)
        {
            Cluster &cluster = clusters[index];
            for (CompressedVoxel &voxel : cluster.compressedVoxels)
            {
                u32 numVoxels = PopCount((u32)voxel.bitMask) + PopCount(voxel.bitMask >> 32u);
                for (u32 vertexIndex = voxel.vertexOffset;
                     vertexIndex < voxel.vertexOffset + numVoxels; vertexIndex++)
                {
                    Assert(vertexIndex < clusterGroup.numVertices);
                    Vec3f &pos = GetPosition(clusterGroup.vertexData, vertexIndex);

                    for (int z = 0; z < 2; z++)
                    {
                        for (int y = 0; y < 2; y++)
                        {
                            for (int x = 0; x < 2; x++)
                            {
                                Vec3f p = pos + voxelSize * Vec3f(x, y, -z);
                                levelMesh.p[vertexOffset++] = p;
                            }
                        }
                    }
                    for (int i = 0; i < 36; i++)
                    {
                        levelMesh.indices[indexOffset + i] = cubeTable[i] + vertexOffset - 8;
                    }
                    indexOffset += 36;
                }
            }
        }
    }

    levelMesh.numVertices = vertexOffset;
    levelMesh.numIndices  = indexOffset;
    WriteTriOBJ(levelMesh, PushStr8F(scratch.temp.arena, "%S_test_%u.obj",
                                     RemoveFileExtension(filename), depth));
}

// TODO: the builder is no longer deterministic after adding edge quadrics?
// also prevent the builder from solving to a garbage position
void CreateClusters(Mesh *meshes, u32 numMeshes, StaticArray<u32> &materialIndices,
                    string filename)
{
    // if (OS_FileExists(filename))
    // {
    //     Print("%S skipped\n", filename);
    //     return;
    // }

    const u32 minGroupSize = 8;
    const u32 maxGroupSize = 32;

    for (u32 i = 0; i < numMeshes; i++)
    {
        Assert(meshes[i].numIndices % 3 == 0);
        meshes[i].numFaces = meshes[i].numIndices / 3;
    }

    const u32 numAttributes = 0;

    auto GetVertexData = [numAttributes](f32 *ptr, u32 index) {
        return ptr + (3 + numAttributes) * index;
    };

    auto GetPosition = [numAttributes](f32 *ptr, u32 index) -> Vec3f & {
        return *(Vec3f *)(ptr + (3 + numAttributes) * index);
    };

    auto GetTriangle = [GetPosition](f32 *ptr, u32 *indices, u32 triangle, Vec3f &p0,
                                     Vec3f &p1, Vec3f &p2) {
        p0 = GetPosition(ptr, indices[3 * triangle + 0]);
        p1 = GetPosition(ptr, indices[3 * triangle + 1]);
        p2 = GetPosition(ptr, indices[3 * triangle + 2]);
    };

    ScratchArena scratch;

    RecordAOSSplits record;

    StaticArray<Vec2u> meshVertexOffsets(scratch.temp.arena, numMeshes);
    u32 totalNumVertices  = 0;
    u32 totalNumTriangles = 0;
    for (int i = 0; i < numMeshes; i++)
    {
        Vec2u offsets(totalNumVertices, totalNumTriangles);
        meshVertexOffsets.Push(offsets);
        totalNumVertices += meshes[i].numVertices;
        totalNumTriangles += meshes[i].numIndices / 3;
    }

    u32 totalClustersEstimate = ((totalNumTriangles) >> (MAX_CLUSTER_TRIANGLES_BIT - 1)) * 3;
    u32 totalGroupsEstimate   = (totalClustersEstimate + minGroupSize - 1) / minGroupSize;

    f32 *vertexData =
        PushArrayNoZero(scratch.temp.arena, f32, (3 + numAttributes) * totalNumVertices);
    u32 *indexData = PushArrayNoZero(scratch.temp.arena, u32, totalNumTriangles * 3);

    ParallelFor(0, numMeshes, 1, [&](int jobID, int start, int count) {
        for (int meshIndex = start; meshIndex < start + count; meshIndex++)
        {
            Mesh &mesh = meshes[meshIndex];
            // TODO: attributes
            u32 vertexOffset = meshVertexOffsets[meshIndex].x;
            u32 indexOffset  = 3 * meshVertexOffsets[meshIndex].y;
            MemoryCopy(vertexData + (3 + numAttributes) * vertexOffset, mesh.p,
                       sizeof(Vec3f) * mesh.numVertices);

            for (int indexIndex = 0; indexIndex < mesh.numIndices; indexIndex++)
            {
                indexData[indexOffset + indexIndex] = mesh.indices[indexIndex] + vertexOffset;
            }
        }
    });

    Mesh combinedMesh       = {};
    combinedMesh.p          = (Vec3f *)vertexData;
    combinedMesh.indices    = indexData;
    combinedMesh.numIndices = totalNumTriangles * 3;
    combinedMesh.numFaces   = totalNumTriangles;

    u32 hashSize = NextPowerOfTwo(totalNumVertices * 3);
    HashIndex vertexHashSet(scratch.temp.arena, hashSize, hashSize);
    StaticArray<u32> remap(scratch.temp.arena, totalNumVertices, totalNumVertices);
    u32 newNumVertices = 0;
    for (int vertexIndex = 0; vertexIndex < totalNumVertices; vertexIndex++)
    {
        Vec3f p  = combinedMesh.p[vertexIndex];
        int hash = Hash(p);

        bool found = false;
        for (int hashIndex = vertexHashSet.FirstInHash(hash); hashIndex != -1;
             hashIndex     = vertexHashSet.NextInHash(hashIndex))
        {
            if (combinedMesh.p[hashIndex] == p)
            {
                remap[vertexIndex] = hashIndex;
                found              = true;
                break;
            }
        }
        if (!found)
        {
            u32 newVertexIndex = newNumVertices++;
            remap[vertexIndex] = newVertexIndex;
            vertexHashSet.AddInHash(hash, newVertexIndex);
            combinedMesh.p[newVertexIndex] = p;
        }
    }

    for (int indexIndex = 0; indexIndex < totalNumTriangles * 3; indexIndex++)
    {
        combinedMesh.indices[indexIndex] = remap[combinedMesh.indices[indexIndex]];
    }

    combinedMesh.numVertices = newNumVertices;

    Arena **arenas = GetArenaArray(scratch.temp.arena);

    for (int i = 0; i < OS_NumProcessors(); i++)
    {
        arenas[i]->align = 16;
    }

    PrimitiveIndices *primIndices =
        PushArrayNoZero(scratch.temp.arena, PrimitiveIndices, numMeshes);
    for (int i = 0; i < numMeshes; i++)
    {
        PrimitiveIndices ind = {};
        ind.materialID       = MaterialHandle(materialIndices[i]);
        ind.alphaTexture     = 0;
        primIndices[i]       = ind;
    }
    ScenePrimitives scene = {};
    scene.primitives      = meshes;
    scene.numPrimitives   = numMeshes;
    scene.primIndices     = primIndices;
    scene.BuildTriangleBVH(arenas);

    Bounds sceneBounds = scene.GetBounds();

    StaticArray<u32> triangleGeomIDs(scratch.temp.arena, totalNumTriangles);

    u32 currentMesh = 0;
    for (int tri = 0; tri < totalNumTriangles; tri++)
    {
        u32 limit = currentMesh == numMeshes - 1 ? totalNumTriangles
                                                 : meshVertexOffsets[currentMesh + 1].y;
        if (tri >= limit)
        {
            currentMesh++;
        }
        triangleGeomIDs.Push(currentMesh);
    }

    GraphPartitionResult partitionResult;
    bool success =
        GenerateValidTriClusters(scratch.temp.arena, combinedMesh, MAX_CLUSTER_TRIANGLES, ~0u,
                                 triangleGeomIDs, partitionResult);
    Assert(success);

    StaticArray<u32> newTriangleGeomIDs(scratch.temp.arena, totalNumTriangles);

    for (u32 i = 0; i < partitionResult.clusterIndices.Length(); i++)
    {
        int triIndex = partitionResult.clusterIndices[i];
        newTriangleGeomIDs.Push(triangleGeomIDs[triIndex]);
    }

    Array<Cluster> clusters(scratch.temp.arena, totalClustersEstimate);
    Array<ClusterGroup> clusterGroups(scratch.temp.arena, totalGroupsEstimate);

    ClusterGroup clusterGroup = {};
    clusterGroup.vertexData   = vertexData;
    clusterGroup.indices      = indexData;
    clusterGroup.numIndices   = combinedMesh.numIndices;
    clusterGroup.isLeaf       = true;

    clusterGroups.Push(clusterGroup);

#if 0
    // Sort the clusters for determinism
    struct Handle
    {
        u32 sortKey;
        u32 index;
    };
    Handle *handles = PushArrayNoZero(scratch.temp.arena, Handle, numClusters);
    for (int i = 0; i < numClusters; i++)
    {
        handles[i].sortKey = clusterRecords[i].start;
        handles[i].index   = i;
    }

    SortHandles(handles, numClusters);
    RecordAOSSplits *sortedClusterRecords =
        PushArrayNoZero(scratch.temp.arena, RecordAOSSplits, numClusters);

    for (int i = 0; i < numClusters; i++)
    {
        sortedClusterRecords[i] = clusterRecords[handles[i].index];
    }
#endif

#if 0
    StaticArray<u32> numIslandsInCluster(scratch.temp.arena, partitionResult.ranges.Length());
    for (PartitionRange partitionRange : partitionResult.ranges)
    {
        u32 num = partitionRange.end - partitionRange.begin;
        HashIndex cornerHash(scratch.temp.arena, NextPowerOfTwo(num), NextPowerOfTwo(num));
        for (u32 i = 0; i < num; i++) // range.begin; i < range.end; i++)
        {
            u32 tri = partitionResult.clusterIndices[partitionRange.begin + i];
            for (u32 vertIndex = 0; vertIndex < 3; vertIndex++)
            {
                Vec3f p  = combinedMesh.p[combinedMesh.indices[3 * tri + vertIndex]];
                int hash = Hash(p);
                cornerHash.AddInHash(hash, 3 * i + vertIndex);
            }
        }

        BitVector visited(scratch.temp.arena, num);
        u32 numIslands = 0;

        StaticArray<u32> stack(scratch.temp.arena, num);
        for (int index = 0; index < num; index++)
        {
            stack.Empty();

            u32 islandIndex = numIslands;
            if (!visited.GetBit(index))
            {
                numIslands++;
                visited.SetBit(index);
                stack.Push(index);
            }
            while (stack.Length())
            {
                u32 triIndex = stack.Pop();
                u32 tri      = partitionResult.clusterIndices[partitionRange.begin + triIndex];

                for (u32 vert = 0; vert < 3; vert++)
                {
                    Vec3f p  = combinedMesh.p[combinedMesh.indices[3 * tri + vert]];
                    int hash = Hash(p);
                    for (int hashIndex = cornerHash.FirstInHash(hash); hashIndex != -1;
                         hashIndex     = cornerHash.NextInHash(hashIndex))
                    {
                        int otherTri  = hashIndex / 3;
                        int otherVert = hashIndex % 3;
                        int realTri =
                            partitionResult.clusterIndices[partitionRange.begin + otherTri];
                        if (combinedMesh.p[combinedMesh.indices[3 * realTri + otherVert]] == p)
                        {
                            if (!visited.GetBit(otherTri))
                            {
                                visited.SetBit(otherTri);
                                stack.Push(otherTri);
                            }
                        }
                    }
                }
            }
        }
        numIslandsInCluster.Push(numIslands);
    }
#endif

    for (int i = 0; i < partitionResult.ranges.Length(); i++)
    {
        ScratchArena clusterScratch(&scratch.temp.arena, 1);
        Cluster cluster     = {};
        cluster.mipLevel    = 0;
        cluster.headerIndex = i;

        PartitionRange range = partitionResult.ranges[i];
        StaticArray<int> triangleIndices(partitionResult.clusterIndices.data + range.begin,
                                         range.end - range.begin);
        StaticArray<u32> geomIDs(newTriangleGeomIDs.data + range.begin,
                                 range.end - range.begin);
        cluster.triangleIndices = triangleIndices;
        cluster.geomIDs         = geomIDs;

        // Construct the lod bounds
        StaticArray<Vec3f> points(clusterScratch.temp.arena, 3 * (range.end - range.begin));

        Bounds bounds;

        for (u32 triDataIndex = range.begin; triDataIndex < range.end; triDataIndex++)
        {
            u32 tri = partitionResult.clusterIndices[triDataIndex];
            for (int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
            {
                u32 index = indexData[3 * tri + vertexIndex];
                Vec3f p   = GetPosition(vertexData, index);
                points.Push(p);

                bounds.Extend(Lane4F32(p));
            }
        }
        cluster.bounds    = bounds;
        cluster.lodBounds = ConstructSphereFromPoints(points.data, points.Length());
        clusters.Push(cluster);
    }

    // Create clusters
    StaticArray<DenseGeometryBuildData> buildDatas(scratch.temp.arena, OS_NumProcessors());
    for (int i = 0; i < buildDatas.capacity; i++)
    {
        buildDatas.Push(DenseGeometryBuildData(arenas[i]));
    }

    Bounds bounds;
    for (Cluster &cluster : clusters)
    {
        buildDatas[0].WriteTriangleData(cluster.triangleIndices, cluster.geomIDs, combinedMesh,
                                        materialIndices);
    }

    std::atomic<u32> numVoxelClusters(0);
    u32 depth = 0;
    {
        // 1. Split triangles into clusters (mesh remains)

        // 2. Group clusters based on how many shared edges they have (METIS) (mesh remains)
        //      - also have edges between clusters that are close enough
        // 3. Simplify the cluster group (effectively creates num groups different meshes)
        // 4. Split simplified group into clusters

        struct Edge
        {
            Vec3f p0;
            Vec3f p1;

            int clusterIndex;
        };

        ArrayView<Cluster> levelClusters(clusters, 0, clusters.Length());

        u32 prevClusterArrayEnd = 0;
        Bounds bounds;

        for (;;)
        {
            Print("depth: %u num clusters: %u\n", depth, levelClusters.num);
            if (levelClusters.Length() < 2)
            {
                levelClusters[0].groupIndex = clusterGroups.Length();
                ClusterGroup rootGroup;
                rootGroup.vertexData        = 0;
                rootGroup.indices           = 0;
                rootGroup.buildDataIndex    = 0;
                rootGroup.isLeaf            = false;
                rootGroup.maxParentError    = 1e10;
                rootGroup.lodBounds         = levelClusters[0].lodBounds;
                rootGroup.parentStartIndex  = ~0u;
                rootGroup.parentCount       = ~0u;
                rootGroup.clusterStartIndex = prevClusterArrayEnd;
                rootGroup.clusterCount      = 1;

                rootGroup.numVertices = 0;
                rootGroup.numIndices  = 0;
                rootGroup.mipLevel    = depth++;
                rootGroup.hasVoxels   = bool(levelClusters[0].compressedVoxels.Length());

                clusterGroups.Push(rootGroup);

                break;
            }

            u32 hashSize = NextPowerOfTwo(3 * MAX_CLUSTER_TRIANGLES * levelClusters.Length());
            HashIndex edgeHash(scratch.temp.arena, hashSize, hashSize);

            // Calculate the number of edges per group
            u32 edgeOffset = 0;
            StaticArray<u32> clusterEdgeOffsets(scratch.temp.arena, levelClusters.Length());
            for (int clusterIndex = 0; clusterIndex < levelClusters.Length(); clusterIndex++)
            {
                Cluster &cluster = levelClusters[clusterIndex];
                u32 numEdges     = cluster.triangleIndices.Length() * 3;
                clusterEdgeOffsets.Push(edgeOffset);
                edgeOffset += numEdges;
            }
            StaticArray<Edge> edges(scratch.temp.arena, edgeOffset, edgeOffset);

            u32 numClusters = levelClusters.Length();
            ParallelFor(0, numClusters, 32, 32, [&](int jobID, int start, int count) {
                for (int clusterIndex = start; clusterIndex < start + count; clusterIndex++)
                {
                    const Cluster &cluster     = levelClusters[clusterIndex];
                    ClusterGroup &clusterGroup = clusterGroups[cluster.childGroupIndex];
                    u32 edgeOffset             = clusterEdgeOffsets[clusterIndex];

                    for (int primID : cluster.triangleIndices)
                    {
                        for (int edgeIndexIndex = 0; edgeIndexIndex < 3; edgeIndexIndex++)
                        {
                            u32 index0 = clusterGroup.indices[3 * primID + edgeIndexIndex];
                            u32 index1 =
                                clusterGroup.indices[3 * primID + (edgeIndexIndex + 1) % 3];

                            Vec3f p0 = GetPosition(clusterGroup.vertexData, index0);
                            Vec3f p1 = GetPosition(clusterGroup.vertexData, index1);

                            int hash  = HashEdge(p0, p1);
                            Edge edge = {p0, p1, clusterIndex};

                            edges[edgeOffset] = edge;
                            edgeHash.AddConcurrent(hash, edgeOffset);
                            edgeOffset++;
                        }
                    }
                }
            });

            StaticArray<Array<int>> clusterNeighbors(scratch.temp.arena, numClusters,
                                                     numClusters);
            StaticArray<Array<int>> clusterEdgeWeights(scratch.temp.arena, numClusters,
                                                       numClusters);
            StaticArray<StaticArray<int>> clusterExternalEdges(scratch.temp.arena, numClusters,
                                                               numClusters);

            const int numSpatialLinks = 5;
            u32 numAttributes         = 0;
            u32 vertexDataLen         = sizeof(f32) * (3 + numAttributes);

            ParallelForLoop(0, numClusters, 32, 32, [&](int jobID, int clusterIndex) {
                Arena *arena = arenas[GetThreadIndex()];
                ScratchArena threadScratch;

                Cluster &cluster  = levelClusters[clusterIndex];
                int triangleCount = cluster.triangleIndices.Length();

                struct Handle
                {
                    int sortKey;
                };

                Array<Handle> neighbors(threadScratch.temp.arena, 3 * triangleCount);
                Array<int> externalEdges(threadScratch.temp.arena, 3 * triangleCount);

                u32 edgeOffset = clusterEdgeOffsets[clusterIndex];
                for (int edgeIndex = edgeOffset; edgeIndex < edgeOffset + 3 * triangleCount;
                     edgeIndex++)
                {
                    Edge &edge = edges[edgeIndex];
                    int hash   = HashEdge(edge.p0, edge.p1);

                    for (int otherEdgeIndex = edgeHash.FirstInHash(hash); otherEdgeIndex != -1;
                         otherEdgeIndex     = edgeHash.NextInHash(otherEdgeIndex))
                    {
                        Edge &otherEdge = edges[otherEdgeIndex];
                        if (edge.p0 == otherEdge.p0 && edge.p1 == otherEdge.p1 &&
                            edge.clusterIndex != otherEdge.clusterIndex)
                        {
                            neighbors.Push(Handle{otherEdge.clusterIndex});
                            externalEdges.Push(otherEdgeIndex);
                        }
                    }
                }

                if (neighbors.Length() == 0)
                {
                    clusterNeighbors[clusterIndex]   = Array<int>(arena, 2 * numSpatialLinks);
                    clusterEdgeWeights[clusterIndex] = Array<int>(arena, 2 * numSpatialLinks);
                    return;
                }

                int compactedNumNeighbors = 0;
                u32 numNeighbors          = neighbors.Length();
                SortHandles(neighbors.data, neighbors.Length());

                int *weights = PushArray(threadScratch.temp.arena, int, numNeighbors);

                int prev   = neighbors[0].sortKey;
                weights[0] = 1;

                for (int neighborIndex = 1; neighborIndex < numNeighbors; neighborIndex++)
                {
                    int neighbor = neighbors[neighborIndex].sortKey;
                    if (neighbor != prev)
                    {
                        compactedNumNeighbors++;

                        neighbors[compactedNumNeighbors].sortKey = neighbor;
                        prev                                     = neighbor;
                    }
                    weights[compactedNumNeighbors]++;
                }
                compactedNumNeighbors++;

                clusterNeighbors[clusterIndex]      = Array<int>(arena, compactedNumNeighbors);
                clusterNeighbors[clusterIndex].size = compactedNumNeighbors;
                clusterEdgeWeights[clusterIndex]    = Array<int>(arena, compactedNumNeighbors);
                clusterEdgeWeights[clusterIndex].size = compactedNumNeighbors;
                clusterExternalEdges[clusterIndex] =
                    StaticArray<int>(arena, numNeighbors, numNeighbors);

                MemoryCopy(clusterNeighbors[clusterIndex].data, neighbors.data,
                           sizeof(int) * compactedNumNeighbors);
                MemoryCopy(clusterEdgeWeights[clusterIndex].data, weights,
                           sizeof(int) * compactedNumNeighbors);
                MemoryCopy(clusterExternalEdges[clusterIndex].data, externalEdges.data,
                           sizeof(int) * numNeighbors);
            });

            // Create edges between spatially close but separated clusters
            struct Handle
            {
                u32 sortKey;
                u32 index;
            };

            Bounds totalBounds;
            for (int clusterIndex = 0; clusterIndex < levelClusters.Length(); clusterIndex++)
            {
                Cluster &cluster = levelClusters[clusterIndex];
                totalBounds.Extend(cluster.bounds);
            }

            StaticArray<Vec3f> clusterCenters(scratch.temp.arena, levelClusters.Length(),
                                              levelClusters.Length());

            ParallelFor(0, levelClusters.Length(), 4096, 4096,
                        [&](u32 jobID, u32 start, u32 count) {
                            for (int clusterIndex = start; clusterIndex < start + count;
                                 clusterIndex++)
                            {
                                Cluster &cluster = levelClusters[clusterIndex];
                                Vec3f center     = ToVec3f(cluster.bounds.Centroid());

                                clusterCenters[clusterIndex] = center;
                            }
                        });

            AddSpatialLinks(scratch.temp.arena, levelClusters.Length(), totalBounds,
                            clusterCenters, clusterNeighbors, clusterEdgeWeights);

            GraphPartitionResult partitionResult;
            if (numClusters <= maxGroupSize)
            {
                partitionResult.ranges = StaticArray<PartitionRange>(scratch.temp.arena, 1);
                partitionResult.clusterIndices =
                    StaticArray<int>(scratch.temp.arena, numClusters);
                partitionResult.ranges.Push(PartitionRange{0, numClusters});

                for (int i = 0; i < numClusters; i++)
                {
                    partitionResult.clusterIndices.Push(i);
                }
            }
            else
            {
                u32 maxNumPartitions = (numClusters + minGroupSize - 1) / minGroupSize;

                i32 *clusterOffsets =
                    PushArrayNoZero(scratch.temp.arena, i32, numClusters + 1);
                clusterOffsets[0]     = 0;
                i32 *clusterOffsets1  = &clusterOffsets[1];
                u32 totalNumNeighbors = 0;

                for (int clusterIndex = 0; clusterIndex < numClusters; clusterIndex++)
                {
                    u32 num                       = clusterNeighbors[clusterIndex].Length();
                    clusterOffsets1[clusterIndex] = totalNumNeighbors;
                    totalNumNeighbors += num;
                }

                i32 *clusterData = PushArrayNoZero(scratch.temp.arena, i32, totalNumNeighbors);
                i32 *clusterWeights =
                    PushArrayNoZero(scratch.temp.arena, i32, totalNumNeighbors);

                ParallelFor(0, numClusters, 32, 32, [&](int jobID, int start, int count) {
                    for (int clusterIndex = start; clusterIndex < start + count;
                         clusterIndex++)
                    {
                        i32 offset = clusterOffsets1[clusterIndex];

                        u32 numNeighbors = clusterNeighbors[clusterIndex].Length();
                        Assert(numNeighbors == clusterEdgeWeights[clusterIndex].Length());

                        MemoryCopy(clusterData + offset, clusterNeighbors[clusterIndex].data,
                                   sizeof(int) * numNeighbors);
                        MemoryCopy(clusterWeights + offset,
                                   clusterEdgeWeights[clusterIndex].data,
                                   sizeof(int) * numNeighbors);

                        clusterOffsets1[clusterIndex] += numNeighbors;
                    }
                });

                // Recursively partition the clusters into two groups until each group
                // satisfies constraints

                partitionResult = RecursivePartitionGraph(
                    scratch.temp.arena, clusterOffsets, clusterData, clusterWeights,
                    numClusters, totalNumNeighbors, minGroupSize, maxGroupSize);

                BitVector clusterOnce(scratch.temp.arena, numClusters);
                for (int i = 0; i < numClusters; i++)
                {
                    int index = partitionResult.clusterIndices[i];
                    Assert(!clusterOnce.GetBit(index));
                    clusterOnce.SetBit(index);
                }
            }

            Print("num groups: %u\n", partitionResult.ranges.Length());

            StaticArray<u32> clusterToGroupID(scratch.temp.arena, numClusters, numClusters);
            for (int groupIndex = 0; groupIndex < partitionResult.ranges.Length();
                 groupIndex++)
            {
                PartitionRange &range = partitionResult.ranges[groupIndex];
                for (int i = range.begin; i < range.end; i++)
                {
                    clusterToGroupID[partitionResult.clusterIndices[i]] = groupIndex;
                }
            }

            u32 totalNumClusters = clusters.Length();
            u32 totalNumGroups   = clusterGroups.Length();

            clusters.Resize(totalNumClusters + numClusters);
            clusterGroups.Resize(totalNumGroups + partitionResult.ranges.Length());

            ArrayView<Cluster> nextLevelClusters(clusters, totalNumClusters, numClusters);

            std::atomic<u32> numLevelClusters(0);
            std::atomic<u32> numVertices(0);
            std::atomic<u32> numIndices(0);
            std::atomic<u32> numVoxels(0);

            ParallelForLoop(
                0, partitionResult.ranges.Length(), 1, 1, [&](int jobID, int groupIndex) {
                    u32 threadIndex = GetThreadIndex();
                    Arena *arena    = arenas[threadIndex];
                    ScratchArena scratch;

                    PartitionRange range  = partitionResult.ranges[groupIndex];
                    u32 groupNumTriangles = 0;
                    u32 newGroupIndex     = groupIndex + totalNumGroups;

                    for (int clusterIndexIndex = range.begin; clusterIndexIndex < range.end;
                         clusterIndexIndex++)
                    {
                        int clusterIndex = partitionResult.clusterIndices[clusterIndexIndex];
                        const Cluster &cluster = levelClusters[clusterIndex];
                        groupNumTriangles += cluster.triangleIndices.Length();
                    }

                    Vec4f *clusterSpheres =
                        PushArrayNoZero(scratch.temp.arena, Vec4f, range.end - range.begin);
                    Bounds parentBounds;

                    // Set the child start index of last level's clusters
                    for (int clusterIndexIndex = range.begin; clusterIndexIndex < range.end;
                         clusterIndexIndex++)
                    {
                        int clusterIndex = partitionResult.clusterIndices[clusterIndexIndex];

                        levelClusters[clusterIndex].groupIndex = newGroupIndex;

                        clusterSpheres[clusterIndexIndex - range.begin] =
                            levelClusters[clusterIndex].lodBounds;
                    }

                    Vec4f parentSphereBounds =
                        ConstructSphereFromSpheres(clusterSpheres, range.end - range.begin);

                    f32 *groupVertices =
                        PushArrayNoZero(scratch.temp.arena, f32,
                                        (groupNumTriangles * 3) * (3 + numAttributes));
                    u32 *indices =
                        PushArrayNoZero(scratch.temp.arena, u32, groupNumTriangles * 3);
                    u32 *geomIDs = PushArrayNoZero(scratch.temp.arena, u32, groupNumTriangles);
                    u32 vertexCount            = 0;
                    u32 indexCount             = 0;
                    u32 triangleCount          = 0;
                    u32 clusterTotalVoxelCount = 0;

                    u32 numHash = NextPowerOfTwo(groupNumTriangles * 3);

                    f32 maxLodError = 0.f;

                    HashIndex vertexHash(scratch.temp.arena, numHash, numHash);
                    HashIndex cornerHash(scratch.temp.arena, numHash, numHash);

                    // Merge clusters into a single vertex and index buffer
                    bool hasVoxels = false;
                    for (int clusterIndexIndex = range.begin; clusterIndexIndex < range.end;
                         clusterIndexIndex++)
                    {
                        int clusterIndex = partitionResult.clusterIndices[clusterIndexIndex];
                        u32 groupID      = clusterToGroupID[clusterIndex];
                        const Cluster &cluster = levelClusters[clusterIndex];

                        ClusterGroup &prevClusterGroup =
                            clusterGroups[cluster.childGroupIndex];

                        hasVoxels |= (bool)cluster.compressedVoxels.Length();
                        maxLodError = Max(cluster.lodError, maxLodError);

                        for (const CompressedVoxel &voxel : cluster.compressedVoxels)
                        {
                            clusterTotalVoxelCount +=
                                PopCount(voxel.bitMask) + PopCount(voxel.bitMask >> 32u);
                        }

                        for (u32 index = 0; index < cluster.triangleIndices.Length(); index++)
                        {
                            u32 triangleIndex = triangleCount++;

                            u32 primID             = cluster.triangleIndices[index];
                            geomIDs[triangleIndex] = cluster.geomIDs[index];

                            for (int vertIndex = 0; vertIndex < 3; vertIndex++)
                            {
                                u32 indexIndex  = 3 * primID + vertIndex;
                                u32 vertexIndex = prevClusterGroup.indices[indexIndex];

                                f32 *clusterVertexData =
                                    GetVertexData(prevClusterGroup.vertexData, vertexIndex);
                                Vec3f pos =
                                    GetPosition(prevClusterGroup.vertexData, vertexIndex);
                                int hashP = Hash(pos);
                                cornerHash.AddInHash(hashP, 3 * triangleIndex + vertIndex);

                                int hash = MurmurHash32((const char *)clusterVertexData,
                                                        vertexDataLen, 0);

                                u32 newVertexIndex = ~0u;
                                for (int hashIndex = vertexHash.FirstInHash(hash);
                                     hashIndex != -1;
                                     hashIndex = vertexHash.NextInHash(hashIndex))
                                {
                                    f32 *otherVertexData =
                                        GetVertexData(groupVertices, hashIndex);

                                    if (memcmp(otherVertexData, clusterVertexData,
                                               vertexDataLen) == 0)
                                    {
                                        newVertexIndex = (u32)hashIndex;
                                        break;
                                    }
                                }

                                if (newVertexIndex == ~0u)
                                {
                                    newVertexIndex = vertexCount++;
                                    MemoryCopy(GetVertexData(groupVertices, newVertexIndex),
                                               clusterVertexData, vertexDataLen);
                                    vertexHash.AddInHash(hash, newVertexIndex);
                                }

                                indices[indexCount++] = newVertexIndex;
                            }
                        }
                    }
                    Assert(triangleCount == groupNumTriangles);
                    Assert(indexCount == groupNumTriangles * 3);

                    // Calculate the average surface area of all the triangles
                    f32 totalSurfaceArea = 0.f;
                    u32 numTris          = indexCount / 3;
                    for (u32 tri = 0; tri < numTris; tri++)
                    {
                        Vec3f p0 = GetPosition(groupVertices, indices[3 * tri + 0]);
                        Vec3f p1 = GetPosition(groupVertices, indices[3 * tri + 1]);
                        Vec3f p2 = GetPosition(groupVertices, indices[3 * tri + 2]);
                        f32 area = 0.5f * Length(Cross(p1 - p0, p2 - p0));
                        totalSurfaceArea += area;
                    }

                    u32 targetNumVoxels = ((vertexCount + clusterTotalVoxelCount) * 3) / 4;
                    f32 voxelSize       = Sqrt(totalSurfaceArea / targetNumVoxels) * .75f;
                    voxelSize           = Max(maxLodError, voxelSize);

                    StaticArray<u32> islands(scratch.temp.arena, triangleCount, triangleCount);
                    BitVector visited(scratch.temp.arena, triangleCount);
                    u32 numIslands = 0;

                    for (u32 triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
                    {
                        StaticArray<u32> stack(scratch.temp.arena, triangleCount);
                        u32 islandIndex = numIslands;
                        if (!visited.GetBit(triangleIndex))
                        {
                            visited.SetBit(triangleIndex);
                            stack.Push(triangleIndex);
                            numIslands++;
                        }
                        while (stack.Length())
                        {
                            u32 currentTri      = stack.Pop();
                            islands[currentTri] = islandIndex;
                            for (int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
                            {
                                Vec3f p  = GetPosition(groupVertices,
                                                       indices[3 * currentTri + vertexIndex]);
                                int hash = Hash(p);
                                for (int hashIndex = cornerHash.FirstInHash(hash);
                                     hashIndex != -1;
                                     hashIndex = cornerHash.NextInHash(hashIndex))
                                {
                                    Vec3f otherP =
                                        GetPosition(groupVertices, indices[hashIndex]);
                                    u32 otherTriangle = hashIndex / 3;
                                    if (otherP == p && !visited.GetBit(otherTriangle))
                                    {
                                        visited.SetBit(otherTriangle);
                                        stack.Push(otherTriangle);
                                    }
                                }
                            }
                        }
                    }

                    u32 maxClusterTriangles = MAX_CLUSTER_TRIANGLES + 2;
                    for (;;)
                    {
                        maxClusterTriangles -= 2;
                        Mesh simplifiedMesh = {};
                        f32 simplifyError   = pos_inf;
                        if (indexCount && clusterTotalVoxelCount == 0)
                        {
                            // Normalize the positions
                            struct Float
                            {
                                union
                                {
                                    struct
                                    {
                                        u32 mantissa : 23;
                                        u32 exponent : 8;
                                        u32 sign : 1;
                                    };
                                    f32 value;
                                };
                                Float(f32 f) : value(f) {}
                            };

                            f32 *tempVertices = PushArrayNoZero(
                                scratch.temp.arena, f32, (3 + numAttributes) * vertexCount);
                            MemoryCopy(tempVertices, groupVertices,
                                       vertexCount * sizeof(f32) * (3 + numAttributes));

                            f32 triangleSize = Sqrt(totalSurfaceArea / (float)numTris);
                            Float currentSize(Max(triangleSize, .00002f));
                            Float desired(.25f);
                            Float scale(1.f);
                            int exponent = Clamp(
                                (int)desired.exponent - (int)currentSize.exponent, -126, 127);
                            scale.exponent = exponent + 127;
                            float posScale = scale.value;
                            for (int i = 0; i < vertexCount; i++)
                            {
                                GetPosition(tempVertices, i) *= posScale;
                            }

                            // Simplify the clusters
                            u32 targetNumParents =
                                (groupNumTriangles + MAX_CLUSTER_TRIANGLES * 2 - 1) /
                                (MAX_CLUSTER_TRIANGLES * 2);
                            u32 targetNumTris = targetNumParents * maxClusterTriangles;

                            f32 targetError = 0.f;
                            u32 *tempIndices =
                                PushArrayNoZero(scratch.temp.arena, u32, indexCount);
                            MemoryCopy(tempIndices, indices, sizeof(u32) * indexCount);

                            MeshSimplifier simplifier(scratch.temp.arena, tempVertices,
                                                      vertexCount, tempIndices, indexCount,
                                                      numAttributes);

                            // Lock edges shared with other groups
                            for (int clusterIndexIndex = range.begin;
                                 clusterIndexIndex < range.end; clusterIndexIndex++)
                            {
                                int clusterIndex =
                                    partitionResult.clusterIndices[clusterIndexIndex];
                                u32 groupID            = clusterToGroupID[clusterIndex];
                                const Cluster &cluster = levelClusters[clusterIndex];
                                auto &externalEdges    = clusterExternalEdges[clusterIndex];

                                for (int edgeIndex : externalEdges)
                                {
                                    Edge &edge = edges[edgeIndex];
                                    if (clusterToGroupID[edge.clusterIndex] != groupID)
                                    {
                                        simplifier.LockVertex(edge.p0 * posScale);
                                        simplifier.LockVertex(edge.p1 * posScale);
                                    }
                                }
                            }

                            f32 invScale  = 1.f / posScale;
                            simplifyError = simplifier.Simplify(
                                vertexCount, targetNumTris, Sqr(targetError), 0, 0, FLT_MAX);
                            f32 preError  = simplifyError;
                            simplifyError = Sqrt(simplifyError) * invScale;

                            simplifier.Finalize(simplifiedMesh.numVertices,
                                                simplifiedMesh.numIndices, geomIDs);

                            // TODO: attributes
                            simplifiedMesh.p =
                                PushArrayNoZero(arena, Vec3f, simplifiedMesh.numVertices);
                            simplifiedMesh.indices =
                                PushArrayNoZero(arena, u32, simplifiedMesh.numIndices);

                            MemoryCopy(simplifiedMesh.p, simplifier.vertexData,
                                       sizeof(Vec3f) * simplifiedMesh.numVertices);
                            MemoryCopy(simplifiedMesh.indices, simplifier.indices,
                                       sizeof(u32) * simplifiedMesh.numIndices);

                            for (int i = 0; i < simplifiedMesh.numVertices; i++)
                            {
                                simplifiedMesh.p[i] *= invScale;
                            }

                            numVertices.fetch_add(simplifiedMesh.numVertices);
                            numIndices.fetch_add(simplifiedMesh.numIndices);
                        }

                        // Persists to future clusters
                        StaticArray<CompressedVoxel> compressedVoxels;

                        // Temp
                        StaticArray<Voxel> voxels;
                        StaticArray<Vec3i> extraVoxels;
                        Mesh voxelMesh    = {};
                        f32 *coverages    = 0;
                        SGGXCompact *sggx = 0;
                        StaticArray<u32> voxelGeomIDs;
                        f32 error = 0.f;
                        u32 numSlots =
                            NextPowerOfTwo(groupNumTriangles + clusterTotalVoxelCount);

                        if (numIslands > 1 || hasVoxels)
                        {
                            u32 targetNumBricks =
                                (range.end - range.begin) * MAX_CLUSTER_TRIANGLES;
                            Assert(numSlots);

                            while (voxelSize < simplifyError)
                            {
                                SimpleHashSet<Vec3i> voxelHashSet(scratch.temp.arena,
                                                                  numSlots);
                                f32 rcpVoxelSize = 1.f / voxelSize;
                                // Gather voxels
                                if (groupNumTriangles)
                                {
                                    VoxelizeTriangles(
                                        scratch.temp.arena, voxelHashSet, groupVertices,
                                        indices, groupNumTriangles, numAttributes, voxelSize);
                                }

                                for (int clusterIndexIndex = range.begin;
                                     clusterIndexIndex < range.end; clusterIndexIndex++)
                                {
                                    int clusterIndex =
                                        partitionResult.clusterIndices[clusterIndexIndex];
                                    Cluster &cluster = levelClusters[clusterIndex];
                                    ClusterGroup &childGroup =
                                        clusterGroups[cluster.childGroupIndex];
                                    for (CompressedVoxel &voxel : cluster.compressedVoxels)
                                    {
                                        u32 vertexOffset   = voxel.vertexOffset;
                                        u32 brickNumVoxels = PopCount((u32)voxel.bitMask) +
                                                             PopCount(voxel.bitMask >> 32u);
                                        for (u32 i = 0; i < brickNumVoxels; i++)
                                        {
                                            u32 vertexIndex = vertexOffset + i;
                                            Vec3f minP = GetPosition(childGroup.vertexData,
                                                                     vertexIndex);

                                            Vec3i minVoxel = Floor(minP * rcpVoxelSize);
                                            Vec3i maxVoxel = Floor((minP + cluster.lodError) *
                                                                   rcpVoxelSize);

                                            for (int x = minVoxel.x; x <= maxVoxel.x; x++)
                                            {
                                                for (int y = minVoxel.y; y <= maxVoxel.y; y++)
                                                {
                                                    for (int z = minVoxel.z; z <= maxVoxel.z;
                                                         z++)
                                                    {
                                                        Vec3i p  = Vec3i(x, y, z);
                                                        int hash = Hash(p);
                                                        voxelHashSet.AddUnique(
                                                            scratch.temp.arena, hash,
                                                            Vec3i(x, y, z));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if (!cluster.extraVoxels.Length()) continue;
                                    for (Vec3i voxel : cluster.extraVoxels)
                                    {
                                        Vec3i minVoxel = Vec3i(Floor(
                                            Vec3f(voxel) * cluster.lodError * rcpVoxelSize));
                                        Vec3i maxVoxel =
                                            Vec3i(Floor(Vec3f(voxel + 1) * cluster.lodError *
                                                        rcpVoxelSize));

                                        for (int x = minVoxel.x; x <= maxVoxel.x; x++)
                                        {
                                            for (int y = minVoxel.y; y <= maxVoxel.y; y++)
                                            {
                                                for (int z = minVoxel.z; z <= maxVoxel.z; z++)
                                                {
                                                    Vec3i p  = Vec3i(x, y, z);
                                                    int hash = Hash(p);
                                                    voxelHashSet.AddUnique(scratch.temp.arena,
                                                                           hash,
                                                                           Vec3i(x, y, z));
                                                }
                                            }
                                        }
                                    }
                                }

                                CheckVoxelOccupancy(scratch.temp.arena, &scene, voxelHashSet,
                                                    voxels, extraVoxels, voxelSize);

                                if (voxels.Length())
                                {
                                    voxelMesh = CompressVoxels(scratch.temp.arena, voxels,
                                                               voxelGeomIDs, compressedVoxels,
                                                               voxelSize, coverages, sggx);
                                    // Print("num bricks: %u\n", compressedVoxels.Length());
                                }

                                if (voxels.Length() < targetNumVoxels &&
                                    compressedVoxels.Length() < targetNumBricks)
                                {
                                    break;
                                }
                                voxelSize *= 1.1f;
                                // Print("voxel again\n");
                            }
                            if (voxelSize < simplifyError)
                            {
                                error          = voxelSize;
                                simplifiedMesh = voxelMesh;
                            }
                            else
                            {
                                error = simplifyError;
                                compressedVoxels.Clear();
                                voxels.Clear();
                            }
                        }

                        DenseGeometryBuildData *groupBuildData = &buildDatas[threadIndex];
                        u32 headerOffset      = groupBuildData->headers.Length();
                        int parentStartIndex  = -1;
                        int numParentClusters = -1;

                        if (compressedVoxels.Length())
                        {
                            RecordAOSSplits record;
                            PrimRef *newPrimRefs = PushArrayNoZero(scratch.temp.arena, PrimRef,
                                                                   compressedVoxels.Length());

                            GenerateVoxelRefs(newPrimRefs, compressedVoxels, voxelMesh, record,
                                              voxelSize);
                            record.SetRange(0, compressedVoxels.Length());

                            StaticArray<RecordAOSSplits> records =
                                ClusterBuilder::BuildClusters(scratch.temp.arena, newPrimRefs,
                                                              record, MAX_CLUSTER_TRIANGLES,
                                                              range.end - range.begin);

                            if (records.Length() > range.end - range.begin)
                            {
                                Print("again\n");
                                continue;
                            }

                            Vec3f *newP =
                                PushArrayNoZero(arena, Vec3f, simplifiedMesh.numVertices);

                            MemoryCopy(newP, simplifiedMesh.p,
                                       sizeof(Vec3f) * simplifiedMesh.numVertices);
                            simplifiedMesh.p = newP;

                            numVoxels.fetch_add(simplifiedMesh.numVertices,
                                                std::memory_order_relaxed);

                            numParentClusters = records.Length();
                            parentStartIndex  = numLevelClusters.fetch_add(
                                numParentClusters, std::memory_order_relaxed);
                            Assert(parentStartIndex + numParentClusters <= numClusters);

                            for (u32 clusterIndex = 0; clusterIndex < records.Length();
                                 clusterIndex++)
                            {
                                RecordAOSSplits &clusterRecord = records[clusterIndex];

                                Cluster &cluster =
                                    nextLevelClusters[parentStartIndex + clusterIndex];
                                cluster.bounds          = Bounds(clusterRecord.geomBounds);
                                cluster.mipLevel        = depth + 1;
                                cluster.childGroupIndex = newGroupIndex;
                                cluster.lodError        = error;
                                cluster.lodBounds       = parentSphereBounds;
                                cluster.headerIndex     = headerOffset + clusterIndex;

                                StaticArray<CompressedVoxel> clusterCompressedVoxels(
                                    arena, clusterRecord.count);

                                for (u32 primRefIndex = clusterRecord.start;
                                     primRefIndex < clusterRecord.start + clusterRecord.count;
                                     primRefIndex++)
                                {
                                    PrimRef &ref = newPrimRefs[primRefIndex];

                                    clusterCompressedVoxels.Push(compressedVoxels[ref.primID]);
                                }

                                cluster.compressedVoxels = clusterCompressedVoxels;

                                groupBuildData->WriteVoxelData(cluster.compressedVoxels,
                                                               voxelMesh, materialIndices,
                                                               voxelGeomIDs, coverages, sggx);
                            }
                            numVoxelClusters.fetch_add(records.Length(),
                                                       std::memory_order_relaxed);

                            if (extraVoxels.Length())
                            {
                                Graph<Vec3i> groupExtraVoxels;
                                groupExtraVoxels.InitializeStatic(
                                    scratch.temp.arena, extraVoxels.Length(),
                                    numParentClusters,
                                    [&](u32 voxelIndex, u32 *offsets, Vec3i *data = 0) {
                                        Vec3i &extraVoxel = extraVoxels[voxelIndex];
                                        Vec3f pos     = (Vec3f(extraVoxel) + 0.5f) * voxelSize;
                                        f32 bestDist  = pos_inf;
                                        int bestIndex = -1;
                                        for (int parentIndex = parentStartIndex;
                                             parentIndex <
                                             parentStartIndex + numParentClusters;
                                             parentIndex++)
                                        {
                                            Cluster &cluster = nextLevelClusters[parentIndex];
                                            Vec3f centroid =
                                                ToVec3f(cluster.bounds.Centroid());
                                            f32 dist = LengthSquared(centroid - pos);
                                            if (dist < bestDist)
                                            {
                                                bestIndex = parentIndex - parentStartIndex;
                                                bestDist  = dist;
                                            }
                                        }

                                        u32 dataIndex = offsets[bestIndex]++;
                                        if (data) data[dataIndex] = extraVoxel;
                                        return 1;
                                    });

                                for (int parentIndex = 0; parentIndex < numParentClusters;
                                     parentIndex++)
                                {
                                    u32 count = groupExtraVoxels.offsets[parentIndex + 1] -
                                                groupExtraVoxels.offsets[parentIndex];
                                    Cluster &cluster =
                                        nextLevelClusters[parentStartIndex + parentIndex];
                                    cluster.extraVoxels = {};
                                    if (count)
                                    {
                                        StaticArray<Vec3i> clusterExtraVoxels(arena, count,
                                                                              count);
                                        MemoryCopy(clusterExtraVoxels.data,
                                                   groupExtraVoxels.data +
                                                       groupExtraVoxels.offsets[parentIndex],
                                                   sizeof(Vec3i) * count);
                                        cluster.extraVoxels = clusterExtraVoxels;
                                    }
                                }
                            }
                        }
                        else
                        {
                            simplifiedMesh.numFaces = simplifiedMesh.numIndices / 3;
                            GraphPartitionResult clusterGroupPartitionResult;
                            bool success = GenerateValidTriClusters(
                                scratch.temp.arena, simplifiedMesh, MAX_CLUSTER_TRIANGLES,
                                range.end - range.begin,
                                StaticArray<u32>(geomIDs, simplifiedMesh.numFaces),
                                clusterGroupPartitionResult);

                            if (!success)
                            {
                                Print(
                                    "Triangle clusterization exceeded limits; trying again\n");
                                continue;
                            }

                            numParentClusters = clusterGroupPartitionResult.ranges.Length();
                            parentStartIndex  = numLevelClusters.fetch_add(
                                numParentClusters, std::memory_order_relaxed);
                            Assert(parentStartIndex + numParentClusters <= numClusters);

                            for (u32 clusterIndex = 0;
                                 clusterIndex < clusterGroupPartitionResult.ranges.Length();
                                 clusterIndex++)
                            {
                                Cluster &cluster =
                                    nextLevelClusters[parentStartIndex + clusterIndex];
                                cluster.mipLevel        = depth + 1;
                                cluster.childGroupIndex = newGroupIndex;
                                cluster.lodError        = error;
                                cluster.lodBounds       = parentSphereBounds;
                                cluster.headerIndex     = headerOffset + clusterIndex;

                                PartitionRange clusterRange =
                                    clusterGroupPartitionResult.ranges[clusterIndex];
                                StaticArray<int> triangleIndices(
                                    arena, clusterRange.end - clusterRange.begin);

                                for (u32 i = clusterRange.begin; i < clusterRange.end; i++)
                                {
                                    triangleIndices.Push(
                                        clusterGroupPartitionResult.clusterIndices[i]);
                                }

                                Bounds bounds;
                                StaticArray<u32> newGeomIDs(arena, clusterRange.end -
                                                                       clusterRange.begin);

                                for (u32 triDataIndex = clusterRange.begin;
                                     triDataIndex < clusterRange.end; triDataIndex++)
                                {
                                    u32 tri = clusterGroupPartitionResult
                                                  .clusterIndices[triDataIndex];
                                    newGeomIDs.Push(geomIDs[tri]);

                                    for (u32 vert = 0; vert < 3; vert++)
                                    {
                                        Vec3f p =
                                            simplifiedMesh
                                                .p[simplifiedMesh.indices[3 * tri + vert]];
                                        bounds.Extend(Lane4F32(p));
                                    }
                                }

                                cluster.bounds          = bounds;
                                cluster.triangleIndices = triangleIndices;
                                cluster.geomIDs         = newGeomIDs;

                                groupBuildData->WriteTriangleData(
                                    cluster.triangleIndices, cluster.geomIDs, simplifiedMesh,
                                    materialIndices);
                            }
                        }

                        ClusterGroup newClusterGroup;
                        newClusterGroup.vertexData       = (f32 *)simplifiedMesh.p;
                        newClusterGroup.indices          = simplifiedMesh.indices;
                        newClusterGroup.buildDataIndex   = threadIndex;
                        newClusterGroup.isLeaf           = false;
                        newClusterGroup.hasVoxels        = hasVoxels;
                        newClusterGroup.maxParentError   = Max(error, voxelSize);
                        newClusterGroup.lodBounds        = parentSphereBounds;
                        newClusterGroup.parentStartIndex = parentStartIndex;
                        newClusterGroup.parentCount      = numParentClusters;

                        newClusterGroup.numVertices = simplifiedMesh.numVertices;
                        newClusterGroup.numIndices  = simplifiedMesh.numIndices;
                        newClusterGroup.mipLevel    = depth;

                        clusterGroups[newGroupIndex] = newClusterGroup;
                        break;
                    }
                });

            // Write obj to disk
#if 0
            ArrayView<ClusterGroup> levelClusterGroups(clusterGroups, totalNumGroups,
                                                       partitionResult.ranges.Length());
            u32 vertexCount = numVertices.load();
            u32 indexCount  = numIndices.load();
            u32 voxelCount  = numVoxels.load();

            WriteClustersToOBJ(levelClusterGroups, nextLevelClusters, filename, vertexCount,
                               indexCount, voxelCount, numAttributes, depth);

#endif

            u32 numNextLevelClusters = numLevelClusters.load();

            clusters.Resize(totalNumClusters + numLevelClusters.load());

            Assert(numLevelClusters.load() <= levelClusters.Length());

            StaticArray<Cluster> reorderedClusters(scratch.temp.arena, levelClusters.Length(),
                                                   levelClusters.Length());
            for (int groupIndex = 0; groupIndex < partitionResult.ranges.Length();
                 groupIndex++)
            {
                PartitionRange &range   = partitionResult.ranges[groupIndex];
                ClusterGroup &group     = clusterGroups[totalNumGroups + groupIndex];
                group.clusterStartIndex = prevClusterArrayEnd + range.begin;
                group.clusterCount      = range.end - range.begin;
                ErrorExit(group.clusterStartIndex + group.clusterCount <= clusters.Length(),
                          "%u %u %u\n", group.clusterStartIndex, group.clusterCount,
                          clusters.Length());

                for (int i = range.begin; i < range.end; i++)
                {
                    int clusterIndex = partitionResult.clusterIndices[i];
                    Assert(levelClusters[clusterIndex].groupIndex ==
                           totalNumGroups + groupIndex);
                    reorderedClusters[i] = levelClusters[clusterIndex];
                }
            }
            MemoryCopy(levelClusters.data, reorderedClusters.data,
                       levelClusters.Length() * sizeof(Cluster));

            for (int groupIndex = 0; groupIndex < clusterGroups.Length(); groupIndex++)
            {
                ClusterGroup &group = clusterGroups[groupIndex];
                for (int i = group.clusterStartIndex;
                     i < group.clusterStartIndex + group.clusterCount; i++)
                {
                    Assert(clusters[i].groupIndex == groupIndex);
                }
            }

            prevClusterArrayEnd = totalNumClusters;
            u32 clusterOffset   = 0;
            for (int groupIndex = 0; groupIndex < partitionResult.ranges.Length();
                 groupIndex++)
            {
                PartitionRange &range = partitionResult.ranges[groupIndex];
                ClusterGroup &group   = clusterGroups[totalNumGroups + groupIndex];

                u32 newStartIndex = clusterOffset;
                for (int parentIndex = group.parentStartIndex;
                     parentIndex < group.parentStartIndex + group.parentCount; parentIndex++)
                {
                    Assert(clusters[totalNumClusters + parentIndex].childGroupIndex ==
                           totalNumGroups + groupIndex);
                    reorderedClusters[clusterOffset++] =
                        clusters[totalNumClusters + parentIndex];
                }

                group.parentStartIndex = newStartIndex;
            }
            MemoryCopy(clusters.data + totalNumClusters, reorderedClusters.data,
                       clusterOffset * sizeof(Cluster));

            bool skip = numLevelClusters.load() == levelClusters.Length();

            levelClusters = ArrayView<Cluster>(clusters, totalNumClusters,
                                               clusters.Length() - totalNumClusters);
            depth++;

            if (skip)
            {
                break;
            }
        }
    }

    // Write clusters to disk
    StaticArray<u8 *> geoByteDatasBuffer(scratch.temp.arena, buildDatas.Length());
    StaticArray<u8 *> shadingByteDatasBuffer(scratch.temp.arena, buildDatas.Length());
    StaticArray<PackedDenseGeometryHeader *> headersBuffer(scratch.temp.arena,
                                                           buildDatas.Length());

    for (auto &buildData : buildDatas)
    {
        u8 *geoByteData =
            PushArrayNoZero(scratch.temp.arena, u8, buildData.geoByteBuffer.Length());
        u8 *shadingByteData =
            PushArrayNoZero(scratch.temp.arena, u8, buildData.shadingByteBuffer.Length());
        PackedDenseGeometryHeader *headers = PushArrayNoZero(
            scratch.temp.arena, PackedDenseGeometryHeader, buildData.headers.Length());

        buildData.geoByteBuffer.Flatten(geoByteData);
        buildData.shadingByteBuffer.Flatten(shadingByteData);
        buildData.headers.Flatten(headers);

        geoByteDatasBuffer.Push(geoByteData);
        shadingByteDatasBuffer.Push(shadingByteData);
        headersBuffer.Push(headers);
    }

    int startIndexIndex          = 0;
    u32 currentGeoBufferSize     = 0;
    u32 currentShadingBufferSize = 0;

    Vec3f minP(pos_inf);
    Vec3f maxP(neg_inf);
    for (int groupIndex = 0; groupIndex < clusterGroups.Length(); groupIndex++)
    {
        ClusterGroup &group = clusterGroups[groupIndex];
        minP                = Min(group.lodBounds.xyz, minP);
        maxP                = Max(group.lodBounds.xyz, maxP);
    }

    Vec3f dist         = maxP - minP;
    float maxComponent = Max(dist.x, Max(dist.y, dist.z));
    float scale        = 1023.f / maxComponent;

    struct GroupHandle
    {
        u64 sortKey;
        int index;
    };

    GroupHandle *groupHandles =
        PushArrayNoZero(scratch.temp.arena, GroupHandle, clusterGroups.Length());

    for (int groupIndex = 0; groupIndex < clusterGroups.Length(); groupIndex++)
    {
        ClusterGroup &group = clusterGroups[groupIndex];

        Vec3i quantizedCenter =
            Clamp(Vec3i((group.lodBounds.xyz - minP) * scale + 0.5f), Vec3i(0), Vec3i(1023));

        u32 key = (MortonCode3(quantizedCenter.z) << 2) |
                  (MortonCode3(quantizedCenter.y) << 1) | MortonCode3(quantizedCenter.x);

        if (group.mipLevel & 1)
        {
            key ^= ~0u;
        }

        GroupHandle handle;
        handle.sortKey = ((u64)group.mipLevel << 32u) | key;
        handle.index   = groupIndex;

        groupHandles[groupIndex] = handle;
    }

    SortHandles(groupHandles, clusterGroups.Length());

    string outFilename =
        PushStr8F(scratch.temp.arena, "%S.geo", RemoveFileExtension(filename));
    StringBuilderMapped builder(outFilename);
    u64 fileHeaderOffset = AllocateSpace(&builder, sizeof(ClusterFileHeader));

    struct PageInfo
    {
        u32 partStartIndex;
        u32 partCount;
        u32 numClusters;
    };

    u32 clusterPageStartIndex = 0;
    u32 numClustersInPage     = 0;
    u32 partStartIndex        = 0;
    StaticArray<GroupPart> parts(scratch.temp.arena, clusters.Length());
    StaticArray<PageInfo> pageInfos(scratch.temp.arena, clusterGroups.Length() * 4);

    auto GetGeoByteSize = [&](int headerIndex, int buildDataIndex) {
        u8 *geoByteData                    = geoByteDatasBuffer[buildDataIndex];
        PackedDenseGeometryHeader *headers = headersBuffer[buildDataIndex];
        u32 numHeaders                     = buildDatas[buildDataIndex].headers.Length();
        Assert(headerIndex < numHeaders);
        return (headerIndex == numHeaders - 1
                    ? buildDatas[buildDataIndex].geoByteBuffer.Length()
                    : headers[headerIndex + 1].a) -
               headers[headerIndex].a;
    };

    auto GetShadByteSize = [&](int headerIndex, int buildDataIndex) {
        u8 *shadingByteData                = shadingByteDatasBuffer[buildDataIndex];
        PackedDenseGeometryHeader *headers = headersBuffer[buildDataIndex];
        u32 numHeaders                     = buildDatas[buildDataIndex].headers.Length();
        Assert(headerIndex < numHeaders);
        return (headerIndex == numHeaders - 1
                    ? buildDatas[buildDataIndex].shadingByteBuffer.Length()
                    : headers[headerIndex + 1].z) -
               headers[headerIndex].z;
    };

    for (int handleIndex = 0; handleIndex < clusterGroups.Length(); handleIndex++)
    {
        GroupHandle handle  = groupHandles[handleIndex];
        u32 groupIndex      = handle.index;
        ClusterGroup &group = clusterGroups[groupIndex];
        if (group.isLeaf) continue;

        group.pageStartIndex = pageInfos.Length();
        group.partStartIndex = parts.Length();

        u32 clusterStartIndex = 0;

        for (int clusterGroupIndex = 0; clusterGroupIndex < group.clusterCount;
             clusterGroupIndex++)
        {
            u32 clusterMetadataSize =
                (numClustersInPage + 1) * NUM_CLUSTER_HEADER_FLOAT4S * sizeof(Vec4f);

            u32 clusterIndex = group.clusterStartIndex + clusterGroupIndex;
            Cluster &cluster = clusters[clusterIndex];
            ErrorExit(cluster.groupIndex == groupIndex, "%u %u\n", cluster.groupIndex,
                      groupIndex);
            ClusterGroup &childGroup = clusterGroups[cluster.childGroupIndex];
            int buildDataIndex       = childGroup.buildDataIndex;
            u32 geoByteSize          = GetGeoByteSize(cluster.headerIndex, buildDataIndex);
            u32 shadByteSize         = GetShadByteSize(cluster.headerIndex, buildDataIndex);

            u32 totalSize = sizeof(ClusterPageHeader) + clusterMetadataSize +
                            currentGeoBufferSize + currentShadingBufferSize + geoByteSize +
                            shadByteSize;

            if (totalSize > CLUSTER_PAGE_SIZE || numClustersInPage == MAX_CLUSTERS_PER_PAGE)
            {
                if (clusterGroupIndex > 0)
                {
                    GroupPart part;
                    part.groupIndex            = groupIndex;
                    part.clusterStartIndex     = clusterStartIndex;
                    part.clusterCount          = clusterGroupIndex - clusterStartIndex;
                    part.clusterPageStartIndex = clusterPageStartIndex;
                    part.pageIndex             = pageInfos.Length();

                    parts.Push(part);
                    clusterStartIndex = clusterGroupIndex;
                }

                PageInfo pageInfo;
                pageInfo.partStartIndex = partStartIndex;
                pageInfo.partCount      = parts.Length() - partStartIndex;
                pageInfo.numClusters    = numClustersInPage;
                pageInfos.Push(pageInfo);

                partStartIndex           = parts.Length();
                currentGeoBufferSize     = 0;
                currentShadingBufferSize = 0;
                numClustersInPage        = 0;
                clusterPageStartIndex    = 0;
            }

            numClustersInPage++;
            currentGeoBufferSize += geoByteSize;
            currentShadingBufferSize += shadByteSize;
        }

        group.numPages = (pageInfos.Length() - group.pageStartIndex) + 1;

        GroupPart part;
        part.groupIndex            = groupIndex;
        part.clusterStartIndex     = clusterStartIndex;
        part.clusterCount          = group.clusterCount - clusterStartIndex;
        part.clusterPageStartIndex = clusterPageStartIndex;
        part.pageIndex             = pageInfos.Length();

        clusterPageStartIndex = numClustersInPage;

        parts.Push(part);
        group.numParts = parts.Length() - group.numParts;
    }

    PageInfo finalPageInfo;
    finalPageInfo.partStartIndex = partStartIndex;
    finalPageInfo.partCount      = parts.Length() - partStartIndex;
    finalPageInfo.numClusters    = numClustersInPage;
    pageInfos.Push(finalPageInfo);

    Graph<ClusterFixup> pageToParentClusterGraph;
    u32 numParentClusters = pageToParentClusterGraph.InitializeStatic(
        scratch.temp.arena, pageInfos.Length(),
        [&](u32 pageIndex, u32 *offsets, ClusterFixup *data = 0) {
            PageInfo &pageInfo = pageInfos[pageIndex];
            u32 num            = 0;
            for (int partIndex = pageInfo.partStartIndex;
                 partIndex < pageInfo.partStartIndex + pageInfo.partCount; partIndex++)
            {
                GroupPart &part     = parts[partIndex];
                ClusterGroup &group = clusterGroups[part.groupIndex];

                // need to find the parent groups/clusters somehow
                for (int clusterIndex = part.clusterStartIndex;
                     clusterIndex < part.clusterStartIndex + part.clusterCount; clusterIndex++)
                {
                    Cluster &cluster = clusters[group.clusterStartIndex + clusterIndex];
                    if (cluster.childGroupIndex == 0) continue;
                    Assert(cluster.groupIndex == part.groupIndex);
                    ClusterGroup &childGroup = clusterGroups[cluster.childGroupIndex];

                    ClusterFixup fixup(part.pageIndex,
                                       part.clusterPageStartIndex + clusterIndex -
                                           part.clusterStartIndex,
                                       childGroup.pageStartIndex, childGroup.numPages);

                    for (u32 childPageIndex = childGroup.pageStartIndex;
                         childPageIndex < childGroup.pageStartIndex + childGroup.numPages;
                         childPageIndex++)
                    {
                        u32 dataIndex = offsets[childPageIndex]++;
                        num++;

                        if (data) data[dataIndex] = fixup;
                    }
                }
            }
            return num;
        });

    Graph<u32> pageToParentPageGraph;
    u32 numParentPages = pageToParentPageGraph.InitializeStatic(
        scratch.temp.arena, pageInfos.Length(),
        [&](u32 pageIndex, u32 *offsets, u32 *data = 0) {
            u32 num = 0;
            for (int clusterIndex = pageToParentClusterGraph.offsets[pageIndex];
                 clusterIndex < pageToParentClusterGraph.offsets[pageIndex + 1];
                 clusterIndex++)
            {
                ClusterFixup fixup(pageToParentClusterGraph.data[clusterIndex]);
                if (fixup.GetPageIndex() == pageIndex) continue;
                bool add = 1;
                for (int otherClusterIndex = clusterIndex + 1;
                     otherClusterIndex < pageToParentClusterGraph.offsets[pageIndex + 1];
                     otherClusterIndex++)
                {
                    ClusterFixup otherFixup =
                        ClusterFixup{pageToParentClusterGraph.data[otherClusterIndex]};
                    if (fixup.GetPageIndex() == otherFixup.GetPageIndex())
                    {
                        add = 0;
                        break;
                    }
                }
                if (add)
                {
                    u32 dataIndex = offsets[pageIndex]++;
                    num++;

                    if (data) data[dataIndex] = fixup.GetPageIndex();
                }
            }
            return num;
        });

    u32 *clusterVoxelOffsets = PushArrayNoZero(scratch.temp.arena, u32, clusters.Length());
    u32 totalVoxelClusters   = 0;
    for (u32 groupIndex = 0; groupIndex < clusterGroups.Length(); groupIndex++)
    {
        auto &group          = clusterGroups[groupIndex];
        u32 numVoxelClusters = 0;
        for (int clusterIndex = group.clusterStartIndex;
             clusterIndex < group.clusterStartIndex + group.clusterCount; clusterIndex++)
        {
            Cluster &cluster = clusters[clusterIndex];
            if (cluster.compressedVoxels.Length()) numVoxelClusters++;
        }
        clusterVoxelOffsets[groupIndex] = totalVoxelClusters;
        totalVoxelClusters += numVoxelClusters;
    }

    u32 *clusterVoxelCounts = PushArray(scratch.temp.arena, u32, clusters.Length());
    u32 *clusterVoxelData   = PushArrayNoZero(scratch.temp.arena, u32, totalVoxelClusters);
    u32 id                  = 0;
    u32 *clusterIndices     = PushArrayNoZero(scratch.temp.arena, u32, clusters.Length());
    for (u32 groupIndex = 0; groupIndex < clusterGroups.Length(); groupIndex++)
    {
        auto &group = clusterGroups[groupIndex];
        for (int clusterIndex = group.clusterStartIndex;
             clusterIndex < group.clusterStartIndex + group.clusterCount; clusterIndex++)
        {
            Cluster &cluster = clusters[clusterIndex];
            if (cluster.compressedVoxels.Length())
            {
                ClusterGroup &childGroup = clusterGroups[cluster.childGroupIndex];
                // u32 newID                    = id++;
                // clusterIndices[clusterIndex] = newID;
                if (!childGroup.hasVoxels)
                {
                    u32 dataIndex =
                        clusterVoxelOffsets[groupIndex] + clusterVoxelCounts[groupIndex]++;
                    u32 newID                    = id++;
                    clusterVoxelData[dataIndex]  = newID;
                    clusterIndices[clusterIndex] = newID;
                }
                else
                {
                    u32 offset = clusterVoxelOffsets[cluster.childGroupIndex]++;
                    Assert(offset < clusterVoxelOffsets[cluster.childGroupIndex + 1]);
                    Assert(clusterVoxelCounts[cluster.childGroupIndex]);
                    u32 newID = clusterVoxelData[offset];
                    clusterVoxelCounts[cluster.childGroupIndex]++;

                    u32 dataIndex =
                        clusterVoxelOffsets[groupIndex] + clusterVoxelCounts[groupIndex]++;
                    clusterVoxelData[dataIndex]  = newID;
                    clusterIndices[clusterIndex] = newID;
                }
            }
        }
    }

    // Write the data to the pages
    for (auto &pageInfo : pageInfos)
    {
        u32 numClustersInPage = pageInfo.numClusters;
        u64 fileOffset        = AllocateSpace(&builder, CLUSTER_PAGE_SIZE);
        u8 *ptr               = (u8 *)GetMappedPtr(&builder, fileOffset);

        u32 baseGeoOffset = sizeof(ClusterPageHeader) +
                            numClustersInPage * NUM_CLUSTER_HEADER_FLOAT4S * sizeof(Vec4u);
        u32 currentGeoOffset = baseGeoOffset;
        MemoryCopy(ptr, &numClustersInPage, sizeof(ClusterPageHeader));

        for (int partIndex = pageInfo.partStartIndex;
             partIndex < pageInfo.partStartIndex + pageInfo.partCount; partIndex++)
        {
            GroupPart &part     = parts[partIndex];
            ClusterGroup &group = clusterGroups[part.groupIndex];

            for (int clusterGroupIndex = part.clusterStartIndex;
                 clusterGroupIndex < part.clusterStartIndex + part.clusterCount;
                 clusterGroupIndex++)
            {
                Cluster &cluster = clusters[group.clusterStartIndex + clusterGroupIndex];
                ClusterGroup &childGroup = clusterGroups[cluster.childGroupIndex];
                u32 geoByteSize =
                    GetGeoByteSize(cluster.headerIndex, childGroup.buildDataIndex);
                currentGeoOffset += geoByteSize;
            }
        }

        u32 currentShadOffset = currentGeoOffset;
        u32 baseShadOffset    = currentGeoOffset;
        currentGeoOffset      = baseGeoOffset;

        // Write headers in SOA
        u32 stride            = sizeof(Vec4u);
        u32 soaStride         = numClustersInPage * stride;
        u32 currentPageOffset = sizeof(ClusterPageHeader);

        for (int partIndex = pageInfo.partStartIndex;
             partIndex < pageInfo.partStartIndex + pageInfo.partCount; partIndex++)
        {
            GroupPart &part     = parts[partIndex];
            ClusterGroup &group = clusterGroups[part.groupIndex];

            for (int clusterGroupIndex = part.clusterStartIndex;
                 clusterGroupIndex < part.clusterStartIndex + part.clusterCount;
                 clusterGroupIndex++)
            {
                int clusterIndex         = group.clusterStartIndex + clusterGroupIndex;
                Cluster &cluster         = clusters[clusterIndex];
                ClusterGroup &childGroup = clusterGroups[cluster.childGroupIndex];
                int headerIndex          = cluster.headerIndex;
                int buildDataIndex       = childGroup.buildDataIndex;

                PackedDenseGeometryHeader header = headersBuffer[buildDataIndex][headerIndex];
                header.z                         = currentShadOffset;
                header.a                         = currentGeoOffset;

                MemoryCopy(ptr + currentPageOffset, &cluster.lodBounds, sizeof(Vec4f));

                for (u32 i = 1; i < 4; i++)
                {
                    u32 copySize = Min(stride, (u32)sizeof(header) - (i - 1) * stride);
                    u32 *src     = (u32 *)&header + 4u * (i - 1);
                    MemoryCopy(ptr + currentPageOffset + i * soaStride, src, copySize);
                }

                u32 flags = CLUSTER_STREAMING_LEAF_FLAG;

                MemoryCopy(ptr + currentPageOffset + (4 - 1) * soaStride + sizeof(Vec3f),
                           &cluster.lodError, sizeof(float));
                MemoryCopy(ptr + currentPageOffset + 4 * soaStride, &flags, sizeof(u32));
                MemoryCopy(ptr + currentPageOffset + 4 * soaStride + sizeof(u32),
                           &clusterIndices[clusterIndex], sizeof(u32));
                currentPageOffset += sizeof(Vec4u);

                currentGeoOffset += GetGeoByteSize(headerIndex, buildDataIndex);
                currentShadOffset += GetShadByteSize(headerIndex, buildDataIndex);
            }
        }

        currentPageOffset = baseGeoOffset;

        for (int partIndex = pageInfo.partStartIndex;
             partIndex < pageInfo.partStartIndex + pageInfo.partCount; partIndex++)
        {
            GroupPart &part     = parts[partIndex];
            ClusterGroup &group = clusterGroups[part.groupIndex];

            for (int clusterGroupIndex = part.clusterStartIndex;
                 clusterGroupIndex < part.clusterStartIndex + part.clusterCount;
                 clusterGroupIndex++)
            {
                int clusterIndex         = group.clusterStartIndex + clusterGroupIndex;
                Cluster &cluster         = clusters[clusterIndex];
                ClusterGroup &childGroup = clusterGroups[cluster.childGroupIndex];
                int headerIndex          = cluster.headerIndex;
                int buildDataIndex       = childGroup.buildDataIndex;

                PackedDenseGeometryHeader &header = headersBuffer[buildDataIndex][headerIndex];
                u8 *geoByteData                   = geoByteDatasBuffer[buildDataIndex];

                u32 geoByteSize = GetGeoByteSize(headerIndex, buildDataIndex);
                u32 geoOffset   = header.a;

                MemoryCopy(ptr + currentPageOffset, geoByteData + geoOffset, geoByteSize);
                currentPageOffset += geoByteSize;
            }
        }

        Assert(currentPageOffset == baseShadOffset);

        for (int partIndex = pageInfo.partStartIndex;
             partIndex < pageInfo.partStartIndex + pageInfo.partCount; partIndex++)
        {
            GroupPart &part     = parts[partIndex];
            ClusterGroup &group = clusterGroups[part.groupIndex];

            for (int clusterGroupIndex = part.clusterStartIndex;
                 clusterGroupIndex < part.clusterStartIndex + part.clusterCount;
                 clusterGroupIndex++)
            {
                int clusterIndex         = group.clusterStartIndex + clusterGroupIndex;
                Cluster &cluster         = clusters[clusterIndex];
                ClusterGroup &childGroup = clusterGroups[cluster.childGroupIndex];
                int headerIndex          = cluster.headerIndex;
                int buildDataIndex       = childGroup.buildDataIndex;

                PackedDenseGeometryHeader &header = headersBuffer[buildDataIndex][headerIndex];
                u8 *shadingByteData               = shadingByteDatasBuffer[buildDataIndex];

                u32 shadByteSize = GetShadByteSize(headerIndex, buildDataIndex);
                u32 shadOffset   = header.z;

                MemoryCopy(ptr + currentPageOffset, shadingByteData + shadOffset,
                           shadByteSize);
                currentPageOffset += shadByteSize;
            }
        }
    }

    // Build hierarchies over cluster groups
    PrimRef *hierarchyPrimRefs = PushArrayNoZero(scratch.temp.arena, PrimRef, parts.Length());
    Bounds geomBounds;
    Bounds centBounds;

    StaticArray<u32> partStarts(scratch.temp.arena, depth + 1);
    StaticArray<RecordAOSSplits> records(scratch.temp.arena, depth, depth);
    partStarts.Push(0);

    u32 groupDepth = 0;
    for (int i = 0; i < parts.Length(); i++)
    {
        PrimRef &primRef = hierarchyPrimRefs[i];
        primRef.primID   = i;

        GroupPart &part     = parts[i];
        ClusterGroup &group = clusterGroups[part.groupIndex];

        if (group.mipLevel != groupDepth)
        {
            Assert(groupDepth + 1 == group.mipLevel);
            u32 start = partStarts[partStarts.Length() - 1];

            RecordAOSSplits hierarchyRecord;
            hierarchyRecord.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
            hierarchyRecord.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
            hierarchyRecord.start      = start;
            hierarchyRecord.count      = i - start;

            records[groupDepth] = hierarchyRecord;
            partStarts.Push(i);
            groupDepth++;

            geomBounds = Bounds();
            centBounds = Bounds();
        }

        Bounds partBounds;

        for (int clusterGroupIndex = part.clusterStartIndex;
             clusterGroupIndex < part.clusterStartIndex + part.clusterCount;
             clusterGroupIndex++)
        {
            Cluster &cluster = clusters[group.clusterStartIndex + clusterGroupIndex];

            partBounds.Extend(cluster.bounds);
        }

        primRef.minX = -partBounds.minP[0];
        primRef.minY = -partBounds.minP[1];
        primRef.minZ = -partBounds.minP[2];
        primRef.maxX = partBounds.maxP[0];
        primRef.maxY = partBounds.maxP[1];
        primRef.maxZ = partBounds.maxP[2];

        geomBounds.Extend(partBounds);
        centBounds.Extend(partBounds.minP + partBounds.maxP);
    }

    u32 start = partStarts[partStarts.Length() - 1];
    RecordAOSSplits hierarchyRecord;
    hierarchyRecord.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    hierarchyRecord.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    hierarchyRecord.start      = start;
    hierarchyRecord.count      = parts.Length() - start;
    partStarts.Push(partStarts.Length());
    records[groupDepth] = hierarchyRecord;
    geomBounds          = Bounds();
    centBounds          = Bounds();

    Arena *arena = ArenaAlloc();
    u32 numNodes = 0;

    // First build hierarchy over each level
    StaticArray<HierarchyNode> depthHierarchyNodes(scratch.temp.arena, depth);
    for (int index = 0; index < depth; index++)
    {
        HierarchyNode node = BuildHierarchy(arena, clusters, clusterGroups, parts,
                                            hierarchyPrimRefs, records[index], numNodes);
        depthHierarchyNodes.Push(node);
    }

    PrimRef *topLevelRefs =
        PushArrayNoZero(scratch.temp.arena, PrimRef, depthHierarchyNodes.Length());
    for (int index = 0; index < depth; index++)
    {
        HierarchyNode &node = depthHierarchyNodes[index];
        PrimRef &primRef    = topLevelRefs[index];
        for (int i = 0; i < 3; i++)
        {
            primRef.min[i] = neg_inf;
            primRef.max[i] = neg_inf;
        }

        for (int childIndex = 0; childIndex < node.numChildren; childIndex++)
        {
            primRef.minX = Max(-node.bounds[childIndex].minP[0], primRef.minX);
            primRef.minY = Max(-node.bounds[childIndex].minP[1], primRef.minY);
            primRef.minZ = Max(-node.bounds[childIndex].minP[2], primRef.minZ);

            primRef.maxX = Max(node.bounds[childIndex].maxP[0], primRef.maxX);
            primRef.maxY = Max(node.bounds[childIndex].maxP[1], primRef.maxY);
            primRef.maxZ = Max(node.bounds[childIndex].maxP[2], primRef.maxZ);
            geomBounds.Extend(node.bounds[childIndex]);
            centBounds.Extend(node.bounds[childIndex].minP + node.bounds[childIndex].maxP);
        }
        primRef.primID = index;
    }

    RecordAOSSplits topLevelRecord;
    topLevelRecord.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    topLevelRecord.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    topLevelRecord.start      = 0;
    topLevelRecord.count      = depthHierarchyNodes.Length();

    // Then build over the root nodes of each level
    HierarchyNode rootNode = BuildTopLevelHierarchy(arena, depthHierarchyNodes, topLevelRefs,
                                                    topLevelRecord, numNodes);

    // Flatten tree to array
    StaticArray<PackedHierarchyNode> hierarchy(arena, numNodes);

    struct StackEntry
    {
        HierarchyNode node;

        u32 parentIndex;
        u32 childIndex;
    };

    const u32 maxClustersPerSubtree = MAX_CLUSTERS_PER_BLAS;

    StaticArray<StackEntry> queue(scratch.temp.arena, numNodes, numNodes);
    u32 readOffset  = 0;
    u32 writeOffset = 0;

    StackEntry root;
    root.node        = rootNode;
    root.parentIndex = ~0u;
    root.childIndex  = ~0u;

    queue[writeOffset++] = root;

    // 1. instance culling hierarchy. procedural aabbs as proxies for instance groups.
    // closest hit intersections of these aabbs are saved, and a bvh is built over just the
    // instances inside these proxies. the intersections are then repeated with this
    // smaller set.
    // 2. partial rebraiding
    // 3. instance proxy combining

    // interesting idea:
    // 1. create a bounding sphere/aabb hierarchy over instances. the leaves contain
    // instance ids.
    // 2. use previous frame's rays to traverse this hierarchy. instances in intersected
    // leaves are added to the tlas. this would be in addition to standard
    // occlusion/frustum/small element culling (maybe? i'm not sure about this last part)
    // 3. since we control this hierarchy, instead of normal instancing you could use
    // instanced submeshes (cluster groups), reducing overlap between instances (just like
    // partial rebraiding)
    // 4. thus instance data can be very highly compressed, and decompressed when needed

    // 5. maybe have some form of feedback? like with virtual textures (idk how this fits
    // in to the rest of the system)
    // 6. when the instance is small enough, use some smaller proxy/proxies (i.e., somehow
    // combine and simplify the instances)
    // 7. partitioned tlas (obviously)

    // so if the instance is outside the frustum or occluded, and wasn't intersected, then
    // it should be removed

    u32 numLeaves    = 0;
    u32 numParts     = 0;
    u32 numLeafParts = 0;

    for (;;)
    {
        if (writeOffset == readOffset) break;

        u32 readIndex    = readOffset++;
        StackEntry entry = queue[readIndex];

        u32 childOffset = hierarchy.Length();
        u32 parentIndex = entry.parentIndex;

        if (parentIndex != ~0u)
        {
            hierarchy[parentIndex].childRef[entry.childIndex] = childOffset;
        }

        HierarchyNode &child = entry.node;

        PackedHierarchyNode packed = {};
        for (int i = 0; i < CHILDREN_PER_HIERARCHY_NODE; i++)
        {
            packed.childRef[i] = ~0u;
            packed.leafInfo[i] = ~0u;
        }
        numLeaves += !(bool)child.children;

        u32 clusterTotal = 0;
        for (int i = 0; i < child.numChildren; i++)
        {
            clusterTotal += child.clusterTotals[i];
        }

        for (int i = 0; i < child.numChildren; i++)
        {
            packed.lodBounds[i] = child.lodBounds[i];
            packed.center[i]    = ToVec3f(child.bounds[i].Centroid());
            packed.extents[i] = ToVec3f((child.bounds[i].maxP - child.bounds[i].minP) * 0.5f);
            packed.maxParentError[i] = child.maxParentError[i];

            if (child.children)
            {
                StackEntry newEntry;
                newEntry.node        = child.children[i];
                newEntry.parentIndex = childOffset;
                newEntry.childIndex  = i;

                u32 writeIndex    = writeOffset++;
                queue[writeIndex] = newEntry;
            }
            else
            {
                numParts++;
                u32 partIndex   = child.partIndices[i];
                GroupPart &part = parts[partIndex];
                Assert(part.clusterPageStartIndex < MAX_CLUSTERS_PER_PAGE);
                Assert(part.clusterCount <= 32);

                Assert(part.pageIndex < (1u << 16));
                u32 numPages = clusterGroups[part.groupIndex].numPages;
                ErrorExit(numPages < (1u << MAX_PARTS_PER_GROUP_BITS), "%u\n", numPages);
                Assert(numPages != 0);

                u32 pageStartIndex = clusterGroups[part.groupIndex].pageStartIndex;
                u32 leafInfo       = 0;
                u32 bitOffset      = 0;
                Assert(part.clusterPageStartIndex < MAX_CLUSTERS_PER_PAGE);
                leafInfo = BitFieldPackU32(leafInfo, part.clusterPageStartIndex, bitOffset,
                                           MAX_CLUSTERS_PER_PAGE_BITS);
                Assert(part.clusterCount - 1 < MAX_CLUSTERS_PER_GROUP);
                leafInfo = BitFieldPackU32(leafInfo, part.clusterCount - 1, bitOffset,
                                           MAX_CLUSTERS_PER_GROUP_BITS);
                Assert(numPages < MAX_PARTS_PER_GROUP);
                leafInfo =
                    BitFieldPackU32(leafInfo, numPages, bitOffset, MAX_PARTS_PER_GROUP_BITS);
                leafInfo =
                    BitFieldPackU32(leafInfo, pageStartIndex, bitOffset, 32u - bitOffset);

                packed.leafInfo[i] = leafInfo;
                packed.childRef[i] = part.pageIndex;
            }
        }

        hierarchy.Push(packed);
    }

    Assert(numNodes != 0);
    Print("num nodes: %u\nnum parts: %u %u, num leaves: %u %u\n", numNodes, parts.Length(),
          numParts, numLeaves, numLeafParts);
    Assert(hierarchy.Length() == numNodes);

    u32 totalNumVoxelClusters = numVoxelClusters.load();
    ClusterFileHeader *fileHeader =
        (ClusterFileHeader *)GetMappedPtr(&builder, fileHeaderOffset);
    fileHeader->magic            = CLUSTER_FILE_MAGIC;
    fileHeader->numPages         = pageInfos.Length();
    fileHeader->numNodes         = numNodes;
    fileHeader->numVoxelClusters = totalNumVoxelClusters;
    fileHeader->boundsMin        = scene.boundsMin;
    fileHeader->boundsMax        = scene.boundsMax;

    Print("num voxel clusters: %u\n", totalNumVoxelClusters);

    // Write hierarchy to disk
    u64 hierarchyOffset = AllocateSpace(&builder, sizeof(PackedHierarchyNode) * numNodes);
    u8 *ptr             = (u8 *)GetMappedPtr(&builder, hierarchyOffset);
    MemoryCopy(ptr, hierarchy.data, sizeof(PackedHierarchyNode) * numNodes);

    // Graphs
    u32 offsetsSize = sizeof(u32) * (pageInfos.Length() + 1);
    u64 pageToParentPageOffset =
        AllocateSpace(&builder, offsetsSize + sizeof(u32) * numParentPages);
    ptr = (u8 *)GetMappedPtr(&builder, pageToParentPageOffset);
    MemoryCopy(ptr, pageToParentPageGraph.offsets, offsetsSize);
    ptr += offsetsSize;
    MemoryCopy(ptr, pageToParentPageGraph.data, sizeof(u32) * numParentPages);

    u64 pageToParentClusterOffset =
        AllocateSpace(&builder, offsetsSize + sizeof(ClusterFixup) * numParentClusters);
    ptr = (u8 *)GetMappedPtr(&builder, pageToParentClusterOffset);
    MemoryCopy(ptr, pageToParentClusterGraph.offsets, offsetsSize);
    ptr += offsetsSize;
    MemoryCopy(ptr, pageToParentClusterGraph.data, sizeof(ClusterFixup) * numParentClusters);

    OS_UnmapFile(builder.ptr);
    OS_ResizeFile(builder.filename, builder.totalSize);

    ReleaseArenaArray(arenas);
}

struct ClusterInfo
{
    StaticArray<Cluster> mipClusters;
};

struct InstanceType2
{
};

struct IntermediateInfo
{
    AffineSpace *transforms;
    InstanceType2 *instances;
};

struct KDTreeNode
{
    Bounds bounds;
    union
    {
        struct
        {
            int axis;
            f32 split;
        };
        struct
        {
            int start;
            int count;
        };
    };
    KDTreeNode *left;
    KDTreeNode *right;
};

#if 0
KDTreeNode BuildKDTree(Arena **arenas, PrimRef *refs, PrimRef *doubleBufferRefs, u32 start,
                       u32 count)
{
    struct Handle
    {
        u32 sortKey;
        u32 index;
    };
    struct Float
    {
        f32 f;
        u32 i;
    };

    // Build KDTree
    ScratchArena scratch;
    Handle *handles = PushArrayNoZero(scratch.temp.arena, Handle, count);

    // Chose axis with max extent
    int bestAxis  = 0;
    f32 maxExtent = neg_inf;
    for (int axis = 0; axis < 3; axis++)
    {
        f32 extent = bounds.maxP[axis] - bounds.minP[axis];
        if (extent > maxExtent)
        {
            bestAxis  = axis;
            maxExtent = extent;
        }
    }

    for (u32 refIndex = start; refIndex < start + count; refIndex++)
    {
        PrimRef &ref = refs[refIndex];
        f32 center   = (ref.min[bestAxis] + ref.max[bestAxis]) / 2.f;
        Float fl;
        fl.f = center;

        Handle handle;
        handle.sortKey            = fl.i;
        handle.index              = refIndex;
        handles[refIndex - start] = handle;
    }

    SortHandles(handles, count);

    PrimRef &centerRef = refs[handle[count / 2].index];
    f32 centerSplit    = (ref.min[bestAxis] + ref.max[bestAxis]) / 2.f;

    u32 leftStart               = start;
    u32 rightStart              = start + count / 2;
    std::atomic<u32> leftCount  = 0;
    std::atomic<u32> rightCount = 0;

    KDTreeNode *node = PushStructNoZero(arenas[GetThreadIndex()], KDTreeNode);
    node.split       = centerSplit;
    node.axis        = bestAxis;

    ParallelFor(0, count, 4096, 32, [&](u32 jobID, u32 start, u32 count) {
        for (u32 i = start; i < start + count; i++)
        {
            PrimRef &ref            = refs[i];
            f32 center              = (ref.min[bestAxis] + ref.max[bestAxis]) / 2.f;
            u32 index               = center >= centerSplit
                                          ? leftStart + leftCount.fetch_add(1, std::memory_order_relaxed)
                                          : rightStart + rightCount.fetch_add(1, std::memory_order_relaxed);
            doubleBufferRefs[index] = ref;
        }
    });
    Assert(leftCount == count / 2);
    Assert(rightCount == count - count / 2);

    if (parallel)
    {
        ParallelFor(0, 2, 1, [&](u32 jobID, u32 start, u32 count) {
            Assert(count == 1);
            BuildKDTree(arenas, doubleBufferRefs, refs, start, leftCount);
            BuildKDTree(arenas, doubleBufferRefs, refs, start + leftCount, rightCount);
        });
    }
    else
    {
        BuildKDTree(arenas, doubleBufferRefs, refs, start, leftCount);
        BuildKDTree(arenas, doubleBufferRefs, refs, start + leftCount, rightCount);
    }
}
#endif

template <typename LeafFunc>
void TraverseKDTree(PrimRef *refs, KDTreeNode *rootNode, Bounds bounds, LeafFunc &&func)
{
    FixedArray<KDTreeNode *, 128> stack;
    stack.Push(rootNode);

    while (stack.Length())
    {
        KDTreeNode *node = stack.Pop();
        bool leaf        = node->left == 0 && node->right == 0;
        if (leaf)
        {
            for (u32 primRefIndex = node->start; primRefIndex < node->start + node->count;
                 primRefIndex++)
            {
                func(refs[primRefIndex]);
            }
        }
        else
        {
            Bounds testBounds = node->bounds;
            testBounds.Intersect(bounds);
            if (!testBounds.Empty())
            {
                if (node->left) stack.Push(node->left);
                if (node->right) stack.Push(node->right);
            }
        }
    }
}

template <typename T>
__forceinline Mask<T> TriangleAABB_SAT(Vec3<T> v[3], Mask<T> &resultMask)
{
    Vec3<T> edges[3] = {
        v[1] - v[0],
        v[2] - v[1],
        v[0] - v[2],
    };

    for (int i = 0; i < 3; i++)
    {
        int y = (1 << i) & 3;
        int z = (1 << y) & 3;

        for (int j = 0; j < 3; j++)
        {
            int nextAxis = (1 << j) & 3;
            int nextNext = (1 << nextAxis) & 3;

            T p0 = v[j][z] * v[nextAxis][y] - v[j][y] * v[nextAxis][z];
            T p2 = edges[j][y] * v[nextNext][z] - edges[j][z] * v[nextNext][y];

            T r = 0.5f * (Abs(edges[j][z]) + Abs(edges[j][y]));

            resultMask |= (Min(p0, p2) > r) | (Max(p0, p2) < -r);
            if (All(resultMask))
            {
                return resultMask;
            }
        }
    }
    return resultMask;
}

static u32 EvolveSobolSeed(u32 &seed)
{
    // constant from:
    // https://www.pcg-random.org/posts/does-it-beat-the-minimal-standard.html
    const u32 MCG_C = 2739110765;
    seed += MCG_C;

    // Generated using https://github.com/skeeto/hash-prospector
    // Estimated Bias ~583
    u32 hash = seed;
    hash *= 0x92955555u;
    hash ^= hash >> 15;
    return hash;
}

static u32 FastOwenScrambling(u32 index, u32 seed)
{
    index += seed;
    index ^= index * 0x9c117646u;
    index ^= index * 0xe0705d72u;
    return ReverseBits32(index);
}

static Vec2f GetSobolSample(u32 sampleIndex, u32 &seed)
{
    u32 sobolIndex = FastOwenScrambling(sampleIndex, EvolveSobolSeed(seed));
    Vec2u result(sobolIndex);
    result.y ^= result.y >> 16;
    result.y ^= (result.y & 0xFF00FF00) >> 8;
    result.y ^= (result.y & 0xF0F0F0F0) >> 4;
    result.y ^= (result.y & 0xCCCCCCCC) >> 2;
    result.y ^= (result.y & 0xAAAAAAAA) >> 1;

    result.x = FastOwenScrambling(result.x, EvolveSobolSeed(seed));
    result.y = FastOwenScrambling(result.y, EvolveSobolSeed(seed));

    return Vec2f(result.x >> 8, result.y >> 8) * 5.96046447754e-08f;
}

static void VoxelizeTriangles(Arena *arena, SimpleHashSet<Vec3i> &voxelHashSet,
                              f32 *vertexData, u32 *indices, u32 triangleCount,
                              u32 numAttributes, f32 voxelSize)
{

    f32 rcpVoxelSize = 1.f / voxelSize;
    u32 voxelCount   = 0;

    auto GetPosition = [&](u32 index) {
        return *(Vec3f *)(vertexData + (3 + numAttributes) * index);
    };

    for (int triIndex = 0; triIndex < triangleCount; triIndex++)
    {
        Lane4F32 rcpVoxelSizeV(1.f);

        Vec3f pos[3];
        pos[0] = GetPosition(indices[3 * triIndex + 0]) * rcpVoxelSize;
        pos[1] = GetPosition(indices[3 * triIndex + 1]) * rcpVoxelSize;
        pos[2] = GetPosition(indices[3 * triIndex + 2]) * rcpVoxelSize;

        Vec3lf8 v[3];
        for (int i = 0; i < 3; i++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                v[i][axis] = Lane8F32(pos[i][axis]);
            }
        }

        Bounds bounds;
        bounds.minP = Lane4F32(Min(pos[0], Min(pos[1], pos[2])));
        bounds.maxP = Lane4F32(Max(pos[0], Max(pos[1], pos[2])));

        Lane4F32 minVoxel = Floor(bounds.minP);
        Lane4F32 maxVoxel = Ceil(bounds.maxP);

        Vec3lf8 minP(Lane8F32(Min(pos[0].x, Min(pos[1].x, pos[2].x))),
                     Lane8F32(Min(pos[0].y, Min(pos[1].y, pos[2].y))),
                     Lane8F32(Min(pos[0].z, Min(pos[1].z, pos[2].z))));
        Vec3lf8 maxP(Lane8F32(Max(pos[0].x, Max(pos[1].x, pos[2].x))),
                     Lane8F32(Max(pos[0].y, Max(pos[1].y, pos[2].y))),
                     Lane8F32(Max(pos[0].z, Max(pos[1].z, pos[2].z))));

        Vec3f norm = Normalize(Cross(pos[1] - pos[0], pos[2] - pos[0]));

        Lane8F32 e = Dot(Abs(norm), Vec3f(0.5f));
        Vec3lf8 triPlane(Lane8F32(norm.x), Lane8F32(norm.y), Lane8F32(norm.z));
        Lane8F32 d(-Dot(pos[0], norm));

        const int stepSizeX = 8;
        // First, test voxel AABB around triangle AABB
        for (float voxelZ = minVoxel[2]; voxelZ < maxVoxel[2]; voxelZ++)
        {
            for (float voxelY = minVoxel[1]; voxelY < maxVoxel[1]; voxelY++)
            {
                Vec3lf8 voxelCenter;
                voxelCenter.y = Lane8F32(voxelY + 0.5f);
                voxelCenter.z = Lane8F32(voxelZ + 0.5f);

                for (float voxelX = minVoxel[0]; voxelX < maxVoxel[0]; voxelX += stepSizeX)
                {
                    voxelCenter.x = Lane8F32(voxelX + 0.5f) +
                                    Lane8F32(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);

                    Vec3lf8 voxelMin = Max(voxelCenter - Vec3lf8(Lane8F32(0.5f)), minP);
                    Vec3lf8 voxelMax = Min(voxelCenter + Vec3lf8(Lane8F32(0.5f)), maxP);

                    Mask<Lane8F32> mask = voxelMin.x < voxelMax.x & voxelMin.y < voxelMax.y &
                                          voxelMin.z < voxelMax.z;

                    if (None(mask))
                    {
                        continue;
                    }

                    Lane8F32 s = Dot(voxelCenter, triPlane) + d;

                    mask &= s <= e & s >= -e;

                    if (None(mask))
                    {
                        continue;
                    }

                    Vec3lf8 verts[3] = {
                        v[0] - voxelCenter,
                        v[1] - voxelCenter,
                        v[2] - voxelCenter,
                    };

                    mask = ~mask;
                    TriangleAABB_SAT(verts, mask);
                    u32 bits = ~Movemask(mask) & ((1u << stepSizeX) - 1u);
                    while (bits)
                    {
                        u32 increment = Bsf(bits);
                        Assert(voxelX + increment < maxVoxel[0]);
                        bits &= bits - 1;
                        voxelCount++;

                        Vec3i voxel((i32)voxelX, (i32)voxelY, (i32)voxelZ);
                        u32 hash = Hash(voxel);
                        voxelHashSet.AddUnique(arena, hash, voxel);
                    }
                }
            }
        }
    }
}

static void CheckVoxelOccupancy(Arena *arena, ScenePrimitives *scene,
                                SimpleHashSet<Vec3i> &voxelHashSet,
                                StaticArray<Voxel> &outVoxels, StaticArray<Vec3i> &extraVoxels,
                                f32 voxelSize)
{
    ScratchArena scratch(&arena, 1);
    const u32 numRays = 16;

    struct SGGX
    {
        float nxx;
        float nxy;
        float nxz;

        float nyy;
        float nyz;

        float nzz;

        SGGX()
        {
            nxx = 0.f;
            nxy = 0.f;
            nxz = 0.f;
            nyy = 0.f;
            nyz = 0.f;
            nzz = 0.f;
        }

        void Add(const Vec3f &n)
        {
            nxx += n.x * n.x;
            nxy += n.x * n.y;
            nxz += n.x * n.z;

            nyy += n.y * n.y;
            nyz += n.y * n.z;

            nzz += n.z * n.z;
        }

        Vec3f Fit(u32 hitCount, Vec2f &alpha)
        {
            f32 hits = f32(hitCount);

            nxx /= hits;
            nxy /= hits;
            nxz /= hits;

            nyy /= hits;
            nyz /= hits;

            nzz /= hits;

            float m[9] = {
                nxx, nxy, nxz, nxy, nyy, nyz, nxz, nyz, nzz,
            };
            float eigenVectors[9] = {};

            float eigenValues[3] = {};
            Eigen::jacobiEigenSolver(m, eigenValues, eigenVectors, 1e-8f);

            float scale[3];
            for (int axis = 0; axis < 3; axis++)
            {
                scale[axis] = Sqrt(Abs(eigenValues[axis]));
            }

            f32 maxRatio = 0.f;
            int maxIndex = 0;

            int lut[3] = {1, 2, 0};
            for (int axis = 0; axis < 3; axis++)
            {
                f32 ratio =
                    Min(scale[axis], scale[lut[axis]]) / Max(scale[axis], scale[lut[axis]]);
                if (ratio > maxRatio)
                {
                    maxRatio = ratio;
                    maxIndex = axis;
                }
            }

            int chosenAxis = lut[lut[maxIndex]];
            Vec3f result;
            for (int axis = 0; axis < 3; axis++)
            {
                result[axis] = eigenVectors[3 * axis + chosenAxis];
            }
            alpha[0] = 0.5f * (scale[maxIndex] + scale[lut[maxIndex]]);
            alpha[1] = scale[chosenAxis];

            return result;
        }

        SGGXCompact Compact()
        {
            SGGXCompact compact;
            f32 sigmaX = Sqrt(nxx);
            f32 sigmaY = Sqrt(nyy);
            f32 sigmaZ = Sqrt(nzz);

            f32 rxy = nxy / Sqrt(nxx * nyy);
            f32 rxz = nxz / Sqrt(nxx * nzz);
            f32 ryz = nyz / Sqrt(nyy * nzz);

            compact.packed[0] = u8(Clamp(u32(sigmaX * 255.f + 0.5f), 0u, 255u));
            compact.packed[1] = u8(Clamp(u32(sigmaY * 255.f + 0.5f), 0u, 255u));
            compact.packed[2] = u8(Clamp(u32(sigmaZ * 255.f + 0.5f), 0u, 255u));

            compact.packed[3] = u8(Clamp(u32((rxy + 1.f) * 0.5f * 255.f + 0.5f), 0u, 255u));
            compact.packed[4] = u8(Clamp(u32((rxz + 1.f) * 0.5f * 255.f + 0.5f), 0u, 255u));
            compact.packed[5] = u8(Clamp(u32((ryz + 1.f) * 0.5f * 255.f + 0.5f), 0u, 255u));

            return compact;
        }
    };

    auto GenerateRay = [&](u32 sampleIndex, u32 &seed, float voxelSize, Vec3f &outOrigin,
                           Vec3f &outDir, float &outTMax) {
        for (;;)
        {
            Vec3f dir = SampleUniformSphere(GetSobolSample(sampleIndex, seed));

            Vec3f tx, ty;
            CoordinateSystem(dir, &tx, &ty);

            Vec2f disk = SampleUniformDiskConcentric(GetSobolSample(sampleIndex, seed));
            disk *= voxelSize * Sqrt(3) * 0.5f;

            Vec3f invDir = 1.f / dir;

            Vec3f origin = tx * disk.x + ty * disk.y;
            Vec3f extent = Abs(invDir) * (0.5f * voxelSize);
            Vec3f center = -origin * invDir;

            Vec3f maxP = center + extent;
            Vec3f minP = center - extent;

            float minT = Max(minP.x, Max(minP.y, minP.z));
            float maxT = Min(maxP.x, Min(maxP.y, maxP.z));

            if (minT < maxT)
            {
                outOrigin = origin + dir * minT;
                outDir    = dir;
                outTMax   = maxT - minT;
                break;
            }
        }
    };

    struct Handle
    {
        u32 sortKey;
        u32 index;
    };

    StaticArray<Voxel> voxels(arena, voxelHashSet.totalCount);
    StaticArray<Vec3i> failedVoxels(arena, voxelHashSet.totalCount);
    StaticArray<Handle> handles(scratch.temp.arena, voxelHashSet.totalCount);
    HashIndex hitVoxelHash(scratch.temp.arena, NextPowerOfTwo(voxelHashSet.totalCount),
                           NextPowerOfTwo(voxelHashSet.totalCount));

    f32 totalCoverage = 0.f;
    for (u32 slotIndex = 0; slotIndex < voxelHashSet.numSlots; slotIndex++)
    {
        auto *node = &voxelHashSet.nodes[slotIndex];
        while (node->next)
        {
            SGGX sggx;
            Vec3i voxel       = node->value;
            Vec3f voxelCenter = (Vec3f(voxel) + 0.5f) * voxelSize;
            u32 hitCount      = 0;
            u32 geomID        = 0;
            Vec3f n;

            for (u32 sample = 0; sample < numRays; sample++)
            {
                u32 seed        = 0;
                u32 sampleIndex = ReverseBits32((MortonCode3(voxel.x & 1023) |
                                                 (MortonCode3(voxel.y & 1023) << 1) |
                                                 (MortonCode3(voxel.z & 1023) << 2)) *
                                                    numRays +
                                                sample);
                SurfaceInteraction si;
                Ray2 ray;

                GenerateRay(sampleIndex, seed, voxelSize, ray.o, ray.d, ray.tFar);
                ray.o += voxelCenter;

                bool intersect = scene->Intersect(ray, si);

                if (intersect)
                {
                    n      = si.n;
                    geomID = si.geomID;
                    hitCount++;

                    sggx.Add(n);
                }
            }
            if (hitCount)
            {
                union Float
                {
                    f32 f;
                    u32 u;
                };
                f32 coverage = (f32)hitCount / numRays;
                totalCoverage += coverage;

                Float fl;
                fl.f = coverage;

                Vec2f alpha;
                Voxel v;
                v.loc    = voxel;
                v.normal = sggx.Fit(hitCount, alpha);

                v.coverage = coverage;
                // v.coverage = alpha.x > alpha.y ? 1.f - 0.5f * alpha.y / alpha.x
                //                                : 0.5f * alpha.x / alpha.y;
                v.sggx   = sggx.Compact();
                v.geomID = geomID;

                voxels.Push(v);

                u32 hash = Hash(voxel);

                hitVoxelHash.AddInHash(hash, voxels.Length() - 1);
            }
            else
            {
                failedVoxels.Push(voxel);
            }
            node = node->next;
        }
    }

    outVoxels   = voxels;
    extraVoxels = failedVoxels;

#if 0
    Assert(voxels.Length());
    SortHandles(handles.data, handles.Length());

    u32 handleIndex = 0;
    u32 numVoxels   = voxels.Length();
    while ((f32)numVoxels > totalCoverage)
    {
        Voxel &voxel = voxels[handles[handleIndex].index];

        FixedArray<u32, 27> neighbors;
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    Vec3i neighborLoc = voxel.loc + Vec3i(x, y, z);
                    u32 hash          = Hash(neighborLoc);
                    for (int hashIndex = hitVoxelHash.FirstInHash(hash); hashIndex != -1;
                         hashIndex     = hitVoxelHash.NextInHash(hashIndex))
                    {
                        if (voxels[hashIndex].loc == neighborLoc)
                        {
                            neighbors.Push((u32)hashIndex);
                            break;
                        }
                    }
                }
            }
        }
        f32 coverage = voxel.coverage / neighbors.Length();
        for (u32 neighbor : neighbors)
        {
            Voxel &neighborVoxel   = voxels[neighbor];
            f32 newCoverage        = 1.f - (1.f - neighborVoxel.coverage) * (1.f - coverage);
            neighborVoxel.coverage = newCoverage;
        }

        numVoxels--;
    }
#endif
}

#if 0
void CreateInstanceHierarchy()
{
    AffineSpace *transforms;
    Cluster lastLevelClusters;

    // basically, you just take all of the cluster groups, transform all the clusters
    // in all of the cluster groups, and then

    struct Handle
    {
        f32 sortKey;
        u32 index;
    };

    ScratchArena scratch;
    ScenePrimitives *scene;

    Instance *instances = (Instance *)scene->primitives;

    KDTreeNode kdTreeRootNodes;

    // Flatten meshes into an array
    struct TriangleKDTree
    {
        KDTreeNode *root;
        PrimRef *refs;
    };

    Arena **arenas = GetArenaArray(scratch.temp.arena);

    StaticArray<TriangleKDTree> trees(scratch.temp.arena, scene->numChildScenes,
                                      scene->numChildScenes);

    // Build KD tree over triangles in object space
    ParallelFor(0, scene->numChildScenes, 1, [&](u32 jobID, u32 start, u32 count) {
        for (u32 sceneIndex = start; sceneIndex < start + count; sceneIndex++)
        {
            ScenePrimitives *childScene = scene->childScenes[sceneIndex];
            RecordAOSSplits record;
            PrimRef *refs = ParallelGenerateMeshRefs<GeometryType::TriangleMesh>(
                scratch.temp.arena, (Mesh *)childScene->primitives, childScene->numPrimitives,
                record, false);
            PrimRef *doubleBufferRefs =
                PushArrayNoZero(scratch.temp.arena, childScene->numPrimitives);
            KDTreeNode *root = BuildKDTree(arenas, refs, 0, childScene->numPrimitives);

            TriangleKDTree tree;
            tree.root = root;
            tree.refs = refs;

            trees[sceneIndex] = tree;
        }
    });

    u32 numInstances;
    PrimRef *primRefs = PushArrayNoZero(scratch.temp.arena, PrimRef, numInstances);

    // - form kd tree over instances
    // - iterate over every instance through the kd tree, find all oher instances with bounding
    // boxes that intersect OR that are within a certain distance
    // - build a triangle kd tree once per instanced
    // - if the instance pair has not been considered yet, find the triangle distances
    // between the instances, and if a virtual edge is detected

    ParallelFor(0, numInstances, 32, [&](u32 jobID, u32 start, u32 count) {
        for (u32 instanceIndex = start; instanceIndex < start + count; instanceIndex++)
        {
            Instance &instance     = instances[instanceIndex];
            AffineSpace &transform = transforms[instance.transformIndex];

            PrimRef ref;
            ref.primID = instanceIndex;
            for (u32 axis = 0; axis < 3; axis++)
            {
                ref.min[axis] = bb.minP[axis];
                ref.max[axis] = bb.maxP[axis];
            }
            primRefs[instanceIndex] = ref;
        }
    });
    PrimRef *doubleBufferRefs = PushArrayNoZero(scratch.temp.arena, PrimRef, numInstances);

    KDTreeNode *rootInstanceNode =
        BuildKDTree(arenas, primRefs, doubleBufferRefs, 0, numInstances);

    u32 **instanceNeighbors     = PushArrayNoZero(scratch.temp.arena, u32 *, numInstances);
    u32 *instanceNeighborCounts = PushArrayNoZero(scratch.temp.arena, u32, numInstances);

    ParallelFor(0, numInstances, 32, [&](u32 jobID, u32 start, u32 count) {
        Arena *arena = arenas[GetThreadIndex()];
        ScratchArena scratch;

        for (u32 instanceIndex = start; instanceIndex < start + count; instanceIndex++)
        {
            Array<u32> neighbors(scratch.temp.arena, 8);
            PrimRef &ref = primRefs[instanceIndex];
            Bounds bounds;
            for (u32 axis = 0; axis < 3; axis++)
            {
                bounds.minP[axis] = ref.min[axis];
                bounds.maxP[axis] = ref.max[axis];
            }
            u32 instanceIndex      = ref.primID;
            Instance &instance     = instances[instanceIndex];
            AffineSpace &transform = transforms[instance.transformIndex];

            TraverseKDTree(primRefs, rootNode, bounds, [&](PrimRef &otherRef) {
                Lane8F32 intersection = Min(otherRef.Load(), ref.Load());
                if (All(-Extract4<0>(intersection) <= Extract4<1>(intersection)))
                {
                    neighbors.Push(otherRef.primID);
                }
            });

            u32 *outNeighbors = PushArrayNoZero(arena, u32, neighbors.Length());
            MemoryCopy(outNeighbors, neighbors.data, sizeof(u32) * neighbors.Length());
            instanceNeighbors[instanceIndex] = outNeighbors;
            instanceNeighborCounts[instanceIndex] neighbors.Length();
        }
    });

    // DFS to find islands
    BitVector visited(scratch.temp.arena, numInstances);

    u32 *islandOffsets     = PushArrayNoZero(scratch.temp.arena, u32, numInstances + 1);
    u32 *islandOffsets1    = &islandOffsets[1];
    u32 *islandData        = PushArrayNoZero(scratch.temp.arena, u32, numInstances);
    u32 currentIsland      = 0;
    u32 currentIslandCount = 0;

    for (u32 instanceIndex = 0; instanceIndex < numInstances; instanceIndex++)
    {
        if (visited.GetBit(instanceIndex)) continue;

        FixedArray<u32, 128> stack;
        stack.Push(instanceIndex);
        visited.SetBit(instanceIndex);
        while (stack.Length())
        {
            u32 instance = stack.Pop();
            for (u32 neighborIndex = 0; neighborIndex < instanceNeighborCounts[instance];
                 neighborIndex++)
            {
                u32 neighbor = instanceNeighbors[instance][neighborIndex];
                if (!visited.GetBit(neighbor))
                {
                    visited.SetBit(neighbor);
                    stack.Push(neighbor);
                    islandData[currentIslandCount++] = neighbor;
                }
            }
        }
        virtualEdgesFound.SetBit(instanceIndex);

        islandOffsets1[currentIsland] = currentIslandCount;
        currentIsland++;
    }

    BitVector virtualEdgesFound(scratch.temp.arena, numInstances);
    // Merge instances in the same island and simplify
    for (u32 island = 0; island < currentIsland; island++)
    {
        for (u32 islandMemberIndex = islandOffsets[island];
             islandMemberIndex < islandOffsets[island + 1]; islandMemberIndex++)
        {
            u32 instanceIndex = islandData[islandMemberIndex];
            u32 *neighbors    = instanceNeighbors[instanceIndex];
            u32 neighborCount = instanceNeighborCounts[instanceIndex];
            for (u32 neighborIndex = 0; neighborIndex < neighborCount; neighborIndex++)
            {
                u32 neighborInstance = neighbors[neighborIndex];
                if (!virtualEdgesFound.GetBit(neighborInstance))
                {
                    KDTreeNode *root = rootNodes[];
                    TraverseKDTree(PrimRef * refs, KDTreeNode * rootNode, Bounds bounds,
                                   LeafFunc && func);
                }
            }
            virtualEdgesFound.SetBit(instanceIndex);
        }
    }

    CreateClusters(Mesh * meshes, u32 numMeshes, StaticArray<u32> & materialIndices,
                   string filename);
}
#endif

} // namespace rt
