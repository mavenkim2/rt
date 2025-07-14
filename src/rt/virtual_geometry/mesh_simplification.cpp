#include "../hash.h"
#include "../memory.h"
#include "../thread_context.h"
#include "../dgfs.h"
#include "../radix_sort.h"
#include <atomic>
#include "../mesh.h"
#include <cstring>
#include "../scene_load.h"
#include "../bvh/bvh_types.h"
#include "../bvh/bvh_build.h"
#include "../bvh/bvh_aos.h"
#include "../parallel.h"
#include "mesh_simplification.h"
#include "../shader_interop/as_shaderinterop.h"
#include "../../third_party/METIS/include/metis.h"

namespace rt
{
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

    if (adjCorners.Length() >= 1000)
    {
        Print("num corners: %u, num remaining tris: %u, num tris: %u\n", adjCorners.Length(),
              remainingNumTriangles, numIndices / 3);
        for (int i = 0; i < adjCorners.Length(); i++)
        {
            Print("%u %u\n", i, adjCorners[i]);
            u32 index = indices[adjCorners[i]];
            Print("%u\n", index);
            Print("%f %f %f\n", GetPosition(index).x, GetPosition(index).y,
                  GetPosition(index).z);
        }
        Assert(0);
    }

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
struct KDTreeNode
{
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
    u32 left;
    u32 right;
};

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

void MeshSimplifier::BuildKDTree(Bounds bounds, u32 start, u32 count)
{
    struct Handle
    {
        f32 sortKey;
        u32 index;
    };

    // Build KDTree
    ScratchArena scratch;
    Handle *handles = PushArrayNoZero(scratch.temp.arnea, Handle, count);

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

    for (int tri = start; tri < start + count; tri++)
    {
        Vec3f p0 = GetPosition(indices[3 * tri + 0]);
        Vec3f p1 = GetPosition(indices[3 * tri + 1]);
        Vec3f p2 = GetPosition(indices[3 * tri + 2]);

        Vec3f center               = (p0 + p1 + p2) / 3.f;
        handles[tri - start].key   = center[bestAxis];
        handles[tri - start].index = tri;
    }
    SortHandles(handles, count);

    u32 leftCount  = count / 2;
    u32 rightCount = count - leftCount;

    Bounds leftBounds;
    Bounds rightBounds;
    for (int i = 0; i < leftCount; i++)
    {
        for (int vertIndex = 0; vertIndex < 3; vertIndex++)
        {
            leftBounds.Extend(GetPosition(indices[3 * i + vertIndex]);
        }
    }
    for (int i = leftCount; i < count; i++)
    {
        for (int vertIndex = 0; vertIndex < 3; vertIndex++)
        {
            rightBounds.Extend(GetPosition(indices[3 * i + vertIndex]);
        }
    }

    BuildKDTree(leftBounds, start, leftCount);
    BuildKDTree(rightBounds, start + leftCount, rightCount);
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
        // ErrorExit(LengthSquared(sphere0.xyz - newCenter) <=
        //               Sqr(newRadius + tolerance - sphere0.w),
        //           "%f %f %f %f %f %f %f %f\n", sphere0.x, sphere0.y, sphere0.z, sphere0.w,
        //           newCenter.x, newCenter.y, newCenter.z, newRadius);
        // ErrorExit(LengthSquared(sphere1.xyz - newCenter) <=
        //               Sqr(newRadius + tolerance - sphere1.w),
        //           "%f %f %f %f %f %f %f %f\n", sphere1.x, sphere1.y, sphere1.z, sphere1.w,
        //           newCenter.x, newCenter.y, newCenter.z, newRadius);

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
    // u32 clusterOffset;
    // u32 numClusters;

    Vec4f lodBounds;
    PrimRef *primRefs;
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

    // Debug
    u32 numVertices;
    u32 numIndices;

    u32 mipLevel;

    bool isLeaf;
};

struct Cluster
{
    RecordAOSSplits record;
    Vec4f lodBounds;
    u32 mipLevel;

    u32 groupIndex;
    u32 childGroupIndex;

    u32 headerIndex;

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

static const u32 minGroupSize = 8;
static const u32 maxGroupSize = 32;

struct Range
{
    u32 begin;
    u32 end;
};

template <typename T>
struct ArrayView
{
    T *data;
    u32 num;

    ArrayView(Array<T> &array, u32 offset, u32 num) : num(num)
    {
        Assert(offset + num <= array.Length());
        data = array.data + offset;
    }
    ArrayView(Array<T> &array) : num(array.Length()) { data = array.data; }
    ArrayView(StaticArray<T> &array, u32 offset, u32 num) : num(num)
    {
        Assert(offset + num <= array.Length());
        data = array.data + offset;
    }
    ArrayView(StaticArray<T> &array) : num(array.Length()) { data = array.data; }
    ArrayView(ArrayView<T> &view, u32 offset, u32 num) : num(num)
    {
        Assert(offset + num <= view.num);
        data = view.data + offset;
    }
    ArrayView(T *data, u32 num) : data(data), num(num) {}
    T &operator[](u32 index)
    {
        Assert(index < num);
        return data[index];
    }
    const T &operator[](u32 index) const
    {
        Assert(index < num);
        return data[index];
    }
    u32 Length() const { return num; }

    void Copy(StaticArray<T> &array)
    {
        Assert(array.capacity >= num);
        MemoryCopy(array.data, data, sizeof(T) * num);
        array.size() = num;
    }
};

void PartitionGraph(ArrayView<int> clusterIndices, ArrayView<idx_t> clusterOffsets,
                    ArrayView<idx_t> clusterData, ArrayView<idx_t> clusterWeights,
                    ArrayView<int> newClusterIndices, ArrayView<idx_t> newClusterOffsets,
                    ArrayView<idx_t> newClusterData, ArrayView<idx_t> newClusterWeights,
                    int numClusters, Range &left, Range &right, u32 &newAdjOffset,
                    u32 numAdjacency, Vec2u &newNumAdjacency)

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
    options[METIS_OPTION_UFACTOR] = 200;

    int result =
        METIS_PartGraphRecursive(&numClusters, &numConstraints, tempClusterOffsets.data,
                                 clusterData.data, NULL, NULL, clusterWeights.data, &numParts,
                                 partitionWeights, NULL, options, &edgesCut, partitionIDs);

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

void RecursivePartitionGraph(ArrayView<int> clusterIndices, ArrayView<idx_t> clusterOffsets,
                             ArrayView<idx_t> clusterData, ArrayView<idx_t> clusterWeights,
                             ArrayView<int> newClusterIndices,
                             ArrayView<idx_t> newClusterOffsets,
                             ArrayView<idx_t> newClusterData,
                             ArrayView<idx_t> newClusterWeights, int numClusters, int numSwaps,
                             int globalClusterOffset, StaticArray<Range> &ranges,
                             std::atomic<int> &numPartitions, u32 numAdjacency)
{
    u32 newAdjOffset;
    Vec2u newNumAdjacency;
    Range left, right;

    PartitionGraph(clusterIndices, clusterOffsets, clusterData, clusterWeights,
                   newClusterIndices, newClusterOffsets, newClusterData, newClusterWeights,
                   numClusters, left, right, newAdjOffset, numAdjacency, newNumAdjacency);

    u32 numLeft  = left.end - left.begin;
    u32 numRight = right.end - right.begin;
    if (numLeft <= maxGroupSize && numRight <= maxGroupSize)
    {
        int rangeIndex = numPartitions.fetch_add(2, std::memory_order_relaxed);

        left.begin += globalClusterOffset;
        left.end += globalClusterOffset;
        right.begin += globalClusterOffset;
        right.end += globalClusterOffset;

        ranges[rangeIndex]     = left;
        ranges[rangeIndex + 1] = right;

        if (numSwaps & 1)
        {
            MemoryCopy(clusterIndices.data, newClusterIndices.data, sizeof(int) * numClusters);
            // MemoryCopy(clusterOffsets, newClusterOffsets, sizeof(idx_t) * extent);
            // MemoryCopy(clusterData, newClusterData, sizeof(idx_t) * extent);
            // MemoryCopy(clusterWeights, newClusterWeights, sizeof(idx_t) * extent);
        }

        return;
    }

    auto Recurse = [&](int jobID) {
        u32 clusterOffset  = jobID == 0 ? 0 : right.begin;
        u32 newNumClusters = jobID == 0 ? numLeft : numRight;
        u32 adjOffset      = jobID == 0 ? 0 : newAdjOffset;
        u32 numAdjacency   = newNumAdjacency[jobID];

        RecursivePartitionGraph(
            ArrayView<idx_t>(newClusterIndices, clusterOffset, newNumClusters),
            ArrayView<idx_t>(newClusterOffsets, clusterOffset, newNumClusters),
            ArrayView<idx_t>(newClusterData, adjOffset, numAdjacency),
            ArrayView<idx_t>(newClusterWeights, adjOffset, numAdjacency),
            ArrayView<idx_t>(clusterIndices, clusterOffset, newNumClusters),
            ArrayView<idx_t>(clusterOffsets, clusterOffset, newNumClusters),
            ArrayView<idx_t>(clusterData, adjOffset, numAdjacency),
            ArrayView<idx_t>(clusterWeights, adjOffset, numAdjacency), newNumClusters,
            numSwaps + 1, globalClusterOffset + clusterOffset, ranges, numPartitions,
            numAdjacency);
    };

    // TODO: for whatever reason multithreading METIS causes inscrutable errors. fix this if
    // speedup needed
    //
    // if (numClusters > 256)
    // {
    //     scheduler.ScheduleAndWait(2, 1, Recurse);
    // }
    // else
    // {
    Recurse(0);
    Recurse(1);
    // }
}

struct GraphPartitionResult
{
    StaticArray<Range> ranges;
    StaticArray<int> clusterIndices;
};

GraphPartitionResult RecursivePartitionGraph(Arena *arena, idx_t *clusterOffsets,
                                             idx_t *clusterData, idx_t *clusterWeights,
                                             int numClusters, u32 dataSize)
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

    StaticArray<Range> ranges(arena, maxNumPartitions, maxNumPartitions);

    RecursivePartitionGraph(
        ArrayView<int>(clusterIndices), ArrayView<idx_t>(clusterOffsets, (u32)numClusters),
        ArrayView<idx_t>(clusterData, dataSize), ArrayView<idx_t>(clusterWeights, dataSize),
        ArrayView<int>(newClusterIndices), ArrayView<idx_t>(newClusterOffsets),
        ArrayView<idx_t>(newClusterData), ArrayView<idx_t>(newClusterWeights), numClusters, 0,
        0, ranges, numPartitions, dataSize);

    ranges.size() = numPartitions.load();

    GraphPartitionResult result;
    result.ranges         = ranges;
    result.clusterIndices = clusterIndices;

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

    HeuristicObjectBinning<PrimRef> heuristic(primRefs, Log2Int(4));

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
        Vec4f *spheres = PushArrayNoZero(scratch.temp.arena, Vec4f, childNode.numChildren);
        Bounds bounds;
        for (int j = 0; j < childNode.numChildren; j++)
        {
            bounds.Extend(childNode.bounds[j]);
            spheres[j]     = childNode.lodBounds[j];
            maxParentError = Max(maxParentError, childNode.maxParentError[j]);
        }

        node.bounds[i]         = bounds;
        node.lodBounds[i]      = ConstructSphereFromSpheres(spheres, childNode.numChildren);
        node.maxParentError[i] = maxParentError;
    }

    numNodes++;
    return node;
}

static_assert((sizeof(PackedDenseGeometryHeader) + 4) % 16 == 0, "bad header size");

// TODO: the builder is no longer deterministic after adding edge quadrics?
// also prevent the builder from solving to a garbage position
void CreateClusters(Mesh *meshes, u32 numMeshes, StaticArray<u32> &materialIndices,
                    string filename)
{
    const u32 numAttributes = 0;

    auto GetVertexData = [numAttributes](f32 *ptr, u32 index) {
        return ptr + (3 + numAttributes) * index;
    };

    auto GetPosition = [numAttributes](f32 *ptr, u32 index) -> Vec3f & {
        return *(Vec3f *)(ptr + (3 + numAttributes) * index);
    };

    ScratchArena scratch;

    RecordAOSSplits record;

    StaticArray<Vec2u> meshVertexOffsets(scratch.temp.arena, numMeshes);
    u32 totalNumVertices = 0;
    u32 totalNumIndices  = 0;
    for (int i = 0; i < numMeshes; i++)
    {
        Vec2u offsets(totalNumVertices, totalNumIndices);
        meshVertexOffsets.Push(offsets);
        totalNumVertices += meshes[i].numVertices;
        totalNumIndices += meshes[i].numIndices;
    }

    u32 totalClustersEstimate = ((totalNumIndices / 3) >> (MAX_CLUSTER_TRIANGLES_BIT - 1)) * 3;
    u32 totalGroupsEstimate   = (totalClustersEstimate + minGroupSize - 1) / minGroupSize;

    f32 *vertexData =
        PushArrayNoZero(scratch.temp.arena, f32, (3 + numAttributes) * totalNumVertices);
    u32 *indexData = PushArrayNoZero(scratch.temp.arena, u32, totalNumIndices);

    ParallelFor(0, numMeshes, 1, [&](int jobID, int start, int count) {
        for (int meshIndex = start; meshIndex < start + count; meshIndex++)
        {
            Mesh &mesh = meshes[meshIndex];
            // TODO: attributes
            u32 vertexOffset = meshVertexOffsets[meshIndex].x;
            u32 indexOffset  = meshVertexOffsets[meshIndex].y;
            MemoryCopy(vertexData + (3 + numAttributes) * vertexOffset, mesh.p,
                       sizeof(Vec3f) * mesh.numVertices);

            for (int indexIndex = 0; indexIndex < mesh.numIndices; indexIndex++)
            {
                indexData[indexOffset + indexIndex] = mesh.indices[indexIndex] + vertexOffset;
            }
        }
    });

    Mesh combinedMesh        = {};
    combinedMesh.p           = (Vec3f *)vertexData;
    combinedMesh.indices     = indexData;
    combinedMesh.numVertices = totalNumVertices;
    combinedMesh.numIndices  = totalNumIndices;
    combinedMesh.numFaces    = totalNumIndices / 3;

    PrimRef *primRefs = ParallelGenerateMeshRefs<GeometryType::TriangleMesh>(
        scratch.temp.arena, &combinedMesh, 1, record, false);

    u32 currentMesh = 0;
    for (int primRefIndex = 0; primRefIndex < record.Count(); primRefIndex++)
    {
        u32 primID = primRefs[primRefIndex].primID;
        u32 limit  = currentMesh == numMeshes - 1 ? totalNumIndices
                                                  : meshVertexOffsets[currentMesh + 1].y;
        if (primID >= limit)
        {
            currentMesh++;
        }
        primRefs[primRefIndex].geomID = currentMesh;
    }

    ClusterBuilder clusterBuilder(scratch.temp.arena, primRefs);
    clusterBuilder.BuildClusters(record, true);

    Array<Cluster> clusters(scratch.temp.arena, totalClustersEstimate);
    Array<ClusterGroup> clusterGroups(scratch.temp.arena, totalGroupsEstimate);

    int numClusters = 0;
    for (auto &list : clusterBuilder.threadClusters)
    {
        numClusters += list.l.Length();
    }

    RecordAOSSplits *clusterRecords =
        PushArrayNoZero(scratch.temp.arena, RecordAOSSplits, numClusters);

    u32 clusterOffset = 0;
    for (auto &list : clusterBuilder.threadClusters)
    {
        list.l.Flatten(clusterRecords + clusterOffset);
        clusterOffset += list.l.Length();
    }

    ClusterGroup clusterGroup = {};
    clusterGroup.primRefs     = primRefs;
    clusterGroup.vertexData   = vertexData;
    clusterGroup.indices      = indexData;
    clusterGroup.isLeaf       = true;

    clusterGroups.Push(clusterGroup);

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

    Assert(totalClustersEstimate > clusterOffset);
    for (int i = 0; i < clusterOffset; i++)
    {
        ScratchArena clusterScratch;
        Cluster cluster     = {};
        cluster.record      = sortedClusterRecords[i];
        cluster.mipLevel    = 0;
        cluster.headerIndex = handles[i].index;

        // Construct the lod bounds
        Vec3f *points =
            PushArrayNoZero(clusterScratch.temp.arena, Vec3f, 3 * cluster.record.Count());
        u32 start = cluster.record.Start();
        for (u32 j = 0; j < cluster.record.count; j++)
        {
            PrimRef &ref     = primRefs[start + j];
            u32 vertexIndex0 = indexData[3 * ref.primID + 0];
            u32 vertexIndex1 = indexData[3 * ref.primID + 1];
            u32 vertexIndex2 = indexData[3 * ref.primID + 2];

            points[3 * j + 0] = GetPosition(vertexData, vertexIndex0);
            points[3 * j + 1] = GetPosition(vertexData, vertexIndex1);
            points[3 * j + 2] = GetPosition(vertexData, vertexIndex2);
        }
        cluster.lodBounds = ConstructSphereFromPoints(points, 3 * cluster.record.Count());
        clusters.Push(cluster);
    }

    // Create clusters
    Arena **arenas = GetArenaArray(scratch.temp.arena);
    StaticArray<DenseGeometryBuildData> buildDatas(scratch.temp.arena, OS_NumProcessors());
    for (int i = 0; i < buildDatas.capacity; i++)
    {
        buildDatas.Push(DenseGeometryBuildData());
    }

    Bounds bounds;
    clusterBuilder.CreateDGFs(materialIndices, &buildDatas[0], &combinedMesh, 1, bounds);

    // u32 debugNumLevelClusters[20] = {
    //     177245, 96085, 52377, 28685, 15709, 8551, 4698, 2489, 1355, 740,
    //     405,    218,   121,   71,    38,    21,   12,   6,    4,    1,
    // };

    {
        // 1. Split triangles into clusters (mesh remains)

        // 2. Group clusters based on how many shared edges they have (METIS) (mesh remains)
        //      - also have edges between clusters that are close enough
        // 3. Simplify the cluster group (effectively creates num groups different meshes)
        // 4. Split simplified group into clusters

        u32 depth = 0;
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
            if (levelClusters.Length() < 2) break;

            u32 hashSize = NextPowerOfTwo(3 * MAX_CLUSTER_TRIANGLES * levelClusters.Length());
            HashIndex edgeHash(scratch.temp.arena, hashSize, hashSize);

            // ROADMAP
            // 2. verify the grouping / simplifying / clustering somehow
            //      - still need to add edge locking + attributes
            // 3. build clas over each lod level of clusters
            // 4. write dag/hierarchy to disk
            // 5. run time hierarchy selection
            // 6. streaming
            // 7. impostors/ptex baking?
            // vertex references to deduplicate vertices used at different lod levels

            // Calculate the number of edges per group
            u32 edgeOffset = 0;
            StaticArray<u32> clusterEdgeOffsets(scratch.temp.arena, levelClusters.Length());
            for (int clusterIndex = 0; clusterIndex < levelClusters.Length(); clusterIndex++)
            {
                Cluster &cluster = levelClusters[clusterIndex];
                u32 numEdges     = 3 * cluster.record.Count();
                clusterEdgeOffsets.Push(edgeOffset);
                edgeOffset += numEdges;
            }
            StaticArray<Edge> edges(scratch.temp.arena, edgeOffset, edgeOffset);

            u32 numClusters = levelClusters.Length();
            ParallelFor(0, numClusters, 32, 32, [&](int jobID, int start, int count) {
                for (int clusterIndex = start; clusterIndex < start + count; clusterIndex++)
                {
                    Cluster &cluster           = levelClusters[clusterIndex];
                    ClusterGroup &clusterGroup = clusterGroups[cluster.childGroupIndex];

                    RecordAOSSplits &record = cluster.record;
                    int triangleStart       = record.start;
                    int triangleCount       = record.count;

                    u32 edgeOffset = clusterEdgeOffsets[clusterIndex];
                    for (int triangle = triangleStart;
                         triangle < triangleStart + triangleCount; triangle++)
                    {
                        PrimRef &primRef = clusterGroup.primRefs[triangle];

                        for (int edgeIndexIndex = 0; edgeIndexIndex < 3; edgeIndexIndex++)
                        {
                            u32 index0 =
                                clusterGroup.indices[3 * primRef.primID + edgeIndexIndex];
                            u32 index1 =
                                clusterGroup
                                    .indices[3 * primRef.primID + (edgeIndexIndex + 1) % 3];

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

            StaticArray<u32> clusterNeighborCounts(scratch.temp.arena, numClusters,
                                                   numClusters);

            struct ClusterData
            {
                int *neighbors;
                int *weights;
                int *externalEdges;
                int numExternalEdges;
                int numNeighbors;
            };

            StaticArray<ClusterData> clusterDatas(scratch.temp.arena, numClusters,
                                                  numClusters);

            u32 numAttributes = 0;
            u32 vertexDataLen = sizeof(f32) * (3 + numAttributes);

            ParallelFor(0, numClusters, 32, 32, [&](int jobID, int start, int count) {
                for (int clusterIndex = start; clusterIndex < start + count; clusterIndex++)
                {
                    ScratchArena threadScratch;

                    Cluster &cluster        = levelClusters[clusterIndex];
                    RecordAOSSplits &record = cluster.record;
                    int triangleStart       = record.start;
                    int triangleCount       = record.count;

                    struct Handle
                    {
                        int sortKey;
                    };

                    Array<Handle> neighbors(threadScratch.temp.arena, 3 * triangleCount);
                    Array<int> externalEdges(threadScratch.temp.arena, 3 * triangleCount);

                    u32 edgeOffset = clusterEdgeOffsets[clusterIndex];
                    for (int edgeIndex = edgeOffset;
                         edgeIndex < edgeOffset + 3 * triangleCount; edgeIndex++)
                    {
                        Edge &edge = edges[edgeIndex];
                        int hash   = HashEdge(edge.p0, edge.p1);

                        for (int otherEdgeIndex = edgeHash.FirstInHash(hash);
                             otherEdgeIndex != -1;
                             otherEdgeIndex = edgeHash.NextInHash(otherEdgeIndex))
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
                        clusterDatas[clusterIndex] = {};
                        continue;
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
                    Arena *arena             = arenas[GetThreadIndex()];
                    ClusterData &clusterData = clusterDatas[clusterIndex];
                    clusterData.neighbors = PushArrayNoZero(arena, int, compactedNumNeighbors);
                    clusterData.weights   = PushArrayNoZero(arena, int, compactedNumNeighbors);
                    clusterData.externalEdges    = PushArrayNoZero(arena, int, numNeighbors);
                    clusterData.numNeighbors     = compactedNumNeighbors;
                    clusterData.numExternalEdges = numNeighbors;

                    MemoryCopy(clusterData.neighbors, neighbors.data,
                               sizeof(int) * compactedNumNeighbors);
                    MemoryCopy(clusterData.weights, weights,
                               sizeof(int) * compactedNumNeighbors);
                    MemoryCopy(clusterData.externalEdges, externalEdges.data,
                               sizeof(int) * numNeighbors);
                }
            });

            GraphPartitionResult partitionResult;
            if (numClusters <= maxGroupSize)
            {
                partitionResult.ranges = StaticArray<Range>(scratch.temp.arena, 1);
                partitionResult.clusterIndices =
                    StaticArray<int>(scratch.temp.arena, numClusters);
                partitionResult.ranges.Push(Range{0, numClusters});

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
                    ClusterData &data             = clusterDatas[clusterIndex];
                    u32 num                       = data.numNeighbors;
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
                        const ClusterData &cluster = clusterDatas[clusterIndex];
                        i32 offset                 = clusterOffsets1[clusterIndex];
                        MemoryCopy(clusterData + offset, cluster.neighbors,
                                   sizeof(int) * cluster.numNeighbors);
                        MemoryCopy(clusterWeights + offset, cluster.weights,
                                   sizeof(int) * cluster.numNeighbors);

                        clusterOffsets1[clusterIndex] += cluster.numNeighbors;
                    }
                });

                // Recursively partition the clusters into two groups until each group
                // satisfies constraints
                partitionResult =
                    RecursivePartitionGraph(scratch.temp.arena, clusterOffsets, clusterData,
                                            clusterWeights, numClusters, totalNumNeighbors);
            }

            Print("num groups: %u\n", partitionResult.ranges.Length());

            StaticArray<u32> clusterToGroupID(scratch.temp.arena, numClusters, numClusters);
            for (int groupIndex = 0; groupIndex < partitionResult.ranges.Length();
                 groupIndex++)
            {
                Range &range = partitionResult.ranges[groupIndex];
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

            // Simplify every group
            ParallelFor(
                0, partitionResult.ranges.Length(), 1, 1,
                [&](int jobID, int start, int count) {
                    u32 threadIndex = GetThreadIndex();
                    Arena *arena    = arenas[threadIndex];
                    for (int groupIndex = start; groupIndex < start + count; groupIndex++)
                    {
                        ScratchArena scratch;

                        Range range           = partitionResult.ranges[groupIndex];
                        u32 groupNumTriangles = 0;

                        for (int clusterIndexIndex = range.begin;
                             clusterIndexIndex < range.end; clusterIndexIndex++)
                        {
                            int clusterIndex =
                                partitionResult.clusterIndices[clusterIndexIndex];
                            const Cluster &cluster = levelClusters[clusterIndex];
                            groupNumTriangles += cluster.record.Count();
                        }

                        f32 *groupVertices =
                            PushArrayNoZero(scratch.temp.arena, f32,
                                            (groupNumTriangles * 3) * (3 + numAttributes));
                        u32 *indices =
                            PushArrayNoZero(scratch.temp.arena, u32, groupNumTriangles * 3);
                        u32 *geomIDs =
                            PushArrayNoZero(scratch.temp.arena, u32, groupNumTriangles);
                        u32 vertexCount   = 0;
                        u32 indexCount    = 0;
                        u32 triangleCount = 0;

                        u32 numHash = NextPowerOfTwo(groupNumTriangles * 3);

                        HashIndex vertexHash(scratch.temp.arena, numHash, numHash);

                        // Merge clusters into a single vertex and index buffer
                        u32 total = 0;
                        for (int clusterIndexIndex = range.begin;
                             clusterIndexIndex < range.end; clusterIndexIndex++)
                        {
                            int clusterIndex =
                                partitionResult.clusterIndices[clusterIndexIndex];
                            u32 groupID            = clusterToGroupID[clusterIndex];
                            const Cluster &cluster = levelClusters[clusterIndex];
                            const ClusterGroup &prevClusterGroup =
                                clusterGroups[cluster.childGroupIndex];

                            total += cluster.record.count;

                            // if (depth == 7 && groupIndex == 90 && clusterIndex == 829)
                            // {
                            //     Print("count %u\n", cluster.record.count);
                            //     Print("clusterIndex: %u groupIndex %u\n",
                            //           clusterIndex + totalNumClusters,
                            //           cluster.childGroupIndex);
                            //     for (int testClusterIndex =
                            //     prevClusterGroup.parentStartIndex;
                            //          testClusterIndex < prevClusterGroup.parentStartIndex +
                            //                                 prevClusterGroup.parentCount;
                            //          testClusterIndex++)
                            //     {
                            //         auto &record = levelClusters[testClusterIndex].record;
                            //         Print("%i %u %u\n", testClusterIndex, record.Start(),
                            //               record.Count());
                            //     }
                            // }

                            for (int refID = cluster.record.Start();
                                 refID < cluster.record.End(); refID++)
                            {
                                PrimRef &ref = prevClusterGroup.primRefs[refID];

                                geomIDs[triangleCount++] = ref.geomID;

                                Assert(ref.minX <= cluster.record.geomMin[0]);
                                Assert(ref.minY <= cluster.record.geomMin[1]);
                                Assert(ref.minZ <= cluster.record.geomMin[2]);
                                Assert(ref.maxX <= cluster.record.geomMax[0]);
                                Assert(ref.maxY <= cluster.record.geomMax[1]);
                                Assert(ref.maxZ <= cluster.record.geomMax[2]);

                                for (int vertIndex = 0; vertIndex < 3; vertIndex++)
                                {
                                    u32 indexIndex  = 3 * ref.primID + vertIndex;
                                    u32 vertexIndex = prevClusterGroup.indices[indexIndex];

                                    f32 *clusterVertexData = GetVertexData(
                                        prevClusterGroup.vertexData, vertexIndex);

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
                                        MemoryCopy(
                                            GetVertexData(groupVertices, newVertexIndex),
                                            clusterVertexData, vertexDataLen);
                                        vertexHash.AddInHash(hash, newVertexIndex);
                                    }

                                    indices[indexCount++] = newVertexIndex;
                                }
                            }
                        }

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

                        f32 triangleSize = Sqrt(totalSurfaceArea / (float)numTris);
                        Float currentSize(Max(triangleSize, .00002f));
                        Float desired(.25f);
                        Float scale(1.f);
                        int exponent = Clamp((int)desired.exponent - (int)currentSize.exponent,
                                             -126, 127);
                        scale.exponent = exponent + 127;
                        float posScale = scale.value;
                        for (int i = 0; i < vertexCount; i++)
                        {
                            GetPosition(groupVertices, i) *= posScale;
                        }

                        // Simplify the clusters
                        u32 targetNumParents =
                            (groupNumTriangles + MAX_CLUSTER_TRIANGLES * 2 - 1) /
                            (MAX_CLUSTER_TRIANGLES * 2);
                        u32 targetNumTris = targetNumParents * MAX_CLUSTER_TRIANGLES;

                        f32 targetError = 0.f;
                        MeshSimplifier simplifier(scratch.temp.arena, (f32 *)groupVertices,
                                                  vertexCount, indices, indexCount,
                                                  numAttributes);

                        // Lock edges shared with other groups
                        for (int clusterIndexIndex = range.begin;
                             clusterIndexIndex < range.end; clusterIndexIndex++)
                        {
                            int clusterIndex =
                                partitionResult.clusterIndices[clusterIndexIndex];
                            u32 groupID            = clusterToGroupID[clusterIndex];
                            const Cluster &cluster = levelClusters[clusterIndex];

                            const ClusterData &clusterData = clusterDatas[clusterIndex];

                            for (int externalEdgeIndex = 0;
                                 externalEdgeIndex < clusterData.numExternalEdges;
                                 externalEdgeIndex++)
                            {
                                int edgeIndex = clusterData.externalEdges[externalEdgeIndex];
                                Edge &edge    = edges[edgeIndex];
                                if (clusterToGroupID[edge.clusterIndex] != groupID)
                                {
                                    simplifier.LockVertex(edge.p0 * posScale);
                                    simplifier.LockVertex(edge.p1 * posScale);
                                    // simplifier.LockVertex(edge.p0);
                                    // simplifier.LockVertex(edge.p1);
                                }
                            }
                        }

                        // if (depth == 7 && groupIndex == 90)
                        // {
                        //     Print(" %u %u\n", vertexCount, indexCount);
                        //     // for (int i = 0; i < vertexCount; i++)
                        //     // {
                        //     //     Vec3f v = ((Vec3f *)groupVertices)[i];
                        //     //     Print("%f %f %f\n", v.x, v.y, v.z);
                        //     // }
                        //     // for (int i = 0; i < indexCount; i += 3)
                        //     // {
                        //     //     Print("%u %u %u \n", indices[i], indices[i + 1],
                        //     //           indices[i + 2]);
                        //     // }
                        //     // Print("%u %u\n", groupIndex, numParentClusters);
                        //     DebugBreak();
                        // }

                        // if (groupIndex + totalNumGroups == 21894)
                        // {
                        //     for (int i = 0; i < vertexCount; i++)
                        //     {
                        //         Vec3f p = ((Vec3f *)groupVertices)[i];
                        //         Print("%f %f %f\n", p.x, p.y, p.z);
                        //     }
                        //     for (int i = 0; i < indexCount; i += 3)
                        //     {
                        //         Print("%u %u %u\n", indices[i], indices[i + 1],
                        //               indices[i + 2]);
                        //     }
                        // }

                        f32 invScale = 1.f / posScale;
                        f32 error    = simplifier.Simplify(vertexCount, targetNumTris,
                                                           Sqr(targetError), 0, 0, FLT_MAX);
                        f32 preError = error;
                        error        = Sqrt(error) * invScale;

                        Mesh simplifiedMesh = {};
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

                        // Split the simplified meshes into clusters
                        u32 numFaces = simplifiedMesh.numIndices / 3;

                        simplifiedMesh.numFaces = numFaces;
                        RecordAOSSplits record;
                        PrimRef *newPrimRefs = PushArrayNoZero(arena, PrimRef, numFaces);

                        GenerateMeshRefs<GeometryType::TriangleMesh>(
                            &simplifiedMesh, newPrimRefs, 0, numFaces, 0, 1, record);
                        record.SetRange(0, numFaces);

                        // if (groupIndex + totalNumGroups == 21894)
                        // {
                        //     Print("num faces: %u\n", numFaces);
                        //     for (int i = 0; i < numFaces; i++)
                        //     {
                        //         Print("prim %f %f %f %f %f %f\n", newPrimRefs[i].min[0],
                        //               newPrimRefs[i].min[1], newPrimRefs[i].min[2],
                        //               newPrimRefs[i].max[0], newPrimRefs[i].max[1],
                        //               newPrimRefs[i].max[2]);
                        //     }
                        // }

                        for (int primRefIndex = 0; primRefIndex < numFaces; primRefIndex++)
                        {
                            newPrimRefs[primRefIndex].geomID = geomIDs[primRefIndex];
                        }

                        ClusterBuilder clusterBuilder(arena, newPrimRefs);
                        clusterBuilder.BuildClusters(record, false);

                        u32 numParentClusters = 0;
                        for (auto &list : clusterBuilder.threadClusters)
                        {
                            numParentClusters += list.l.Length();
                        }

                        // if (depth == 7 && groupIndex == 90)
                        // {
                        //     Print("num parent clusters: %u\n", numParentClusters);
                        // }

                        u32 parentStartIndex = numLevelClusters.fetch_add(
                            numParentClusters, std::memory_order_relaxed);
                        Assert(parentStartIndex + numParentClusters < numClusters);

                        u32 offset        = 0;
                        u32 newGroupIndex = groupIndex + totalNumGroups;

                        Vec4f *clusterSpheres = PushArrayNoZero(scratch.temp.arena, Vec4f,
                                                                range.end - range.begin);
                        Bounds parentBounds;
                        // Set the child start index of last level's clusters
                        for (int clusterIndexIndex = range.begin;
                             clusterIndexIndex < range.end; clusterIndexIndex++)
                        {
                            int clusterIndex =
                                partitionResult.clusterIndices[clusterIndexIndex];

                            levelClusters[clusterIndex].groupIndex = newGroupIndex;

                            clusterSpheres[clusterIndexIndex - range.begin] =
                                levelClusters[clusterIndex].lodBounds;

                            // Parent error should always be >= to child error
                            error = Max(error, levelClusters[clusterIndex].lodError);
                        }

                        Vec4f parentSphereBounds = ConstructSphereFromSpheres(
                            clusterSpheres, range.end - range.begin);

                        // Add the new clusters
                        DenseGeometryBuildData *groupBuildData = &buildDatas[threadIndex];
                        u32 headerOffset = groupBuildData->headers.Length();
                        for (auto &list : clusterBuilder.threadClusters)
                        {
                            RecordAOSSplits *newClusterRecords = PushArrayNoZero(
                                scratch.temp.arena, RecordAOSSplits, numParentClusters);
                            list.l.Flatten(newClusterRecords);
                            for (int i = 0; i < list.l.Length(); i++)
                            {
                                Cluster &cluster =
                                    nextLevelClusters[parentStartIndex + offset + i];
                                cluster.record          = newClusterRecords[i];
                                cluster.mipLevel        = depth + 1;
                                cluster.childGroupIndex = newGroupIndex;
                                cluster.lodError        = error;
                                cluster.lodBounds       = parentSphereBounds;
                                cluster.headerIndex     = headerOffset + offset + i;
                            }
                            offset += list.l.Length();
                        }

                        ClusterGroup newClusterGroup;
                        newClusterGroup.vertexData       = (f32 *)simplifiedMesh.p;
                        newClusterGroup.indices          = simplifiedMesh.indices;
                        newClusterGroup.primRefs         = newPrimRefs;
                        newClusterGroup.buildDataIndex   = threadIndex;
                        newClusterGroup.isLeaf           = false;
                        newClusterGroup.maxParentError   = error;
                        newClusterGroup.lodBounds        = parentSphereBounds;
                        newClusterGroup.parentStartIndex = parentStartIndex;
                        newClusterGroup.parentCount      = numParentClusters;

                        newClusterGroup.numVertices = simplifiedMesh.numVertices;
                        newClusterGroup.numIndices  = simplifiedMesh.numIndices;
                        newClusterGroup.mipLevel    = depth;

                        clusterGroups[newGroupIndex] = newClusterGroup;

                        clusterBuilder.CreateDGFs(materialIndices, groupBuildData,
                                                  &simplifiedMesh, 1, bounds);

                        ReleaseArenaArray(clusterBuilder.arenas);
                    }
                });

            // Write obj to disk
#if 0
            u32 vertexCount   = numVertices.load();
            u32 indexCount    = numIndices.load();
            Mesh levelMesh    = {};
            u32 vertexOffset  = 0;
            u32 indexOffset   = 0;
            levelMesh.p       = PushArrayNoZero(scratch.temp.arena, Vec3f, vertexCount);
            levelMesh.indices = PushArrayNoZero(scratch.temp.arena, u32, indexCount);
            HashIndex vertexHash(scratch.temp.arena, NextPowerOfTwo(vertexCount),
                                 NextPowerOfTwo(vertexCount));

            // for (int clusterIndex = totalNumClusters;
            //      clusterIndex < totalNumClusters + numLevelClusters.load(); clusterIndex++)
            // {
            //     Cluster &cluster         = clusters[clusterIndex];
            //     RecordAOSSplits &record  = cluster.record;
            //     ClusterGroup &childGroup = clusterGroups[cluster.childGroupIndex];
            //
            //     for (int i = record.start; i < record.End(); i++)
            //     {
            //         for (int vertIndex = 0; vertIndex < 3; vertIndex++)
            //         {
            //             u32 index       = childGroup.indices[3 * i + vertIndex];
            //             f32 *data       = childGroup.vertexData + (3 + numAttributes) *
            //             index; Vec3f p         = *(Vec3f *)data; int hash        = Hash(p);
            //             int vertexIndex = -1;
            //             for (int hashIndex = vertexHash.FirstInHash(hash); hashIndex != -1;
            //                  hashIndex     = vertexHash.NextInHash(hashIndex))
            //             {
            //                 if (levelMesh.p[hashIndex] == p)
            //                 {
            //                     vertexIndex = hashIndex;
            //                     break;
            //                 }
            //             }
            //             if (vertexIndex == -1)
            //             {
            //                 vertexIndex              = vertexOffset++;
            //                 levelMesh.p[vertexIndex] = p;
            //                 vertexHash.AddInHash(hash, vertexIndex);
            //             }
            //             levelMesh.indices[indexOffset++] = vertexIndex;
            //         }
            //     }
            // }
            for (int groupIndex = totalNumGroups; groupIndex < clusterGroups.Length();
                 groupIndex++)
            {
                ClusterGroup &clusterGroup = clusterGroups[groupIndex];
                for (int i = 0; i < clusterGroup.numIndices; i++)
                {
                    f32 *data = clusterGroup.vertexData +
                                (3 + numAttributes) * clusterGroup.indices[i];
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
            }

            levelMesh.numVertices = vertexOffset;
            levelMesh.numIndices  = indexOffset;
            WriteTriOBJ(levelMesh,
                        PushStr8F(scratch.temp.arena,
                                  "../../data/island/pbrt-v4/obj/osOcean/test_%u.obj", depth));
#endif

            u32 numNextLevelClusters = numLevelClusters.load();
            // if (debugNumLevelClusters[depth + 1] != numNextLevelClusters)
            // {
            //     DebugBreak();
            //     Print("%u\n", numNextLevelClusters);
            //     for (int groupIndex = 0; groupIndex < partitionResult.ranges.Length();
            //          groupIndex++)
            //     {
            //         Print("%u %u %u\n", groupIndex,
            //         partitionResult.ranges[groupIndex].begin,
            //               partitionResult.ranges[groupIndex].end);
            //     }
            //     DebugBreak();
            // }

            clusters.Resize(totalNumClusters + numLevelClusters.load());

            Assert(numLevelClusters.load() < levelClusters.Length());
            u32 clusterOffset = 0;
            StaticArray<Cluster> reorderedClusters(scratch.temp.arena, levelClusters.Length(),
                                                   levelClusters.Length());
            for (int groupIndex = 0; groupIndex < partitionResult.ranges.Length();
                 groupIndex++)
            {
                Range &range            = partitionResult.ranges[groupIndex];
                ClusterGroup &group     = clusterGroups[totalNumGroups + groupIndex];
                group.clusterStartIndex = prevClusterArrayEnd + range.begin;
                group.clusterCount      = range.end - range.begin;
                ErrorExit(group.clusterStartIndex + group.clusterCount <= clusters.Length(),
                          "%u %u %u\n", group.clusterStartIndex, group.clusterCount,
                          clusters.Length());

                for (int i = range.begin; i < range.end; i++)
                {
                    int clusterIndex                   = partitionResult.clusterIndices[i];
                    reorderedClusters[clusterOffset++] = levelClusters[clusterIndex];
                }
            }
            MemoryCopy(levelClusters.data, reorderedClusters.data,
                       clusterOffset * sizeof(Cluster));

            prevClusterArrayEnd = totalNumClusters;
            clusterOffset       = 0;
            for (int groupIndex = 0; groupIndex < partitionResult.ranges.Length();
                 groupIndex++)
            {
                Range &range        = partitionResult.ranges[groupIndex];
                ClusterGroup &group = clusterGroups[totalNumGroups + groupIndex];

                u32 newStartIndex = clusterOffset;
                for (int parentIndex = group.parentStartIndex;
                     parentIndex < group.parentStartIndex + group.parentCount; parentIndex++)
                {
                    reorderedClusters[clusterOffset++] =
                        clusters[totalNumClusters + parentIndex];
                }

                group.parentStartIndex = newStartIndex;
            }
            MemoryCopy(clusters.data + totalNumClusters, reorderedClusters.data,
                       clusterOffset * sizeof(Cluster));

            levelClusters = ArrayView<Cluster>(clusters, totalNumClusters,
                                               clusters.Length() - totalNumClusters);
            depth++;

            // u32 numEdges = 0;
            // for (int i = 0; i < OS_NumProcessors(); i++)
            // {
            //     numEdges += threadLocalStatistics[i].test;
            // }
            // Print("num edges: %u\n", numEdges);
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

    StaticArray<int> sortedClusterIndices(scratch.temp.arena, numClusters);
    // TODO: morton order sort
    for (int i = 0; i < numClusters; i++)
    {
        sortedClusterIndices.Push(i);
    }

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

    for (int groupIndex = 0; groupIndex < clusterGroups.Length(); groupIndex++)
    {
        ClusterGroup &group = clusterGroups[groupIndex];
        if (group.isLeaf) continue;

        group.pageStartIndex = pageInfos.Length();

        u32 clusterStartIndex = 0;

        for (int clusterGroupIndex = 0; clusterGroupIndex < group.clusterCount;
             clusterGroupIndex++)
        {
            u32 clusterMetadataSize =
                (numClustersInPage + 1) * NUM_CLUSTER_HEADER_FLOAT4S * sizeof(Vec4f);

            Cluster &cluster         = clusters[group.clusterStartIndex + clusterGroupIndex];
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
    }

    PageInfo pageInfo;
    pageInfo.partStartIndex = partStartIndex;
    pageInfo.partCount      = parts.Length() - partStartIndex;
    pageInfo.numClusters    = numClustersInPage;
    pageInfos.Push(pageInfo);

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

                // TODO: frustum culling bounds
                MemoryCopy(ptr + currentPageOffset, &cluster.lodBounds, sizeof(Vec4f));

                for (u32 i = 1; i < NUM_CLUSTER_HEADER_FLOAT4S; i++)
                {
                    u32 copySize = Min(stride, (u32)sizeof(header) - (i - 1) * stride);
                    u32 *src     = (u32 *)&header + 4u * (i - 1);
                    MemoryCopy(ptr + currentPageOffset + i * soaStride, src, copySize);
                }

                MemoryCopy(ptr + currentPageOffset +
                               (NUM_CLUSTER_HEADER_FLOAT4S - 1) * soaStride + sizeof(Vec3f),
                           &cluster.lodError, sizeof(float));
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

    for (int i = 0; i < parts.Length(); i++)
    {
        PrimRef &primRef = hierarchyPrimRefs[i];
        primRef.primID   = i;

        GroupPart &part     = parts[i];
        ClusterGroup &group = clusterGroups[part.groupIndex];

        Bounds partBounds;

        for (int clusterGroupIndex = part.clusterStartIndex;
             clusterGroupIndex < part.clusterStartIndex + part.clusterCount;
             clusterGroupIndex++)
        {
            Cluster &cluster        = clusters[group.clusterStartIndex + clusterGroupIndex];
            RecordAOSSplits &record = cluster.record;

            Lane4F32 clusterMin(-record.geomMin[0], -record.geomMin[1], -record.geomMin[2],
                                0.f);
            Lane4F32 clusterMax(record.geomMax[0], record.geomMax[1], record.geomMax[2], 0.f);

            partBounds.Extend(clusterMin, clusterMax);
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

    RecordAOSSplits hierarchyRecord;
    hierarchyRecord.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    hierarchyRecord.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    hierarchyRecord.start      = 0;
    hierarchyRecord.count      = parts.Length();

    Arena *arena           = ArenaAlloc();
    u32 numNodes           = 0;
    HierarchyNode rootNode = BuildHierarchy(arena, clusters, clusterGroups, parts,
                                            hierarchyPrimRefs, hierarchyRecord, numNodes);

    // Flatten tree to array
    StaticArray<PackedHierarchyNode> hierarchy(arena, numNodes);
    PackedHierarchyNode rootPacked = {};
    for (int i = 0; i < CHILDREN_PER_HIERARCHY_NODE; i++)
    {
        rootPacked.childRef[i] = ~0u;
        rootPacked.leafInfo[i] = ~0u;
    }
    for (int i = 0; i < rootNode.numChildren; i++)
    {
        rootPacked.lodBounds[i]      = rootNode.lodBounds[i];
        rootPacked.maxParentError[i] = rootNode.maxParentError[i];
    }
    hierarchy.Push(rootPacked);

    struct StackEntry
    {
        HierarchyNode node;

        u32 parentIndex;
        u32 childIndex;
    };

    StaticArray<StackEntry> queue(scratch.temp.arena, numNodes, numNodes);
    u32 readOffset  = 0;
    u32 writeOffset = 0;

    for (int i = 0; i < rootNode.numChildren; i++)
    {
        StackEntry root;
        root.node        = rootNode.children[i];
        root.parentIndex = 0;
        root.childIndex  = i;

        queue[writeOffset++] = root;
    }

    // 1. instance culling hierarchy. procedural aabbs as proxies for instance groups. closest
    // hit intersections of these aabbs are saved, and a bvh is built over just the instances
    // inside these proxies. the intersections are then repeated with this smaller set.
    // 2. partial rebraiding
    // 3. instance proxy combining

    // interesting idea:
    // 1. create a bounding sphere/aabb hierarchy over instances. the leaves contain instance
    // ids.
    // 2. use previous frame's rays to traverse this hierarchy. instances in intersected leaves
    // are added to the tlas. this would be in addition to standard occlusion/frustum/small
    // element culling (maybe? i'm not sure about this last part)
    // 3. since we control this hierarchy, instead of normal instancing you could use instanced
    // submeshes (cluster groups), reducing overlap between instances (just like partial
    // rebraiding)
    // 4. thus instance data can be very highly compressed, and decompressed when needed

    // 5. maybe have some form of feedback? like with virtual textures (idk how this fits in to
    // the rest of the system)
    // 6. when the instance is small enough, use some smaller proxy/proxies (i.e., somehow
    // combine and simplify the instances)
    // 7. partitioned tlas (obviously)

    // so if the instance is outside the frustum or occluded, and wasn't intersected, then it
    // should be removed

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

        hierarchy[parentIndex].childRef[entry.childIndex] = childOffset;

        HierarchyNode &child       = entry.node;
        PackedHierarchyNode packed = {};
        for (int i = 0; i < CHILDREN_PER_HIERARCHY_NODE; i++)
        {
            packed.childRef[i] = ~0u;
            packed.leafInfo[i] = ~0u;
        }
        numLeaves += !(bool)child.children;
        for (int i = 0; i < child.numChildren; i++)
        {
            packed.lodBounds[i]      = child.lodBounds[i];
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

                if (clusterGroups[part.groupIndex].mipLevel == 1)
                {
                    packed.flags |= (1 << i);
                    numLeafParts++;
                }
            }
        }

        hierarchy.Push(packed);
    }

    Assert(numNodes != 0);
    Print("num nodes: %u\nnum parts: %u %u, num leaves: %u %u\n", numNodes, parts.Length(),
          numParts, numLeaves, numLeafParts);
    Assert(hierarchy.Length() == numNodes);

    ClusterFileHeader *fileHeader =
        (ClusterFileHeader *)GetMappedPtr(&builder, fileHeaderOffset);
    fileHeader->magic    = CLUSTER_FILE_MAGIC;
    fileHeader->numPages = pageInfos.Length();
    fileHeader->numNodes = numNodes;

    // Write hierarchy to disk
    u64 hierarchyOffset = AllocateSpace(&builder, sizeof(PackedHierarchyNode) * numNodes);
    u8 *ptr             = (u8 *)GetMappedPtr(&builder, hierarchyOffset);
    MemoryCopy(ptr, hierarchy.data, sizeof(PackedHierarchyNode) * numNodes);

    OS_UnmapFile(builder.ptr);
    OS_ResizeFile(builder.filename, builder.totalSize);
}

#if 0
struct GetHierarchyNode
{
    HierarchyNode *hierarchyNodes;
    using NodeType = HierarchyNode;
    __forceinline NodeType *operator()(const BRef &ref)
    {
        u32 nodeIndex = ref.nodePtr.data & ~0u;
        return &hierarchyNodes[nodeIndex];
    }
};

struct InstanceLeaf
{
    u32 instanceID[CHILDREN_PER_HIERARCHY_NODE];
    u32 children[CHILDREN_PER_HIERARCHY_NODE];

    __forceinline void Fill(const ScenePrimitives *scene, BuildRef<N> *refs, u32 &begin,
                            u32 end)
    {
    }
};

void GenerateInstanceHierarchy()
{
    ScenePrimitive *scene;

    Arena *arena   = ArenaAlloc();
    Arena **arenas = GetArenaArray(arena);

    // Flatten scene

    HierarchyNode *hierarchyNodes;
    BuildRef<CHILDREN_PER_HIERARCHY_NODE> *refs;

    using BuildType =
        BuildFuncs<CHILDREN_PER_HIERARCHY_NODE, HeuristicPartialRebraid<GetHierarchyNode>,
                   QuantizedCompressedNode<N>, CreateQuantizedNode<N>,
                   UpdateQuantizedCompressedNode<N>, InstanceLeaf>;
    using Builder = BVHBuilder<N, BuildType>;

    Builder builder;
    using Heuristic = typename Builder::Heuristic;
    new (&builder.heuristic) Heuristic(scene, refs, settings.logBlockSize);
    builder.heuristic.getNode.hierarchyNodes = hierarchyNodes;
    builder.primRefs                         = refs;
    builder.scene                            = scene;

    BVHNodeN rootNode = builder.BuildCompressedBVH(settings, arenas, record);
}
#endif

} // namespace rt
