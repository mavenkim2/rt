#include "../hash.h"
#include "../memory.h"
#include "../thread_context.h"
#include "../bit_packing.h"
#include "../dgfs.h"
#include "mesh_simplification.h"

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

void Quadric::InitializeEdge(const Vec3f &p0, const Vec3f &p1)
{
#if 0
    Vec3f n = Cross(p0, p1);

    Vec3f p01 = p1 - p0;
    Vec3f p02 = p2 - p0;

    Vec3f n = Cross(p01, p02);

    gVol = 0.f;
    dVol = 0.f;

    f32 length = Length(n);
    area       = 0.5f * length;

    if (length < 1e-8f) return;

    n /= length;

    OuterProduct(n, c00, c01, c02, c11, c12, c22);

    f32 distToPlane = -Dot(n, p0);
    dn              = distToPlane * n;
    d2              = Sqr(distToPlane);

    // Multiply quadric by area (in preparation to be summed by other faces)
    c00 *= area;
    c01 *= area;
    c02 *= area;

    c11 *= area;
    c12 *= area;
    c22 *= area;

    dn *= area;
    d2 *= area;
#endif
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

void AddQuadric(QuadricGrad *g, const QuadricGrad *other, u32 numAttributes)
{
    for (int i = 0; i < numAttributes; i++)
    {
        g[i].g += other[i].g;
        g[i].d += other[i].d;
    }
}

int Pair::Hash() { return rt::Hash(index0, index1); }

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

MeshSimplifier::MeshSimplifier(f32 *vertexData, u32 numVertices, u32 *indices, u32 numIndices)
    : vertexData(vertexData), indices(indices), numVertices(numVertices),
      numIndices(numIndices)
{
}

Vec3f &MeshSimplifier::GetPosition(u32 vertexIndex)
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

bool MeshSimplifier::CheckInversion(const Vec3f &newPosition, u32 *adjTris, u32 count)
{
    for (int i = 0; i < count; i++)
    {
        u32 triangle = adjTris[i];
        indices[3 * triangle];
        if (adjTris[i])
        {
        }
    }

    for (int i = 0; i < 2; i++)
    {
        int vertexIndex      = i == 0 ? index0 : index1;
        int otherVertexIndex = i == 0 ? index1 : index0;
        while (vertexIndex != -1)
        {
            VertexGraphNode *node = &vertexNodes[vertexIndex];
            for (int j = node->offset; j < node->offset + node->count; j++)
            {
                u32 indexIndex0 = indexData[j];
                u32 indexIndex1 = NextInTriangle(indexIndex0, 1);
                u32 indexIndex2 = NextInTriangle(indexIndex0, 2);

                u32 vertexIndex0 = indices[indexIndex0];
                u32 vertexIndex1 = indices[indexIndex1];
                u32 vertexIndex2 = indices[indexIndex2];

                // This triangle will be collapsed if this pair is collapsed
                if (vertexIndex1 == otherVertexIndex || vertexIndex2 == otherVertexIndex)
                    continue;

                Vec3f p0 = GetPosition(vertexIndex0);
                Vec3f p1 = GetPosition(vertexIndex1);
                Vec3f p2 = GetPosition(vertexIndex2);

                Vec3f p21      = p2 - p1;
                Vec3f p01      = p0 - p1;
                Vec3f pNewEdge = newPosition - p1;

                bool result = Dot(Cross(pNewEdge, p21), Cross(p01, p21)) >= 0.f;
                if (!result) return true;
            }
            vertexIndex = node->next;
        }
    }

    return false;
}

static const int next[3] = {1, 2, 0};

template <typename Func>
void IterateHashBreak(HashIndex &index, int hash, const &Func func)
{
    for (int i = index.FirstInHash(hash); i != -1; i = index.NextInHash(i))
    {
        if (func(i)) break;
    }
}

template <typename Func>
void IterateHash(HashIndex &index, int hash, const &Func func)
{
    for (int i = index.FirstInHash(hash); i != -1; i = index.NextInHash(i))
    {
        func(i);
    }
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

bool MeshSimplifier::AddUniquePair(Pair &pair)
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
        pairs.Push(pair);
        int pairIndex = pairs.Length() - 1;
        pairHash0.AddInHash(p0Hash, pairIndex);
        pairHash1.AddInHash(p1Hash, pairIndex);
    }

    return !duplicate;
}

f32 MeshSimplifier::EvaluatePair(Pair &pair, Vec3f *outP)
{
    VertexGraphNode *node = &vertexNodes[pair.index0];

    // Find the set of triangles adjacent to the pair
    ScratchArena scratch;

    Array<u32> adjTris(scratch.temp.arena, 24);

    int hashes[] = {
        Hash(pair.p0),
        Hash(pair.p1),
    };

    Vec3f pos[] = {
        pair.p0,
        pair.p1,
    };

    for (int i = 0; i < 2; i++)
    {
        IterateHash(cornerHash, hashes[i],
                    [&](int cornerIndex) { adjTris.PushUnique(cornerIndex / 3); });
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

    // Add edge quadric
    // Quadric edgeQuadric(0);
    // for (int i = 0; i < 2; i++)
    // {
    // int nodeIndex = indices[pair.indexIndex0];
    // }

    // TODO: handle locked edges/verts + preserving boundary edges

    Vec3f p;

    f32 error  = 0.f;
    bool valid = false;

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
                valid = !CheckInversion(p + basePosition, pair.index0, pair.index1);
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
                valid = !CheckInversion(p + basePosition, pair.index0, pair.index1);
            }
        }
    }

    if (!valid)
    {
        p     = (pair.p0 + pair.p1) / 2.f;
        valid = !CheckInversion(p, pair.index0, pair.index1);
        p -= basePosition;
    }

    if (!valid)
    {
        error += inversionPenalty;
    }

    // Evaluate the error for the optimal position
    error += EvaluateQuadric(p, quadric, quadricGrad, 0, 0, numAttributes);

    if (outP) *outP = p + basePosition;

    return error;
}

f32 MeshSimplifier::Simplify(Arena *arena, u32 targetNumVerts, u32 targetNumTris,
                             f32 targetError, u32 limitNumVerts, u32 limitNumTris,
                             f32 limitError)
{
    ScratchArena scratch;
    // Follows section 4.1 of the quadric error paper

    // For every edge, compute the optimal contraction target and its cost

    // Packed vertex data AOS:
    // p.x p.y p.z n.x n.y n.z uv.x uv.y uv.z

    numAttributes = 0;

    u32 numTriangles = numIndices / 3;

    attributeWeights = 0;

    triangleQuadrics = PushArrayNoZero(scratch.temp.arena, Quadric, numTriangles);

    triangleAttrQuadrics = 0;
    if (numAttributes)
        triangleAttrQuadrics =
            PushArrayNoZero(scratch.temp.arena, QuadricGrad, numTriangles * numAttributes);

    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
        CalculateTriQuadrics(triIndex);
    }

    u32 cornerHashSize = NextPowerOfTwo(numIndices);
    cornerHash         = HashIndex(scratch.temp.arena, cornerHashSize, cornerHashSize);

    for (int corner = 0; corner < numIndices; corner++)
    {
        Vec3f position = GetPosition(indices[corner]);
        int hash       = Hash(position);
        cornerHashSize.AddInHash(hash, corner);
    }

    pairs = StaticArray<Pair>(scratch.temp.arena, 3 * numTriangles);
    Heap<Pair> heap(scratch.temp.arena, 3 * numTriangles);

    u32 pairHashSize = NextPowerOfTwo(3 * numTriangles);
    pairHash0        = HashIndex(scratch.temp.arena, pairHashSize, pairHashSize);
    pairHash1        = HashIndex(scratch.temp.arena, pairHashSize, pairHashSize);

    // Add unique pairs to hash
    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
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

            AddUniquePair(pair);
        }
    }

    for (int pairIndex = 0; pairIndex < pairs.size() - 1; pairIndex++)
    {
        f32 error              = EvaluatePair(pairs[pairIndex]);
        pairs[pairIndex].error = error;
        heap.Add(pairs.data, pairIndex);
    }

    BitVector triangleIsRemoved(scratch.temp.arena, numTriangles);
    BitVector vertexIsRemoved(scratch.temp.arena, numVertices);

    f32 maxError              = 0.f;
    u32 remainingNumVertices  = numVertices;
    u32 remainingNumTriangles = numTriangles;
    for (;;)
    {
        int pairIndex = heap.Pop(pairs.data);
        if (pairIndex == -1) break;

        Pair &pair = pairs[pairIndex];

        Vec3f newPosition;
        f32 error = EvaluatePair(pair, &newPosition);

        maxError = Max(maxError, error);

        Assert(maxError < 100);

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
        std::vector<int> movedPairs;

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

        for (int i = 0; i < 2; i++)
        {
            IterateHash(cornerHash, hashes[i], [&](int corner) {
                if (indices[corner] == pos[i])
                {
                    movedCorners.PushUnique(corner);
                    cornerHash.RemoveFromHash(Hash(indices[corner]), corner);
                }
            });

            IterateHash(pairHash0, hashes[i], [&](int otherPairIndex) {
                if (pairs[otherPairIndex].p0 == pos[i])
                {
                    movedPairs.push_back(otherPairIndex);
                    pairHash0.RemoveFromHash(Hash(pairs[otherPairIndex].p0), otherPairIndex);
                    pairHash1.RemoveFromHash(Hash(pairs[otherPairIndex].p1), otherPairIndex);
                }
            });

            IterateHash(pairHash1, hashes[i], [&](int otherPairIndex) {
                if (pairs[otherPairIndex].p1 == pos[i])
                {
                    movedPairs.push_back(otherPairIndex);
                    pairHash0.RemoveFromHash(Hash(pairs[otherPairIndex].p0), otherPairIndex);
                    pairHash1.RemoveFromHash(Hash(pairs[otherPairIndex].p1), otherPairIndex);
                }
            });
        }

        for (int pairIndex : movedPairs)
        {
            Pair &movedPair = pairs[pairIndex];
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

        for (int pairIndex : movedPairs)
        {
            Pair &pair = pairs[pairIndex];
            if (pair.p0 == pair.p1 || !AddUniquePair(pair))
            {
                heap.Remove(pairs, pairIndex);
            }
        }

        // Recalculate quadrics
        for (u32 corner : movedCorners)
        {
            GetPosition(indices[corner]) = newPosition;

            u32 triangle = corner / 3;

            Vec3f p0 = GetPosition(indices[3 * triangle]);
            Vec3f p1 = GetPosition(indices[3 * triangle + 1]);
            Vec3f p2 = GetPosition(indices[3 * triangle + 2]);

            if (!(p0 == p1 || p0 == p2 || p1 == p2))
            {
            }

            CalculateTriQuadrics(triangle);
        }
    }

    // Compact the vertex buffer
    StaticArray<u32> remap(scratch.temp.arena, numVertices);
    u32 vertexDataStride = (3 + numAttributes) * sizeof(f32);
    u32 vertexCount      = 0;
    for (int i = 0; i < numVertices; i++)
    {
        if (!vertexIsRemoved.GetBit(i))
        {
            remap[i] = vertexCount;
            MemoryCopy(vertexData + vertexDataStride * vertexCount,
                       vertexData + vertexDataStride * i, vertexDataStride);
            vertexCount++;
        }
    }

    // Compact the index buffer
    u32 indexCount = 0;
    for (int i = 0; i < numTriangles; i++)
    {
        if (!triangleIsRemoved.GetBit(i))
        {
            Assert(!vertexIsRemoved.GetBit(indices[3 * i]));
            Assert(!vertexIsRemoved.GetBit(indices[3 * i + 1]));
            Assert(!vertexIsRemoved.GetBit(indices[3 * i + 2]));

            indices[indexCount]     = remap[indices[3 * i]];
            indices[indexCount + 1] = remap[indices[3 * i + 1]];
            indices[indexCount + 2] = remap[indices[3 * i + 2]];
            indexCount += 3;
        }
    }

    u32 *finalIndices = PushArrayNoZero(arena, u32, indexCount);
    MemoryCopy(finalIndices, indices, sizeof(u32) * indexCount);
    f32 *finalVertexData = PushArrayNoZero(arena, f32, vertexCount * (3 + numAttributes));
    MemoryCopy(finalVertexData, vertexData, vertexDataStride * vertexCount);

    return maxError;
}

#if 0
struct ClusterGroup
{
};

struct Cluster
{
};

void CreateClusters(Mesh &mesh)
{
    // 1. Split triangles into clusters
    // 2. Group clusters based on how many shared edges they have (METIS)
    // 3. Simplify the cluster group
    // 4. Split into clusters

    ClusterBuilder builder;

    for (;;)
    {
        ScratchArena scratch;

        RecordAOSSplits record;
        builder.BuildClusters(record, true);

        PrimRef *refs = PushArrayNoZero(scratch.temp.arena, PrimRef, mesh.numFaces);
        u32 offset    = 0;
        u32 start     = 0;
        u32 count     = mesh.numFaces;
        GenerateMeshRefsHelper<GeometryType::TriangleMesh>{mesh->p, mesh->indices}(
            refs, offset, 0, start, count, record);
    }
}
#endif

} // namespace rt
