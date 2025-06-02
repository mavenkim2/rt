#include "../dgfs.h"
#include "../scene.h"
#include "mesh_simplification.h"

namespace rt
{
// https://en.wikipedia.org/wiki/LU_decomposition
template <typename T>
int LUPDecompose(T **__restrict A, int N, double Tol, int *__restrict P)
{
    for (int i = 0; i <= N; i++) P[i] = i; // Unit permutation matrix, P[N] initialized with N

    for (int i = 0; i < N; i++)
    {
        T maxA   = 0.0;
        int imax = i;

        for (k = i; k < N; k++)
        {
            T absA = Abs(A[k][i]);
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
            Swap(A[i], A[imax]);

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (int j = i + 1; j < N; j++)
        {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++) A[j][k] -= A[j][i] * A[i][k];
        }
    }

    return 1; // decomposition done
}

/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
void LUPSolve(double **A, int *P, double *b, int N, double *x)
{
    for (int i = 0; i < N; i++)
    {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++) x[i] -= A[i][k] * x[k];
    }

    for (int i = N - 1; i >= 0; i--)
    {
        for (int k = i + 1; k < N; k++) x[i] -= A[i][k] * x[k];

        x[i] /= A[i][i];
    }
}

// Due to floating point inaccuracy, use residuals to minimize error
template <typename T>
bool LUPSolveIterate(T **__restrict A, int *__restrict P, T *__restrict b, int N,
                     T *__restrict x, u32 numIters)
{
    LUPSolve(A, P, b, N, x);

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
                residual[i] -= A[i][j] * x[j];
            }
        }

        LUPSolve(A, P, residual, N, error);

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

Quadric::Quadric(u32 numAttributes) : numAttributes(numAttributes)
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

Quadric::Quadric(const Vec3f &p0, const Vec3f &p1, const Vec3f &p2, f32 *__restrict attr0,
                 f32 *__restrict attr1, f32 *__restrict attr2,
                 f32 *__restrict attributeWeights, u32 numAttributes)
    : numAttributes(numAttributes)
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

    OuterProduct(n, c00, c01, c02, c11, c12, c22);

    f32 distToPlane = -Dot(n, p0);
    dn              = distToPlane * n;
    d2              = Sqr(distToPlane);

    // Solve system of equations to find gradient for each attribute
    // (p1 - p0) * g = a1 - a0
    // (p2 - p0) * g = a2 - a0
    // n * g = 0

    f32 M[3][3] = {
        {p01.x, p01.y, p01.z},
        {p02.x, p02.y, p02.z},
        {n.x, n.y, n.z},
    };

    int pivots[3];
    bool isInvertible = LUPDecompose(M, 3, 1e-12, pivots);

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

        if (isInvertible) LUPSolveIterate(M, pivots, b, 3, grad.e, 1);

        gradients[i] = grad;
        d[i]         = a0 - Dot(grad, p);

        OuterProduct(gradients[i], c00, c01, c02, c11, c12, c22);

        dn += d[i] * gradients[i];
        d2 += Sqr(d[i]);
    }

    // Multiply quadric by area (in preparation to be summed by other faces)
    c00 *= area;
    c01 *= area;
    c02 *= area;

    c11 *= area;
    c12 *= area;
    c22 *= area;

    dn *= area;
    d2 *= area;

    for (u32 i = 0; i < numAttributes; i++)
    {
        gradients[i] *= area;
        d[i] *= area;
    }
}

f32 Quadric::Evaluate(const Vec3f &p, f32 *__restrict attributes,
                      f32 *__restrict attributeWeights)
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

    f32 x = Dot(Vec3f(c00, c01, c02), p);
    f32 y = Dot(Vec3f(c01, c11, c12), p);
    f32 z = Dot(Vec3f(c02, c12, c22), p);

    f32 error = Dot(Vec3f(x, y, z) + 2 * dn, p) + d2;

    f32 invArea = 1.f / area;

    for (int i = 0; i < numAttributes; i++)
    {
        f32 pgd       = d[i] + Dot(gradients[i], p);
        f32 s         = pgd * invArea;
        attributes[i] = s / attributeWeights[i];

        f32 gp = Dot(gradients[i], p);

        // 2s * Dot(-g, p) + -2s * d + dj2 + s^2 * area
        //
        // 1/area(d^2 + 2gp + gp^2 + 2d * -gp - 2gp^2 - 2d^2 - 2gp)
        // 1/area(-d^2 -gp^2 -2dgp)
        // -1/area(pgd^2)
        // -pgd * s

        error -= pgd * s;

        // f32 attributeVal = Dot(Vec3f(gradients[i]), p) + d[i];
        // f32 s            = attributeVal / area;
        //
        // error += s * (a * s - 2 * attributeVal);
        // attributes[i] = attributeVal / attributeWeight;
    }

    return error;
}

void Quadric::InitializeEdge(const Vec3f &p0, const Vec3f &p1)
{
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
}

void Quadric::Add(Quadric &other)
{
    Assert(numAttributes == other.numAttributes);
    c00 += other.c00;
    c01 += other.c01;
    c02 += other.c02;

    c11 += other.c11;
    c12 += other.c02;
    c22 += other.c22;

    dn += other.dn;

    d2 += other.d2;

    gVol += other.gVol;
    dVol += other.dVol;

    // Volume optimization
    area = other.area;

    for (int i = 0; i < numAttributes; i++)
    {
        gradients[i] += other.gradients[i];
        d[i] += other.d[i];
    }
}

bool Quadric::Optimize(Vec3f &p, bool volume)
{
    if (a < 1e-12) return false;

    // https://hhoppe.com/minqem.pd
    // Solve linear subsystem for v according to above paper

    // (C - 1/a * BBt) * v = b1 - 1/a * B * b2

    // C is the 3x3 outer product sum of all the plane normals and gradients for all the
    // quadrics
    // 1/a is the inv area
    // B is the 3xnumAttributes matrix of attribute gradients
    // v is the vector of length 3 + numAttributes containin the position and attributes
    //
    // b1 is -dn + sum djgj
    // b2 is dj

    f32 invA = 1.f / a;

    f32 BBt00 = 0.f;
    f32 BBt01 = 0.f;
    f32 BBt02 = 0.f;
    f32 BBt11 = 0.f;
    f32 BBt12 = 0.f;
    f32 BBt22 = 0.f;

    Vec3f b1 = dn;
    Vec3f Bb2(0.f);

    for (int i = 0; i < numAttributes; i++)
    {
        OuterProduct(gradients[i], BBt00, BBt01, BBt02, BBt11, BBt12, BBt22);

        Bb2 += gradients[i] * d[i];
    }

    // A = (C - 1/a * BBt)
    f32 A00 = c00 - BBt00 * invA;
    f32 A01 = c01 - BBt01 * invA;
    f32 A02 = c02 - BBt02 * invA;

    f32 A11 = c11 - BBt11 * invA;
    f32 A12 = c12 - BBt12 * invA;

    f32 A22 = c12 - BBt12 * invA;

    // b = b1 - 1/a * B * b2
    Vec3f b = b1 - invA * Bb2;

    // Now add the lagrange multiplier volume constraint:
    // v is now 4-dim position and lagrange multiplier
    // Dot(gVol, p) + dVol = 0

    if (volume)
    {
        f32 A[4][4] = {
            {A00, A01, A02, gVol.x},
            {A01, A11, A12, gVol.y},
            {A02, A12, A22, gVol.z},
            {gVol.x, gVol.y, gVol.z, 0},
        };

        f32 b[4] = {-b.x, -b.y, -b.z, -dVol};

        // Solve the 4x4 linear system
        int pivots[4];
        if (LUPDecompose(A, 4, 1e-8f, pivots))
        {
            f32 result[4];
            if (LUPSolveIterate(A, pivots, b, 4, result))
            {
                p.x = result[0];
                p.y = result[1];
                p.z = result[2];
                return true;
            }
        }
    }
    else
    {
        f32 A[3][3] = {
            {A00, A01, A02},
            {A01, A11, A12},
            {A02, A12, A22},
        };

        f32 b[3] = {-b.x, -b.y, -b.z};

        // Solve the 4x4 linear system
        int pivots[3];
        if (LUPDecompose(A, 3, 1e-8f, pivots))
        {
            f32 result[3];
            if (LUPSolveIterate(A, pivots, b, 3, result))
            {
                p.x = result[0];
                p.y = result[1];
                p.z = result[2];
                return true;
            }
        }
    }

    return false;
}

template <typename T>
struct Heap
{
    u32 *indices;
    u32 *indicesIndex;
    T *values;

    u32 heapNum;
    u32 numValues;

    u32 maxSize;

    Heap(Arena *arena, u32 arraySize)
    {
        indices      = PushArrayNoZero(arena, u32, arraySize);
        indicesIndex = PushArrayNoZero(arena, u32, arraySize);
        values       = (T *)PushArrayNoZero(arena, u8, sizeof(T) * arraySize);
        heapNum      = 0;
        maxSize      = arraySize;
    }

    int GetParent(int index) const { return index == 0 ? 0 : (index - 1) >> 1; }

    int Add(const T &element)
    {
        values[numValues] = element;

        int index                      = numValues++;
        int result                     = index;
        indices[heapNum]               = index;
        indicesIndex[indices[heapNum]] = heapNum;

        UpHeap(heapNum);

        heapNum++;
        return result;
    }

    int Pop()
    {
        if (values.empty()) return -1;

        if (indices.size() == 1)
        {
            values.size_ = 0;
            return indices[0];
        }

        // Down heap
        int index = indices[0];
        Assert(indicesIndex[index] == 0);

        indices[0]               = indices.Pop();
        indicesIndex[indices[0]] = 0;
        indicesIndex[index]      = -1;

        DownHeap(0);

        return index;
    }

    void Remove(int index)
    {
        int indexIndex = indicesIndex[index];

        Assert(values[indices[indexIndex]] < values[indices.Last()]);
        indices[indexIndex]               = indices.Pop();
        indicesIndex[indices[indexIndex]] = indexIndex;
        indicesIndex[index]               = -1;

        DownHeap(indexIndex);
    }

    void UpHeap(int startIndex)
    {
        int index  = startIndex;
        int parent = GetParent(startIndex);
        T &element = values[indices[startIndex]];

        Assert(indicesIndex.Length() == values.Length());

        while (parent != 0 && element < values[indices[parent]])
        {
            Assert(indicesIndex[indices[parent]] == parent);
            Assert(indicesIndex[indices[index]] == index);

            Swap(indices[parent], indices[index]);
            indicesIndex[indices[parent]] = parent;
            indicesIndex[indices[index]]  = index;

            index  = parent;
            parent = GetParent(index);
        }
    }

    void DownHeap(int startIndex)
    {
        T &addedVal = values[indices[startIndex]];

        int parent = startIndex;
        while (parent < indices.Length() - 1)
        {
            int left  = (parent << 1) + 1;
            int right = left + 1;
            int minIndex =
                left < indices.Length() && values[indices[left]] < addedVal ? left : parent;
            minIndex = right < indices.Length() && values[indices[right]] < addedVal
                           ? right
                           : minIndex;
            if (minIndex == parent) break;

            Assert(indicesIndex[indices[parent]] == parent);
            Assert(indicesIndex[indices[minIndex]] == minIndex);

            Swap(indices[parent], indices[minIndex]);
            indicesIndex[indices[parent]]   = parent;
            indicesIndex[indices[minIndex]] = minIndex;

            parent = minIndex;
        }
    }

    void FixHeap(int index)
    {
        int startIndex = indicesIndex[index];

        Assert(indicesIndex[indices[startIndex]] == startIndex);

        int left  = (startIndex << 1) + 1;
        int right = left + 1;

        T &value  = values[indices[startIndex]];
        T &parent = values[indices[GetParent(startIndex)]];

        if (value < parent)
        {
            UpHeap(startIndex);
        }
        else
        {
            DownHeap(startIndex);
        }
    }
};

Vec3f &MeshSimplifier::GetPosition(u32 vertexIndex)
{
    return *(Vec3f *)(attributeData + (3 + numAttributes) * vertexIndex);
}

bool MeshSimplifier::CheckInversion(const Vec3f &newPosition, u32 vertexIndex)
{
    // p0 is the vertex being replaced
    VertexGraphNode *node = &vertexNodes[vertexIndex];

    while (node->next != -1)
    {
        for (int i = node->offset; i < node->offset + node->count; i++)
        {
            u32 indexIndex0 = indexData[i];
            u32 indexIndex1 = (indexIndex0 & ~0x3) + (indexIndex0 + 1) & 3;
            u32 indexIndex2 = (indexIndex0 & ~0x3) + (indexIndex0 + 2) & 3;

            u32 vertexIndex0 = indices[indexIndex0];
            u32 vertexIndex1 = indices[indexIndex1];
            u32 vertexIndex2 = indices[indexIndex2];

            Vec3f p0 = GetPosition(vertexIndex0);
            Vec3f p1 = GetPosition(vertexIndex1);
            Vec3f p2 = GetPosition(vertexIndex2);

            Vec3f p21      = p2 - p1;
            Vec3f p01      = p0 - p1;
            Vec3f pNewEdge = newPosition - p1;

            bool result = Dot(Cross(pNewEdge, p21), Cross(p01, p21)) >= 0.f;
            if (!result) return true;
        }
    }

    return false;
}

f32 MeshSimplifier::EvaluatePair(Pair &pair, Vec3f *outP)
{
    VertexGraphNode *node = &vertexNodes[pair.index0];

    // Find the set of triangles adjacent to the pair
    u32 maxAdjTris = 0;

    int nodeIndex = pair.index0;
    while (nodeIndex != -1)
    {
        VertexGraphNode *travNode = &vertexNodes[nodeIndex];
        maxAdjTris += travNode->count;
        nodeIndex = travNode->next;
    }
    nodeIndex = pair.index1;
    while (nodeIndex != -1)
    {
        VertexGraphNode *travNode = &vertexNodes[nodeIndex];
        maxAdjTris += travNode->count;
        nodeIndex = travNode->next;
    }

    StaticArray<u32> adjTris(scratch.temp.arena, maxAdjTris);

    for (int i = 0; i < 2; i++)
    {
        nodeIndex = pair.GetIndex(i);
        while (nodeIndex != -1)
        {
            VertexGraphNode *travNode = &vertexNodes[nodeIndex];
            for (int i = travNode->offset; i < travNode->offset + travNode->count; i++)
            {
                u32 adjTri     = indexData[i] / 3;
                bool duplicate = false;
                for (int j = 0; j < adjTris.Length(); j++)
                {
                    if (adjTris[j] == adjTri)
                    {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate)
                {
                    adjTris.push_back(adjTri);
                }
            }
            nodeIndex = travNode->next;
        }
    }

    // Add triangle quadrics
    Quadric quadric(numAttributes);
    for (int i = 0; i < adjTris.Length(); i++)
    {
        quadric.Add(triangleQuadrics[adjTris[i]]);
    }

    // Add edge quadric
    Quadric edgeQuadric(0);
    for (int i = 0; i < 2; i++)
    {
        int nodeIndex = indices[pair.indexIndex0];
    }

    // TODO: handle locked edges, preserving boundary edges, rebase to new coordinate system to
    // for floating point accuracy

    Vec3f newPosition;

    f32 error = 0.f;

    // Try optimize volume
    bool valid = quadric.Optimize(newPosition, true) && !CheckInversion(newPosition);

    if (!valid)
    {
        // Try optimize wout volume
        valid = quadric.Optimize(newPosition, false) && !CheckInversion(newPosition);
    }

    if (!valid)
    {
        newPosition = (position0 + position1) / 2.f;
        valid       = CheckInversion(newPosition);
    }

    if (!valid)
    {
        error += inversionPenalty;
    }

    // Evaluate the error for the optimal position
    f32 error = quadric.Evaluate(newPosition);

    if (p) *outP = newPosition;

    return error;
}

void MeshSimplifier::Simplify(Mesh &mesh, u32 limitNumVerts, u32 limitNumTris, u32 targetError,
                              u32 limitError);
{
    ScratchArena scratch;
    // Follows section 4.1 of the quadric error paper

    // For every edge, compute the optimal contraction target and its cost

    // Packed vertex data AOS:
    // p.x p.y p.z n.x n.y n.z uv.x uv.y uv.z

    const u32 numAttributes = 6;

    vertexData =
        PushArrayNoZero(scratch.temp.arena, f32, (3 * numAttributes) * mesh.numVertices);
    int numTriangles = mesh.numIndices / 3;

    triangleQuadrics = StaticArray<Quadric>(scratch.temp.arena, numTriangles);

    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
        int index0 = 3 * triIndex + 0;
        int index1 = 3 * triIndex + 1;
        int index2 = 3 * triIndex + 2;
        Vec3f p0   = mesh.p[mesh.indices[index0]];
        Vec3f p1   = mesh.p[mesh.indices[index1]];
        Vec3f p2   = mesh.p[mesh.indices[index2]];

        f32 *attributeWeights;
        triangleQuadrics.push_back(
            Quadric(p0, p1, p2, &vertexData[(3 + numAttributes) * index0],
                    &vertexData[(3 + numAttributes) * index1],
                    &vertexData[(3 + numAttributes) * index2], attributeWeights));
    }

    // Generate graph of vertices to triangles. These point into the triangleData array.

    int numVertices   = mesh.numVertices;
    vertexNodes       = PushArray(scratch.temp.arena, VertexGraphNode, numVertices);
    vertexToPairNodes = PushArray(scratch.temp.arena, VertexGraphNode, numVertices);

    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
        for (int vertIndex = 0; vertIndex < 3; vertIndex++)
        {
            int index = mesh.indices[3 * triIndex + vertIndex];
            vertexNodes[index].count++;

            vertexToPairNodes[index].count += 2;
        }
    }

    u32 total      = 0;
    u32 totalPairs = 0;
    for (int vertIndex = 0; vertIndex < numVertices; vertIndex++)
    {
        vertexNodes[vertIndex].offset = total;
        vertexNodes[vertIndex].next   = -1;
        total += vertexNodes[vertIndex].count;

        vertexToPairNodes[vertIndex].offset = totalPairs;
        vertexToPairNodes[vertIndex].next   = -1;
        totalPairs                          = vertexToPairNodes[vertIndex].count;
    }

    indexData   = PushArray(scratch.temp.arena, u32, total);
    pairIndices = PushArray(scratch.temp.arena, u32, totalPairs);

    Heap<Pair> heap(scratch.temp.arena, totalPairs / 2);

    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
        for (int vertIndex = 0; vertIndex < 3; vertIndex++)
        {
            int indexIndex0 = 3 * triIndex + vertIndex;
            int indexIndex1 = (indexIndex0 & ~0x3) + ((indexIndex0 + 1) & 0x3);

            Pair pair;
            pair.index0 = indexIndex0;
            pair.index1 = indexIndex1;

            int vertexIndex0 = mesh.indices[indexIndex0];
            int vertexIndex1 = mesh.indices[indexIndex1];

            int pairIndex                                         = heap.Add(pair);
            pairIndices[vertexToPairNodes[vertexIndex0].offset++] = pairIndex;
            pairIndices[vertexToPairNodes[vertexIndex1].offset++] = pairIndex;

            indexData[vertexNodes[vertexIndex0].offset++] = indexIndex0;
        }
    }

    for (int vertIndex = 0; vertIndex < numVertices; vertIndex++)
    {
        vertexNodes[vertIndex].offset -= vertexNodes[vertIndex].count;

        vertexToPairNodes[vertIndex].offset -= vertexToPairNodes[vertIndex].count;
    }

    for (;;)
    {
        int pairIndex = heap.Pop();
        if (pairIndex == -1) break;

        Pair &pair = heap.values[pairIndex];

        Vec3f newPosition;
        f32 error = EvaluatePair(pair, &newPosition);

        // Move the position and change the attribute data
        int index0 = indices[pair.indexIndex0];
        int index1 = indices[pair.indexIndex0];

        GetPosition(index0)            = newPosition;
        vertexNodes[index0].next       = index1;
        vertexToPairNodes[index0].next = index1;

        // Remove duplicate pairs and reevaluate remaining pairs.
        nodeIndex = pair.index1;
        while (nodeIndex != -1)
        {
            VertexGraphNode *vertexToPairNode = vertexToPairNodes[nodeIndex];
            for (int i = 0; i < vertexToPairNode->count; i++)
            {
                u32 pairIndexIndex = vertexToPairNode->offset + i;
                if (pairIndices[pairIndexIndex] == pairIndex)
                {
                    pairIndices[pairIndexIndex] =
                        pairIndices[vertexToPairNode->offset + vertexToPairNode->count - 1];
                }
                else
                {
                    Pair &changedPair    = heap.values[pairIndices[pairIndexIndex]];
                    u32 firstVertexIndex = indices[pair.indexIndex0];

                    if (indices[changedPair.indexIndex0] == indices[pair.indexIndex1])
                        indices[changedPair.indexIndex0] = firstVertexIndex;
                    else if (changedPair.index1 == pair.index1)
                        indices[changedPair.indexIndex1] = firstVertexIndex;
                    else
                    {
                        ErrorExit(0, "Bug in graph state");
                    }

                    changedPair.error = EvaluatePair(changedPair);
                    heap.FixHeap(pairIndices[pairIndexIndex]);
                }
            }
        }

        // Remove degenerate triangles
        for (int i = 0; i < 2; i++)
        {
            int indexIndex  = pair.GetIndex(i);
            int vertexIndex = indices[indexIndex];

            while (vertexIndex != -1)
            {
                VertexGraphNode *node = &vertexNodes[vertexIndex];
                for (int j = 0; j < node->count;)
                {
                    u32 tri           = indexData[node->offset + j] / 3;
                    u32 triIndices[3] = {
                        indices[3 * tri + 0],
                        indices[3 * tri + 1],
                        indices[3 * tri + 2],
                    };

                    if (triIndices[0] == triIndices[1] || triIndices[1] == triIndices[2] ||
                        triIndices[0] == triIndices[2])
                    {
                        indexData[node->offset + j] =
                            indexData[node->offset + node->count - 1];
                        node->count--;
                    }
                    else
                    {
                        j++;
                    }
                }
                vertexIndex = node->next;
            }
        }
    }
}

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

    // mesh.
    for (;;)
    {
        RecordAOSSplits record;
        builder.BuildClusters(RecordAOSSplits & record, true);
    }
}

} // namespace rt
