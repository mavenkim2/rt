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

// https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf Quadric error
// https://www.cs.cmu.edu/~garland/Papers/quadric2.pdf Quadric error w/ attributes
f32 Quadric::Evaluate(const Vec3f &p)
{
    // The error is: vt * K * v where
    // v is the column vector position of the vertex
    // K is the outer product matrix of the plane vector p = [a b c d] for
    // plane ax + by + cz + d = 0. The Kp for all the triangles planes at a vertex is simply
    // the sum of the matrices.

    // Simplified calculation:
    // vt * K * v =
    // vt * A * v + 2 * nd * v + d2

    f32 x = Dot(Vec3f(nxx, nxy, nxz), p);
    f32 y = Dot(Vec3f(nxx, nxy, nxz), p);
    f32 z = Dot(Vec3f(nxx, nxy, nxz), p);

    f32 error = Dot(Vec3f(x, y, z) + 2 * dn, p) + d2;
    return error;
}

template <u32 numAttributes>
QuadricAttr::QuadricAttr(const Vec3f &p0, const Vec3f &p1, const Vec3f &p2,
                         f32 *__restrict attr0, f32 *__restrict attr1, f32 *__restrict attr2,
                         f32 *__restrict attributeWeights)
{
    Vec3f p01 = p1 - p0;
    Vec3f p02 = p2 - p0;

    Vec3f n = Cross(p01, p02);

    gVol = n;
    dVol = -Dot(gVol, p0);

    f32 length = Length(n);
    f32 area   = 0.5f * length;

    if (length < 1e-8f)
    {
        return;
    }

    n /= length;

    a2 = Sqr(n.x);
    ab = n.x * n.y;
    ac = n.x * n.z;

    b2 = Sqr(n.y);
    bc = n.y * n.z;

    c2 = Sqr(n.z);

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

        if (isInvertible)
        {
            LUPSolve(M, pivots, b, 3, grad.e);
            f32 residual[] = {
                b[0] - Dot(grad, p01),
                b[1] - Dot(grad, p02),
                b[2] - Dot(grad, n),
            };

            Vec3f error;
            LUPSolve(M, pivots, residual, 3, error.e);
            grad += error;
        }

        gradients[i] = grad;
        d[i]         = a0 - Dot(grad, p);

        a2 += Sqr(g[i].x);
        ab += g[i].x * g[i].y;
        ac += g[i].x * g[i].z;

        b2 += Sqr(g[i].y);
        bc += g[i].y * g[i].z;

        c2 += Sqr(g[i].z);

        dn += d[i] * grad;
        d2 += Sqr(d[i]);
    }
}

template <u32 numAttributes>
bool QuadricAttr<numAttributes>::OptimizeVolume(Vec3f &p)
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
        BBt00 += Sqr(gradients[i].x);
        BBt01 += gradients[i].x * gradients[i].y;
        BBt02 += gradients[i].x * gradients[i].z;

        BBt11 += Sqr(gradients[i].y);
        BBt12 += gradients[i].y * gradients[i].z;

        BBt22 += Sqr(gradients[i].z);

        b1 += gradients[i] * d[i];
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
    return false;
}

template <u32 numAttributes>
f32 QuadricAttr<numAttributes>::Evaluate(const Vec3f &p, f32 *__restrict attributes,
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

    f32 x = Dot(Vec3f(nxx, nxy, nxz), p);
    f32 y = Dot(Vec3f(nxx, nxy, nxz), p);
    f32 z = Dot(Vec3f(nxx, nxy, nxz), p);

    f32 error = Dot(Vec3f(x, y, z) + 2 * dn, p) + d2;

    for (int i = 0; i < numAttributes; i++)
    {
        f32 attributeVal = Dot(Vec3f(gradients[i]), p) + d[i];
        f32 s            = attributeVal / area;

        error += s * (a * s - 2 * attributeVal);

        attributes[i] = attributeVal / attributeWeight;
    }

    return error;
}

template <typename T>
struct Heap
{
    StaticArray<T> array;

    Heap(Arena *arena, u32 arraySize) { array = StaticArray<T>(arena, arraySize); }

    int GetParent(int index) const { return (index - 1) >> 1; }

    void Add(const T &element)
    {
        array.push_back(element);
        int index  = array.size() - 1;
        int parent = GetParent(index);
        while (parent != 0 && element < array[parent])
        {
            Swap(array[parent], array[index]);
            index  = parent;
            parent = GetParent(index);
        }
    }

    bool Pop(T &element)
    {
        if (array.empty()) return false;

        if (array.size() == 1)
        {
            element = array.back();
            array.pop_back();
            return true;
        }

        // Down heap
        element  = array[0];
        array[0] = array.back();
        array.pop_back();

        int parent = 0;
        while (parent < array.Length() - 1)
        {
            int left  = (parent << 1) + 1;
            int right = left + 1;
            int minIndex =
                left < array.Length() && array[left] < array[parent] ? left : parent;
            minIndex =
                right < array.Length() && array[right] < array[minIndex] ? right : minIndex;
            if (minIndex == parent) break;

            Swap(array[parent], array[minIndex]);
            parent = minIndex;
        }

        return true;
    }
};

Vec3f MeshSimplifier::GetPosition(u32 vertexIndex)
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

void MeshSimplifier::Simplify(Mesh &mesh)
{
    ScratchArena scratch;
    // Follows section 4.1 of the quadric error paper

    struct Pair
    {
        int index0;
        int index1;

        float error;
        bool operator<(const Pair &other) { return error < other.error; }
        u32 GetIndex(u32 index)
        {
            Assert(index < 2);
            return index == 0 ? index0 : index1;
        }
    };

    // For every edge, compute the optimal contraction target and its cost

    // Packed vertex data AOS:
    // p.x p.y p.z n.x n.y n.z uv.x uv.y uv.z

    const u32 numAttributes = 6;

    f32 *vertexData =
        PushArrayNoZero(scratch.temp.arena, f32, (3 * numAttributes) * mesh.numVertices);
    int numTriangles = mesh.numIndices / 3;

    StaticArray<QuadricAttrb<6>> triangleQuadrics(scratch.temp.arena, numTriangles);

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
            QuadricAttr(p0, p1, p2, &vertexData[(3 + numAttributes) * index0],
                        &vertexData[(3 + numAttributes) * index1],
                        &vertexData[(3 + numAttributes) * index2], attributeWeights));
    }

    // Generate graph of vertices to triangles. These point into the triangleData array.

    int numVertices = mesh.numVertices;
    vertexNodes     = PushArray(scratch.temp.arena, VertexGraphNode, numVertices);

    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
        for (int vertIndex = 0; vertIndex < 3; vertIndex++)
        {
            int index = mesh.indices[3 * triIndex + vertIndex];
            vertexNodes[index].count++;
        }
    }

    u32 total = 0;
    for (int vertIndex = 0; vertIndex < numVertices; vertIndex++)
    {
        vertexNodes[vertIndex].offset = total;
        vertexNodes[vertIndex].next   = -1;
        total += vertexNodes[vertIndex].count;
    }

    indexData = PushArray(scratch.temp.arena, u32, total);

    for (int triIndex = 0; triIndex < numTriangles; triIndex++)
    {
        for (int vertIndex = 0; vertIndex < 3; vertIndex++)
        {
            int indexIndex                               = 3 * triIndex + vertIndex;
            int vertexIndex                              = mesh.indices[indexIndex];
            indexData[vertexNodes[vertexIndex].offset++] = indexIndex;
        }
    }

    for (int vertIndex = 0; vertIndex < numVertices; vertIndex++)
    {
        vertexNodes[index].offset -= vertexNodes[index].count;
    }

    Heap<Pair> heap;

    for (;;)
    {
        Pair pair;
        bool remaining = heap.Pop(pair);
        if (!remaining) break;

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

        for (int pairIndex = 0; pairIndex < 2; pairIndex++)
        {
            nodeIndex = pair.GetIndex(pairIndex);
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

        // Add quadrics together
        QuadricAttr<6> quadric;
        for (int i = 0; i < adjTris.Length(); i++)
        {
            quadric.Add(triangleQuadrics[adjTris[i]]);
        }

        // TODO: handle locked edges, preserving boundary edges, inversion prevention,
        // rebase to new coordinate system to for floating point accuracy
        Vec3f newPosition;

        // Try optimize volume
        bool valid = quadric.OptimizeVolume(newPosition) && !CheckInversion(newPosition);

        if (!valid)
        {
            // Try optimize linear
            valid = quadric.OptimizeLinear(newPosition) && !CheckInversion(newPosition);
        }

        if (!valid)
        {
            newPosition = (position0 + position1) / 2.f;
            valid       = CheckInversion(newPosition);
        }

        if (!valid)
        {
        }
    }
}

} // namespace rt
