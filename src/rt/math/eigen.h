// Copyright Contributors to the OpenEXR Project. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//
//     Redistributions in binary form must reproduce the above copyright notice, this list of
//     conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//
//     Neither the name of the copyright holder nor the names of its contributors may be used
//     to endorse or promote products derived from this software without specific prior written
//     permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef EIGEN_H_
#define EIGEN_H_

#include "../base.h"
#include "basemath.h"

namespace rt
{

namespace Eigen
{

template <int j, int k, typename T>
inline void jacobiRotateRight(T A[9], T s, T tau)
{
    for (unsigned int i = 0; i < 3; ++i)
    {
        const T nu1 = A[3 * i + j];
        const T nu2 = A[3 * i + k];
        A[3 * i + j] -= s * (nu2 + tau * nu1);
        A[3 * i + k] += s * (nu1 - tau * nu2);
    }
}

template <int j, int k, int l, typename T>
bool jacobiRotation(T A[9], T V[9], T Z[3], const T tol)
{
    // Load everything into local variables to make things easier on the
    // optimizer:
    const T x = A[3 * j + j];
    const T y = A[3 * j + k];
    const T z = A[3 * k + k];

    // The first stage diagonalizes,
    //   [ c  s ]^T [ x y ] [ c -s ]  = [ d1   0 ]
    //   [ -s c ]   [ y z ] [ s  c ]    [  0  d2 ]
    const T mu1 = z - x;
    const T mu2 = T(2) * y;

    if (Abs(mu2) <= tol * Abs(mu1))
    {
        // We've decided that the off-diagonal entries are already small
        // enough, so we'll set them to zero.  This actually appears to result
        // in smaller errors than leaving them be, possibly because it prevents
        // us from trying to do extra rotations later that we don't need.
        A[3 * j + k] = 0;
        return false;
    }
    const T rho = mu1 / mu2;
    const T t   = (rho < 0 ? T(-1) : T(1)) / (Abs(rho) + Sqrt(1 + rho * rho));
    const T c   = T(1) / Sqrt(T(1) + t * t);
    const T s   = t * c;
    const T tau = s / (T(1) + c);
    const T h   = t * y;

    // Update diagonal elements.
    Z[j] -= h;
    Z[k] += h;
    A[3 * j + j] -= h;
    A[3 * k + k] += h;

    // For the entries we just zeroed out, we'll just set them to 0, since
    // they should be 0 up to machine precision.
    A[3 * j + k] = 0;

    // We only update upper triagnular elements of A, since
    // A is supposed to be symmetric.
    T &offd1    = l < j ? A[3 * l + j] : A[3 * j + l];
    T &offd2    = l < k ? A[3 * l + k] : A[3 * k + l];
    const T nu1 = offd1;
    const T nu2 = offd2;
    offd1       = nu1 - s * (nu2 + tau * nu1);
    offd2       = nu2 + s * (nu1 - tau * nu2);

    // Apply rotation to V
    jacobiRotateRight<j, k>(V, s, tau);

    return true;
}

template <typename T>
inline T maxOffDiagSymm(const T *A, u32 size)
{
    T result = 0;
    for (unsigned int i = 0; i < size; ++i)
        for (unsigned int j = i + 1; j < size; ++j) result = Max(result, Abs(A[size * i + j]));

    return result;
}

template <typename T>
void jacobiEigenSolver(T A[9], T S[3], T V[9], const T tol)
{
    for (int i = 0; i < 3; ++i)
    {
        S[i]         = A[3 * i + i];
        V[3 * i + i] = T(0);
    }

    const int maxIter = 20; // In case we get really unlucky, prevents infinite loops
    const T absTol    = tol * maxOffDiagSymm(A, 3); // Tolerance is in terms of the maximum
    if (absTol != 0)                                // _off-diagonal_ entry.
    {
        int numIter = 0;
        do
        {
            // Z is for accumulating small changes (h) to diagonal entries
            // of A for one sweep. Adding h's directly to A might cause
            // a cancellation effect when h is relatively very small to
            // the corresponding diagonal entry of A and
            // this will increase numerical errors
            T Z[3] = {};
            ++numIter;
            bool changed = jacobiRotation<0, 1, 2>(A, V, Z, tol);
            changed      = jacobiRotation<0, 2, 1>(A, V, Z, tol) || changed;
            changed      = jacobiRotation<1, 2, 0>(A, V, Z, tol) || changed;
            // One sweep passed. Add accumulated changes (Z) to singular values (S)
            // Update diagonal elements of A for better accuracy as well.
            for (int i = 0; i < 3; ++i)
            {
                A[3 * i + i] = S[i] += Z[i];
            }
            if (!changed) break;
        } while (maxOffDiagSymm(A, 3) > absTol && numIter < maxIter);
    }
}

} // namespace Eigen

} // namespace rt

#endif
