// Stand-in for the parts of MKL that DiskANN includes, so that its in-memory index builds without MKL.
// DiskANN calls MKL only for product quantization, k-means and the disk index (pq.cpp, math_utils.cpp,
// disk_utils.cpp), which the in-memory Vamana build, insert, delete and consolidate never reach. pq.cpp and
// math_utils.cpp are still linked through the PQ data store, so the routines exist: plain but correct, except
// LAPACKE_sgesdd, which stops the program.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

typedef long long MKL_INT;

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;

#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

inline float cblas_snrm2(const MKL_INT count, const float* values, const MKL_INT stride) {
    double sum = 0.0;
    for (MKL_INT index = 0; index < count; index++) {
        sum += double(values[index * stride]) * values[index * stride];
    }
    return float(std::sqrt(sum));
}

// row-major only, as DiskANN calls it
inline void cblas_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transpose_a, const CBLAS_TRANSPOSE transpose_b,
                        const MKL_INT rows, const MKL_INT columns, const MKL_INT inner, const float alpha,
                        const float* matrix_a, const MKL_INT lead_a, const float* matrix_b, const MKL_INT lead_b,
                        const float beta, float* matrix_c, const MKL_INT lead_c) {
    if (layout != CblasRowMajor) {
        std::fprintf(stderr, "mkl.h stand-in: cblas_sgemm supports row-major only\n");
        std::abort();
    }
    for (MKL_INT row = 0; row < rows; row++) {
        for (MKL_INT column = 0; column < columns; column++) {
            double sum = 0.0;
            for (MKL_INT position = 0; position < inner; position++) {
                const float left = transpose_a == CblasNoTrans ? matrix_a[row * lead_a + position]
                                                               : matrix_a[position * lead_a + row];
                const float right = transpose_b == CblasNoTrans ? matrix_b[position * lead_b + column]
                                                                : matrix_b[column * lead_b + position];
                sum += double(left) * right;
            }
            float& target = matrix_c[row * lead_c + column];
            target = float(alpha * sum + (beta == 0.0f ? 0.0 : double(beta) * target));
        }
    }
}

inline MKL_INT LAPACKE_sgesdd(int, char, MKL_INT, MKL_INT, float*, MKL_INT, float*, float*, MKL_INT, float*, MKL_INT) {
    std::fprintf(stderr, "mkl.h stand-in: LAPACKE_sgesdd (OPQ) is not available\n");
    std::abort();
}
