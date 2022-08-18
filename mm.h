// Advanced matrix multiplication techniques
// note: all techniques are parallelized using OpenMP

#ifndef MM_H
#define MM_H

#include "arr.h" // get the Matrix struct

Matrix p_matmul(Matrix a, Matrix b); // parallel naive matmul
Matrix cache_tiled_matmul(Matrix a, Matrix b);
Matrix blas_mm(Matrix a, Matrix b); // wrapper for official OpenBLAS matmul
Matrix goto_mm(Matrix a, Matrix b); // custom implementation of the GotoBLAS matmul algorithm
Matrix vector_transpose_mm(Matrix a, Matrix b); // george hotz matmul implementation

#endif
