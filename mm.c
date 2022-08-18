#include <stdlib.h>
#include <math.h>
#include "arr.h"
#include "mm.h"


Matrix p_matmul(Matrix a, Matrix b) {
  int m, n, p;
  m = a.m;
  n = a.n;
  p = b.n;
  float *c = (float *) calloc(m*p, sizeof(float));
  int i, j, k;

#pragma omp parallel for
  for (i = 0; i < m; ++i)
    for (j = 0; j < n; ++j)
      for (k = 0; k < p; ++k)
        c[i*p+k] += a.w[i*n+j] * b.w[j*p+k]; // c[i][k] (M, P) = a[i][j] (M, N) * b[j][k] (N, P)
  Matrix ret = {c, m, p};
  return ret;
}

Matrix cache_tiled_matmul(Matrix a, Matrix b) {
  int block_size = 32;
  int m, n, p;
  m = a.m;
  n = a.n;
  p = b.n;
  float *c = (float *) calloc(m*p, sizeof(float));
  int i, j, k, i2, j2; // i/j/k loop over whole matrix, i2/j2/k2 loop over cache tile
  int i_block_max = block_size*(m/block_size);
  int j_block_max = block_size*(n/block_size);

  /* Main cache-tiled matrix multiplication */
#pragma omp parallel for
  for (i = 0; i < i_block_max; i += block_size)
    for (j = 0; j < j_block_max; j += block_size)

      for (i2 = i; i2 < i+block_size; ++i2)
        for (j2 = j; j2 < j+block_size; ++j2)

          for (k = 0; k < p; ++k)
              c[i2*p + k] += a.w[i2*n + j2] * b.w[j2*p + k];

  /* rightmost-columns of A matmuled with lowest rows of B */
#pragma omp parallel for
  for (i = 0; i < m; ++i)
    for (j = j_block_max; j < n; ++j)
      for (k = 0; k < p; ++k)
        c[i*p + k] += a.w[i*n + j] * b.w[j*p + k];

  /* lowest rows of A matmuled with upper blocks of B */
#pragma omp parallel for
  for (i = i_block_max; i < m; ++i)
    for (j = 0; j < j_block_max; ++j)
      for (k = 0; k < p; ++k)
        c[i*p + k] += a.w[i*n + j] * b.w[j*p + k];

  Matrix ret = {c, m, p};
  return ret;
}

#ifdef HAVE_BLAS
#include <cblas.h>
Matrix blas_mm(Matrix a, Matrix b) {
  /*
  void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
  */
  int m, n, p;
  m = a.m;
  n = a.n;
  p = b.n;

  Matrix ret = initMat(m, p);
  cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, p, n,
      1,
      a.w, n,
      b.w, p,
      0,
      ret.w, p
  );

  return ret;
}
#endif

#ifdef __AVX__
#include <immintrin.h>
#define VECTOR_LEN 8 // 8 floats per vector register if CPU supports AVX (256 bits)
#define V_DTYPE __m256
#define _MM_SET1_PS _mm256_set1_ps
#define _MM_SETZERO_PS _mm256_setzero_ps
#define _MM_LOADU_PS _mm256_loadu_ps
#define _MM_STOREU_PS _mm256_storeu_ps

#elif __SSE__
#include <xmmintrin.h>
#define VECTOR_LEN 4 // 4 if CPU supports SSE (128 bits)
#define V_DTYPE __m128
#define _MM_SET1_PS _mm_set1_ps
#define _MM_SETZERO_PS _mm_setzero_ps
#define _MM_LOADU_PS _mm_loadu_ps
#define _MM_STOREU_PS _mm_storeu_ps
#endif

// GotoBLAS implementation

static float * pack_a(Matrix a, int start_i, int start_j, int height, int width, int block_y) {
  int ld = a.n; // ld = leading dimension, a[i*lda + j] = a[i][j]
  float *ret = (float *) malloc(height*width*sizeof(float));

  for (int i = 0; i < height/block_y; ++i)
    for (int j = 0; j < width; ++j)
      for (int i2 = 0; i2 < block_y; ++i2) {
        int unpacked_idx = (start_i + i*block_y + i2)*ld + (start_j+j); // a[si+i*by+i2][sj+j]
        int packed_idx = i*(block_y*width) + j*block_y + i2; // a_packed[i][j][i2]
        ret[packed_idx] = a.w[unpacked_idx];
      }

  return ret;
}

static void pack_b(Matrix b, int start_i, int start_j, int height, int width, int block_x, V_DTYPE ret[height*width]) {
  int ld = b.n / VECTOR_LEN;

  for (int j = 0; j < width/block_x; ++j)
    for (int i = 0; i < height; ++i)
      for (int j2 = 0; j2 < block_x; ++j2) {
        int unpacked_idx = (start_i+i)*ld + (start_j + j*block_x + j2); // b[si+i][sj+j*bx+j2]
        int packed_idx = j*(block_x*height) + i*block_x + j2; // b_packed[j][i][j2]
        V_DTYPE v = _MM_LOADU_PS(b.w + VECTOR_LEN*unpacked_idx);
        ret[packed_idx] = v;
      }
}

Matrix goto_mm(Matrix a, Matrix b) {
  const int M_R = 4; // size of m-block (R stands for register)
  const int N_R = 2; // n-block
  const int M_C_DEFAULT = 16; // m-block (C stands for cache)
  const int K_C_DEFAULT = 256; // k-block
  const int N_C_DEFAULT = 64; // n-block

  // Determine matrix padding and change cached m/k/n blocks to maximize performance
  int pad_m = a.m % M_R == 0 ? 0 : M_R - (a.m % M_R); // make sure padded A.m is divisible by M_R
  int M_C = fmin(a.m + pad_m, M_C_DEFAULT); // set I block to be equal to A.m if it is smaller than the default value
  pad_m = (a.m + pad_m) % M_C == 0 ? pad_m : pad_m + M_C - ((a.m + pad_m) % M_C); // update the pad value if A.m isn't divisible by I block value

  int K_C = fmin(a.n, K_C_DEFAULT);
  int pad_n = a.n % K_C == 0 ? 0 : K_C - (a.n % K_C);

  int p_div = N_R * VECTOR_LEN; // b.n (N) must be divisible by n_div
  int pad_p = b.n % p_div == 0 ? 0 : p_div - (b.n % p_div);
  int N_C = fmin((b.n + pad_p)/VECTOR_LEN, N_C_DEFAULT);
  pad_p = (b.n + pad_p) % N_C == 0 ? pad_p : pad_p + N_C - VECTOR_LEN*((b.n + pad_p)/VECTOR_LEN % N_C);

  Matrix a_padded, b_padded;

  if (pad_m + pad_n == 0) // avoid unnecessary matrix copying if no padding is needed
    a_padded = a;
  else
    a_padded = pad_matrix(a, pad_m, pad_n);

  if (pad_n + pad_p == 0)
    b_padded = b;
  else
    b_padded = pad_matrix(b, pad_n, pad_p);

  int M = a.m + pad_m;
  int N = a.n + pad_n;
  int P = b.n + pad_p;

  float *c = (float *) calloc(M*P, sizeof(float));

#pragma omp parallel for
  for (int j = 0; j < P/VECTOR_LEN; j += N_C) {
    // get B[j]
    for (int k = 0; k < N; k += K_C) {
      // get A[k], B[j][k], pack B[j][k] -> Bp
      V_DTYPE bp[K_C*N_C]; // segfault when B_packed is a V_DTYPE pointer (i think from unaligned memory)
      pack_b(b_padded, k, j, K_C, N_C, N_R, bp);
      for (int i = 0; i < M; i += M_C) {
        // get and pack A[k][i] -> Ap
        float *ap = pack_a(a_padded, i, k, M_C, K_C, M_R);

        for (int j2 = 0; j2 < N_C; j2 += N_R) {
          // get Bp[j2]
          for (int i2 = 0; i2 < M_C; i2 += M_R) {
            // get Ap[i2]

            V_DTYPE accum[M_R][N_R];
            for (int acc_idx_i = 0; acc_idx_i < M_R; ++acc_idx_i) // zero out accumulation vectors
              for (int acc_idx_j = 0; acc_idx_j < N_R; ++acc_idx_j)
                accum[acc_idx_i][acc_idx_j] = _MM_SETZERO_PS();

            //microkernel
            for (int k2 = 0; k2 < K_C; ++k2) {
              for (int i3 = 0; i3 < M_R; ++i3) {
                // get Ap[i2][k2][i3] -> av (A vector)
                V_DTYPE av = _MM_SET1_PS(ap[i2*K_C + k2*M_R + i3]);
                for (int j3 = 0; j3 < N_R; ++j3) {
                  // get Bp[j2][k2][j3] -> bv (B vector)
                  V_DTYPE bv = bp[j2*K_C + k2*N_R + j3];
                  accum[i3][j3] += av * bv;
                }
              }
            }

            for (int i3 = 0; i3 < M_R; i3++)
              for (int j3 = 0; j3 < N_R; j3++) {
                float *c_idx = c + (i+i2+i3)*P + VECTOR_LEN*(j+j2+j3);
                V_DTYPE c_vec = _MM_LOADU_PS(c_idx);
                c_vec += accum[i3][j3];
                _MM_STOREU_PS(c_idx, c_vec);
              }
          }
        }
        free(ap);
      }
    }
  }

  Matrix ret = {c, M, P};

  if (pad_m + pad_n + pad_p == 0)
    return ret;

  Matrix ret_cropped = pad_matrix(ret, -pad_m, -pad_p);

  free(ret.w);
  if (pad_m + pad_n != 0)
    free(a_padded.w);
  if (pad_n + pad_p != 0)
    free(b_padded.w);

  return ret_cropped;
}

// Vector-transposed implementation

/*
   * Reshapes matrix such that the
   * result returns the transpose of the matrix 
   * if contiguous sections of VECTOR_LEN elements
   * were treated as one element.
   *
   * Fused multiply-add operations can
   * be accumulated into a sum to be placed in C without 
   * reducement of vectors into scalars (i.e. sum of {1, 2, 3, 4} = 10).
   * Offers the best of non-transposed and transposed matmul.
   * 
   * For 4x4 matrix with vector size 2        (not cache-friendly v0-v3 because columns)
   * 0 1 2 3                                  [ v0 ][ v4 ]
   * 4 5 6 7    -(vector representation)->    [ v1 ][ v5 ]
   * 8 9 a b                                  [ v2 ][ v6 ]
   * c d e f                                  [ v3 ][ v7 ]
   *
   * and the vector-transposed matrix would be:
   * 0 1 4 5 8 9 c d -> [ v0 ][ v1 ][ v2 ][ v3 ] (cache-friendly v0-v3!)
   * 2 3 4 5 a b e f    [ v4 ][ v5 ][ v6 ][ v7 ]
 */
static float * vector_transpose(Matrix self) {
  int m = self.m;
  int n = self.n;
  float *w = self.w;
  float *ret = (float *) malloc(m*n*sizeof(float));

  for (int j = 0; j < n/VECTOR_LEN; ++j)
    for (int i = 0; i < m; ++i)
      for (int j2 = 0; j2 < VECTOR_LEN; ++j2)
        // ret.shape = (N/VECTOR_LEN, M) where each element of ret is a vector of length VECTOR_LEN
        ret[j*(m*VECTOR_LEN) + i*VECTOR_LEN + j2] = w[i*n + j*VECTOR_LEN + j2];

  return ret;
}

Matrix vector_transpose_mm(Matrix a, Matrix b) {
  // block configuration
  // each superblock contains (SUPERBLOCK_I, SUPERBLOCK_J) blocks 
  // each block will output (BLOCK_I, BLOCK_J) vectors in the return matrix
  const int SUPERBLOCK_I = 4;
  const int SUPERBLOCK_J = 4;
  const int BLOCK_I = 8;
  const int BLOCK_J = 2;

  int m_div = SUPERBLOCK_I * BLOCK_I; 
  int pad_m = a.m % m_div == 0 ? 0 : m_div - (a.m % m_div); // make sure padded A.m is divisible by BY
  int p_div = SUPERBLOCK_J * BLOCK_J * VECTOR_LEN; 
  int pad_p = b.n % p_div == 0 ? 0 : p_div - (b.n % p_div); // make sure padded A.m is divisible by BY

  // Pad matrix A and B to make sure they can be processed properly
  float *aw, *bw;
  if (pad_m == 0) { // avoid unnecessary matrix copying if matrix doesn't need to be padded
    aw = a.w;
  }
  else {
    Matrix a_padded = pad_matrix(a, pad_m, 0); // pad only M dimension
    aw = a_padded.w;
  }

  if (pad_p == 0) {
    bw = vector_transpose(b); // reshaped B, bw.shape = (P/VECTOR_LEN, N) where each element has length VECTOR_LEN
  }
  else {
    Matrix b_padded = pad_matrix(b, 0, pad_p); // pad only N dimension
    bw = vector_transpose(b_padded); // reshaped B, bw.shape = (P/VECTOR_LEN, N) where each element has length VECTOR_LEN
    free(b_padded.w);
  }

  int M = a.m + pad_m;
  int N = a.n;
  int P = b.n + pad_p;

  float *c = (float *) malloc(M*P*sizeof(float)); // c.shape = (M, P) or (M, P/VECTOR_LEN) when elements are viewed as vectors for contiguous elements

  // Matrix multiplication loops
#pragma omp parallel for
  for (int i = 0; i < M; i += SUPERBLOCK_I*BLOCK_I) { // loop over row super-blocks of A
    for (int j = 0; j < P/VECTOR_LEN; j += SUPERBLOCK_J*BLOCK_J) { // loop over row super-blocks of reshaped B
                                                      
      for (int i2 = 0; i2 < SUPERBLOCK_I*BLOCK_I; i2 += BLOCK_I) { // loop over row blocks of A
        for (int j2 = 0; j2 < SUPERBLOCK_J*BLOCK_J; j2 += BLOCK_J) { // loop over row blocks of reshaped B
                                                                     
          // Fused-multiply add loop
          V_DTYPE accum[BLOCK_I][BLOCK_J];
          for (int acc_idx = 0; acc_idx < BLOCK_I*BLOCK_J; ++acc_idx) // zero out accumulation vectors
            accum[acc_idx/BLOCK_J][acc_idx%BLOCK_J] = _MM_SETZERO_PS();

          for (int k = 0; k < N; ++k) { // k = axis of reduction (k associates to N if A.shape=(M,N) and B.shape=(N,P))
            for (int i3 = 0; i3 < BLOCK_I; ++i3) { // loop over individual rows of A
              V_DTYPE av = _MM_SET1_PS(aw[(i+i2+i3)*N + k]); // a[i][k]
              for (int j3 = 0; j3 < BLOCK_J; ++j3) { // loop over individual rows of B
                int vector_idx = (j+j2+j3)*N + k; // vector at bw[j][k]
                V_DTYPE bv = _MM_LOADU_PS(bw + VECTOR_LEN*vector_idx); // each vector is size VECTOR_LEN so scale vector_idx accordingly
                accum[i3][j3] += av * bv; // equivalent to c[i][k] += a[i][j] * b[j:j+VECTOR_LEN][k] for naive matmul algorithm
              }
            }
          }

          // Assign accumulated values into return matrix
          for (int i3 = 0; i3 < BLOCK_I; ++i3)
            for (int j3 = 0; j3 < BLOCK_J; ++j3)
              _MM_STOREU_PS(c + ((i+i2+i3)*P + VECTOR_LEN*(j+j2+j3)), accum[i3][j3]); // store vector at c[i][j]
        }
      }
    }
  }

  Matrix ret = {c, M, P};

  free(bw); // bw must always be freed since it was generated within the function
  if (pad_m + pad_p == 0)
    return ret;

  if (pad_m != 0)
    free(aw); // only free aw if it was from a padded matrix

  Matrix ret_cropped = pad_matrix(ret, -pad_m, -pad_p);
  free(ret.w);
  return ret_cropped;
}
