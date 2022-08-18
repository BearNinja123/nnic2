#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "arr.h"

Matrix initMat(int m, int n) {
  float *w = (float *) calloc(m*n, sizeof(float));
  Matrix ret = {w, m, n};
  return ret;
}

Matrix nullMat() {
  return (Matrix){NULL, -1, -1};
}

Matrix copyMat(Matrix self) {
  int m, n;
  m = self.m;
  n = self.n;
  float *w = (float *) malloc(m*n*sizeof(float));
  for(int i = 0; i < m*n; ++i)
    w[i] = self.w[i];
  Matrix ret = {w, m, n};
  return ret;
}

static void print_arr(float* arr, int size) {
  printf("[");
  for (int i = 0; i < size-1; ++i)
    printf("%f, ", arr[i]);
  printf("%f]\n", arr[size-1]);
}

void print_mat(Matrix self) {
  int m, n;
  m = self.m;
  n = self.n;
  for (int i = 0; i < m*n; i += n)
    print_arr(self.w + i, n);
}

Matrix matmul(Matrix a, Matrix b) {
  int m, n, p;
  m = a.m;
  n = a.n;
  p = b.n;
  float *c = (float *) calloc(m*p, sizeof(float));
  int i, j, k;

  for (i = 0; i < m; ++i)
    for (j = 0; j < n; ++j)
      for (k = 0; k < p; ++k)
        c[i*p+k] += a.w[i*n+j] * b.w[j*p+k]; // c[i][k] (M, P) = a[i][j] (M, N) * b[j][k] (N, P)
  Matrix ret = {c, m, p};
  return ret;
}

Matrix transpose(Matrix self) {
  Matrix ret = initMat(self.n, self.m);
  for (int i = 0; i < self.n; ++i)
    for (int j = 0; j < self.m; ++j)
      ret.w[i*self.m + j] = self.w[j*self.n + i];
  return ret;
}

Matrix pointwise_inplace(Matrix self, float f(float)) {
  for (int i = 0; i < self.m*self.n; ++i)
    self.w[i] = f(self.w[i]);
  return self;
}

Matrix pointwise(Matrix self, float f(float)) {
  Matrix ret = copyMat(self);
  return pointwise_inplace(ret, f);
}

static float apply_op(float a, float b, int op) {
  switch (op) {
    case 0: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return a / b;
  }
}

/* matrix-matrix ops */
static Matrix op_mm_inplace(Matrix a, Matrix b, int op) {
  for (int i = 0; i < a.m*a.n; ++i)
    a.w[i] = apply_op(a.w[i], b.w[i], op);
  return a;
}

static Matrix op_mm(Matrix a, Matrix b, int op) {
  Matrix ret = copyMat(a);
  return op_mm_inplace(ret, b, op);
}

Matrix add_mm_inplace(Matrix a, Matrix b) { return op_mm_inplace(a, b, 0); }
Matrix sub_mm_inplace(Matrix a, Matrix b) { return op_mm_inplace(a, b, 1); }
Matrix mul_mm_inplace(Matrix a, Matrix b) { return op_mm_inplace(a, b, 2); }
Matrix div_mm_inplace(Matrix a, Matrix b) { return op_mm_inplace(a, b, 3); }
Matrix add_mm(Matrix a, Matrix b) { return op_mm(a, b, 0); }
Matrix sub_mm(Matrix a, Matrix b) { return op_mm(a, b, 1); }
Matrix mul_mm(Matrix a, Matrix b) { return op_mm(a, b, 2); }
Matrix div_mm(Matrix a, Matrix b) { return op_mm(a, b, 3); }

/* matrix-scalar ops */
static Matrix op_ms_inplace(Matrix a, float b, int op) {
  for (int i = 0; i < a.m*a.n; ++i)
    a.w[i] = apply_op(a.w[i], b, op);
  return a;
}

static Matrix op_ms(Matrix a, float b, int op) {
  Matrix ret = copyMat(a);
  return op_ms_inplace(ret, b, op);
}

Matrix add_ms_inplace(Matrix a, float b) { return op_ms_inplace(a, b, 0); }
Matrix sub_ms_inplace(Matrix a, float b) { return op_ms_inplace(a, b, 1); }
Matrix mul_ms_inplace(Matrix a, float b) { return op_ms_inplace(a, b, 2); }
Matrix div_ms_inplace(Matrix a, float b) { return op_ms_inplace(a, b, 3); }
Matrix add_ms(Matrix a, float b) { return op_ms(a, b, 0); }
Matrix sub_ms(Matrix a, float b) { return op_ms(a, b, 1); }
Matrix mul_ms(Matrix a, float b) { return op_ms(a, b, 2); }
Matrix div_ms(Matrix a, float b) { return op_ms(a, b, 3); }

/* matrix-vector ops */
static Matrix op_mv_inplace(Matrix a, Matrix b, int op) {
  int m = a.m;
  int n = a.n;

  int i, j;
  int *vec_idx;
  if (b.m == m)
    vec_idx = &i;
  else
    vec_idx = &j;

  int idx = 0;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      a.w[idx] = apply_op(a.w[idx], b.w[*vec_idx], op);
      ++idx;
    }
  }

  return a;
}

static Matrix op_mv(Matrix a, Matrix b, int op) {
  Matrix ret = copyMat(a);
  return op_mv_inplace(ret, b, op);
}

Matrix add_mv_inplace(Matrix a, Matrix b) { return op_mv_inplace(a, b, 0); }
Matrix sub_mv_inplace(Matrix a, Matrix b) { return op_mv_inplace(a, b, 1); }
Matrix mul_mv_inplace(Matrix a, Matrix b) { return op_mv_inplace(a, b, 2); }
Matrix div_mv_inplace(Matrix a, Matrix b) { return op_mv_inplace(a, b, 3); }
Matrix add_mv(Matrix a, Matrix b) { return op_mv(a, b, 0); }
Matrix sub_mv(Matrix a, Matrix b) { return op_mv(a, b, 1); }
Matrix mul_mv(Matrix a, Matrix b) { return op_mv(a, b, 2); }
Matrix div_mv(Matrix a, Matrix b) { return op_mv(a, b, 3); }

float sum(Matrix self) {
  float ret = 0;
  for (int i = 0; i < self.m*self.n; ++i)
    ret += self.w[i];
  return ret;
}

float mean(Matrix self) {
  return sum(self) / (self.m * self.n);
}

Matrix sum_1d(Matrix self, int dim) {
  int m = self.m;
  int n = self.n;

  Matrix ret;
  int i, j;
  int *vec_idx;
  if (dim == 0) {
    ret = initMat(1, n);
    vec_idx = &j;
  }
  else {
    ret = initMat(m, 1);
    vec_idx = &i;
  }

  int idx = 0;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      ret.w[*vec_idx] += self.w[idx];
      ++idx;
    }
  }

  return ret;
}

Matrix mean_1d(Matrix self, int dim) {
  Matrix matrix_sum = sum_1d(self, dim);
  float div_gain;
  if (dim == 0)
    div_gain = (int)self.m;
  else
    div_gain = (int)self.n;
  return div_ms_inplace(matrix_sum, div_gain);
}

Matrix pad_matrix(Matrix a, int pad_m, int pad_n) {
  if (pad_m == 0 && pad_n == 0)
    return copyMat(a);
  int m = a.m;
  int n = a.n;
  int new_m = m+pad_m;
  int new_n = n+pad_n;
  float *c = (float *) malloc(new_m*new_n*sizeof(float));
  int min_m = fmin(m, new_m); // if new_m < m, don't want to overflow return matrix
  int min_n = fmin(n, new_n);

  for (int i = 0; i < min_m; ++i) {
    for (int j = 0; j < min_n; ++j)
      c[i*new_n+j] = a.w[i*n+j];
    for (int j = n; j < new_n; ++j) // if matrix crop, this loop does not run
      c[i*new_n+j] = 0;
  }

  for (int i = m; i < new_m; ++i) // if matrix crop, this loop does not run
    for (int j = 0; j < new_n; ++j)
      c[i*new_n+j] = 0;

  Matrix ret = {c, new_m, new_n};
  return ret;
}

Matrix axpby_inplace(float a, Matrix A, float b, Matrix B) {
  for (int i = 0; i < A.m*A.n; ++i)
      A.w[i] = a*A.w[i] + b*B.w[i];
  return A;
}

Matrix axpby(float a, Matrix A, float b, Matrix B) {
  Matrix ret = copyMat(A);
  return axpby_inplace(a, ret, b, B);
}
