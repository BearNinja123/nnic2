#ifndef ARR_H
#define ARR_H

typedef struct Matrix {
  float *w;
  int m, n;
} Matrix;

Matrix initMat(int m, int n);
Matrix nullMat();
Matrix copyMat(Matrix self);
void print_mat(Matrix self);

Matrix matmul(Matrix a, Matrix b);
Matrix transpose(Matrix self);
Matrix pointwise_inplace(Matrix self, float (*f)(float));
Matrix pointwise(Matrix self, float (*f)(float));

Matrix add_mm_inplace(Matrix a, Matrix b);
Matrix sub_mm_inplace(Matrix a, Matrix b);
Matrix mul_mm_inplace(Matrix a, Matrix b);
Matrix div_mm_inplace(Matrix a, Matrix b);
Matrix add_mm(Matrix a, Matrix b);
Matrix sub_mm(Matrix a, Matrix b);
Matrix mul_mm(Matrix a, Matrix b);
Matrix div_mm(Matrix a, Matrix b);

Matrix add_ms_inplace(Matrix a, float b);
Matrix sub_ms_inplace(Matrix a, float b);
Matrix mul_ms_inplace(Matrix a, float b);
Matrix div_ms_inplace(Matrix a, float b);
Matrix add_ms(Matrix a, float b);
Matrix sub_ms(Matrix a, float b);
Matrix mul_ms(Matrix a, float b);
Matrix div_ms(Matrix a, float b);

/* matrix-vector ops, vector is a slight misnomer (refers to a 1xN or a Mx1 matrix)
 * dim represents the dimension where the vector shape
 * is aligned with the matrix shape
 * e.g. 3x4 matrix A + 3-element vector B -> add_mv(A, B, 0)
 * e.g. 32x4 matrix A + 4-element vector B -> add_mv(A, B, 1)
 */
Matrix add_mv_inplace(Matrix a, Matrix b);
Matrix sub_mv_inplace(Matrix a, Matrix b);
Matrix mul_mv_inplace(Matrix a, Matrix b);
Matrix div_mv_inplace(Matrix a, Matrix b);
Matrix add_mv(Matrix a, Matrix b);
Matrix sub_mv(Matrix a, Matrix b);
Matrix mul_mv(Matrix a, Matrix b);
Matrix div_mv(Matrix a, Matrix b);

float sum(Matrix self);
float mean(Matrix self);
Matrix sum_1d(Matrix self, int dim);
Matrix mean_1d(Matrix self, int dim);

/* right-bottom zero-pad a matrix to a multiple of (pad_m, pad_n)
 * if pad_m or pad_n is negative, it crops the image from the right and bottom
*/
Matrix pad_matrix(Matrix a, int pad_m, int pad_n);

Matrix axpby_inplace(float a, Matrix x, float b, Matrix y); // a*X + b*Y
Matrix axpby(float a, Matrix x, float b, Matrix y); // a*X + b*Y

#endif
