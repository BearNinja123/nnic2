#include <sys/time.h>
#include <arpa/inet.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include "network.h"
#include "arr.h"

//float init_rand(float placeholder) { return (int)(10 * (float)rand() / RAND_MAX); }
float square(float x) { return x * x; }
float f_sqrt(float x) { return (float)sqrt((double)x); }

double gettime() {
  struct timeval ret;
  gettimeofday(&ret, NULL);
  return (double)ret.tv_sec + (double)ret.tv_usec / 1000000.;
}

Matrix read_file(char *fname, int magic_number, int is_x) {
  FILE *fp = fopen(fname, "rb");
  int m_num, len, dim1, dim2;
  fread(&m_num, sizeof(int), 1, fp);
  m_num = ntohl(m_num); // swap bytes if CPU uses little endian
  assert(m_num == magic_number);

  fread(&len, sizeof(int), 1, fp);
  len = ntohl(len);

  if (is_x == 1) {
    fread(&dim1, sizeof(int), 1, fp);
    dim1 = ntohl(dim1);
    fread(&dim2, sizeof(int), 1, fp);
    dim2 = ntohl(dim2);
  }
  else {
    dim1 = dim2 = 1;
  }

  float *ret = malloc(len*dim1*dim2*sizeof(float)); // unsigned char (byte) = [0, 255]
  unsigned char *tmp = malloc(len*dim1*dim2*sizeof(char)); // unsigned char (byte) = [0, 255]
  fread(tmp, sizeof(char), len*dim1*dim2, fp);

  for (int i = 0; i < len*dim1*dim2; ++i)
    if (is_x == 1)
      ret[i] = tmp[i] / 127.5 - 1;
    else
      ret[i] = tmp[i];

  free(tmp);
  fclose(fp);

  return (Matrix){ret, len, dim1*dim2};
}

Matrix to_categorical(Matrix labels, int num_classes) {
  int m = labels.m;
  float *ret = calloc(m*num_classes, sizeof(float));

  for (int i = 0; i < m; ++i)
    ret[i*num_classes + (int)labels.w[i]] = 1.0;

  return (Matrix){ret, m, num_classes};
}

float identity(float x) { return x; }
float identity_backward(float x) { return 1; }
Act Linear = {identity, identity_backward};

float relu(float x) { return x > 0 ? x : 0; }
float relu_backward(float x) { return x > 0 ? 1 : 0; }
Act ReLU = {relu, relu_backward};

float sig(float x) { return 1 / (1 + (float)exp(-x)); }
float sig_backward(float x) { return sig(x) * (1 - sig(x)); }
Act Sigmoid = {sig, sig_backward};

float mse(Matrix y_pred, Matrix y_true) {
  Matrix diff = sub_mm(y_pred, y_true);
  float ret = mean(pointwise_inplace(diff, square));
  free(diff.w);
  return ret;
}
Matrix mse_backward(Matrix y_pred, Matrix y_true) {
  return sub_mm(y_pred, y_true);
}
Loss MSE = {mse, mse_backward};

Updates get_sgd_updates(MLP model, Updates moments, float lr, float momentum, int nag) {
  for (int layer_idx = 0; layer_idx < model.num_layers; ++layer_idx) {
    Matrix w_grad = model.w_grads[layer_idx];
    Matrix b_grad = model.b_grads[layer_idx];

    Matrix w_moment = moments.w_updates[layer_idx];
    Matrix b_moment = moments.b_updates[layer_idx];

    w_moment = axpby_inplace(momentum, w_moment, lr, w_grad);
    b_moment = axpby_inplace(momentum, b_moment, lr, b_grad);

    if (nag == 1) {
      w_moment = axpby_inplace(momentum, w_moment, lr, w_grad);
      w_moment = axpby_inplace(momentum, b_moment, lr, b_grad);
    }
  }

  return moments;
}

Updates get_adam_updates(MLP model, Updates param_updates, Updates moments, Updates vars, int *t, float lr, float b1, float b2, float eps) {
  *t += 1;
  for (int layer_idx = 0; layer_idx < model.num_layers; ++layer_idx) {
    Matrix *w_update = &param_updates.w_updates[layer_idx];
    Matrix *b_update = &param_updates.b_updates[layer_idx];

    Matrix w_grad = model.w_grads[layer_idx];
    Matrix b_grad = model.b_grads[layer_idx];

    Matrix w_moment = moments.w_updates[layer_idx];
    Matrix b_moment = moments.b_updates[layer_idx];

    Matrix w_var = vars.w_updates[layer_idx];
    Matrix b_var = vars.b_updates[layer_idx];

    w_moment = axpby_inplace(b1, w_moment, (1 - b1), w_grad);
    b_moment = axpby_inplace(b1, b_moment, (1 - b1), b_grad);

    w_var = axpby_inplace(b2, w_var, (1 - b2), pointwise_inplace(w_grad, square));
    b_var = axpby_inplace(b2, b_var, (1 - b2), pointwise_inplace(b_grad, square));

    float unbiased_lr = lr * f_sqrt(1-pow(b2,*t)) / (1-pow(b1,*t));

    Matrix sqrt_w_var = add_ms_inplace(pointwise(w_var, f_sqrt), eps);
    Matrix sqrt_b_var = add_ms_inplace(pointwise(b_var, f_sqrt), eps);

    free(w_update->w);
    free(b_update->w);
    *w_update = div_mm_inplace(mul_ms(w_moment, unbiased_lr), sqrt_w_var);
    *b_update = div_mm_inplace(mul_ms(b_moment, unbiased_lr), sqrt_b_var);
    //*w_update = mul_ms(w_moment, unbiased_lr);
    //*b_update = mul_ms(b_moment, unbiased_lr);
    free(sqrt_w_var.w);
    free(sqrt_b_var.w);
  }

  return param_updates;
}

int argmax(float *arr, int n) {
  int max_idx = 0;
  for (int i = 1; i < n; ++i) {
    if (arr[i] > arr[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}

int main(int argc, char *argv[])
{
  srand((unsigned int)(gettime() * 1000000));
  char *train_x_fname = "data/train-images-idx3-ubyte";
  char *train_y_fname = "data/train-labels-idx1-ubyte";
  char *test_x_fname = "data/t10k-images-idx3-ubyte";
  char *test_y_fname = "data/t10k-labels-idx1-ubyte";
  Matrix train_x = read_file(train_x_fname, 2051, 1);
  Matrix train_y_sparse = read_file(train_y_fname, 2049, 0);
  Matrix train_y = to_categorical(train_y_sparse, 10);
  free(train_y_sparse.w);

  Matrix test_x = read_file(test_x_fname, 2051, 1);
  Matrix test_y_sparse = read_file(test_y_fname, 2049, 0);
  Matrix test_y = to_categorical(test_y_sparse, 10);
  free(test_y_sparse.w);

  int arch[3] = {784, 300, 10};
  Act acts[2] = {ReLU, Sigmoid};
  MLP net = initMLP(2, arch, acts, MSE);
  Updates param_updates = initUpdates(net);
  Updates moments = initUpdates(net);
  Updates vars = initUpdates(net);

  int t = 0;
  int epochs = 100;
  float lr = 1e-4;
  int batch_size = 256;

  for (int epoch_idx = 0; epoch_idx < epochs; ++epoch_idx) {
    float cost = 0.0;
    int train_num_batches = train_x.m / batch_size;
    char *char_print_str = malloc(100 * sizeof(char));
    for (int i = 0; i < train_num_batches; ++i) {
      Matrix batch = (Matrix){train_x.w + batch_size*784*i, batch_size, 784};
      Matrix batch_y = (Matrix){train_y.w + batch_size*10*i, batch_size, 10};
      Matrix out = forward(net, batch);
      cost += get_loss(net, batch_y);
      sprintf(char_print_str, "epoch %d/%d, batch %d/%d, epoch loss: %f", epoch_idx+1, epochs, i+1, train_num_batches, cost / (i+1));
      printf("%s               \r", char_print_str );
      backward(net, batch_y);
      //Updates param_updates = get_sgd_updates(net, moments, 0.001, 0.0, 0);
      param_updates = get_adam_updates(net, param_updates, moments, vars, &t, lr, 0.9, 0.999, 1e-8);
      update_params(net, param_updates);
    }

    int test_num_batches = test_x.m / batch_size;
    float test_cost = 0.0;
    int num_correct = 0;
    for (int i = 0; i < test_num_batches; ++i) {
      float *batch_y_ptr = test_y.w + batch_size*10*i;
      Matrix batch = (Matrix){test_x.w + batch_size*784*i, batch_size, 784};
      Matrix batch_y = (Matrix){batch_y_ptr, batch_size, 10};
      Matrix out = forward(net, batch);
      for (int elem = 0; elem < batch_size; ++elem)
        if (argmax(out.w + elem*10, 10) == argmax(batch_y_ptr + elem*10, 10))
          num_correct += 1;
      test_cost += get_loss(net, batch_y);
    }
    printf("%s -- test cost: %f -- test accuracy: %f\n", char_print_str, test_cost/test_num_batches, (float)num_correct/(test_num_batches*batch_size));
  }
}
