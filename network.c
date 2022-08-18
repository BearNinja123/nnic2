#include <stdlib.h>
#include <math.h>
#include "arr.h"
#include "mm.h"
#include "network.h"

float init_rand(float placeholder) { return 2.0 * ((float)rand() / RAND_MAX) - 1; }
MLP initMLP(int num_layers, int *arch, Act *act_fns, Loss loss_fn) {
  Matrix *weights, *biases, *buffers, *w_grads, *b_grads;
  weights = malloc(num_layers * sizeof(Matrix));
  biases = malloc(num_layers * sizeof(Matrix));
  buffers = malloc((1 + 2*num_layers) * sizeof(Matrix)); // input, N*(W*x+b, activation)
  w_grads = malloc(num_layers * sizeof(Matrix));
  b_grads = malloc(num_layers * sizeof(Matrix));

  buffers[0] = nullMat();
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    int fan_in = arch[layer_idx];
    int fan_out = arch[layer_idx+1];
    int buffer_idx = 2 * layer_idx + 1;

    float mul_gain = sqrt(6.0 / (fan_in + fan_out));
    weights[layer_idx] = mul_ms_inplace(pointwise_inplace(initMat(fan_in, fan_out), init_rand), mul_gain);
    biases[layer_idx] = initMat(1, fan_out);

    buffers[buffer_idx] = nullMat();
    buffers[buffer_idx+1] = nullMat();

    w_grads[layer_idx] = initMat(fan_in, fan_out);
    b_grads[layer_idx] = initMat(1, fan_out);
  }

  return (MLP){weights, biases, buffers, w_grads, b_grads, act_fns, loss_fn, num_layers};
}

Matrix forward(MLP model, Matrix input) {
  int batch_size = input.m; 
  free(model.buffers[0].w);
  model.buffers[0] = copyMat(input);
  for (int layer_idx = 0; layer_idx < model.num_layers; ++layer_idx) {
    int buffer_idx = 2 * layer_idx + 1;

    free(model.buffers[buffer_idx].w);
    free(model.buffers[buffer_idx+1].w);

    Matrix prev_act = model.buffers[buffer_idx-1];
    Matrix xw_b = p_matmul(prev_act, model.weights[layer_idx]);
    xw_b = add_mv_inplace(xw_b, model.biases[layer_idx]);
    model.buffers[buffer_idx] = xw_b;
    Matrix act_xw_b = pointwise(xw_b, model.act_fns[layer_idx].forward_fn);
    model.buffers[buffer_idx+1] = act_xw_b;
  }
  
  return model.buffers[2*model.num_layers];
}

float get_loss(MLP model, Matrix y_true) {
  return model.loss_fn.forward_fn(y_true, model.buffers[2*model.num_layers]);
}

void backward(MLP model, Matrix y_true) {
  Matrix delta = model.loss_fn.backward_fn(model.buffers[2*model.num_layers], y_true);
  int batch_size = model.buffers[0].m;
  for (int layer_idx = model.num_layers-1; layer_idx >= 0; --layer_idx) {
    int buffer_idx = 2 * (layer_idx + 1);
    free(model.w_grads[layer_idx].w);
    free(model.b_grads[layer_idx].w);

    Matrix wx_b = model.buffers[buffer_idx-1];
    Matrix activation_delta = pointwise(wx_b, model.act_fns[layer_idx].backward_fn);
    delta = mul_mm_inplace(delta, activation_delta);
    free(activation_delta.w);

    Matrix b_grad = div_ms_inplace(sum_1d(delta, 0), batch_size);
    model.b_grads[layer_idx] = b_grad;

    Matrix prev_act_T = transpose(model.buffers[buffer_idx-2]);
    Matrix w_grad = div_ms_inplace(p_matmul(prev_act_T, delta), batch_size);
    model.w_grads[layer_idx] = w_grad;
    free(prev_act_T.w);

    Matrix w_T = transpose(model.weights[layer_idx]);
    Matrix new_delta = p_matmul(delta, w_T);
    free(delta.w);
    free(w_T.w);
    delta = new_delta;

  }
  free(delta.w);
}

void update_params(MLP model, Updates updates) {
  for (int layer_idx = 0; layer_idx < model.num_layers; ++layer_idx) {
    sub_mm_inplace(model.weights[layer_idx], updates.w_updates[layer_idx]);
    sub_mm_inplace(model.biases[layer_idx], updates.b_updates[layer_idx]);
  }
}

Updates initUpdates(MLP model) {
  Matrix *w_updates = malloc(model.num_layers * sizeof(Matrix));
  Matrix *b_updates = malloc(model.num_layers * sizeof(Matrix));
  for (int layer_idx = 0; layer_idx < model.num_layers; ++layer_idx) {
    int fan_in = model.weights[layer_idx].m;
    int fan_out = model.weights[layer_idx].n;
    w_updates[layer_idx] = initMat(fan_in, fan_out);
    b_updates[layer_idx] = initMat(1, fan_out);
  }

  return (Updates){w_updates, b_updates};
}
