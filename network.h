// Neural network stuff
#include "arr.h"

#ifndef NETWORK_H
#define NETWORK_H

typedef float (*pointwise_fn)(float);
typedef struct Act {
  pointwise_fn forward_fn;
  pointwise_fn backward_fn;
} Act;

typedef struct Loss {
  float (*forward_fn)(Matrix y_pred, Matrix y_true);
  Matrix (*backward_fn)(Matrix y_pred, Matrix y_true);
} Loss;

typedef struct MLP {
  Matrix *weights;
  Matrix *biases;
  Matrix *buffers;
  Matrix *w_grads;
  Matrix *b_grads;
  Act *act_fns;
  Loss loss_fn;
  int num_layers;
} MLP;

typedef struct Updates {
  Matrix *w_updates;
  Matrix *b_updates;
} Updates;

MLP initMLP(int num_layers, int *arch, Act *act_fns, Loss loss_fn);
Matrix forward(MLP model, Matrix input);
float get_loss(MLP model, Matrix y_true);
void backward(MLP model, Matrix y_true); // backprop, setting w_grads and b_grads
void update_params(MLP model, Updates updates);
Updates initUpdates(MLP model);

#endif
