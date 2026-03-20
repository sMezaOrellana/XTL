#include "tensor_internal.h"

/* Create a tensor with all elements set to 1.0. */
Tensor* create_tensor_ones(int* shape, int ndim) {
  Tensor* t = init_tensor(shape, ndim);
  if (!t) return NULL;

  fill_tensor(t, 1.0f, t->size);
  return t;
}

/* Create a tensor with all elements set to 0.0. */
Tensor* create_tensor_zeros(int* shape, int ndim) {
  Tensor* t = init_tensor(shape, ndim);
  if (!t) return NULL;

  fill_tensor(t, 0.0f, t->size);
  return t;
}

/* Create a tensor with all elements set to a uniform random value in [0, 1]. */
Tensor* create_tensor_rand(int* shape, int ndim) {
  Tensor* t = init_tensor(shape, ndim);
  if (!t) return NULL;

  fill_tensor_random(t, 0.0f, 1.0f, t->size);
  return t;
}

/* Create a tensor by copying from a flat float array (row-major order). */
Tensor* tensor_from_data(float* data, int* shape, int ndim) {
  Tensor* t = init_tensor(shape, ndim);
  if (!t) return NULL;

  for (int i = 0; i < t->size; i++)
    t->data[i] = data[i];

  return t;
}
