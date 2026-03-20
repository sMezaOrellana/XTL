#include "tensor_internal.h"
#include <stdlib.h>
#include <stdio.h>

int tensor_reshape(Tensor* a, int* new_shape, int new_ndim) {
  if (!a->contiguous) {
    fprintf(stderr, "tensor_reshape: tensor is not contiguous\n");
    return -1;
  }

  int new_size = 1;
  for (int i = 0; i < new_ndim; i++)
    new_size *= new_shape[i];

  if (a->size != new_size) {
    fprintf(stderr, "tensor_reshape: size mismatch (%d vs %d)\n", a->size, new_size);
    return -1;
  }

  int* new_shape_buf = malloc(sizeof(int) * new_ndim);
  if (!new_shape_buf) {
    fprintf(stderr, "tensor_reshape: malloc failed for shape\n");
    return -1;
  }

  int* new_strides_buf = malloc(sizeof(int) * new_ndim);
  if (!new_strides_buf) {
    fprintf(stderr, "tensor_reshape: malloc failed for strides\n");
    free(new_shape_buf);
    return -1;
  }

  free(a->shape);
  a->shape = new_shape_buf;
  for (int i = 0; i < new_ndim; i++)
    a->shape[i] = new_shape[i];

  free(a->strides);
  a->strides = new_strides_buf;

  a->ndim = new_ndim;

  if (new_ndim > 0) {
    a->strides[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--)
      a->strides[i] = a->shape[i + 1] * a->strides[i + 1];
  }

  return 0;
}

void tensor_transpose(Tensor* a) {
  if (a->ndim < 2) {
    fprintf(stderr, "tensor_transpose: ndim must be >= 2 (got %d)\n", a->ndim);
    return;
  }
  int ndim = a->ndim;
  int last = a->shape[ndim - 1];
  int second_to_last = a->shape[ndim - 2];

  a->shape[ndim - 1] = second_to_last;
  a->shape[ndim - 2] = last;
  
  int x = a->strides[ndim - 1];
  a->strides[ndim - 1] = a->strides[ndim - 2];
  a->strides[ndim - 2] = x;

  a->contiguous = false;
}

Tensor* tensor_contiguous(Tensor* a) {

  Tensor* out = create_tensor_zeros(a->shape, a->ndim);
  if (!out) return NULL;

  for (int i = 0; i < out->size; i++) {
    int offset_a = 0;
    int remaining = i;
    for (int d = out->ndim - 1; d >= 0; d--) {
      int coord = remaining % out->shape[d];
      remaining /= out->shape[d];
      offset_a += coord * a->strides[d];
    }
    out->data[i] = a->data[offset_a];
  }

  return out;
}
