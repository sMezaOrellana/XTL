#include <stdio.h>
#include <stdlib.h>
#include "tensor_internal.h"

/* Allocate and initialise a Tensor: copies shape, allocates data and strides,
   computes row-major strides. Returns NULL on any allocation failure,
   freeing all previously allocated memory before returning. */
Tensor* init_tensor(int* shape, int ndim) {
  int total_elements = 1;

  Tensor* t = malloc(sizeof(Tensor));
  if (!t) {
    fprintf(stderr, "init_tensor: malloc failed for Tensor\n");
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    total_elements *= shape[i];
  }

  t->ndim = ndim;
  t->size = total_elements;
  t->contiguous = true;
  t->data = NULL;
  t->shape = NULL;
  t->strides = NULL;

  t->shape = malloc(sizeof(int) * ndim);
  if (!t->shape) {
    fprintf(stderr, "init_tensor: malloc failed for shape\n");
    free(t);
    return NULL;
  }
  for (int i = 0; i < ndim; i++) {
    t->shape[i] = shape[i];
  }

  t->data = malloc(sizeof(float) * total_elements);
  if (!t->data) {
    fprintf(stderr, "init_tensor: malloc failed for data\n");
    free(t->shape);
    free(t);
    return NULL;
  }

  t->strides = malloc(sizeof(int) * ndim);
  if (!t->strides) {
    fprintf(stderr, "init_tensor: malloc failed for strides\n");
    free(t->data);
    free(t->shape);
    free(t);
    return NULL;
  }

  /* Row-major strides: innermost stride is 1, each outer stride is the
     product of all inner dimension sizes. */
  if (ndim > 0) {
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
      t->strides[i] = t->shape[i + 1] * t->strides[i + 1];
    }
  }

  return t;
}

/* Free all heap memory owned by the tensor, including the struct itself. */
void free_tensor(Tensor* t) {
  if (!t) return;
  free(t->data);
  free(t->shape);
  free(t->strides);
  free(t);
}
