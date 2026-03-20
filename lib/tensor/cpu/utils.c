#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
/* Returns a new shape array padding shorter's shape with leading 1s to match
   ref->ndim. Caller is responsible for freeing. */
static int* tensor_match_dimensions(Tensor* shorter, Tensor* ref) {
  if (shorter->ndim > ref->ndim) {
    fprintf(stderr, "tensor_match_dimensions: shorter->ndim (%d) > ref->ndim (%d)\n",
            shorter->ndim, ref->ndim);
    return NULL;
  }

  int dim_diff = ref->ndim - shorter->ndim;
  int* shape = malloc(sizeof(int) * ref->ndim);
  if (!shape) {
    fprintf(stderr, "tensor_match_dimensions: malloc failed\n");
    return NULL;
  }

  for (int i = 0; i < dim_diff; i++)
    shape[i] = 1;
  for (int i = dim_diff; i < ref->ndim; i++)
    shape[i] = shorter->shape[i - dim_diff];

  return shape;
}

/* Returns a broadcast strides array: stride[i] = 0 where shape[i] == 1,
   otherwise copies from strides. Caller is responsible for freeing. */
static int* tensor_broadcast_strides(int* shape, int* strides, int ndim) {
  int* bstrides = malloc(sizeof(int) * ndim);
  if (!bstrides) {
    fprintf(stderr, "tensor_broadcast_strides: malloc failed\n");
    return NULL;
  }

  for (int i = 0; i < ndim; i++)
    bstrides[i] = shape[i] == 1 ? 0 : strides[i];

  return bstrides;
}

int broadcast_prepare(Tensor* ref, Tensor* shorter, BroadcastCtx* ctx) {
  int dim_diff = ref->ndim - shorter->ndim;

  ctx->matched_shape = tensor_match_dimensions(shorter, ref);
  if (!ctx->matched_shape) return -1;

  int* raw_strides = malloc(sizeof(int) * ref->ndim);
  if (!raw_strides) { free(ctx->matched_shape); return -1; }
  for (int i = 0; i < dim_diff; i++)
    raw_strides[i] = 0;
  for (int i = dim_diff; i < ref->ndim; i++)
    raw_strides[i] = shorter->strides[i - dim_diff];

  ctx->matched_strides = tensor_broadcast_strides(ctx->matched_shape, raw_strides, ref->ndim);
  free(raw_strides);
  if (!ctx->matched_strides) { free(ctx->matched_shape); return -1; }

  ctx->orig_shape   = shorter->shape;
  ctx->orig_strides = shorter->strides;
  ctx->orig_ndim    = shorter->ndim;

  shorter->shape   = ctx->matched_shape;
  shorter->strides = ctx->matched_strides;
  shorter->ndim    = ref->ndim;

  return 0;
}

void broadcast_cleanup(Tensor* shorter, BroadcastCtx* ctx) {
  shorter->shape   = ctx->orig_shape;
  shorter->strides = ctx->orig_strides;
  shorter->ndim    = ctx->orig_ndim;
  free(ctx->matched_shape);
  free(ctx->matched_strides);
}
