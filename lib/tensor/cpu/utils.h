#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "../tensor.h"

/* Holds the temporary state needed to broadcast a shorter tensor against a
   reference tensor. Pass to broadcast_prepare / broadcast_cleanup. */
typedef struct {
  int* matched_shape;    /* shorter's shape padded with leading 1s to ref->ndim */
  int* matched_strides;  /* strides zeroed out where matched_shape[i] == 1 */
  int* orig_shape;       /* shorter's original shape pointer */
  int* orig_strides;     /* shorter's original strides pointer */
  int  orig_ndim;        /* shorter's original ndim */
} BroadcastCtx;

/* Pad `shorter` to match `ref->ndim`, compute broadcast strides, and swap
   shorter's shape/strides in place. Fills ctx for later cleanup.
   Returns 0 on success, -1 on allocation failure. */
int broadcast_prepare(Tensor* ref, Tensor* shorter, BroadcastCtx* ctx);

/* Restore shorter's original shape/strides and free ctx allocations. */
void broadcast_cleanup(Tensor* shorter, BroadcastCtx* ctx);

#endif
