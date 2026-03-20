#include "tensor_internal.h"
#include "utils.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* Defined in matmul_kernel.c, compiled without -fopenmp for full AVX2 vectorisation. */
void matmul_tiled_serial(float* restrict C, const float* restrict A,
                         const float* restrict B, int M, int K, int N);

bool tensor_check_op_dim(Tensor* a, Tensor* b) {
  Tensor* larger  = a->ndim >= b->ndim ? a : b;
  Tensor* shorter = a->ndim <  b->ndim ? a : b;

  int dim_diff = larger->ndim - shorter->ndim;
  for (int i = shorter->ndim - 1; i >= 0; i--) {
    if (larger->shape[i + dim_diff] != shorter->shape[i] &&
        larger->shape[i + dim_diff] != 1 && shorter->shape[i] != 1)
      return false;
  }
  return true;
}

int* tensor_op_output_shape(Tensor* a, Tensor* b) {
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensor_op_output_shape: ndim mismatch (%d vs %d)\n", a->ndim, b->ndim);
    return NULL;
  }

  int* shape = malloc(sizeof(int) * a->ndim);
  if (!shape) {
    fprintf(stderr, "tensor_op_output_shape: malloc failed\n");
    return NULL;
  }

  for (int i = 0; i < a->ndim; i++)
    shape[i] = a->shape[i] > b->shape[i] ? a->shape[i] : b->shape[i];

  return shape;
}

static Tensor* tensor_matmul_impl(Tensor* a, Tensor* b) {
  if (a->ndim < 2) {
    fprintf(stderr, "tensor_matmul: a->ndim must be >= 2 (got %d)\n", a->ndim);
    return NULL;
  }

  BroadcastCtx ctx;
  if (broadcast_prepare(a, b, &ctx) < 0) return NULL;

  // last dim of a (K) must match second-to-last dim of b (also K)
  if (a->shape[a->ndim - 1] != ctx.matched_shape[a->ndim - 2]) {
    fprintf(stderr, "tensor_matmul: K dimension mismatch (%d vs %d)\n",
            a->shape[a->ndim - 1], ctx.matched_shape[a->ndim - 2]);
    broadcast_cleanup(b, &ctx);
    return NULL;
  }

  // batch dims broadcast as usual, but the last two are [M, N] not max per dim
  int* out_shape = tensor_op_output_shape(a, b);
  if (out_shape) {
    out_shape[a->ndim - 2] = a->shape[a->ndim - 2];  // M (rows of a)
    out_shape[a->ndim - 1] = b->shape[b->ndim - 1];  // N (cols of b)
  }
  Tensor* out = NULL;
  if (out_shape)
    out = init_tensor(out_shape, a->ndim);

  if (!out) goto clean_up;

  int total_batch = 1;
  for (int i = 0; i < out->ndim - 2; i++)
    total_batch *= out->shape[i];

  int M = a->shape[a->ndim - 2];
  int K = a->shape[a->ndim - 1];
  int N = b->shape[b->ndim - 1];

  for (int batch = 0; batch < total_batch; batch++) {
    int offset_a = 0, offset_b = 0;
    int remaining = batch;
    for (int d = out->ndim - 3; d >= 0; d--) {
      int coord = remaining % out->shape[d];
      remaining /= out->shape[d];
      offset_a += coord * (a->shape[d] == 1 ? 0 : a->strides[d]);
      offset_b += coord * b->strides[d];
    }

    float* a_ptr = a->data + offset_a;
    float* b_ptr = b->data + offset_b;
    float* c_ptr = out->data + batch * M * N;

    int a_row_stride = a->strides[a->ndim - 2];
    int a_col_stride = a->strides[a->ndim - 1];
    int b_row_stride = b->strides[b->ndim - 2];
    int b_col_stride = b->strides[b->ndim - 1];

    /* If either slice is non-contiguous (e.g. after transpose), materialise a
       contiguous copy so the tiled fast path can always be used. */
    float* a_data = a_ptr;
    float* b_data = b_ptr;
    float* a_tmp  = NULL;
    float* b_tmp  = NULL;

    if (a_col_stride != 1 || a_row_stride != K) {
      a_tmp = malloc(sizeof(float) * M * K);
      if (!a_tmp) goto clean_up;
      for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
          a_tmp[i * K + k] = a_ptr[i * a_row_stride + k * a_col_stride];
      a_data = a_tmp;
    }

    if (b_col_stride != 1 || b_row_stride != N) {
      b_tmp = malloc(sizeof(float) * K * N);
      if (!b_tmp) { free(a_tmp); goto clean_up; }
      for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
          b_tmp[k * N + j] = b_ptr[k * b_row_stride + j * b_col_stride];
      b_data = b_tmp;
    }

    /* Tiled matmul: loop order i,k,j keeps B accesses sequential (row-major).
       Tile size 32 fits three 32×32 tiles in L1 (~12 KB of 40 KB available).
       collapse(2) on (i0,j0) exposes M/TILE * N/TILE independent tasks so
       OpenMP can fill all cores even when M=1 (single-token decode).
       Threshold: skip OpenMP for tiny matrices where thread overhead > compute. */
    if ((long)M * K * N > 25000000L && (M > 1 || N > 10000)) {
      /* Parallel path: collapse(i0,j0) tiles across all cores.
         omp simd on j ensures AVX2 vectorisation inside each thread. */
      float* restrict ra = a_data;
      float* restrict rb = b_data;
      float* restrict rc = c_ptr;
      #define TILE 32
      for (int i = 0; i < M * N; i++) rc[i] = 0.0f;
      #pragma omp parallel for collapse(2) schedule(static)
      for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
          int i1 = i0 + TILE < M ? i0 + TILE : M;
          int j1 = j0 + TILE < N ? j0 + TILE : N;
          for (int k0 = 0; k0 < K; k0 += TILE) {
            int k1 = k0 + TILE < K ? k0 + TILE : K;
            for (int i = i0; i < i1; i++) {
              for (int k = k0; k < k1; k++) {
                float a_val = ra[i * K + k];
                #pragma omp simd
                for (int j = j0; j < j1; j++)
                  rc[i * N + j] += a_val * rb[k * N + j];
              }
            }
          }
        }
      }
      #undef TILE
    } else {
      /* Serial path: compiled in a separate TU without -fopenmp so gcc
         can freely auto-vectorise with AVX2 FMA. */
      matmul_tiled_serial(c_ptr, a_data, b_data, M, K, N);
    }

    free(a_tmp);
    free(b_tmp);
  }

clean_up:
  broadcast_cleanup(b, &ctx);
  free(out_shape);
  return out;
}

static Tensor* tensor_elementwise_op(Tensor* a, Tensor* b, float (*op)(float, float)) {
  if (!tensor_check_op_dim(a, b)) return NULL;

  Tensor* larger  = a->ndim >= b->ndim ? a : b;
  Tensor* shorter = a->ndim <  b->ndim ? a : b;

  BroadcastCtx ctx;
  if (broadcast_prepare(larger, shorter, &ctx) < 0) return NULL;

  int* out_shape = tensor_op_output_shape(a, b);
  Tensor* out = NULL;
  if (out_shape)
    out = init_tensor(out_shape, larger->ndim);

  if (!out) goto clean_up;

  for (int i = 0; i < out->size; i++) {
    int offset_a = 0, offset_b = 0;
    int remaining = i;
    for (int d = out->ndim - 1; d >= 0; d--) {
      int coord = remaining % out->shape[d];
      remaining /= out->shape[d];
      offset_a += coord * a->strides[d];
      offset_b += coord * b->strides[d];
    }
    out->data[i] = op(a->data[offset_a], b->data[offset_b]);
  }

clean_up:
  broadcast_cleanup(shorter, &ctx);
  free(out_shape);
  return out;
}

static Tensor* tensor_elementwise_op_singular(Tensor* a, float (*op)(float)) {
  Tensor* out = init_tensor(a->shape, a->ndim);
  if (!out) return NULL;

  for (int i = 0; i < a->size; i++) {
    int offset_a = 0;
    int remaining = i;
    for (int d = a->ndim - 1; d >= 0; d--) {
      int coord = remaining % a->shape[d];
      remaining /= a->shape[d];
      offset_a += coord * a->strides[d];
    }
    out->data[i] = op(a->data[offset_a]);
  }

  return out;
}

static float op_add(float x, float y) { return x + y; }
static float op_sub(float x, float y) { return x - y; }
static float op_mul(float x, float y) { return x * y; }
static float op_div(float x, float y) { return x / y; }

static float op_gelu      (float x) { return x * 0.5f * (1.0f + erff(x / 1.4142135f)); } /* sqrt(2) == 2**0.5 */
static float op_gelu_apprx(float x) { float t = tanhf(0.7978845f * (x + 0.044715f * x*x*x)); return 0.5f * x * (1.0f + t); } /* sqrt(2/pi) == 0.7978845 */
static float op_sigmoid   (float x) { return 1.0f / (1.0f + expf(-x)); }
static float op_tanh      (float x) { return tanhf(x); }
static float op_exp       (float x) { return expf(x); }
static float op_sqrt      (float x) { return sqrtf(x); }
static float op_neg       (float x) { return -x; }
static float op_log       (float x) { return logf(x); }

Tensor* tensor_add(Tensor* a, Tensor* b) { return tensor_elementwise_op(a, b, op_add); }
Tensor* tensor_sub(Tensor* a, Tensor* b) { return tensor_elementwise_op(a, b, op_sub); }
Tensor* tensor_mul(Tensor* a, Tensor* b) { return tensor_elementwise_op(a, b, op_mul); }
Tensor* tensor_div(Tensor* a, Tensor* b) { return tensor_elementwise_op(a, b, op_div); }

Tensor* tensor_gelu      (Tensor* a) { return tensor_elementwise_op_singular(a, op_gelu); }
Tensor* tensor_gelu_apprx(Tensor* a) { return tensor_elementwise_op_singular(a, op_gelu_apprx); }
Tensor* tensor_sigmoid   (Tensor* a) { return tensor_elementwise_op_singular(a, op_sigmoid); }
Tensor* tensor_tanh      (Tensor* a) { return tensor_elementwise_op_singular(a, op_tanh); }
Tensor* tensor_exp       (Tensor* a) { return tensor_elementwise_op_singular(a, op_exp); }
Tensor* tensor_sqrt      (Tensor* a) { return tensor_elementwise_op_singular(a, op_sqrt); }
Tensor* tensor_neg       (Tensor* a) { return tensor_elementwise_op_singular(a, op_neg); }
Tensor* tensor_log       (Tensor* a) { return tensor_elementwise_op_singular(a, op_log); }

/* Scalar ops: apply a constant to every element. Return a new tensor. */
Tensor* tensor_add_scalar(Tensor* a, float s) {
  Tensor* out = init_tensor(a->shape, a->ndim);
  if (!out) return NULL;
  for (int i = 0; i < a->size; i++) {
    int offset = 0, remaining = i;
    for (int d = a->ndim - 1; d >= 0; d--) {
      offset += (remaining % a->shape[d]) * a->strides[d];
      remaining /= a->shape[d];
    }
    out->data[i] = a->data[offset] + s;
  }
  return out;
}

Tensor* tensor_mul_scalar(Tensor* a, float s) {
  Tensor* out = init_tensor(a->shape, a->ndim);
  if (!out) return NULL;
  for (int i = 0; i < a->size; i++) {
    int offset = 0, remaining = i;
    for (int d = a->ndim - 1; d >= 0; d--) {
      offset += (remaining % a->shape[d]) * a->strides[d];
      remaining /= a->shape[d];
    }
    out->data[i] = a->data[offset] * s;
  }
  return out;
}

Tensor* tensor_div_scalar(Tensor* a, float s) {
  return tensor_mul_scalar(a, 1.0f / s);
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) { return tensor_matmul_impl(a, b); }

Tensor* tensor_sum(Tensor* a, int axis) {
  if (axis < 0 || axis >= a->ndim) {
    fprintf(stderr, "tensor_sum: axis %d out of range for ndim %d\n", axis, a->ndim);
    return NULL;
  }

  // output shape is input shape with the axis dimension removed
  int out_ndim = a->ndim - 1;
  int* out_shape = malloc(sizeof(int) * out_ndim);
  if (!out_shape) {
    fprintf(stderr, "tensor_sum: malloc failed for out_shape\n");
    return NULL;
  }
  for (int d = 0, od = 0; d < a->ndim; d++) {
    if (d != axis) out_shape[od++] = a->shape[d];
  }

  Tensor* out = create_tensor_zeros(out_shape, out_ndim);
  free(out_shape);
  if (!out) return NULL;

  int* coords = malloc(sizeof(int) * a->ndim);
  if (!coords) {
    fprintf(stderr, "tensor_sum: malloc failed for coords\n");
    free_tensor(out);
    return NULL;
  }

  for (int i = 0; i < a->size; i++) {
    // decompose flat index into coords using input shape
    int remaining = i;
    for (int d = a->ndim - 1; d >= 0; d--) {
      coords[d] = remaining % a->shape[d];
      remaining /= a->shape[d];
    }

    // input offset via strides (handles non-contiguous)
    int in_offset = 0;
    for (int d = 0; d < a->ndim; d++)
      in_offset += coords[d] * a->strides[d];

    // output flat index: same coords but skip axis
    int out_idx = 0, out_d = 0;
    for (int d = 0; d < a->ndim; d++) {
      if (d == axis) continue;
      out_idx += coords[d] * out->strides[out_d++];
    }

    out->data[out_idx] += a->data[in_offset];
  }

  free(coords);
  return out;
}

/* Helper: generic axis reduction. init_val is the starting accumulator value.
   combine(acc, x) returns the new accumulator. */
static Tensor* tensor_reduce_axis(Tensor* a, int axis, float init_val,
                                  float (*combine)(float, float)) {
  if (axis < 0 || axis >= a->ndim) {
    fprintf(stderr, "tensor_reduce_axis: axis %d out of range for ndim %d\n", axis, a->ndim);
    return NULL;
  }

  int out_ndim = a->ndim - 1;
  int* out_shape = malloc(sizeof(int) * out_ndim);
  if (!out_shape) return NULL;
  for (int d = 0, od = 0; d < a->ndim; d++)
    if (d != axis) out_shape[od++] = a->shape[d];

  Tensor* out = init_tensor(out_shape, out_ndim);
  free(out_shape);
  if (!out) return NULL;

  // initialize output with init_val
  for (int i = 0; i < out->size; i++) out->data[i] = init_val;

  int* coords = malloc(sizeof(int) * a->ndim);
  if (!coords) { free_tensor(out); return NULL; }

  for (int i = 0; i < a->size; i++) {
    int remaining = i;
    for (int d = a->ndim - 1; d >= 0; d--) {
      coords[d] = remaining % a->shape[d];
      remaining /= a->shape[d];
    }
    int in_offset = 0;
    for (int d = 0; d < a->ndim; d++)
      in_offset += coords[d] * a->strides[d];

    int out_idx = 0, out_d = 0;
    for (int d = 0; d < a->ndim; d++) {
      if (d == axis) continue;
      out_idx += coords[d] * out->strides[out_d++];
    }
    out->data[out_idx] = combine(out->data[out_idx], a->data[in_offset]);
  }

  free(coords);
  return out;
}

static float combine_max(float acc, float x) { return x > acc ? x : acc; }

Tensor* tensor_max(Tensor* a, int axis) {
  return tensor_reduce_axis(a, axis, -FLT_MAX, combine_max);
}

Tensor* tensor_mean(Tensor* a, int axis) {
  Tensor* s = tensor_sum(a, axis);
  if (!s) return NULL;
  float n = (float)a->shape[axis];
  Tensor* out = tensor_mul_scalar(s, 1.0f / n);
  free_tensor(s);
  return out;
}

Tensor* tensor_var(Tensor* a, int axis) {
  // var = mean((x - mean(x))^2)
  Tensor* mu = tensor_mean(a, axis);
  if (!mu) return NULL;

  // broadcast mu back to a's shape by re-inserting the axis dim
  // We build a "expanded" mu with shape[axis]=1, then broadcast sub
  // Easiest: insert a size-1 dim at position axis in mu, then rely on broadcasting
  int new_ndim = mu->ndim + 1;
  int* new_shape = malloc(sizeof(int) * new_ndim);
  if (!new_shape) { free_tensor(mu); return NULL; }
  for (int d = 0, od = 0; d < new_ndim; d++) {
    if (d == axis) new_shape[d] = 1;
    else           new_shape[d] = mu->shape[od++];
  }
  if (tensor_reshape(mu, new_shape, new_ndim) < 0) {
    free(new_shape); free_tensor(mu); return NULL;
  }
  free(new_shape);

  // diff = a - mu  (broadcasts axis dim 1 -> a->shape[axis])
  Tensor* diff = tensor_sub(a, mu);
  free_tensor(mu);
  if (!diff) return NULL;

  Tensor* sq = tensor_mul(diff, diff);
  free_tensor(diff);
  if (!sq) return NULL;

  Tensor* v = tensor_mean(sq, axis);
  free_tensor(sq);
  return v;
}

Tensor* tensor_softmax(Tensor* a, int axis) {
  // max for numerical stability
  Tensor* m = tensor_max(a, axis);
  if (!m) return NULL;

  // expand m back along axis (insert size-1 dim)
  int new_ndim = m->ndim + 1;
  int* new_shape = malloc(sizeof(int) * new_ndim);
  if (!new_shape) { free_tensor(m); return NULL; }
  for (int d = 0, od = 0; d < new_ndim; d++) {
    if (d == axis) new_shape[d] = 1;
    else           new_shape[d] = m->shape[od++];
  }
  if (tensor_reshape(m, new_shape, new_ndim) < 0) {
    free(new_shape); free_tensor(m); return NULL;
  }
  free(new_shape);

  Tensor* shifted = tensor_sub(a, m);
  free_tensor(m);
  if (!shifted) return NULL;

  Tensor* e = tensor_exp(shifted);
  free_tensor(shifted);
  if (!e) return NULL;

  Tensor* s = tensor_sum(e, axis);
  if (!s) { free_tensor(e); return NULL; }

  // expand s along axis
  new_ndim = s->ndim + 1;
  new_shape = malloc(sizeof(int) * new_ndim);
  if (!new_shape) { free_tensor(e); free_tensor(s); return NULL; }
  for (int d = 0, od = 0; d < new_ndim; d++) {
    if (d == axis) new_shape[d] = 1;
    else           new_shape[d] = s->shape[od++];
  }
  if (tensor_reshape(s, new_shape, new_ndim) < 0) {
    free(new_shape); free_tensor(e); free_tensor(s); return NULL;
  }
  free(new_shape);

  Tensor* out = tensor_div(e, s);
  free_tensor(e);
  free_tensor(s);
  return out;
}

Tensor* tensor_layer_norm(Tensor* a, int axis) {
  float eps = 1e-5f;

  Tensor* mu = tensor_mean(a, axis);
  if (!mu) return NULL;

  Tensor* v = tensor_var(a, axis);
  if (!v) { free_tensor(mu); return NULL; }

  // expand mu along axis
  int new_ndim = mu->ndim + 1;
  int* new_shape = malloc(sizeof(int) * new_ndim);
  if (!new_shape) { free_tensor(mu); free_tensor(v); return NULL; }
  for (int d = 0, od = 0; d < new_ndim; d++) {
    if (d == axis) new_shape[d] = 1;
    else           new_shape[d] = mu->shape[od++];
  }
  if (tensor_reshape(mu, new_shape, new_ndim) < 0) {
    free(new_shape); free_tensor(mu); free_tensor(v); return NULL;
  }
  // reuse new_shape for v (same shape)
  if (tensor_reshape(v, new_shape, new_ndim) < 0) {
    free(new_shape); free_tensor(mu); free_tensor(v); return NULL;
  }
  free(new_shape);

  Tensor* centered = tensor_sub(a, mu);
  free_tensor(mu);
  if (!centered) { free_tensor(v); return NULL; }

  Tensor* var_eps = tensor_add_scalar(v, eps);
  free_tensor(v);
  if (!var_eps) { free_tensor(centered); return NULL; }

  Tensor* std = tensor_sqrt(var_eps);
  free_tensor(var_eps);
  if (!std) { free_tensor(centered); return NULL; }

  Tensor* out = tensor_div(centered, std);
  free_tensor(centered);
  free_tensor(std);
  return out;
}

/* Concatenate n tensors along axis. All must have the same shape except on axis. */
Tensor* tensor_cat(Tensor** tensors, int n, int axis) {
  if (n <= 0) return NULL;
  int ndim = tensors[0]->ndim;

  for (int i = 1; i < n; i++) {
    if (tensors[i]->ndim != ndim) {
      fprintf(stderr, "tensor_cat: ndim mismatch\n");
      return NULL;
    }
    for (int d = 0; d < ndim; d++) {
      if (d != axis && tensors[i]->shape[d] != tensors[0]->shape[d]) {
        fprintf(stderr, "tensor_cat: shape mismatch at dim %d\n", d);
        return NULL;
      }
    }
  }

  int* out_shape = malloc(sizeof(int) * ndim);
  if (!out_shape) return NULL;
  for (int d = 0; d < ndim; d++) out_shape[d] = tensors[0]->shape[d];
  for (int i = 1; i < n; i++) out_shape[axis] += tensors[i]->shape[axis];

  Tensor* out = init_tensor(out_shape, ndim);
  free(out_shape);
  if (!out) return NULL;

  int* coords = malloc(sizeof(int) * ndim);
  if (!coords) { free_tensor(out); return NULL; }

  for (int i = 0; i < out->size; i++) {
    int remaining = i;
    for (int d = ndim - 1; d >= 0; d--) {
      coords[d] = remaining % out->shape[d];
      remaining /= out->shape[d];
    }

    // find which input tensor owns this axis coordinate
    int axis_coord = coords[axis];
    int t = 0;
    while (t < n - 1 && axis_coord >= tensors[t]->shape[axis]) {
      axis_coord -= tensors[t]->shape[axis];
      t++;
    }

    int src_offset = 0;
    for (int d = 0; d < ndim; d++) {
      int c = (d == axis) ? axis_coord : coords[d];
      src_offset += c * tensors[t]->strides[d];
    }
    out->data[i] = tensors[t]->data[src_offset];
  }

  free(coords);
  return out;
}

/* Create a [T, T] causal mask: 0 where j <= i, -1e10 where j > i. */
Tensor* tensor_causal_mask(int T) {
  int shape[2] = {T, T};
  Tensor* out = init_tensor(shape, 2);
  if (!out) return NULL;
  for (int i = 0; i < T; i++)
    for (int j = 0; j < T; j++)
      out->data[i * T + j] = (j <= i) ? 0.0f : -1e10f;
  return out;
}

/* Return the flat index of the maximum element (handles non-contiguous). */
int tensor_argmax(Tensor* a) {
  int best = 0;
  float best_val = -FLT_MAX;
  for (int i = 0; i < a->size; i++) {
    int offset = 0, rem = i;
    for (int d = a->ndim - 1; d >= 0; d--) {
      offset += (rem % a->shape[d]) * a->strides[d];
      rem /= a->shape[d];
    }
    if (a->data[offset] > best_val) { best_val = a->data[offset]; best = i; }
  }
  return best;
}

/* Sample an index from a 1-D probability tensor using the current rand() state. */
int tensor_sample(Tensor* probs) {
  float r = (float)rand() / ((float)RAND_MAX + 1.0f);
  float cumsum = 0.0f;
  for (int i = 0; i < probs->size; i++) {
    int offset = 0, rem = i;
    for (int d = probs->ndim - 1; d >= 0; d--) {
      offset += (rem % probs->shape[d]) * probs->strides[d];
      rem /= probs->shape[d];
    }
    cumsum += probs->data[offset];
    if (r < cumsum) return i;
  }
  return probs->size - 1;
}

/* Seed the C random number generator. */
void tensor_srand(unsigned int seed) { srand(seed); }

/* gather: integer index lookup along axis 0.
   indices[i] selects row i from a. out shape: [num_indices, a->shape[1], ...] */
Tensor* tensor_gather(Tensor* a, int* indices, int num_indices) {
  if (a->ndim < 1) {
    fprintf(stderr, "tensor_gather: ndim must be >= 1\n");
    return NULL;
  }

  int row_size = a->size / a->shape[0];
  int* out_shape = malloc(sizeof(int) * a->ndim);
  if (!out_shape) return NULL;
  out_shape[0] = num_indices;
  for (int d = 1; d < a->ndim; d++) out_shape[d] = a->shape[d];

  Tensor* out = init_tensor(out_shape, a->ndim);
  free(out_shape);
  if (!out) return NULL;

  for (int i = 0; i < num_indices; i++) {
    int idx = indices[i];
    if (idx < 0 || idx >= a->shape[0]) {
      fprintf(stderr, "tensor_gather: index %d out of range [0, %d)\n", idx, a->shape[0]);
      free_tensor(out);
      return NULL;
    }
    // source row offset (axis-0 only, assumes contiguous)
    int src_offset = idx * a->strides[0];
    for (int j = 0; j < row_size; j++) {
      // decompose j into coords for dims 1..ndim-1
      int in_offset = src_offset;
      int rem = j;
      for (int d = a->ndim - 1; d >= 1; d--) {
        in_offset += (rem % a->shape[d]) * a->strides[d];
        rem /= a->shape[d];
      }
      out->data[i * row_size + j] = a->data[in_offset];
    }
  }

  return out;
}
