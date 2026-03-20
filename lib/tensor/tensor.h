#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

typedef struct Tensor Tensor;

/* N-dimensional tensor with row-major (C-order) memory layout. */
struct Tensor {
  float* data;          /* flat array of all elements */
  int* shape;           /* size of each dimension */
  int* strides;         /* number of elements to skip per dimension step */
  int ndim;             /* number of dimensions */
  int size;             /* total number of elements */
  bool contiguous;      /* true if the tensor is contiguously laid out in memory */
};

/* Create a tensor filled with 1.0. Returns NULL on allocation failure. */
Tensor* create_tensor_ones(int* shape, int ndim);

/* Create a tensor filled with 0.0. Returns NULL on allocation failure. */
Tensor* create_tensor_zeros(int* shape, int ndim);

/* Create a tensor filled with uniform random values in [0, 1].
   Caller must seed srand() before use. Returns NULL on allocation failure. */
Tensor* create_tensor_rand(int* shape, int ndim);

/* Create a tensor by copying from a flat float array (row-major order). */
Tensor* tensor_from_data(float* data, int* shape, int ndim);

/* Fill all elements of t with a constant value. */
void fill_tensor(Tensor* t, float value, int size);

/* Fill all elements of t with uniform random values in [lower, upper]. */
void fill_tensor_random(Tensor* t, float lower, float upper, int size);

/* Print the tensor in nested-bracket notation, e.g. [[1, 2], [3, 4]]. */
void print_tensor(Tensor* t);

/* Free all memory associated with t, including the struct itself. */
void free_tensor(Tensor* t);

/* Element-wise addition of a and b with broadcasting. Returns a new tensor. */
Tensor* tensor_add(Tensor* a, Tensor* b);

/* Element-wise subtraction of a and b with broadcasting. Returns a new tensor. */
Tensor* tensor_sub(Tensor* a, Tensor* b);

/* Element-wise multiplication of a and b with broadcasting. Returns a new tensor. */
Tensor* tensor_mul(Tensor* a, Tensor* b);

/* Element-wise divition of a and b with broadcasting. Returns a new tensor. */
Tensor* tensor_div(Tensor* a, Tensor* b);

/* Batched matrix multiplication. Last two dims are [M,K] @ [K,N] -> [M,N].
   Leading dims are broadcast. Returns a new tensor. */
Tensor* tensor_matmul(Tensor* a, Tensor* b);

/* Reshape a in-place to new_shape. Total elements must be unchanged.
   Returns 0 on success, -1 on allocation failure. */
int tensor_reshape(Tensor* a, int* new_shape, int new_ndim);

/* Transpose a in-place by swapping the last two dimensions. ndim must be >= 2. */
void tensor_transpose(Tensor* a);

/* Return a new contiguous copy of a with row-major strides. */
Tensor* tensor_contiguous(Tensor* a);

/* Sum over axis, removing that dimension. Returns a new tensor. */
Tensor* tensor_sum(Tensor* a, int axis);

/* Max over axis, removing that dimension. Returns a new tensor. */
Tensor* tensor_max(Tensor* a, int axis);

/* Mean over axis, removing that dimension. Returns a new tensor. */
Tensor* tensor_mean(Tensor* a, int axis);

/* Variance over axis (unbiased=false), removing that dimension. Returns a new tensor. */
Tensor* tensor_var(Tensor* a, int axis);

/* Softmax over axis. Returns a new tensor with same shape as a. */
Tensor* tensor_softmax(Tensor* a, int axis);

/* Layer normalization over axis (zero mean, unit variance). Returns a new tensor. */
Tensor* tensor_layer_norm(Tensor* a, int axis);

/* Gather rows from a along axis 0 by integer indices. Returns a new tensor. */
Tensor* tensor_gather(Tensor* a, int* indices, int num_indices);

/* Concatenate n tensors along axis. All must share the same shape except on axis. */
Tensor* tensor_cat(Tensor** tensors, int n, int axis);

/* Create a [T, T] causal attention mask: 0 on/below diagonal, -1e10 above. */
Tensor* tensor_causal_mask(int T);

/* Return the flat index of the maximum element. */
int tensor_argmax(Tensor* a);

/* Sample an index from a 1-D probability tensor using rand(). */
int tensor_sample(Tensor* probs);

/* Seed the C random number generator. */
void tensor_srand(unsigned int seed);

/* Elementwise activations. Return a new tensor. */
Tensor* tensor_gelu      (Tensor* a);
Tensor* tensor_gelu_apprx(Tensor* a);
Tensor* tensor_sigmoid   (Tensor* a);
Tensor* tensor_tanh      (Tensor* a);
Tensor* tensor_exp       (Tensor* a);
Tensor* tensor_sqrt      (Tensor* a);
Tensor* tensor_neg       (Tensor* a);
Tensor* tensor_log       (Tensor* a);

/* Scalar ops: broadcast a constant over every element. Return a new tensor. */
Tensor* tensor_add_scalar(Tensor* a, float s);
Tensor* tensor_mul_scalar(Tensor* a, float s);
Tensor* tensor_div_scalar(Tensor* a, float s);

#endif
