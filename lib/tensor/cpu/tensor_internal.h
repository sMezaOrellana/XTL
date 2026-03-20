#ifndef TENSOR_INTERNAL_H
#define TENSOR_INTERNAL_H

#include <stddef.h>
#include "../tensor.h"

Tensor* init_tensor(int* shape, int ndim);
float rand_float(float lower, float upper);
void fill_tensor(Tensor* t, float value, int size);
void fill_tensor_random(Tensor* t, float lower, float upper, int size);

#endif
