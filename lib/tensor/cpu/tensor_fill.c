#include <stdlib.h>
#include "tensor_internal.h"

/* Returns a random float in [lower, upper] using rand(). */
float rand_float(float lower, float upper) {
  return lower + (float)rand() / (float)RAND_MAX * (upper - lower);
}

/* Fill every element with a constant value. */
void fill_tensor(Tensor* t, float value, int size) {
  for (int i = 0; i < size; i++) {
    t->data[i] = value;
  }
}

/* Fill every element with a random value in [lower, upper]. */
void fill_tensor_random(Tensor* t, float lower, float upper, int size) {
  for (int i = 0; i < size; i++) {
    t->data[i] = rand_float(lower, upper);
  }
}
