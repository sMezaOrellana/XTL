#include <stdio.h>
#include <stdlib.h>
#include "tensor_internal.h"

/* Print tensor in nested-bracket notation matching its shape.
   Iterates flat data while tracking per-dimension indices to place
   opening/closing brackets and commas. */
void print_tensor(Tensor* t) {
  if (!t) return;

  int* idx = calloc(t->ndim, sizeof(int));
  if (!idx) {
    fprintf(stderr, "print_tensor: calloc failed\n");
    return;
  }

  for (int i = 0; i < t->size; i++) {
    // opening brackets for each dimension that starts a new row
    for (int d = t->ndim - 1; d >= 0; d--) {
      if (idx[d] == 0) printf("[");
      else break;
    }

    printf("%g", t->data[i]);

    // increment indices from innermost dimension outward;
    // print ']' for each dimension that just completed, ',' otherwise
    int d = t->ndim - 1;
    while (d >= 0) {
      idx[d]++;
      if (idx[d] < t->shape[d]) {
        printf(d < t->ndim - 1 ? ",\n" : ", ");
        break;
      }
      printf("]");
      idx[d] = 0;
      d--;
    }
  }
  printf("\n");
  free(idx);
}
