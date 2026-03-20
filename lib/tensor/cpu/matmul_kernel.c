/* Serial tiled matmul kernel — compiled WITHOUT -fopenmp so the compiler
   can freely auto-vectorise the inner j loop with AVX2 FMA.
   Called from tensor_ops.c when the matrix is too small to benefit from
   OpenMP thread launch overhead. */

#define TILE 32

void matmul_tiled_serial(float* restrict C,
                         const float* restrict A,
                         const float* restrict B,
                         int M, int K, int N) {
  for (int i = 0; i < M * N; i++) C[i] = 0.0f;

  for (int i0 = 0; i0 < M; i0 += TILE) {
    int i1 = i0 + TILE < M ? i0 + TILE : M;
    for (int k0 = 0; k0 < K; k0 += TILE) {
      int k1 = k0 + TILE < K ? k0 + TILE : K;
      for (int j0 = 0; j0 < N; j0 += TILE) {
        int j1 = j0 + TILE < N ? j0 + TILE : N;
        for (int i = i0; i < i1; i++) {
          for (int k = k0; k < k1; k++) {
            float a_val = A[i * K + k];
            for (int j = j0; j < j1; j++)
              C[i * N + j] += a_val * B[k * N + j];
          }
        }
      }
    }
  }
}
