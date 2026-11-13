// lattice_ext_kernels.cuh -- the three device kernels and the cublasLt int8 GEMM of
// upstream cuda/lattice_ext.cu (commit 227f3f6), copied VERBATIM (only TORCH_CHECK ->
// a local check macro) so they can be tested without building the torch extension.
// The upstream file itself is untouched in ../upstream/cuda/lattice_ext.cu.
#pragma once
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define LX_LT(call)                                                                           \
  do {                                                                                        \
    cublasStatus_t s__ = (call);                                                              \
    if (s__ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasLt %d at %d\n", (int)s__, __LINE__); exit(1); } \
  } while (0)

namespace lxk {
constexpr int kThreadsPerBlock = 256;

__global__ void rowmax_cast_kernel(const __nv_bfloat16* __restrict__ xp, int8_t* __restrict__ x8,
                                   float* __restrict__ deq, int M, int K, float alpha) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const long base = static_cast<long>(m) * K;
  float local = 0.0f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) local = fmaxf(local, fabsf(__bfloat162float(xp[base + k])));
  __shared__ float red[kThreadsPerBlock];
  red[threadIdx.x] = local; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float amax = red[0];
  const float sc = amax > 0.0f ? 127.0f / amax : 0.0f;
  if (threadIdx.x == 0) deq[m] = amax / (127.0f * alpha);
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = __float2int_rn(__bfloat162float(xp[base + k]) * sc);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    x8[base + k] = static_cast<int8_t>(q);
  }
}

__global__ void dequant_kernel(const int32_t* __restrict__ C, const float* __restrict__ deq,
                               __nv_bfloat16* __restrict__ y, int M, int N) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<long>(M) * N) return;
  const int m = static_cast<int>(i / N), n = static_cast<int>(i % N);
  y[i] = __float2bfloat16(static_cast<float>(C[static_cast<long>(n) * M + m]) * deq[m]);
}

__global__ void fp8_rowmax_cast_kernel(const __nv_bfloat16* __restrict__ xp, uint8_t* __restrict__ x8,
                                       float* __restrict__ scale_a, int M, int K) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const long base = static_cast<long>(m) * K;
  float local = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) local = fmaxf(local, fabsf(__bfloat162float(xp[base + k])));
  __shared__ float red[kThreadsPerBlock];
  red[threadIdx.x] = local; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float amax = red[0];
  const float fp8_max = 448.f;
  const float inv_sc = amax > 0.f ? fp8_max / amax : 0.f;
  if (threadIdx.x == 0) scale_a[m] = amax / fp8_max;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = __bfloat162float(xp[base + k]) * inv_sc;
    v = fmaxf(-fp8_max, fminf(fp8_max, v));
    x8[base + k] = __nv_fp8_e4m3(v).__x;
  }
}

inline cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t lt = [] { cublasLtHandle_t h; LX_LT(cublasLtCreate(&h)); return h; }();
  return lt;
}
inline void* lt_workspace(size_t bytes) {
  static void* ws = nullptr; static size_t cap = 0;
  if (bytes > cap) { if (ws) cudaFree(ws); cudaMalloc(&ws, bytes); cap = bytes; }
  return ws;
}
inline void int8_gemm_tn(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
  auto lt = lt_handle();
  const size_t ws = 32u << 20; void* wsp = lt_workspace(ws);
  cublasLtMatmulDesc_t desc;
  LX_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  LX_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof opT));
  LX_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof opN));
  cublasLtMatrixLayout_t la, lb, lc;
  LX_LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, M, K));
  LX_LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, K, N, K));
  LX_LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, M, N, M));
  cublasLtMatmulPreference_t pref; LX_LT(cublasLtMatmulPreferenceCreate(&pref));
  LX_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  cublasLtMatmulHeuristicResult_t heur{}; int nh = 0;
  LX_LT(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &nh));
  if (nh <= 0) { fprintf(stderr, "no int8 GEMM algo M=%d N=%d K=%d\n", M, N, K); exit(1); }
  const int32_t one = 1, zero = 0;
  LX_LT(cublasLtMatmul(lt, desc, &one, A, la, B, lb, &zero, C, lc, C, lc, &heur.algo, wsp, ws, 0));
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(desc);
}
}  // namespace lxk
