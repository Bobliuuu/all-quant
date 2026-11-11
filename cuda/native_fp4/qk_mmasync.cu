// QK^T for the UltraQuant FP4 paged-decode op, two implementations on the SAME data:
//   native : mma.sync.m16n8k32 kind::mxf8f6f4 block_scale (A=e4m3 Q, B=e2m1 K, ue8m0 SF, k-block 32)  [sm_120a only]
//   bf16   : mma.sync.m16n8k16 bf16 with software FP4->bf16 dequant (placement trick x 2^(s-1))        [any sm_80+]
// head_dim D=128 (4 scale blocks), M = 16*MT query rows (MT=2,4), K-cache layout of uqmt/attn.py
// (uint8 [key][D/2] low nibble = even elem, ue8m0 [key][D/32]).
// The reduction (k) axis is permuted consistently between A and B (thread tig owns elements
// 8*tig..8*tig+7 of every 32-block), which lets each thread fetch its B bytes with ONE 4-byte load per block.
// usage: qk_mmasync DATADIR [REP] [ITERS]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)
constexpr int D = 128, NKB = 4;

__device__ __forceinline__ uint32_t spread4(uint32_t u16) {   // 4 packed nibbles -> 4 bytes 0b00ABCD00
  uint32_t t = __byte_perm(u16, 0, 0x1100);
  return ((t << 2) & 0x003C003Cu) | ((t >> 2) & 0x3C003C00u);
}
__device__ __forceinline__ uint32_t deq_byte_bf16x2(uint32_t x) {   // byte (2 nibbles) -> bf16x2 of (value * 2^-126)
  return ((x << 6) & 0x1C0u) | ((x << 18) & 0x01C00000u) | ((x << 12) & 0x8000u) | ((x << 24) & 0x80000000u);
}
__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t a, uint32_t b) {
  uint32_t d; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d;
}

__device__ __forceinline__ void mma_native(float* c, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
#ifdef UQ_NATIVE
  const uint16_t z = 0;
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
#else
  __trap();
#endif
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int MT, bool WRITE>
__global__ void __launch_bounds__(128) qk_native(const uint8_t* __restrict__ q8, const uint8_t* __restrict__ qs,
    const uint8_t* __restrict__ kc, const uint8_t* __restrict__ ks, int N, float* S, float* red) {
  const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), nw = gridDim.x * (blockDim.x >> 5);
  uint32_t a[MT][NKB][4], sfa[MT][NKB];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int kb = 0; kb < NKB; kb++) {
      int r0 = mt * 16 + g;
      uint2 v0 = *(const uint2*)(q8 + r0 * D + kb * 32 + tig * 8);
      uint2 v1 = *(const uint2*)(q8 + (r0 + 8) * D + kb * 32 + tig * 8);
      a[mt][kb][0] = v0.x; a[mt][kb][2] = v0.y; a[mt][kb][1] = v1.x; a[mt][kb][3] = v1.y;
      sfa[mt][kb] = qs[(r0 + 8 * (lane & 1)) * NKB + kb];
    }
  float mx[MT][2], sm = 0.f;
#pragma unroll
  for (int mt = 0; mt < MT; mt++) { mx[mt][0] = mx[mt][1] = -1e30f; }
  for (int nt = gw; nt < N / 8; nt += nw) {
    const int key = nt * 8 + g;
    const uint32_t* kw = (const uint32_t*)(kc + (size_t)key * (D / 2));
    uint32_t w[NKB]; 
#pragma unroll
    for (int kb = 0; kb < NKB; kb++) w[kb] = __ldg(kw + kb * 4 + tig);
    const uint32_t sb = __ldg((const uint32_t*)(ks + (size_t)key * NKB));
    uint32_t b[NKB][2];
#pragma unroll
    for (int kb = 0; kb < NKB; kb++) { b[kb][0] = spread4(w[kb] & 0xffffu); b[kb][1] = spread4(w[kb] >> 16); }
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kb = 0; kb < NKB; kb++) mma_native(c, a[mt][kb], b[kb][0], b[kb][1], sfa[mt][kb], (sb >> (8 * kb)) & 0xffu);
      if (WRITE) {
        *(float2*)(S + (size_t)(mt * 16 + g) * N + nt * 8 + 2 * tig) = make_float2(c[0], c[1]);
        *(float2*)(S + (size_t)(mt * 16 + g + 8) * N + nt * 8 + 2 * tig) = make_float2(c[2], c[3]);
      } else {
        mx[mt][0] = fmaxf(mx[mt][0], fmaxf(c[0], c[1])); mx[mt][1] = fmaxf(mx[mt][1], fmaxf(c[2], c[3]));
        sm += c[0] + c[1] + c[2] + c[3];
      }
    }
  }
  if (!WRITE) {
    float t = sm;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) t += mx[mt][0] + mx[mt][1];
    red[blockIdx.x * blockDim.x + threadIdx.x] = t;
  }
}

template <int MT, bool WRITE>
__global__ void __launch_bounds__(128) qk_bf16(const __nv_bfloat16* __restrict__ qd,
    const uint8_t* __restrict__ kc, const uint8_t* __restrict__ ks, int N, float* S, float* red) {
  const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), nw = gridDim.x * (blockDim.x >> 5);
  uint32_t a[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int s2 = 0; s2 < 8; s2++) {
      int r0 = mt * 16 + g, kb = s2 >> 1, h = s2 & 1;
      uint2 v0 = *(const uint2*)(qd + r0 * D + kb * 32 + tig * 8 + h * 4);
      uint2 v1 = *(const uint2*)(qd + (r0 + 8) * D + kb * 32 + tig * 8 + h * 4);
      a[mt][s2][0] = v0.x; a[mt][s2][2] = v0.y; a[mt][s2][1] = v1.x; a[mt][s2][3] = v1.y;
    }
  float mx[MT][2], sm = 0.f;
#pragma unroll
  for (int mt = 0; mt < MT; mt++) { mx[mt][0] = mx[mt][1] = -1e30f; }
  for (int nt = gw; nt < N / 8; nt += nw) {
    const int key = nt * 8 + g;
    const uint32_t* kw = (const uint32_t*)(kc + (size_t)key * (D / 2));
    uint32_t w[NKB];
#pragma unroll
    for (int kb = 0; kb < NKB; kb++) w[kb] = __ldg(kw + kb * 4 + tig);
    const uint32_t sb = __ldg((const uint32_t*)(ks + (size_t)key * NKB));
    uint32_t b[8][2];
#pragma unroll
    for (int kb = 0; kb < NKB; kb++) {
      uint32_t sbits = ((sb >> (8 * kb)) & 0xffu) + 126u;           // bf16 exponent field of 2^(s-1)
      uint32_t sc = (sbits << 7) | (sbits << 23);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t ww = w[kb] >> (16 * h);
        b[kb * 2 + h][0] = hmul2_bf16(deq_byte_bf16x2(ww & 0xffu), sc);
        b[kb * 2 + h][1] = hmul2_bf16(deq_byte_bf16x2((ww >> 8) & 0xffu), sc);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s2 = 0; s2 < 8; s2++) mma_bf16(c, a[mt][s2], b[s2][0], b[s2][1]);
      if (WRITE) {
        *(float2*)(S + (size_t)(mt * 16 + g) * N + nt * 8 + 2 * tig) = make_float2(c[0], c[1]);
        *(float2*)(S + (size_t)(mt * 16 + g + 8) * N + nt * 8 + 2 * tig) = make_float2(c[2], c[3]);
      } else {
        mx[mt][0] = fmaxf(mx[mt][0], fmaxf(c[0], c[1])); mx[mt][1] = fmaxf(mx[mt][1], fmaxf(c[2], c[3]));
        sm += c[0] + c[1] + c[2] + c[3];
      }
    }
  }
  if (!WRITE) {
    float t = sm;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) t += mx[mt][0] + mx[mt][1];
    red[blockIdx.x * blockDim.x + threadIdx.x] = t;
  }
}

static std::vector<char> rd(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb"); if (!f) { printf("cannot open %s\n", p.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> v(n); if (fread(v.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return v;
}
template <class F> static float timeit(F f, int iters) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 3; i++) f();
  CK(cudaDeviceSynchronize()); cudaEventRecord(a);
  for (int i = 0; i < iters; i++) f();
  cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); return ms / iters;
}
template <class T> static T* up(const std::vector<char>& h, int rep = 1) {
  T* d; CK(cudaMalloc(&d, h.size() * rep));
  for (int i = 0; i < rep; i++) CK(cudaMemcpy((char*)d + (size_t)i * h.size(), h.data(), h.size(), cudaMemcpyHostToDevice));
  return d;
}

template <int MT> static void run(const std::string& dir, int N, int rep, int iters, int sms, bool native_ok) {
  const int M = MT * 16;
  auto hkc = rd(dir + "/kc.bin"), hks = rd(dir + "/ks.bin"), hq8 = rd(dir + "/q8.bin"), hqs = rd(dir + "/qs.bin"),
       hqd = rd(dir + "/qdq_bf16.bin"), hS = rd(dir + "/S_ref.bin");
  auto *q8 = up<uint8_t>(hq8), *qs = up<uint8_t>(hqs), *kc = up<uint8_t>(hkc), *ks = up<uint8_t>(hks);
  auto* qd = up<__nv_bfloat16>(hqd);
  float* dS; CK(cudaMalloc(&dS, (size_t)M * N * 4));
  float* dred; CK(cudaMalloc(&dred, 1 << 24));
  const float* ref = (const float*)hS.data();
  std::vector<float> o((size_t)M * N);
  auto check = [&](const char* name) {
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o.data(), dS, o.size() * 4, cudaMemcpyDeviceToHost));
    double maxabs = 0, maxref = 0; size_t bad = 0;
    for (size_t i = 0; i < o.size(); i++) { double d = fabs(o[i] - ref[i]); maxabs = std::max(maxabs, d); maxref = std::max(maxref, (double)fabs(ref[i])); if (d > 2e-3 * maxref + 1e-3) bad++; }
    printf("VALIDATE %-8s M=%d N=%d max|err|=%.3e  max|ref|=%.3f  mismatches(>2e-3rel)=%zu  %s\n", name, M, N, maxabs, maxref, bad, bad == 0 ? "PASS" : "FAIL");
    FILE* f = fopen((dir + "/S_" + name + ".bin").c_str(), "wb"); fwrite(o.data(), 4, o.size(), f); fclose(f);
  };
  int grid_v = N / 8 / 4 > 0 ? std::min(N / 8 / 4, sms * 4) : 1;
  CK(cudaMemset(dS, 0, (size_t)M * N * 4));
  qk_bf16<MT, true><<<grid_v, 128>>>(qd, kc, ks, N, dS, dred); check("bf16");
  if (native_ok) { CK(cudaMemset(dS, 0, (size_t)M * N * 4)); qk_native<MT, true><<<grid_v, 128>>>(q8, qs, kc, ks, N, dS, dred); check("native"); }
  // ---- bench (reduce mode: no S store), tiled K cache
  const int NB = N * rep;
  auto *kcb = up<uint8_t>(hkc, rep), *ksb = up<uint8_t>(hks, rep);
  const double flop = 2.0 * M * NB * D, kb_bytes = (double)NB * (D / 2 + D / 32);
  for (int mult : {2, 4, 8}) {
    int grid = sms * mult;
    float t_b = timeit([&] { qk_bf16<MT, false><<<grid, 128>>>(qd, kcb, ksb, NB, dS, dred); }, iters);
    printf("BENCH bf16   M=%d keys=%d grid=%dx%d: %.4f ms  %.1f keys/us  %.1f TFLOP/s  %.0f GB/s(kv)\n", M, NB, sms, mult, t_b, NB / (t_b * 1e3), flop / (t_b * 1e-3) / 1e12, kb_bytes / (t_b * 1e-3) / 1e9);
    if (native_ok) {
      float t_n = timeit([&] { qk_native<MT, false><<<grid, 128>>>(q8, qs, kcb, ksb, NB, dS, dred); }, iters);
      printf("BENCH native M=%d keys=%d grid=%dx%d: %.4f ms  %.1f keys/us  %.1f TFLOP/s  %.0f GB/s(kv)  speedup_vs_bf16=%.2fx\n", M, NB, sms, mult, t_n, NB / (t_n * 1e3), flop / (t_n * 1e-3) / 1e12, kb_bytes / (t_n * 1e-3) / 1e9, t_b / t_n);
    }
  }
}

int main(int argc, char** argv) {
  std::string dir = argv[1]; int rep = argc > 2 ? atoi(argv[2]) : 64, iters = argc > 3 ? atoi(argv[3]) : 20;
  auto meta = rd(dir + "/meta.json"); std::string ms(meta.begin(), meta.end());
  auto geti = [&](const char* k) { size_t p = ms.find(std::string("\"") + k + "\""); return atoi(ms.c_str() + ms.find(':', p) + 1); };
  int N = geti("N"), M = geti("M"); if (geti("D") != D) { printf("D must be 128\n"); return 1; }
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("device %s sm_%d%d SMs=%d  N=%d M=%d rep=%d\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount, N, M, rep);
#ifdef UQ_NATIVE
  const bool nat = true;
#else
  const bool nat = false;
#endif
  if (M == 64) run<4>(dir, N, rep, iters, pr.multiProcessorCount, nat);
  else if (M == 32) run<2>(dir, N, rep, iters, pr.multiProcessorCount, nat);
  else { printf("M must be 32 or 64\n"); return 1; }
  return 0;
}
