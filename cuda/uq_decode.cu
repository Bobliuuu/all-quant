// UltraQuant native CUDA kernel: split-KV fused K-query FP4/UE8M0 paged decode attention.
// mma.sync m16n8k16 bf16 tensor cores, software FP4 -> bf16 dequant, cp.async multi-stage.
#include "uq_decode.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <algorithm>

#define DEV __device__ __forceinline__
#ifndef UQ_ABL
#define UQ_ABL 0   // ablation bitmask (perf experiments only): 1 = no dequant ALU, 2 = no MMA
#endif

DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
DEV void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> DEV void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

DEV void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEV void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEV void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEV uint32_t pack_bf16(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
DEV uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}

// e2m1x2 byte -> bf16x2 holding (value * 2^-126); lo nibble -> low half.
DEV uint32_t fp4x2_raw(uint32_t b) {
  uint32_t lo = ((b & 7u) << 6) | ((b & 8u) << 12);
  uint32_t hi = ((b & 0x70u) << 2) | ((b & 0x80u) << 8);
  return lo | (hi << 16);
}

// Dequantise one 16-byte chunk (32 elements, one UE8M0 group) into 32 bf16 at dst (16B aligned).
DEV void dequant_chunk(uint4 raw, uint32_t s, uint4* dst) {
#if UQ_ABL & 1
  dst[0] = raw; dst[1] = raw; dst[2] = raw; dst[3] = raw; return;
#endif
  uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
  uint32_t o[16];
  #pragma unroll
  for (int i = 0; i < 4; i++)
    #pragma unroll
    for (int j = 0; j < 4; j++) o[i * 4 + j] = fp4x2_raw((w[i] >> (8 * j)) & 0xFFu);
  if (s <= 128u) {   // 2^(s-1) is representable: single multiply
    uint32_t f = (s + 126u) << 7; f |= f << 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = mul_bf16x2(o[i], f);
  } else {
    uint32_t f1 = 0x7E807E80u;               // 2^126
    uint32_t f2 = s << 7; f2 |= f2 << 16;    // 2^(s-127)
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = mul_bf16x2(mul_bf16x2(o[i], f1), f2);
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) dst[i] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}

#include "uq_decode_w.cuh"

template <int D, int PAGE, int BM, int NW, int ST>
struct Cfg {
  static constexpr int DH = D / 2, NG = D / 32, LD = D + 8;
  static constexpr int NWM = BM / 16;
  static constexpr int NWK = NW / NWM > 0 ? NW / NWM : 1;
  static constexpr int KPW = PAGE / NWK > 0 ? PAGE / NWK : 1;       // keys per warp per page
  static constexpr int NT = KPW / 8;           // S n8 tiles per warp
  static constexpr int STAGE_B = 2 * PAGE * (DH + NG);
  static constexpr int CS = D + 4;
  static constexpr int AB_KV = 2 * PAGE * LD * 2;
  static constexpr int AB_Q = BM * LD * 2;
  static constexpr int AB_C = NWK > 1 ? NWK * BM * CS * 4 + 2 * NWK * BM * 4 : 0;
  static constexpr int AB = AB_KV > AB_Q ? (AB_KV > AB_C ? AB_KV : AB_C) : (AB_Q > AB_C ? AB_Q : AB_C);
  static constexpr int SMEM = AB + ST * STAGE_B;
  static constexpr bool OK = (NW % NWM == 0) && (NW / NWM >= 1) && (PAGE / (NW / NWM)) % 16 == 0 && (PAGE / (NW / NWM)) >= 16;
};

template <int D, int PAGE, int BM, int NW, int ST>
__global__ void __launch_bounds__(NW * 32)
uq_split_kernel(const UQParams a, int G, int gp_shift) {
  using C = Cfg<D, PAGE, BM, NW, ST>;
  constexpr int DH = C::DH, NG = C::NG, LD = C::LD, NWM = C::NWM, NWK = C::NWK, KPW = C::KPW, NT = C::NT;
  constexpr int T = NW * 32;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp % NWM, wk = warp / NWM;
  const int b = blockIdx.x, hk = blockIdx.y, sp = blockIdx.z;
  const int K = a.K, HKV = a.HKV, HQ = a.HQ, nsplit = a.nsplit;
  const int seq_len = a.seq_lens[b];
  const int n_pages = (seq_len + PAGE - 1) / PAGE;
  const int pps = (n_pages + nsplit - 1) / nsplit;
  const int p_lo = sp * pps;
  const int p_hi = min(p_lo + pps, n_pages);
  const size_t rowbase = ((size_t)(b * HKV + hk) * nsplit + sp) * BM;
  if (p_lo >= p_hi) {   // empty split: only the lse marker is needed
    for (int r = tid; r < BM; r += T) a.ml[rowbase + r] = -INFINITY;
    return;
  }
  __nv_bfloat16* sQ = (__nv_bfloat16*)smem;
  __nv_bfloat16* sK = (__nv_bfloat16*)smem;
  __nv_bfloat16* sV = sK + PAGE * LD;
  uint8_t* stages = smem + C::AB;

  // ---- Q tile -> smem (rows j*G_PAD+g, zero padded)
  {
    const __nv_bfloat16* q = (const __nv_bfloat16*)a.q;
    for (int c = tid; c < BM * (D / 8); c += T) {
      int r = c / (D / 8), cc = c % (D / 8);
      int j = r >> gp_shift, g = r & ((1 << gp_shift) - 1);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (j < K && g < G) v = *(const uint4*)(q + ((size_t)(b * K + j) * HQ + hk * G + g) * D + cc * 8);
      *(uint4*)(sQ + r * LD + cc * 8) = v;
    }
  }
  const int32_t* bt = a.block_table + (size_t)b * a.max_pages;
  auto load_stage = [&](int p, int s) {
    uint8_t* st = stages + s * C::STAGE_B;
    size_t page = (size_t)bt[p];
    size_t cb = (page * HKV + hk) * PAGE;
    const uint8_t* src[4] = {a.kc + cb * DH, a.ks + cb * NG, a.vc + cb * DH, a.vs + cb * NG};
    const int sz[4] = {PAGE * DH, PAGE * NG, PAGE * DH, PAGE * NG};
    uint8_t* dst = st;
    #pragma unroll
    for (int t = 0; t < 4; t++) {
      for (int c = tid; c < sz[t] / 16; c += T) cp_async16(smem_u32(dst + c * 16), src[t] + c * 16);
      dst += sz[t];
    }
  };
  #pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (p_lo + s < p_hi) load_stage(p_lo + s, s);
    cp_commit();
  }
  __syncthreads();
  // Q fragments (A operand) -> registers
  uint32_t qa[D / 16][4];
  #pragma unroll
  for (int ks = 0; ks < D / 16; ks++) {
    int row = wm * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    int col = ks * 16 + (lane >> 4) * 8;
    ldsm4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], smem_u32(sQ + row * LD + col));
  }
  // row bookkeeping for this thread's two rows
  const int r0 = wm * 16 + (lane >> 2), r1 = r0 + 8;
  auto lim_of = [&](int r) {
    int j = r >> gp_shift, g = r & ((1 << gp_shift) - 1);
    return (j < K && g < G) ? seq_len - K + j + 1 : 0;
  };
  const int lim0 = lim_of(r0), lim1 = lim_of(r1);
  const float sc = a.sm_scale * 1.4426950408889634f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  float acc[D / 8][4];
  #pragma unroll
  for (int i = 0; i < D / 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

  for (int it = 0; it < p_hi - p_lo; it++) {
    const int p = p_lo + it;
    cp_wait<ST - 2>();
    __syncthreads();   // raw page `it` visible; previous compute done
    {
      int pn = p + ST - 1;
      if (pn < p_hi) load_stage(pn, (it + ST - 1) % ST);
      cp_commit();
    }
    // ---- dequant raw page -> bf16 smem
    {
      const uint8_t* st = stages + (it % ST) * C::STAGE_B;
      const uint8_t* kr = st; const uint8_t* ksc = st + PAGE * DH;
      const uint8_t* vr = ksc + PAGE * NG; const uint8_t* vsc = vr + PAGE * DH;
      for (int c = tid; c < 2 * PAGE * NG; c += T) {
        int which = c >= PAGE * NG; int r = which ? c - PAGE * NG : c;
        int key = r / NG, g = r % NG;
        uint4 raw = *(const uint4*)((which ? vr : kr) + key * DH + g * 16);
        uint32_t s = (which ? vsc : ksc)[key * NG + g];
        dequant_chunk(raw, s, (uint4*)((which ? sV : sK) + key * LD + g * 32));
      }
    }
    __syncthreads();
    // ---- S = Q K^T for this warp's keys
    float s[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
    #pragma unroll
    for (int np = 0; np < KPW / 16; np++) {
      #pragma unroll
      for (int ks = 0; ks < D / 16; ks++) {
        int mi = lane >> 3;
        int key = wk * KPW + np * 16 + (mi >> 1) * 8 + (lane & 7);
        int col = ks * 16 + (mi & 1) * 8;
        uint32_t b0, b1, b2, b3;
        ldsm4(b0, b1, b2, b3, smem_u32(sK + key * LD + col));
#if !(UQ_ABL & 2)
        mma16816(s[2 * np], qa[ks], b0, b1);
        mma16816(s[2 * np + 1], qa[ks], b2, b3);
#else
        s[2*np][0] += __uint_as_float(b0); s[2*np+1][0] += __uint_as_float(b2);
#endif
      }
    }
    // ---- mask + online softmax
    float mx0 = -INFINITY, mx1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      int pos = p * PAGE + wk * KPW + nt * 8 + 2 * (lane & 3);
      s[nt][0] = pos < lim0 ? s[nt][0] * sc : -INFINITY;
      s[nt][1] = pos + 1 < lim0 ? s[nt][1] * sc : -INFINITY;
      s[nt][2] = pos < lim1 ? s[nt][2] * sc : -INFINITY;
      s[nt][3] = pos + 1 < lim1 ? s[nt][3] * sc : -INFINITY;
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
    float al0 = exp2f(m0 - ms0), al1 = exp2f(m1 - ms1);
    m0 = mn0; m1 = mn1;
    float ps0 = 0.f, ps1 = 0.f;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      s[nt][0] = exp2f(s[nt][0] - ms0); s[nt][1] = exp2f(s[nt][1] - ms0);
      s[nt][2] = exp2f(s[nt][2] - ms1); s[nt][3] = exp2f(s[nt][3] - ms1);
      ps0 += s[nt][0] + s[nt][1]; ps1 += s[nt][2] + s[nt][3];
    }
    l0 = l0 * al0 + ps0; l1 = l1 * al1 + ps1;
    #pragma unroll
    for (int i = 0; i < D / 8; i++) { acc[i][0] *= al0; acc[i][1] *= al0; acc[i][2] *= al1; acc[i][3] *= al1; }
    // ---- O += P V
    #pragma unroll
    for (int kk = 0; kk < KPW / 16; kk++) {
      uint32_t pa[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]), pack_bf16(s[2 * kk][2], s[2 * kk][3]),
                        pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
      #pragma unroll
      for (int dt = 0; dt < D / 16; dt++) {
        int key = wk * KPW + kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        int col = dt * 16 + (lane >> 4) * 8;
        uint32_t b0, b1, b2, b3;
        ldsm4t(b0, b1, b2, b3, smem_u32(sV + key * LD + col));
#if !(UQ_ABL & 2)
        mma16816(acc[2 * dt], pa, b0, b1);
        mma16816(acc[2 * dt + 1], pa, b2, b3);
#else
        acc[2*dt][0] += __uint_as_float(b0 ^ pa[0]); acc[2*dt+1][0] += __uint_as_float(b2 ^ pa[1]);
#endif
      }
    }
  }
  // ---- finalize: quad-reduce l
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const int gm = (1 << gp_shift) - 1;
  auto row_ok = [&](int r) { return (r >> gp_shift) < K && (r & gm) < G; };
  if constexpr (NWK == 1) {
    float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    float* mo = a.mo + rowbase * D;
    #pragma unroll
    for (int i = 0; i < D / 8; i++) {
      int col = i * 8 + 2 * (lane & 3);
      if (row_ok(r0)) *(float2*)(mo + (size_t)r0 * D + col) = make_float2(acc[i][0] * i0, acc[i][1] * i0);
      if (row_ok(r1)) *(float2*)(mo + (size_t)r1 * D + col) = make_float2(acc[i][2] * i1, acc[i][3] * i1);
    }
    if ((lane & 3) == 0) {
      a.ml[rowbase + r0] = l0 > 0.f ? m0 + log2f(l0) : -INFINITY;
      a.ml[rowbase + r1] = l1 > 0.f ? m1 + log2f(l1) : -INFINITY;
    }
  } else {
    __syncthreads();
    float* cb = (float*)smem;                       // [NWK][BM][CS]
    float* cm = cb + NWK * BM * C::CS;              // [NWK][BM]
    float* cl = cm + NWK * BM;
    #pragma unroll
    for (int i = 0; i < D / 8; i++) {
      int col = i * 8 + 2 * (lane & 3);
      *(float2*)(cb + ((size_t)wk * BM + r0) * C::CS + col) = make_float2(acc[i][0], acc[i][1]);
      *(float2*)(cb + ((size_t)wk * BM + r1) * C::CS + col) = make_float2(acc[i][2], acc[i][3]);
    }
    if ((lane & 3) == 0) {
      cm[wk * BM + r0] = m0; cm[wk * BM + r1] = m1; cl[wk * BM + r0] = l0; cl[wk * BM + r1] = l1;
    }
    __syncthreads();
    float* mo = a.mo + rowbase * D;
    for (int c = tid; c < BM * (D / 4); c += T) {
      int r = c / (D / 4), d4 = c % (D / 4);
      if (!row_ok(r)) continue;
      float M = -INFINITY;
      #pragma unroll
      for (int w = 0; w < NWK; w++) M = fmaxf(M, cm[w * BM + r]);
      float Ms = M == -INFINITY ? 0.f : M;
      float L = 0.f; float4 o = make_float4(0, 0, 0, 0);
      #pragma unroll
      for (int w = 0; w < NWK; w++) {
        float wt = exp2f(cm[w * BM + r] - Ms);
        L += wt * cl[w * BM + r];
        float4 v = *(const float4*)(cb + ((size_t)w * BM + r) * C::CS + d4 * 4);
        o.x += wt * v.x; o.y += wt * v.y; o.z += wt * v.z; o.w += wt * v.w;
      }
      float inv = L > 0.f ? 1.f / L : 0.f;
      o.x *= inv; o.y *= inv; o.z *= inv; o.w *= inv;
      *(float4*)(mo + (size_t)r * D + d4 * 4) = o;
      if (d4 == 0) a.ml[rowbase + r] = L > 0.f ? Ms + log2f(L) : -INFINITY;
    }
    // rows not valid: still mark lse
    for (int r = tid; r < BM; r += T) if (!row_ok(r)) a.ml[rowbase + r] = -INFINITY;
  }
  if constexpr (NWK == 1) {
    // invalid rows already got lse -inf (lim==0 -> l==0)
  }
}

// one warp per (b, hk, j, g) output row
template <int D>
__global__ void uq_reduce_kernel(const UQParams a, int G, int BM, int gp_shift) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + wid;
  const int KG = a.K * G;
  const int total = a.B * a.HKV * KG;
  if (row >= total) return;
  const int b = row / (a.HKV * KG), rem = row % (a.HKV * KG);
  const int hk = rem / KG, jg = rem % KG;
  const int j = jg / G, g = jg % G;
  const int r = (j << gp_shift) + g;
  const int ns = a.nsplit;
  const size_t base = (size_t)(b * a.HKV + hk) * ns;
  float mx = -INFINITY;
  for (int s = 0; s < ns; s++) mx = fmaxf(mx, a.ml[(base + s) * BM + r]);
  float acc[D / 32];
  #pragma unroll
  for (int i = 0; i < D / 32; i++) acc[i] = 0.f;
  float wsum = 0.f;
  if (mx != -INFINITY) {
    for (int s = 0; s < ns; s++) {
      float lse = a.ml[(base + s) * BM + r];
      if (lse == -INFINITY) continue;
      float w = exp2f(lse - mx);
      wsum += w;
      const float* o = a.mo + ((base + s) * BM + r) * D;
      #pragma unroll
      for (int i = 0; i < D / 32; i++) acc[i] += w * o[lane + 32 * i];
    }
  }
  float inv = wsum > 0.f ? 1.f / wsum : 0.f;
  __nv_bfloat16* out = (__nv_bfloat16*)a.out + ((size_t)(b * a.K + j) * a.HQ + hk * G + g) * D;
  #pragma unroll
  for (int i = 0; i < D / 32; i++) out[lane + 32 * i] = __float2bfloat16(acc[i] * inv);
}

// ------------------------------------------------------------------ host
static bool smem_fits(int bytes) {
  int dev = 0, lim = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&lim, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  return bytes <= lim;
}
static int pow2ceil(int x) { int p = 1; while (p < x) p <<= 1; return p; }
int uq_block_m(int K, int HQ, int HKV) { return std::max(16, pow2ceil(K * pow2ceil(HQ / HKV))); }
size_t uq_mo_elems(const UQParams& p) { return (size_t)p.B * p.HKV * p.nsplit * uq_block_m(p.K, p.HQ, p.HKV) * p.D; }
size_t uq_ml_elems(const UQParams& p) { return (size_t)p.B * p.HKV * p.nsplit * uq_block_m(p.K, p.HQ, p.HKV); }

template <int D, int PAGE, int BM, int NW, int ST>
static cudaError_t launch_cfg(const UQParams& p, cudaStream_t s) {
  using C = Cfg<D, PAGE, BM, NW, ST>;
  if constexpr (!C::OK) { return cudaErrorInvalidValue; } else {
  auto kern = uq_split_kernel<D, PAGE, BM, NW, ST>;
  static bool init = false;
  if (!init) {
    if (!smem_fits(C::SMEM)) return cudaErrorInvalidValue;
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    if (e != cudaSuccess) { cudaGetLastError(); return e; }
    init = true;
  }
  int G = p.HQ / p.HKV, gp = pow2ceil(G), shift = 0;
  while ((1 << shift) < gp) shift++;
  kern<<<dim3(p.B, p.HKV, p.nsplit), NW * 32, C::SMEM, s>>>(p, G, shift);
  int rows = p.B * p.HKV * p.K * G;
  uq_reduce_kernel<D><<<(rows + 3) / 4, 128, 0, s>>>(p, G, BM, shift);
  return cudaGetLastError();
  }
}

#define NW_CASE(D_, PAGE_, BM_, NW_)                                                   \
  case NW_: switch (stages) {                                                         \
    case 2: return launch_cfg<D_, PAGE_, BM_, NW_, 2>(p, s);                          \
    case 3: return launch_cfg<D_, PAGE_, BM_, NW_, 3>(p, s);                          \
    case 4: return launch_cfg<D_, PAGE_, BM_, NW_, 4>(p, s);                          \
    default: return cudaErrorInvalidValue; }

template <int D, int PAGE, int BM>
static cudaError_t dispatch_nw(const UQParams& p, int nw, int stages, cudaStream_t s) {
  if (stages == 0) stages = 3;
  if constexpr (BM == 16) {
    if (nw == 0) nw = 4;
    switch (nw) { NW_CASE(D, PAGE, BM, 1) NW_CASE(D, PAGE, BM, 2) NW_CASE(D, PAGE, BM, 4) NW_CASE(D, PAGE, BM, 8) default: return cudaErrorInvalidValue; }
  } else if constexpr (BM == 32) {
    if (nw == 0) nw = 4;
    switch (nw) { NW_CASE(D, PAGE, BM, 2) NW_CASE(D, PAGE, BM, 4) NW_CASE(D, PAGE, BM, 8) default: return cudaErrorInvalidValue; }
  } else if constexpr (BM == 64) {
    if (nw == 0) nw = 4;
    switch (nw) { NW_CASE(D, PAGE, BM, 4) NW_CASE(D, PAGE, BM, 8) default: return cudaErrorInvalidValue; }
  } else {
    if (nw == 0) nw = 8;
    switch (nw) { NW_CASE(D, PAGE, BM, 8) default: return cudaErrorInvalidValue; }
  }
}

template <int D, int PAGE>
static cudaError_t dispatch_bm(const UQParams& p, int nw, int stages, cudaStream_t s) {
  switch (uq_block_m(p.K, p.HQ, p.HKV)) {
    case 16: return dispatch_nw<D, PAGE, 16>(p, nw, stages, s);
    case 32: return dispatch_nw<D, PAGE, 32>(p, nw, stages, s);
    case 64: return dispatch_nw<D, PAGE, 64>(p, nw, stages, s);
    case 128: return dispatch_nw<D, PAGE, 128>(p, nw, stages, s);
    default: return cudaErrorInvalidValue;
  }
}

template <int D, int PAGE, int BM, int MB, int NW, int ST>
static cudaError_t launch_w(const UQParams& p, cudaStream_t s) {
  using C = CfgW<D, PAGE, BM, MB, NW, ST>;
  if constexpr (!C::OK) { return cudaErrorInvalidValue; } else {
    auto kern = uq_split_w<D, PAGE, BM, MB, NW, ST>;
    static bool init = false;
    if (!init) {
      if (!smem_fits(C::SMEM)) return cudaErrorInvalidValue;
      cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
      if (e != cudaSuccess) { cudaGetLastError(); return e; }
      init = true;
    }
    int G = p.HQ / p.HKV, gp = pow2ceil(G), shift = 0;
    while ((1 << shift) < gp) shift++;
    kern<<<dim3(p.B, p.HKV, p.nsplit), NW * 32, C::SMEM, s>>>(p, G, shift);
    int rows = p.B * p.HKV * p.K * G;
    uq_reduce_kernel<D><<<(rows + 3) / 4, 128, 0, s>>>(p, G, BM, shift);
    return cudaGetLastError();
  }
}

#define W_ST(D_, PAGE_, BM_, MB_, NW_)                                                  \
  case NW_: switch (stages) {                                                           \
    case 2: return launch_w<D_, PAGE_, BM_, MB_, NW_, 2>(p, s);                         \
    case 3: return launch_w<D_, PAGE_, BM_, MB_, NW_, 3>(p, s);                         \
    case 4: return launch_w<D_, PAGE_, BM_, MB_, NW_, 4>(p, s);                         \
    default: return cudaErrorInvalidValue; }
#define W_ST3(D_, PAGE_, BM_, MB_, NW_) case NW_: return launch_w<D_, PAGE_, BM_, MB_, NW_, 3>(p, s);

// full tuning grid for the main shape (D=128, PAGE=64)
template <int BM>
static cudaError_t dispatch_w_main(const UQParams& p, int mb, int nw, int stages, cudaStream_t s) {
  if (stages == 0) stages = 3;
  if constexpr (BM == 16) {
    if (mb == 0) mb = 1; if (nw == 0) nw = 4;
    if (mb == 1) switch (nw) { W_ST(128, 64, 16, 1, 2) W_ST(128, 64, 16, 1, 4) W_ST(128, 64, 16, 1, 8) default: break; }
  } else if constexpr (BM == 32) {
    if (mb == 0) mb = 2; if (nw == 0) nw = 4;
    if (mb == 1) switch (nw) { W_ST(128, 64, 32, 1, 4) W_ST(128, 64, 32, 1, 8) default: break; }
    if (mb == 2) switch (nw) { W_ST(128, 64, 32, 2, 2) W_ST(128, 64, 32, 2, 4) W_ST(128, 64, 32, 2, 8) default: break; }
  } else if constexpr (BM == 64) {
    if (mb == 0) mb = 2; if (nw == 0) nw = 8;
    if (mb == 1) switch (nw) { W_ST(128, 64, 64, 1, 8) W_ST(128, 64, 64, 1, 16) default: break; }
    if (mb == 2) switch (nw) { W_ST(128, 64, 64, 2, 4) W_ST(128, 64, 64, 2, 8) default: break; }
  } else {
    if (mb == 0) mb = 2; if (nw == 0) nw = 8;
    if (mb == 2) switch (nw) { W_ST(128, 64, 128, 2, 8) default: break; }
  }
  return cudaErrorInvalidValue;
}

// other shapes: small default set (ST = 3)
template <int D, int PAGE, int BM>
static cudaError_t dispatch_w_default(const UQParams& p, int mb, int nw, cudaStream_t s) {
  int stages = 3;
  if constexpr (BM == 16) {
    if (mb == 0) mb = 1; if (nw == 0) nw = PAGE >= 64 ? 4 : PAGE / 16;
    if (mb == 1) switch (nw) { W_ST3(D, PAGE, 16, 1, 1) W_ST3(D, PAGE, 16, 1, 2) W_ST3(D, PAGE, 16, 1, 4) W_ST3(D, PAGE, 16, 1, 8) default: break; }
  } else if constexpr (BM == 32) {
    if (mb == 0) mb = 2; if (nw == 0) nw = PAGE >= 32 ? 2 * (PAGE / 32) : 1;
    if (mb == 2) switch (nw) { W_ST3(D, PAGE, 32, 2, 1) W_ST3(D, PAGE, 32, 2, 2) W_ST3(D, PAGE, 32, 2, 4) W_ST3(D, PAGE, 32, 2, 8) default: break; }
    if (mb == 1) switch (nw) { W_ST3(D, PAGE, 32, 1, 2) W_ST3(D, PAGE, 32, 1, 4) W_ST3(D, PAGE, 32, 1, 8) default: break; }
  } else if constexpr (BM == 64) {
    if (mb == 0) mb = 2; if (nw == 0) nw = 2 * (PAGE / 16 > 4 ? 4 : PAGE / 16);
    if (mb == 2) switch (nw) { W_ST3(D, PAGE, 64, 2, 2) W_ST3(D, PAGE, 64, 2, 4) W_ST3(D, PAGE, 64, 2, 8) default: break; }
    if (mb == 1) switch (nw) { W_ST3(D, PAGE, 64, 1, 4) W_ST3(D, PAGE, 64, 1, 8) W_ST3(D, PAGE, 64, 1, 16) default: break; }
  } else {
    if (mb == 0) mb = 2; if (nw == 0) nw = 8;
    if (mb == 2) switch (nw) { W_ST3(D, PAGE, 128, 2, 4) W_ST3(D, PAGE, 128, 2, 8) default: break; }
  }
  return cudaErrorInvalidValue;
}

template <int D, int PAGE>
static cudaError_t dispatch_w(const UQParams& p, int nw, int stages, cudaStream_t s) {
  int bm = uq_block_m(p.K, p.HQ, p.HKV);
  if constexpr (D == 128 && PAGE == 64) {
    switch (bm) {
      case 16: return dispatch_w_main<16>(p, p.mb, nw, stages, s);
      case 32: return dispatch_w_main<32>(p, p.mb, nw, stages, s);
      case 64: return dispatch_w_main<64>(p, p.mb, nw, stages, s);
      case 128: return dispatch_w_main<128>(p, p.mb, nw, stages, s);
    }
  } else {
    switch (bm) {
      case 16: return dispatch_w_default<D, PAGE, 16>(p, p.mb, nw, s);
      case 32: return dispatch_w_default<D, PAGE, 32>(p, p.mb, nw, s);
      case 64: return dispatch_w_default<D, PAGE, 64>(p, p.mb, nw, s);
      case 128: return dispatch_w_default<D, PAGE, 128>(p, p.mb, nw, s);
    }
  }
  return cudaErrorInvalidValue;
}

#ifdef UQ_ENABLE_SM90
#include "uq_decode_sm90.cuh"
#endif

template <int D, int PAGE>
static cudaError_t dispatch_dp(const UQParams& p, int nw, int stages, cudaStream_t s) {
  if (p.variant != 1) {
    cudaError_t e = dispatch_w<D, PAGE>(p, nw, stages, s);
    if (e != cudaErrorInvalidValue || p.variant == 2) return e;
    cudaGetLastError();
  }
  return dispatch_bm<D, PAGE>(p, nw, stages, s);
}

cudaError_t uq_decode_launch(const UQParams& p, int nw, int stages, cudaStream_t s) {
  if (p.nsplit < 1 || p.nsplit > 128) return cudaErrorInvalidValue;
#ifdef UQ_ENABLE_SM90
  if (p.variant == 3) return dispatch_sm90(p, nw, stages, s);   // nw = # MMA warpgroups, stages = dequant stages
#endif
#ifdef UQ_SM90_ONLY
  return cudaErrorInvalidValue;   // dev build: only variant 3
#endif
  if (p.D == 128) {
    switch (p.PAGE) {
      case 16: return dispatch_dp<128, 16>(p, nw, stages, s);
      case 32: return dispatch_dp<128, 32>(p, nw, stages, s);
      case 64: return dispatch_dp<128, 64>(p, nw, stages, s);
      case 128: return dispatch_dp<128, 128>(p, nw, stages, s);
    }
  } else if (p.D == 64) {
    switch (p.PAGE) {
      case 16: return dispatch_dp<64, 16>(p, nw, stages, s);
      case 32: return dispatch_dp<64, 32>(p, nw, stages, s);
      case 64: return dispatch_dp<64, 64>(p, nw, stages, s);
    }
  }
  return cudaErrorInvalidValue;
}

#ifdef UQ_PROF
void uq_prof_read(unsigned long long* out8) { cudaMemcpyFromSymbol(out8, g_prof, 8 * sizeof(unsigned long long)); }
void uq_prof_reset() { unsigned long long z[8] = {0}; cudaMemcpyToSymbol(g_prof, z, sizeof(z)); }
#endif
