// pq-repro fused decode attention over an E8-ball-compressed KV cache (v4).
// Same source builds with nvcc (CUDA) and hipcc (HIP/ROCm): only portable
// intrinsics (__syncthreads, shared memory), no warp shuffles, so the AMD
// wave64 vs NVIDIA warp32 difference does not matter.
//
// Layout (contiguous, not paged):
//   q        f32  [B, Hq, D]                  (unrotated; the kernel rotates it)
//   codes    u8   [B, Hkv, stream_bytes]      LSB-first bitstream, 16 b-bit idx per vector
//   scale    f16  [B, Hkv, T]
//   tab      u32  SoA leader table + radix (pq_ball_decode.h, built by pq_build_soa)
// Formats: FMT_PQ (ball codes), FMT_BF16 ([B,Hkv,T,D] bf16), FMT_FP8 (e4m3 + f32 scale per vector).
// Algorithm: split-KV flash decoding. grid = (nsplit, Hkv, B), 128 threads (= D).
//   per 32-token tile: stage codes in smem -> each thread decodes 4 groups of 8 into
//   int8 lattice coordinates (exact: 2x in [-31, 31]) + one f32 scale per token ->
//   scores for the G query heads (rotated domain) -> online softmax -> same for V ->
//   acc[g] (thread = output dim). The smem tile keeps each format in its compact
//   storage type (int8 / fp8 / bf16) and converts on read, which keeps LDS small and
//   occupancy high. Partials (m, l, acc) per split; pq_reduce merges splits and
//   applies the inverse rotation to the output.
#pragma once
#include <stdint.h>
#define PQ_DEV __host__ __device__ __forceinline__
#include "../common/pq_ball_decode.h"

#define PQ_D 128
#define PQ_TT 32
#define PQ_NT 128

enum { FMT_PQ = 0, FMT_BF16 = 1, FMT_FP8 = 2 };

struct PQArgs {
  const float* q;
  const uint8_t* kc; const uint16_t* ks; const uint32_t* ktab; int bK; int kL; long k_stream;
  const uint8_t* vc; const uint16_t* vs; const uint32_t* vtab; int bV; int vL; long v_stream;
  const uint16_t* kb; const uint16_t* vb;               // bf16
  const uint8_t* k8; const uint8_t* v8; const float* k8s; const float* v8s;  // fp8
  const float* signs;
  float* part_o; float* part_ml; float* out;
  int B, Hq, Hkv, T, nsplit, tokens_per_split, tab_smem;  // tab_smem: copy leader tables to smem
};

// words per tile row: 1-byte formats 33 (132 B >= 128, odd stride -> no bank conflicts), bf16 65
template <int FMT> struct TileCfg { static constexpr int RW = (FMT == FMT_BF16) ? 65 : 33; };

// in-place normalized FWHT of a 128-vector in smem, 128 threads
__device__ __forceinline__ void pq_fwht128(float* x) {
  const int t = threadIdx.x;
#pragma unroll
  for (int h = 1; h < PQ_D; h <<= 1) {
    __syncthreads();
    float a = 0.f, b = 0.f;
    const bool lo = ((t & h) == 0);
    if (lo) { a = x[t]; b = x[t + h]; }
    __syncthreads();
    if (lo) { x[t] = a + b; x[t + h] = a - b; }
  }
  __syncthreads();
  x[t] *= 0.08838834764831845f;  // 1/sqrt(128)
  __syncthreads();
}

__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
  return (uint32_t)(a & 255) | ((uint32_t)(b & 255) << 8) | ((uint32_t)(c & 255) << 16) | ((uint32_t)(d & 255) << 24);
}

// Load (and for PQ, decode) one tile of n tokens into tile32 (compact) + tsc (per-token scale).
template <int FMT, int ND>
__device__ __forceinline__ void load_tile(const PQArgs& a, bool isK, int bh, int t0, int n, uint32_t* wbuf,
                                          const PQTab& tab, uint32_t* tile32, float* tsc) {
  const int tid = threadIdx.x;
  constexpr int RW = TileCfg<FMT>::RW;
  if (FMT == FMT_PQ) {
    const int b = isK ? a.bK : a.bV;
    const uint8_t* stream = (isK ? a.kc : a.vc) + (long)bh * (isK ? a.k_stream : a.v_stream);
    const uint16_t* sc = (isK ? a.ks : a.vs) + (long)bh * a.T;
    const long byte0 = (long)t0 * 2 * b;  // t0 multiple of 32 -> multiple of 64 bytes
    const int nw = (n * 2 * b + 3) / 4 + 2;
    const uint32_t* src = (const uint32_t*)(stream + byte0);
    for (int i = tid; i < nw; i += PQ_NT) wbuf[i] = src[i];  // stream has >= 8 slack bytes
    if (tid < n) tsc[tid] = 0.5f * pq_half_to_float(sc[t0 + tid]);
    __syncthreads();
    for (int gi = tid; gi < n * 16; gi += PQ_NT) {
      const int t = gi >> 4, g = gi & 15;
      const uint32_t idx = pq_get_bits(wbuf, (uint32_t)gi * b, b);
      int y[8];
      pq_ball_decode3<ND>(idx, tab, y);
      uint32_t* dst = tile32 + t * RW + g * 2;
      dst[0] = pack4(y[0], y[1], y[2], y[3]);
      dst[1] = pack4(y[4], y[5], y[6], y[7]);
    }
  } else if (FMT == FMT_BF16) {
    const uint32_t* p2 = (const uint32_t*)((isK ? a.kb : a.vb) + ((long)bh * a.T + t0) * PQ_D);
    for (int i = tid; i < n * PQ_D / 2; i += PQ_NT) tile32[(i >> 6) * RW + (i & 63)] = p2[i];
    if (tid < n) tsc[tid] = 1.f;
  } else {
    const uint32_t* p4 = (const uint32_t*)((isK ? a.k8 : a.v8) + ((long)bh * a.T + t0) * PQ_D);
    for (int i = tid; i < n * PQ_D / 4; i += PQ_NT) tile32[(i >> 5) * RW + (i & 31)] = p4[i];
    if (tid < n) tsc[tid] = (isK ? a.k8s : a.v8s)[(long)bh * a.T + t0 + tid];
  }
  __syncthreads();
}

// element (t, d) of a tile as float, without the per-token scale
template <int FMT>
__device__ __forceinline__ float tile_elem(const uint32_t* tile32, int t, int d) {
  constexpr int RW = TileCfg<FMT>::RW;
  if (FMT == FMT_BF16) {
    const uint32_t w = tile32[t * RW + (d >> 1)];
    return pq_bf16_to_float((uint16_t)((d & 1) ? (w >> 16) : (w & 0xFFFF)));
  }
  const uint32_t w = tile32[t * RW + (d >> 2)];
  const uint8_t byte = (uint8_t)(w >> (8 * (d & 3)));
  if (FMT == FMT_PQ) return (float)(int8_t)byte;
  return pq_e4m3_to_float(byte);
}

// dot(q[0:128], tile row t) without the per-token scale
template <int FMT>
__device__ __forceinline__ float tile_dot(const uint32_t* tile32, int t, const float* qr) {
  constexpr int RW = TileCfg<FMT>::RW;
  const uint32_t* row = tile32 + t * RW;
  float acc = 0.f;
  if (FMT == FMT_BF16) {
#pragma unroll 8
    for (int w = 0; w < 64; ++w) {
      const uint32_t x = row[w];
      acc += qr[2 * w] * pq_bf16_to_float((uint16_t)(x & 0xFFFF)) + qr[2 * w + 1] * pq_bf16_to_float((uint16_t)(x >> 16));
    }
  } else {
#pragma unroll 8
    for (int w = 0; w < 32; ++w) {
      const uint32_t x = row[w];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint8_t byte = (uint8_t)(x >> (8 * j));
        const float v = (FMT == FMT_PQ) ? (float)(int8_t)byte : pq_e4m3_to_float(byte);
        acc += qr[4 * w + j] * v;
      }
    }
  }
  return acc;
}

template <int FMT, int G, int ND>
__global__ void __launch_bounds__(PQ_NT) pq_attn_split(PQArgs a) {
  extern __shared__ float smem[];
  constexpr int RW = TileCfg<FMT>::RW;
  float* qs = smem;                        // G*D (rotated, pre-scaled queries)
  float* sc = qs + G * PQ_D;               // G*TT scores / probs
  float* tsc = sc + G * PQ_TT;             // TT per-token scale
  float* mrow = tsc + PQ_TT;               // G
  float* lrow = mrow + G;                  // G
  float* alpha = lrow + G;                 // G
  uint32_t* tile32 = (uint32_t*)(alpha + G);       // TT*RW
  uint32_t* wbuf = tile32 + PQ_TT * RW;            // up to TT*16*32 bits + 4 words
  uint32_t* ktab_s = wbuf + (PQ_TT * 16 + 4);
  uint32_t* vtab_s = ktab_s + pq_tab_words(a.kL);

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z, tid = threadIdx.x;
  const int bh = b * a.Hkv + h;
  const int tbeg = split * a.tokens_per_split;
  const int tend = min(a.T, tbeg + a.tokens_per_split);

  const uint32_t* kt = a.ktab;
  const uint32_t* vt = a.vtab;
  if (FMT == FMT_PQ && a.tab_smem) {
    const int kw = pq_tab_words(a.kL), vw = pq_tab_words(a.vL);
    for (int i = tid; i < kw; i += PQ_NT) ktab_s[i] = a.ktab[i];
    for (int i = tid; i < vw; i += PQ_NT) vtab_s[i] = a.vtab[i];
    kt = ktab_s;
    vt = vtab_s;
  }
  const PQTab ktab = pq_tab_view(kt, a.kL, a.bK);
  const PQTab vtab = pq_tab_view(vt, a.vL, a.bV);
  const float qscale = 0.08838834764831845f;  // softmax 1/sqrt(D)
  for (int g = 0; g < G; ++g) {
    const float* qp = a.q + ((long)b * a.Hq + h * G + g) * PQ_D;
    qs[g * PQ_D + tid] = qp[tid] * (FMT == FMT_PQ ? a.signs[tid] : 1.f);
    if (FMT == FMT_PQ) pq_fwht128(qs + g * PQ_D);
    __syncthreads();
    qs[g * PQ_D + tid] *= qscale;
  }
  if (tid < G) { mrow[tid] = -1e30f; lrow[tid] = 0.f; }
  float acc[G];
#pragma unroll
  for (int g = 0; g < G; ++g) acc[g] = 0.f;
  __syncthreads();

  constexpr int NJ = (G * PQ_TT + PQ_NT - 1) / PQ_NT;
  for (int t0 = tbeg; t0 < tend; t0 += PQ_TT) {
    const int n = min(PQ_TT, tend - t0);
    load_tile<FMT, ND>(a, true, bh, t0, n, wbuf, ktab, tile32, tsc);
    for (int it = tid; it < G * PQ_TT; it += PQ_NT) {
      const int g = it / PQ_TT, t = it % PQ_TT;
      sc[it] = (t < n) ? tsc[t] * tile_dot<FMT>(tile32, t, qs + g * PQ_D) : -1e30f;
    }
    __syncthreads();
    // online softmax, parallel over (head, token)
    float pv[NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int it = tid + j * PQ_NT;
      if (it < G * PQ_TT) {
        const int g = it / PQ_TT;
        float mx = mrow[g];
        for (int t = 0; t < n; ++t) mx = fmaxf(mx, sc[g * PQ_TT + t]);
        pv[j] = ((it % PQ_TT) < n) ? __expf(sc[it] - mx) : 0.f;
        if ((it % PQ_TT) == 0) alpha[g] = mx;  // new max, read after the barrier
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int it = tid + j * PQ_NT;
      if (it < G * PQ_TT) sc[it] = pv[j];
    }
    __syncthreads();
    if (tid < G) {
      float l = 0.f;
      for (int t = 0; t < n; ++t) l += sc[tid * PQ_TT + t];
      const float mx = alpha[tid];
      const float al = __expf(mrow[tid] - mx);
      lrow[tid] = lrow[tid] * al + l;
      mrow[tid] = mx;
      alpha[tid] = al;
    }
    __syncthreads();  // also protects the K tile before V overwrites it
    load_tile<FMT, ND>(a, false, bh, t0, n, wbuf, vtab, tile32, tsc);
    float vv[PQ_TT];
#pragma unroll
    for (int t = 0; t < PQ_TT; ++t) vv[t] = (t < n) ? tsc[t] * tile_elem<FMT>(tile32, t, tid) : 0.f;
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float s = acc[g] * alpha[g];
      const float* pr = sc + g * PQ_TT;
#pragma unroll
      for (int t = 0; t < PQ_TT; ++t) s += pr[t] * vv[t];
      acc[g] = s;
    }
    __syncthreads();
  }
  for (int g = 0; g < G; ++g) {
    const long pi = ((long)b * a.Hq + h * G + g) * a.nsplit + split;
    a.part_o[pi * PQ_D + tid] = acc[g];
    if (tid == 0) { a.part_ml[pi * 2] = mrow[g]; a.part_ml[pi * 2 + 1] = lrow[g]; }
  }
}

// grid = B*Hq, 128 threads: merge splits, inverse rotation (PQ only), write out f32
template <int FMT>
__global__ void __launch_bounds__(PQ_NT) pq_reduce(PQArgs a) {
  __shared__ float x[PQ_D];
  const int bh = blockIdx.x, tid = threadIdx.x;
  float M = -1e30f;
  for (int s = 0; s < a.nsplit; ++s) M = fmaxf(M, a.part_ml[((long)bh * a.nsplit + s) * 2]);
  float num = 0.f, den = 0.f;
  for (int s = 0; s < a.nsplit; ++s) {
    const long pi = (long)bh * a.nsplit + s;
    const float l = a.part_ml[pi * 2 + 1];
    if (l == 0.f) continue;
    const float w = __expf(a.part_ml[pi * 2] - M);
    num += w * a.part_o[pi * PQ_D + tid];
    den += w * l;
  }
  x[tid] = num / den;
  __syncthreads();
  if (FMT == FMT_PQ) {
    pq_fwht128(x);
    x[tid] *= a.signs[tid];
  }
  a.out[(long)bh * PQ_D + tid] = x[tid];
}

template <int FMT>
inline size_t pq_smem_bytes(int G, int tab_words) {
  return sizeof(float) * (G * PQ_D + G * PQ_TT + PQ_TT + 3 * G) + sizeof(uint32_t) * (PQ_TT * TileCfg<FMT>::RW) +
         sizeof(uint32_t) * (PQ_TT * 16 + 4) + sizeof(uint32_t) * tab_words;
}

// host: 8-u32-row leader table -> SoA + radix layout used by the kernels
#include <vector>
inline std::vector<uint32_t> pq_build_soa(const std::vector<uint32_t>& rows, int b) {
  const int L = (int)rows.size() / 8;
  std::vector<uint32_t> o(pq_tab_words(L), 0);
  for (int i = 0; i < L; ++i) pq_tab_fill_row(o.data(), rows.data(), L, i);
  for (int j = 0; j < (1 << PQ_RADIX_BITS); ++j) pq_tab_fill_radix(o.data(), rows.data(), L, b, j);
  return o;
}
