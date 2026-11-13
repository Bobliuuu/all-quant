// pq-repro fused decode attention over an E8-ball-compressed KV cache.
// Same source builds with nvcc (CUDA) and hipcc (HIP/ROCm): only portable
// intrinsics (__syncthreads, shared memory), no warp shuffles, so the AMD
// wave64 vs NVIDIA warp32 difference does not matter.
//
// Layout (contiguous, not paged):
//   q        f32  [B, Hq, D]                  (unrotated; the kernel rotates it)
//   codes    u8   [B, Hkv, stream_bytes]      LSB-first bitstream, 16 b-bit idx per vector
//   scale    f16  [B, Hkv, T]
//   tab      u32  [L, 8] leader table (pq_ball_decode.h)
// Formats: FMT_PQ (ball codes), FMT_BF16 ([B,Hkv,T,D] bf16), FMT_FP8 (e4m3 + f32 scale per vector).
// Algorithm: split-KV flash decoding. grid = (nsplit, Hkv, B), 128 threads (= D).
//   per 32-token tile: stage codes in smem -> each thread decodes 4 groups of 8 ->
//   K tile (rotated domain) in smem -> scores for the G query heads -> online softmax ->
//   same for V -> acc[g] (thread = output dim). Partials (m, l, acc) per split;
//   pq_reduce merges splits and applies the inverse rotation to the output.
#pragma once
#include <stdint.h>
#define PQ_DEV __host__ __device__ __forceinline__
#include "../common/pq_ball_decode.h"

#define PQ_D 128
#define PQ_TT 32
#define PQ_NT 128
#define PQ_KSTR (PQ_D + 1)

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

// Decode a PQ tile (n tokens) into tile[t*KSTR + d] = scale * y/2 (rotated domain).
__device__ __forceinline__ void pq_decode_tile(const uint8_t* stream, const uint16_t* sc, int t0, int n, int b,
                                               const PQTab& tab, uint32_t* wbuf, float* tile) {
  const int tid = threadIdx.x;
  // tile bitstream: start bit t0*16*b (t0 multiple of 32 -> byte offset t0*2b, multiple of 64)
  const long byte0 = (long)t0 * 2 * b;
  const int nbytes = n * 2 * b;
  const int nw = (nbytes + 3) / 4 + 2;
  const uint32_t* src = (const uint32_t*)(stream + byte0);
  for (int i = tid; i < nw; i += PQ_NT) wbuf[i] = src[i];  // stream has 8 slack bytes
  __syncthreads();
  for (int gi = tid; gi < n * 16; gi += PQ_NT) {
    const int t = gi >> 4, g = gi & 15;
    const uint32_t idx = pq_get_bits(wbuf, (uint32_t)gi * b, b);
    int y[8];
    pq_ball_decode3(idx, tab, y);
    const float s = 0.5f * pq_half_to_float(sc[t0 + t]);
    float* dst = tile + t * PQ_KSTR + g * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[j] = s * (float)y[j];
  }
  __syncthreads();
}

template <int FMT>
__device__ __forceinline__ void load_tile(const PQArgs& a, bool isK, int bh, int t0, int n, uint32_t* wbuf,
                                          const PQTab& tab, float* tile) {
  if (FMT == FMT_PQ) {
    if (isK) pq_decode_tile(a.kc + (long)bh * a.k_stream, a.ks + (long)bh * a.T, t0, n, a.bK, tab, wbuf, tile);
    else pq_decode_tile(a.vc + (long)bh * a.v_stream, a.vs + (long)bh * a.T, t0, n, a.bV, tab, wbuf, tile);
  } else if (FMT == FMT_BF16) {
    const uint16_t* p = (isK ? a.kb : a.vb) + ((long)bh * a.T + t0) * PQ_D;
    // vectorized: each thread loads 2 bf16 at a time
    const uint32_t* p2 = (const uint32_t*)p;
    for (int i = threadIdx.x; i < n * PQ_D / 2; i += PQ_NT) {
      const uint32_t w = p2[i];
      const int e = 2 * i, t = e / PQ_D, d = e % PQ_D;
      tile[t * PQ_KSTR + d] = pq_bf16_to_float((uint16_t)(w & 0xFFFF));
      tile[t * PQ_KSTR + d + 1] = pq_bf16_to_float((uint16_t)(w >> 16));
    }
    __syncthreads();
  } else {
    const uint8_t* p = (isK ? a.k8 : a.v8) + ((long)bh * a.T + t0) * PQ_D;
    const float* s = (isK ? a.k8s : a.v8s) + (long)bh * a.T + t0;
    const uint32_t* p4 = (const uint32_t*)p;
    for (int i = threadIdx.x; i < n * PQ_D / 4; i += PQ_NT) {
      const uint32_t w = p4[i];
      const int e = 4 * i, t = e / PQ_D, d = e % PQ_D;
      const float st = s[t];
#pragma unroll
      for (int j = 0; j < 4; ++j) tile[t * PQ_KSTR + d + j] = st * pq_e4m3_to_float((uint8_t)(w >> (8 * j)));
    }
    __syncthreads();
  }
}

template <int FMT, int G>
__global__ void __launch_bounds__(PQ_NT) pq_attn_split(PQArgs a) {
  extern __shared__ float smem[];
  float* qs = smem;                       // G*D   (rotated, pre-scaled queries)
  float* tile = qs + G * PQ_D;            // TT*KSTR
  float* sc = tile + PQ_TT * PQ_KSTR;     // G*TT  scores / probs
  float* mrow = sc + G * PQ_TT;           // G
  float* lrow = mrow + G;                 // G
  float* alpha = lrow + G;                // G
  uint32_t* wbuf = (uint32_t*)(alpha + G);         // TT*2*32/4 + 4 words max
  uint32_t* ktab_s = wbuf + (PQ_TT * 16 + 4);
  uint32_t* vtab_s = ktab_s + pq_tab_words(a.kL);

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z, tid = threadIdx.x;
  const int bh = b * a.Hkv + h;
  const int tbeg = split * a.tokens_per_split;
  const int tend = min(a.T, tbeg + a.tokens_per_split);

  // a.ktab / a.vtab hold the SoA + radix layout (pq_tab_words words each), built on the host
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

  for (int t0 = tbeg; t0 < tend; t0 += PQ_TT) {
    const int n = min(PQ_TT, tend - t0);
    load_tile<FMT>(a, true, bh, t0, n, wbuf, ktab, tile);
    for (int it = tid; it < G * PQ_TT; it += PQ_NT) {
      const int g = it / PQ_TT, t = it % PQ_TT;
      float s = -1e30f;
      if (t < n) {
        const float* kr = tile + t * PQ_KSTR;
        const float* qr = qs + g * PQ_D;
        float acc2 = 0.f;
#pragma unroll 16
        for (int d = 0; d < PQ_D; ++d) acc2 += qr[d] * kr[d];
        s = acc2;
      }
      sc[it] = s;
    }
    __syncthreads();
    // online softmax, parallel over (head, token): every thread reads its head's 32
    // scores (smem broadcast), so there is no serial per-head chain.
    float pv[(G * PQ_TT + PQ_NT - 1) / PQ_NT];
#pragma unroll
    for (int j = 0; j < (G * PQ_TT + PQ_NT - 1) / PQ_NT; ++j) {
      const int it = tid + j * PQ_NT;
      if (it < G * PQ_TT) {
        const int g = it / PQ_TT;
        const float m = mrow[g];
        float mx = m;
        for (int t = 0; t < n; ++t) mx = fmaxf(mx, sc[g * PQ_TT + t]);
        pv[j] = ((it % PQ_TT) < n) ? __expf(sc[it] - mx) : 0.f;
        if ((it % PQ_TT) == 0) alpha[g] = mx;  // new max, published after the barrier
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < (G * PQ_TT + PQ_NT - 1) / PQ_NT; ++j) {
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
    __syncthreads();  // also protects tile before V overwrites it
    load_tile<FMT>(a, false, bh, t0, n, wbuf, vtab, tile);
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float s = acc[g] * alpha[g];
      const float* pr = sc + g * PQ_TT;
      for (int t = 0; t < n; ++t) s += pr[t] * tile[t * PQ_KSTR + tid];
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

inline size_t pq_smem_bytes(int G, int tab_words) {
  return sizeof(float) * (G * PQ_D + PQ_TT * PQ_KSTR + G * PQ_TT + 3 * G) + sizeof(uint32_t) * (PQ_TT * 16 + 4) +
         sizeof(uint32_t) * tab_words;
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
