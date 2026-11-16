// wq- lane: fused decode-GEMV kernels for re-encoded weights (batch 1..16), SM80+ (tested on SM90).
//
// y[B, N] = x[B, K] @ W^T, W [N, K]. Every codec decodes its weights in registers straight into the A fragments
// of mma.sync m16n8k16 (bf16 in, fp32 accumulate), so the per-weight cost is ONLY the decode; the FMAs run on the
// tensor cores and the decode is amortised over the batch for free. This isolates the question of the lane:
// how many ALU instructions per weight can a codec spend before the bytes it saves stop paying.
//
// Tiling: CTA = 8 warps, one 16-row m-tile, a K range [st0, st1) of 64-column super-tiles (split-K over grid.y);
// warp w takes super-tiles st0 + w, st0 + w + 8, ... Per lane per super-tile: 32 weights =
// 4 k16 steps x 4 A registers x 2 bf16 (row g = lane/4 for registers 0,2 and g+8 for 1,3).
// Weight element order e = s*8 + r*2 + h  -> row g + 8*(r&1), col 16 s + 2 t + 8*(r>>1) + h  (t = lane%4).
// Group params (bf16 sigma | bf16 mu << 16) per (m-tile, group of G columns, row): w = mu + sigma * v(code).
// tile_mod > 0 makes every m-tile read the weights of m-tile (blockIdx.x % tile_mod): L2-resident weights,
// i.e. the decode-bound (compute) ceiling of a codec.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

enum Codec {
  C_BF16 = 0,    // raw bf16 weights in fragment order (roofline reference with the same loop)
  C_INT4 = 1,    // affine 4-bit, magic-number dequant (0x4300|c = 128+c), LOP3 + HSUB2 + HFMA2 per pair
  C_LUT4 = 2,    // non-uniform 4-bit (NF4 / percentile / Lloyd / k-means): 16-entry smem LUT, one lookup per weight
  C_LUT4P = 3,   // same, 256-entry pair LUT replicated per lane (bank-conflict free, FLUTE-style), 1 lookup per pair
  C_ICDF4 = 4,   // percentile code decoded by computing the Gaussian inverse CDF (erfinvf) at runtime, no LUT
  C_LUT3P = 5,   // 3-bit, 64-entry pair LUT replicated per lane
  C_LUT2P = 6,   // 2-bit, 16-entry pair LUT
  C_TCQ1 = 7,    // QTIP-style bitshift trellis, k=2, L=16, tail-biting per lane, 1MAD computed Gaussian code
  C_TCQ3 = 8,    // same with the 3INST code
  C_VLC = 9,     // prefix-coded (Huffman / bitmap-ternary / sparse) stream, warp-interleaved, on-chip table decode
};

struct Args {
  const uint32_t* w;     // packed weights (layout per codec)
  const uint32_t* prm;   // group params
  const uint32_t* offs;  // VLC: word offset of each warp stream (+1 end)
  const uint32_t* tbl;   // LUT / decode table (global, copied to smem)
  const bf16* x;
  float* y;
  int N, K, B, S, G, tile_mod, tbl_n, maxlen, MTr;
};

__device__ __forceinline__ uint32_t ldg_nc(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ldg_nc2(const uint2* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ldg_nc4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t as_u32(bf162 v) { return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ bf162 as_b2(uint32_t v) { return *reinterpret_cast<bf162*>(&v); }
__device__ __forceinline__ uint32_t pack_f2(float a, float b) { return as_u32(__floats2bfloat162_rn(a, b)); }
__device__ __forceinline__ uint32_t fma_pair(uint32_t v, uint32_t sig2, uint32_t mu2) {
  return as_u32(__hfma2(as_b2(v), as_b2(sig2), as_b2(mu2)));
}
__device__ __forceinline__ uint32_t splat_lo(uint32_t p) { return __byte_perm(p, 0, 0x1010); }  // (lo,lo)
__device__ __forceinline__ uint32_t splat_hi(uint32_t p) { return __byte_perm(p, 0, 0x3232); }  // (hi,hi)

// weight register words per lane per super-tile for fixed-rate codecs
template <int C> struct WW { static constexpr int n = 4; };
template <> struct WW<C_BF16> { static constexpr int n = 16; };
template <> struct WW<C_LUT3P> { static constexpr int n = 3; };
template <> struct WW<C_LUT2P> { static constexpr int n = 2; };
template <> struct WW<C_TCQ1> { static constexpr int n = 2; };
template <> struct WW<C_TCQ3> { static constexpr int n = 2; };

template <int C>
__device__ __forceinline__ void load_w(const uint32_t* w, size_t tile_idx, int lane, uint32_t* r) {
  constexpr int n = WW<C>::n;
  const uint32_t* base = w + tile_idx * (32 * n);
  if constexpr (C == C_BF16) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      uint4 v = ldg_nc4(reinterpret_cast<const uint4*>(base + q * 128) + lane);
      r[q * 4] = v.x; r[q * 4 + 1] = v.y; r[q * 4 + 2] = v.z; r[q * 4 + 3] = v.w;
    }
  } else if constexpr (n == 4) {
    uint4 v = ldg_nc4(reinterpret_cast<const uint4*>(base) + lane);
    r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w;
  } else if constexpr (n == 2) {
    uint2 v = ldg_nc2(reinterpret_cast<const uint2*>(base) + lane);
    r[0] = v.x; r[1] = v.y;
  } else {
#pragma unroll
    for (int j = 0; j < n; j++) r[j] = ldg_nc(base + j * 32 + lane);
  }
}

__device__ __forceinline__ float tcq_val(uint32_t s, bool three) {
  if (!three) {
    uint32_t x = s * 34038481u + 76625530u;
    uint32_t sum = __dp4a(x, 0x01010101u, 0u);
    return (float)(int)sum * (1.0f / 147.8f) - (510.0f / 147.8f);
  } else {
    uint32_t x = s * 89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    __half2 h = *reinterpret_cast<__half2*>(&x);
    return __low2float(h) + __high2float(h);
  }
}

// decode 32 weights of one lane into A fragments a[4 steps][4 regs]
template <int C>
__device__ __forceinline__ void decode(const uint32_t* r, uint32_t pg, uint32_t pg8, const uint32_t* slut, int lane,
                                       uint32_t (*a)[4]) {
  const uint32_t s0 = splat_lo(pg), m0 = splat_hi(pg), s1 = splat_lo(pg8), m1 = splat_hi(pg8);
  if constexpr (C == C_BF16) {
#pragma unroll
    for (int s = 0; s < 4; s++)
#pragma unroll
      for (int q = 0; q < 4; q++) a[s][q] = r[s * 4 + q];
  } else if constexpr (C == C_INT4) {
#pragma unroll
    for (int s = 0; s < 4; s++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t v = ((r[s] >> (4 * q)) & 0x000f000fu) | 0x43004300u;  // (128 + c0, 128 + c1), exact in bf16
        v = as_u32(__hsub2(as_b2(v), as_b2(0x43004300u)));               // (c0, c1), exact
        a[s][q] = fma_pair(v, (q & 1) ? s1 : s0, (q & 1) ? m1 : m0);
      }
  } else if constexpr (C == C_LUT4) {
    const uint16_t* l16 = reinterpret_cast<const uint16_t*>(slut);
#pragma unroll
    for (int s = 0; s < 4; s++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t c0 = (r[s] >> (8 * q)) & 15u, c1 = (r[s] >> (8 * q + 4)) & 15u;
        uint32_t v = (uint32_t)l16[c0] | ((uint32_t)l16[c1] << 16);
        a[s][q] = fma_pair(v, (q & 1) ? s1 : s0, (q & 1) ? m1 : m0);
      }
  } else if constexpr (C == C_LUT4P) {
#pragma unroll
    for (int s = 0; s < 4; s++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t idx = (r[s] >> (8 * q)) & 255u;
        a[s][q] = fma_pair(slut[idx * 32 + lane], (q & 1) ? s1 : s0, (q & 1) ? m1 : m0);
      }
  } else if constexpr (C == C_ICDF4) {
    float fs0 = __uint_as_float(pg << 16), fm0 = __uint_as_float(pg & 0xffff0000u);
    float fs1 = __uint_as_float(pg8 << 16), fm1 = __uint_as_float(pg8 & 0xffff0000u);
#pragma unroll
    for (int s = 0; s < 4; s++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t c0 = (r[s] >> (8 * q)) & 15u, c1 = (r[s] >> (8 * q + 4)) & 15u;
        // z = Phi^-1((c + 0.5) / 16) = sqrt(2) erfinv((2c + 1) / 16 - 1)
        float z0 = 1.41421356f * erfinvf((float)(2 * c0 + 1) * 0.0625f - 1.0f);
        float z1 = 1.41421356f * erfinvf((float)(2 * c1 + 1) * 0.0625f - 1.0f);
        float fs = (q & 1) ? fs1 : fs0, fm = (q & 1) ? fm1 : fm0;
        a[s][q] = pack_f2(fmaf(fs, z0, fm), fmaf(fs, z1, fm));
      }
  } else if constexpr (C == C_LUT3P) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
      uint32_t idx = ((r[j >> 3] >> (4 * (j & 7))) & 15u) | (((r[2] >> (2 * j)) & 3u) << 4);
      int q = j & 3;
      a[j >> 2][q] = fma_pair(slut[idx * 32 + lane], (q & 1) ? s1 : s0, (q & 1) ? m1 : m0);
    }
  } else if constexpr (C == C_LUT2P) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
      uint32_t idx = (r[j >> 3] >> (4 * (j & 7))) & 15u;
      int q = j & 3;
      a[j >> 2][q] = fma_pair(slut[idx], (q & 1) ? s1 : s0, (q & 1) ? m1 : m0);
    }
  } else if constexpr (C == C_TCQ1 || C == C_TCQ3) {
    float fs0 = __uint_as_float(pg << 16), fm0 = __uint_as_float(pg & 0xffff0000u);
    float fs1 = __uint_as_float(pg8 << 16), fm1 = __uint_as_float(pg8 & 0xffff0000u);
    const uint64_t v = (uint64_t)r[0] | ((uint64_t)r[1] << 32);
#pragma unroll
    for (int j = 0; j < 16; j++) {
      float z[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int e = 2 * j + h, sh = 2 * e;
        uint64_t rot = sh ? ((v >> sh) | (v << (64 - sh))) : v;
        z[h] = tcq_val((uint32_t)rot & 0xffffu, C == C_TCQ3);
      }
      int q = j & 3;
      float fs = (q & 1) ? fs1 : fs0, fm = (q & 1) ? fm1 : fm0;
      a[j >> 2][q] = pack_f2(fmaf(fs, z[0], fm), fmaf(fs, z[1], fm));
    }
  }
}

template <int NT>
__device__ __forceinline__ void load_x(const bf16* x, int K, int B, int col0, int lane, uint32_t (*b)[4][2]) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    const int n = nt * 8 + g;
#pragma unroll
    for (int s = 0; s < 4; s++) {
      if (n < B) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(x + (size_t)n * K + col0 + 16 * s + 2 * t);
        b[nt][s][0] = __ldg(p);
        b[nt][s][1] = __ldg(p + 4);
      } else {
        b[nt][s][0] = 0u; b[nt][s][1] = 0u;
      }
    }
  }
}

__device__ __forceinline__ void reduce_store(const Args& A, float* acc, int nacc, int warp, int lane, int rt,
                                             float (*red)[32]) {
  // acc[nt*4 + i] for one m-tile; red: smem [8 * 8][32]
  for (int j = 0; j < nacc; j++) red[warp * 8 + j][lane] = acc[j];
  __syncthreads();
  if (warp == 0) {
    const int g = lane >> 2, t = lane & 3;
    for (int j = 0; j < nacc; j++) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < 8; w++) v += red[w * 8 + j][lane];
      const int nt = j >> 2, i = j & 3;
      const int row = rt * 16 + g + ((i & 2) ? 8 : 0);
      const int n = nt * 8 + 2 * t + (i & 1);
      if (n < A.B) atomicAdd(A.y + (size_t)n * A.N + row, v);
    }
  }
  __syncthreads();
}

// ----------------------------------------------------------------------------------------- fixed-rate kernel
// MT m-tiles (16 rows each) per CTA: every activation fragment is reused MT times (amortises x loads + addressing).
template <int C, int NT, int MT>
__global__ void __launch_bounds__(256) k_fixed(Args A) {
  extern __shared__ uint32_t slut[];
  __shared__ float red[64][32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int rtd[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) {
    const int rt = blockIdx.x * MT + m;
    rtd[m] = A.tile_mod > 0 ? (rt % A.tile_mod) : rt;
  }
  for (int i = threadIdx.x; i < A.tbl_n; i += 256) {
    if constexpr (C == C_LUT4P || C == C_LUT3P) {
      uint32_t v = A.tbl[i];
#pragma unroll 4
      for (int l = 0; l < 32; l++) slut[i * 32 + l] = v;
    } else {
      slut[i] = A.tbl[i];
    }
  }
  __syncthreads();
  const int nst = A.K / 64;
  const int per = (nst + A.S - 1) / A.S;
  const int st0 = blockIdx.y * per, st1 = min(nst, st0 + per);
  const int ngrp = A.K / A.G;
  const int g = lane >> 2;
  constexpr int NWW = WW<C>::n;
  float acc[MT][NT * 4];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int i = 0; i < NT * 4; i++) acc[m][i] = 0.f;

  uint32_t wr0[MT][NWW], wr1[MT][NWW];
  uint32_t xb0[NT][4][2], xb1[NT][4][2];
  uint32_t pa0[MT], pb0[MT], pa1[MT], pb1[MT];
  auto issue = [&](uint32_t (*wr)[NWW], uint32_t (*xb)[4][2], uint32_t* pa, uint32_t* pb, int s_) {
    const int grp = (s_ * 64) / A.G;
#pragma unroll
    for (int m = 0; m < MT; m++) {
      load_w<C>(A.w, (size_t)rtd[m] * nst + s_, lane, wr[m]);
      const uint32_t* pp = A.prm + ((size_t)rtd[m] * ngrp + grp) * 16;
      pa[m] = __ldg(pp + g);
      pb[m] = __ldg(pp + g + 8);
    }
    load_x<NT>(A.x, A.K, A.B, s_ * 64, lane, xb);
  };
  auto compute = [&](uint32_t (*wr)[NWW], uint32_t (*xb)[4][2], const uint32_t* pa, const uint32_t* pb) {
#pragma unroll
    for (int m = 0; m < MT; m++) {
      uint32_t a[4][4];
      decode<C>(wr[m], pa[m], pb[m], slut, lane, a);
#pragma unroll
      for (int s = 0; s < 4; s++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++) mma16816(&acc[m][nt * 4], a[s], xb[nt][s][0], xb[nt][s][1]);
    }
  };
  int st = st0 + warp;
  if (st < st1) issue(wr0, xb0, pa0, pb0, st);
#pragma unroll 1
  for (; st < st1; st += 16) {
    if (st + 8 < st1) issue(wr1, xb1, pa1, pb1, st + 8);
    compute(wr0, xb0, pa0, pb0);
    if (st + 8 >= st1) break;
    if (st + 16 < st1) issue(wr0, xb0, pa0, pb0, st + 16);
    compute(wr1, xb1, pa1, pb1);
  }
#pragma unroll
  for (int m = 0; m < MT; m++) reduce_store(A, acc[m], NT * 4, warp, lane, blockIdx.x * MT + m, red);
}

// ----------------------------------------------------------------------------------------- VLC kernel
// One bitstream per (m-tile, split, warp): the warp's 32 lanes decode in lock step; every SPC symbols each lane
// with fewer than 32 buffered bits takes the next 32-bit word, words being handed out in lane order
// (ballot + popc prefix). The encoder replays exactly this schedule, so no per-lane offsets are stored.
// Words are staged through a per-(warp, m-tile) shared ring with cp.async, two super-tiles ahead.
constexpr int RING = 1024;
__device__ __forceinline__ void cp_async4(uint32_t* dst, const uint32_t* src) {
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N_> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N_)); }

template <int NT, int MAXLEN, int MT, int Q>
__global__ void __launch_bounds__(256) k_vlc(Args A) {
  extern __shared__ uint32_t sm[];
  __shared__ float red[64][32];
  uint32_t* stbl = sm;                       // 1 << MAXLEN entries: bf16 value | len << 16
  uint32_t* ring = sm + (1 << MAXLEN);       // 8 warps x MT x RING words
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int rtd[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) {
    const int rt = blockIdx.x * MT + m;
    rtd[m] = A.tile_mod > 0 ? (rt % A.tile_mod) : rt;
  }
  for (int i = threadIdx.x; i < (1 << MAXLEN); i += 256) stbl[i] = A.tbl[i];
  __syncthreads();
  const int nst = A.K / 64;
  const int per = (nst + A.S - 1) / A.S;
  const int st0 = blockIdx.y * per, st1 = min(nst, st0 + per);
  const int ngrp = A.K / A.G;
  const int g = lane >> 2;
  // Q independent sub-streams per lane (ILP: Huffman decode is a serial shift chain); sub-stream q owns
  // elements [q * SUB, (q + 1) * SUB) of every super-tile.
  constexpr int SUB = 32 / Q;
  constexpr int SPC0 = (32 / MAXLEN) >= 16 ? 16 : ((32 / MAXLEN) >= 8 ? 8 : 4);
  constexpr int SPC = SPC0 < SUB ? SPC0 : SUB;
  uint32_t* myring[MT];
  uint32_t base[MT], loaded[MT], send[MT], lo[MT][Q], hi[MT][Q];
  int nbits[MT][Q];
#pragma unroll
  for (int m = 0; m < MT; m++) {
    myring[m] = ring + (warp * MT + m) * RING;
    const size_t sid = ((size_t)rtd[m] * A.S + blockIdx.y) * 8 + warp;
    base[m] = loaded[m] = A.offs[sid];
    send[m] = A.offs[sid + 1];
#pragma unroll
    for (int q = 0; q < Q; q++) { lo[m][q] = hi[m][q] = 0u; nbits[m][q] = 0; }
  }
  float acc[MT][NT * 4];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int i = 0; i < NT * 4; i++) acc[m][i] = 0.f;
  const uint32_t lt = (1u << lane) - 1u;
  bool first = true;
#pragma unroll 1
  for (int st = st0 + warp; st < st1; st += 8) {
#pragma unroll
    for (int m = 0; m < MT; m++) {
      const uint32_t target = min(base[m] + 768u, send[m]);
      for (uint32_t i = loaded[m]; i < target; i += 32)
        if (i + lane < target) cp_async4(myring[m] + ((i + lane) & (RING - 1)), A.w + i + lane);
      loaded[m] = max(loaded[m], target);
    }
    cp_commit();
    if (first) { cp_wait<0>(); first = false; } else { cp_wait<1>(); }
    __syncwarp();
    uint32_t xb[NT][4][2];
    load_x<NT>(A.x, A.K, A.B, st * 64, lane, xb);
    const int grp = (st * 64) / A.G;
#pragma unroll
    for (int m = 0; m < MT; m++) {
      const uint32_t* pp = A.prm + ((size_t)rtd[m] * ngrp + grp) * 16;
      const uint32_t pg = __ldg(pp + g), pg8 = __ldg(pp + g + 8);
      const uint32_t s0 = splat_lo(pg), m0 = splat_hi(pg), s1 = splat_lo(pg8), m1 = splat_hi(pg8);
      uint16_t v[32];
#pragma unroll
      for (int c = 0; c < SUB; c += SPC) {
#pragma unroll
        for (int q = 0; q < Q; q++) {
          const bool need = nbits[m][q] < 32;
          const uint32_t bal = __ballot_sync(0xffffffffu, need);
          if (need) {
            const uint32_t word = myring[m][(base[m] + __popc(bal & lt)) & (RING - 1)];
            hi[m][q] = __funnelshift_l(word, 0u, nbits[m][q]);
            lo[m][q] |= word << nbits[m][q];
            nbits[m][q] += 32;
          }
          base[m] += __popc(bal);
        }
#pragma unroll
        for (int i = 0; i < SPC; i++)
#pragma unroll
          for (int q = 0; q < Q; q++) {
            const uint32_t ent = stbl[lo[m][q] & ((1u << MAXLEN) - 1u)];
            const uint32_t len = ent >> 16;
            v[q * SUB + c + i] = (uint16_t)(ent & 0xffffu);
            lo[m][q] = __funnelshift_r(lo[m][q], hi[m][q], len);
            hi[m][q] >>= len;
            nbits[m][q] -= len;
          }
      }
      uint32_t a[4][4];
#pragma unroll
      for (int s = 0; s < 4; s++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const int e = s * 8 + q * 2;
          a[s][q] = fma_pair((uint32_t)v[e] | ((uint32_t)v[e + 1] << 16), (q & 1) ? s1 : s0, (q & 1) ? m1 : m0);
        }
#pragma unroll
      for (int s = 0; s < 4; s++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++) mma16816(&acc[m][nt * 4], a[s], xb[nt][s][0], xb[nt][s][1]);
    }
  }
  cp_wait<0>();
#pragma unroll
  for (int m = 0; m < MT; m++) reduce_store(A, acc[m], NT * 4, warp, lane, blockIdx.x * MT + m, red);
}

// ----------------------------------------------------------------------------------------- host
static Args mk(torch::Tensor w, torch::Tensor prm, torch::Tensor offs, torch::Tensor tbl, torch::Tensor x,
               torch::Tensor y, int N, int K, int S, int G, int tile_mod, int maxlen, int mt) {
  Args a;
  a.w = reinterpret_cast<const uint32_t*>(w.data_ptr());
  a.prm = reinterpret_cast<const uint32_t*>(prm.data_ptr());
  a.offs = offs.numel() ? reinterpret_cast<const uint32_t*>(offs.data_ptr()) : nullptr;
  a.tbl = tbl.numel() ? reinterpret_cast<const uint32_t*>(tbl.data_ptr()) : nullptr;
  a.x = reinterpret_cast<const bf16*>(x.data_ptr());
  a.y = y.data_ptr<float>();
  a.N = N; a.K = K; a.B = x.size(0); a.S = S; a.G = G; a.tile_mod = tile_mod;
  a.tbl_n = tbl.numel(); a.maxlen = maxlen; a.MTr = mt;
  return a;
}

template <int C, int NT, int MT>
static void launch_fixed(const Args& a, cudaStream_t st) {
  size_t smem = (C == C_LUT4P || C == C_LUT3P) ? (size_t)a.tbl_n * 32 * 4 : (size_t)a.tbl_n * 4;
  if (smem > 40 * 1024) cudaFuncSetAttribute(k_fixed<C, NT, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  dim3 grid(a.N / (16 * MT), a.S);
  k_fixed<C, NT, MT><<<grid, 256, smem, st>>>(a);
}
template <int NT, int ML, int MT, int Q>
static void launch_vlc(const Args& a, cudaStream_t st) {
  size_t smem = ((1 << ML) + 8 * MT * RING) * 4;
  cudaFuncSetAttribute(k_vlc<NT, ML, MT, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  dim3 grid(a.N / (16 * MT), a.S);
  k_vlc<NT, ML, MT, Q><<<grid, 256, smem, st>>>(a);
}

template <int C>
static void disp_fixed(const Args& a, int nt, int mt, cudaStream_t st) {
  if (C == C_BF16 && mt > 2) mt = 2;
  if (nt == 1) {
    if (mt == 1) launch_fixed<C, 1, 1>(a, st);
    else if (mt == 2) launch_fixed<C, 1, 2>(a, st);
    else launch_fixed<C, 1, (C == C_BF16 ? 2 : 4)>(a, st);
  } else {
    if (mt == 1) launch_fixed<C, 2, 1>(a, st);
    else if (mt == 2) launch_fixed<C, 2, 2>(a, st);
    else launch_fixed<C, 2, (C == C_BF16 ? 2 : 4)>(a, st);
  }
}
template <int ML, int Q>
static void disp_vlc_q(const Args& a, int nt, int mt, cudaStream_t st) {
  if (nt == 1) { if (mt == 1) launch_vlc<1, ML, 1, Q>(a, st); else launch_vlc<1, ML, 2, Q>(a, st); }
  else { if (mt == 1) launch_vlc<2, ML, 1, Q>(a, st); else launch_vlc<2, ML, 2, Q>(a, st); }
}
template <int ML>
static void disp_vlc(const Args& a, int nt, int mt, int q, cudaStream_t st) {
  if (q == 1) disp_vlc_q<ML, 1>(a, nt, mt, st); else disp_vlc_q<ML, 4>(a, nt, mt, st);
}

void gemv(int codec, torch::Tensor w, torch::Tensor prm, torch::Tensor offs, torch::Tensor tbl, torch::Tensor x,
          torch::Tensor y, int N, int K, int S, int G, int tile_mod, int maxlen, int mt) {
  Args a = mk(w, prm, offs, tbl, x, y, N, K, S, G, tile_mod, maxlen, mt);
  cudaStream_t st = c10::cuda::getCurrentCUDAStream();
  const int nt = x.size(0) > 8 ? 2 : 1;
  switch (codec) {
    case C_BF16: disp_fixed<C_BF16>(a, nt, mt, st); break;
    case C_INT4: disp_fixed<C_INT4>(a, nt, mt, st); break;
    case C_LUT4: disp_fixed<C_LUT4>(a, nt, mt, st); break;
    case C_LUT4P: disp_fixed<C_LUT4P>(a, nt, mt, st); break;
    case C_ICDF4: disp_fixed<C_ICDF4>(a, nt, mt, st); break;
    case C_LUT3P: disp_fixed<C_LUT3P>(a, nt, mt, st); break;
    case C_LUT2P: disp_fixed<C_LUT2P>(a, nt, mt, st); break;
    case C_TCQ1: disp_fixed<C_TCQ1>(a, nt, mt, st); break;
    case C_TCQ3: disp_fixed<C_TCQ3>(a, nt, mt, st); break;
    case C_VLC:
      TORCH_CHECK(mt <= 2, "vlc mt <= 2");
      // maxlen encodes the sub-stream count too: maxlen + 100 * (Q == 4)
      if (maxlen % 100 == 2) disp_vlc<2>(a, nt, mt, maxlen >= 100 ? 4 : 1, st);
      else if (maxlen % 100 == 5) disp_vlc<5>(a, nt, mt, maxlen >= 100 ? 4 : 1, st);
      else disp_vlc<8>(a, nt, mt, maxlen >= 100 ? 4 : 1, st);
      break;
    default:
      TORCH_CHECK(false, "bad codec");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemv", &gemv); }
