// pq-repro (PearlQuant-style E8 KV codec): on-chip decode of one ball-codebook index
// into 8 E8 coordinates (doubled, i.e. integers 2x). Plain C++ so the same code
// compiles for CUDA, HIP and SYCL (define PQ_DEV before including).
//
// Leader table row (8 x u32): off, nperm, dc, meta, dv_lo, dv_hi, 0, 0
//   dc   : 4-bit counts of the distinct |values|, ascending (k-th nibble)
//   dv   : 5-bit distinct |values|, ascending (k-th 5-bit field), 40 bits
//   meta : shift (log2 nsign) | odd << 8 | ndistinct << 16
// Index layout: idx = off + perm_rank * 2^shift + sign_bits  (see pqrepro/ball.py)
#pragma once
#include <stdint.h>

#ifndef PQ_DECODE
#define PQ_DECODE pq_ball_decode2
#endif

#ifndef PQ_DEV
#define PQ_DEV inline
#endif

PQ_DEV int pq_find_leader(const uint32_t* tab, int L, uint32_t idx) {
  int lo = 0, hi = L - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (tab[mid * 8] <= idx) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// out[i] = 2 * x_i (integers). Returns nothing; out must be a register array.
PQ_DEV void pq_ball_decode(uint32_t idx, const uint32_t* tab, int L, int out[8]) {
  const uint32_t* r = tab + pq_find_leader(tab, L, idx) * 8;
  uint32_t local = idx - r[0];
  const uint32_t meta = r[3];
  const int shift = meta & 255;
  const int odd = (meta >> 8) & 1;
  const int nd = (meta >> 16) & 255;
  const uint32_t s = local & ((1u << shift) - 1u);
  uint32_t rr = local >> shift;
  uint32_t M = r[1];
  uint32_t cnt = r[2];
  const uint64_t dv = (uint64_t)r[4] | ((uint64_t)r[5] << 32);
#pragma unroll
  for (int pos = 0; pos < 8; ++pos) {
    const uint32_t n = 8 - pos;
    int v = 0;
    for (int k = 0; k < 8; ++k) {
      if (k >= nd) break;
      const uint32_t ck = (cnt >> (4 * k)) & 15u;
      if (ck == 0) continue;
      const uint32_t blk = (M * ck) / n;  // exact: multinomial with value k first
      if (rr < blk) {
        M = blk;
        cnt -= 1u << (4 * k);
        v = (int)((dv >> (5 * k)) & 31u);
        break;
      }
      rr -= blk;
    }
    out[pos] = v;
  }
  if (odd) {
    int part = 0;
#pragma unroll
    for (int pos = 0; pos < 7; ++pos) {
      if ((s >> pos) & 1u) out[pos] = -out[pos];
      part += out[pos];
    }
    if (((part + out[7]) & 3) != 0) out[7] = -out[7];
  } else {
    int j = 0;
#pragma unroll
    for (int pos = 0; pos < 8; ++pos) {
      if (out[pos] != 0) {
        if ((s >> j) & 1u) out[pos] = -out[pos];
        ++j;
      }
    }
  }
}

// Branch-free variant (v2), same output as pq_ball_decode. Block k at position pos
// spans [M*C_{k-1}/n, M*C_k/n) with C the prefix counts, so the chosen distinct value
// is v = #{k : M*C_k <= rr*n}: 8 multiply-compares, no division, no data-dependent
// loop, no warp divergence. Prefix counts of the 4-bit count nibbles = cnt * 0x11111111
// (every prefix is <= 8, so nibbles never carry).
PQ_DEV void pq_ball_decode2(uint32_t idx, const uint32_t* tab, int L, int out[8]) {
  const uint32_t* r = tab + pq_find_leader(tab, L, idx) * 8;
  const uint32_t local = idx - r[0];
  const uint32_t meta = r[3];
  const uint32_t shift = meta & 255u;
  const uint32_t s = local & ((1u << shift) - 1u);
  uint32_t rr = local >> shift;
  uint32_t M = r[1];
  uint32_t cnt = r[2];
  const uint64_t dv = (uint64_t)r[4] | ((uint64_t)r[5] << 32);
#pragma unroll
  for (int pos = 0; pos < 8; ++pos) {
    const uint32_t n = 8 - pos;
    const uint32_t C = cnt * 0x11111111u;
    const uint32_t rn = rr * n;
    uint32_t v = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) v += (M * ((C >> (4 * k)) & 15u) <= rn) ? 1u : 0u;
    const uint32_t cv = (cnt >> (4 * v)) & 15u;
    const uint32_t Cprev = v ? ((C >> (4 * (v - 1))) & 15u) : 0u;
    rr -= (M * Cprev) / n;
    M = (M * cv) / n;
    cnt -= 1u << (4 * v);
    out[pos] = (int)((dv >> (5 * v)) & 31u);
  }
  if ((meta >> 8) & 1u) {
    int part = 0;
#pragma unroll
    for (int pos = 0; pos < 7; ++pos) {
      out[pos] = ((s >> pos) & 1u) ? -out[pos] : out[pos];
      part += out[pos];
    }
    out[7] = (((part + out[7]) & 3) != 0) ? -out[7] : out[7];
  } else {
    uint32_t j = 0;
#pragma unroll
    for (int pos = 0; pos < 8; ++pos) {
      const bool nz = out[pos] != 0;
      out[pos] = (nz && ((s >> j) & 1u)) ? -out[pos] : out[pos];
      j += nz ? 1u : 0u;
    }
  }
}

// v3: same math as v2, structure-of-arrays tables (no smem bank conflicts on the
// 32-byte rows) and a radix bucket table instead of the binary search:
//   l = radix[idx >> rsh]; then step forward while off[l+1] <= idx (usually 0-1 steps).
struct PQTab {
  const uint32_t* off; const uint32_t* nperm; const uint32_t* dc; const uint32_t* meta;
  const uint32_t* dvlo; const uint32_t* dvhi; const uint16_t* radix; int rsh; int L;
};
#define PQ_RADIX_BITS 10
// words needed for one SoA table + its radix (u16 x 2^R)
PQ_DEV int pq_tab_words(int L) { return 6 * L + (1 << (PQ_RADIX_BITS - 1)); }

// fill SoA arrays at dst (u32 words) from the 8-u32-row table; radix computed on the fly
// (caller distributes i over threads, then syncs). Entry i < L: rows; radix entries below.
PQ_DEV void pq_tab_fill_row(uint32_t* dst, const uint32_t* rows, int L, int i) {
  const uint32_t* r = rows + i * 8;
  dst[i] = r[0]; dst[L + i] = r[1]; dst[2 * L + i] = r[2]; dst[3 * L + i] = r[3]; dst[4 * L + i] = r[4]; dst[5 * L + i] = r[5];
}
PQ_DEV void pq_tab_fill_radix(uint32_t* dst, const uint32_t* rows, int L, int b, int j) {
  // bucket j covers [j << rsh, (j+1) << rsh): store the leader containing j << rsh
  const int R = b < PQ_RADIX_BITS ? b : PQ_RADIX_BITS;
  if (j >= (1 << R)) return;
  const uint32_t start = (uint32_t)j << (b - R);
  int lo = 0, hi = L - 1;
  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (rows[mid * 8] <= start) lo = mid; else hi = mid - 1; }
  ((uint16_t*)(dst + 6 * L))[j] = (uint16_t)lo;
}
PQ_DEV PQTab pq_tab_view(const uint32_t* base, int L, int b) {
  PQTab t;
  t.off = base; t.nperm = base + L; t.dc = base + 2 * L; t.meta = base + 3 * L; t.dvlo = base + 4 * L; t.dvhi = base + 5 * L;
  t.radix = (const uint16_t*)(base + 6 * L);
  const int R = b < PQ_RADIX_BITS ? b : PQ_RADIX_BITS;
  t.rsh = b - R; t.L = L;
  return t;
}

PQ_DEV void pq_ball_decode3(uint32_t idx, const PQTab& T, int out[8]) {
  int l = T.radix[idx >> T.rsh];
  while (l + 1 < T.L && T.off[l + 1] <= idx) ++l;
  const uint32_t local = idx - T.off[l];
  const uint32_t meta = T.meta[l];
  const uint32_t shift = meta & 255u;
  const uint32_t s = local & ((1u << shift) - 1u);
  uint32_t rr = local >> shift;
  uint32_t M = T.nperm[l];
  uint32_t cnt = T.dc[l];
  const uint64_t dv = (uint64_t)T.dvlo[l] | ((uint64_t)T.dvhi[l] << 32);
#pragma unroll
  for (int pos = 0; pos < 8; ++pos) {
    const uint32_t n = 8 - pos;
    const uint32_t C = cnt * 0x11111111u;
    const uint32_t rn = rr * n;
    uint32_t v = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) v += (M * ((C >> (4 * k)) & 15u) <= rn) ? 1u : 0u;
    const uint32_t cv = (cnt >> (4 * v)) & 15u;
    const uint32_t Cprev = v ? ((C >> (4 * (v - 1))) & 15u) : 0u;
    rr -= (M * Cprev) / n;
    M = (M * cv) / n;
    cnt -= 1u << (4 * v);
    out[pos] = (int)((dv >> (5 * v)) & 31u);
  }
  if ((meta >> 8) & 1u) {
    int part = 0;
#pragma unroll
    for (int pos = 0; pos < 7; ++pos) {
      out[pos] = ((s >> pos) & 1u) ? -out[pos] : out[pos];
      part += out[pos];
    }
    out[7] = (((part + out[7]) & 3) != 0) ? -out[7] : out[7];
  } else {
    uint32_t j = 0;
#pragma unroll
    for (int pos = 0; pos < 8; ++pos) {
      const bool nz = out[pos] != 0;
      out[pos] = (nz && ((s >> j) & 1u)) ? -out[pos] : out[pos];
      j += nz ? 1u : 0u;
    }
  }
}

// extract the g-th b-bit index from an LSB-first bitstream given as u32 words
PQ_DEV uint32_t pq_get_bits(const uint32_t* w, uint32_t bitpos, int b) {
  const uint32_t wi = bitpos >> 5, sh = bitpos & 31u;
  const uint64_t win = (uint64_t)w[wi] | ((uint64_t)w[wi + 1] << 32);
  return (uint32_t)((win >> sh) & ((b == 32) ? 0xFFFFFFFFull : ((1ull << b) - 1ull)));
}

PQ_DEV float pq_half_to_float(uint16_t h) {
  // finite halves only (our scales are finite): shift into float position, rescale by 2^112
  const uint32_t bits = ((uint32_t)(h & 0x7FFFu)) << 13;
  float f;
  __builtin_memcpy(&f, &bits, 4);
  f *= 5.192296858534828e+33f;  // 2^112
  return (h & 0x8000u) ? -f : f;
}

PQ_DEV float pq_bf16_to_float(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

// FP8 e4m3fn -> float (no inf; 0x7F/0xFF = NaN, never produced by our encoder)
PQ_DEV float pq_e4m3_to_float(uint8_t x) {
  const uint32_t sgn = (uint32_t)(x & 0x80u) << 24;
  const uint32_t e = (x >> 3) & 15u, m = x & 7u;
  uint32_t bits;
  if (e == 0) {
    const float f = (float)m * 0.001953125f;  // m * 2^-9
    return (x & 0x80u) ? -f : f;
  }
  bits = sgn | ((e + 120u) << 23) | (m << 20);
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}
