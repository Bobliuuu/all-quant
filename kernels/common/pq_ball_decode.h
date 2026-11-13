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

// extract the g-th b-bit index from an LSB-first bitstream given as u32 words
PQ_DEV uint32_t pq_get_bits(const uint32_t* w, uint32_t bitpos, int b) {
  const uint32_t wi = bitpos >> 5, sh = bitpos & 31u;
  const uint64_t win = (uint64_t)w[wi] | ((uint64_t)w[wi + 1] << 32);
  return (uint32_t)((win >> sh) & ((b == 32) ? 0xFFFFFFFFull : ((1ull << b) - 1ull)));
}

PQ_DEV float pq_half_to_float(uint16_t h) {
  const uint32_t sgn = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 31u, m = h & 1023u;
  uint32_t bits;
  if (e == 0) {
    if (m == 0) bits = sgn;
    else {  // subnormal
      int ee = -1;
      do { m <<= 1; ++ee; } while ((m & 1024u) == 0);
      m &= 1023u;
      bits = sgn | ((uint32_t)(127 - 15 - ee) << 23) | (m << 13);
    }
  } else if (e == 31) bits = sgn | 0x7F800000u | (m << 13);
  else bits = sgn | ((e + 112u) << 23) | (m << 13);
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
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
