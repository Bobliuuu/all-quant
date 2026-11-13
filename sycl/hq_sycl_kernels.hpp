// hq_sycl_kernels.hpp -- SYCL 2020 port of the HyperQuant GPU kernels (upstream commit
// 227f3f6: cuda/stage2_cuda_{encoder,decoder}.cu, cuda/stage3_cuda_decoder.cu,
// cuda/rht_cuda.cu, cuda/lattice_ext.cu, integrations/llama/{fused_decode_gemv,
// rice_fused_attention}.cu, metalrice_device.cuh).  MIT.
//
// Same kernel structure and arithmetic order as CUDA.  Differences (see PORTING_LOG.md):
//  * warp-level code: CUDA hard-codes 32-lane warps.  Here the RHT uses the device's
//    sub-group size SG (16/32/64) with the identical butterfly stage order, so results
//    stay bit-identical; attention keeps "32-lane logical warps" and reduces with
//    shift_group_left inside a sub-group of size >= 32.
//  * the coop decoder's __syncwarp + early per-warp return is restructured so every
//    work-item reaches the work-group barrier.
//  * fp8 e4m3 and bf16 conversions are the portable bit-exact ones in hq_num.hpp.
//  * thrust::exclusive_scan -> a small 3-kernel work-group scan (below).
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include "../common/hq_num.hpp"

namespace hqs {
using u8 = uint8_t; using u16 = uint16_t; using u32 = uint32_t; using u64 = uint64_t;

inline float bf2f(u16 b) { return hqn::bf2f(b); }
inline u16 f2bf(float f) { return hqn::f2bf_rn(f); }

// ───────────────────────────── bit reader (BitReader / MRBitReader)
struct BitReader {
  const u32* words; int word_count, next_word_idx, word_stride;
  u32 c1, c2; int c1_bits, c2_bits; u64 reg; int bits_in_reg;
  static u32 load(const u32* w, int wc, int i) { return (i >= 0 && i < wc) ? w[i] : 0u; }
  static u32 shl(u32 v, int s) { return s >= 32 ? 0u : (v << s); }
  BitReader(const u32* w, int wc, u32 bit_offset) : words(w), word_count(wc) { init((int)(bit_offset >> 5), (int)(bit_offset & 31u), 1); }
  BitReader(const u32* w, int wc, int start_word, int skip, int stride) : words(w), word_count(wc) { init(start_word, skip, stride); }
  void init(int sw, int sb, int stride) {
    word_stride = stride; c1 = load(words, word_count, sw); c2 = load(words, word_count, sw + stride);
    c1_bits = 32; c2_bits = 32; next_word_idx = sw + 2 * stride; reg = 0; bits_in_reg = 0;
    if (sb > 0) { c1 = shl(c1, sb); c1_bits -= sb; if (c1_bits == 0) promote(); }
  }
  void promote() { c1 = c2; c1_bits = c2_bits; c2 = load(words, word_count, next_word_idx); next_word_idx += word_stride; c2_bits = 32; }
  void refill(int need) {
    while (bits_in_reg < need) {
      if (c1_bits == 0) promote();
      const int room = 64 - bits_in_reg, take = c1_bits < room ? c1_bits : room;
      const u32 piece = c1 >> (32 - take);
      reg |= (u64)piece << (64 - bits_in_reg - take);
      c1 = shl(c1, take); c1_bits -= take; bits_in_reg += take;
    }
  }
  u32 read_bits(int n) {
    if (n <= 0) return 0u;
    refill(n); const u32 v = (u32)(reg >> (64 - n)); reg <<= n; bits_in_reg -= n; return v;
  }
  u32 read_unary() {
    u32 q = 0;
    while (true) {
      refill(1);
      u64 probe = reg;
      if (bits_in_reg < 64) probe |= (1ull << (64 - bits_in_reg)) - 1ull;
      const u32 lz = probe == 0ull ? 64u : (u32)sycl::clz(probe);
      q += lz;
      if (lz < (u32)bits_in_reg) { const int c = (int)lz + 1; reg <<= c; bits_in_reg -= c; break; }
      reg = 0; bits_in_reg = 0;
    }
    return q;
  }
};

inline int inv_zz(u32 zz) { return (zz & 1u) ? -(int)((zz + 1u) >> 1) : (int)(zz >> 1); }
inline void inverse_Y(const u8 z[8], int Y[8]) {
  int s[8];
  for (int i = 0; i < 7; ++i) s[i] = inv_zz(z[i]);
  const u32 z7 = z[7]; const int c = (int)(z7 & 1u), t = inv_zz(z7 >> 1u);
  int p = 0; for (int i = 0; i < 7; ++i) p += s[i];
  p &= 1; s[7] = (t << 1) + p;
  for (int i = 0; i < 8; ++i) Y[i] = (s[i] << 1) + c;
}
// OD: 0 bf16 (Y/alpha), 1 fp8 e4m3 (Y/alpha), 2 int8 sat(Y)
template <int OD> inline void inverse_word(const u8 z[8], float alpha, void* out8) {
  int Y[8]; inverse_Y(z, Y);
  for (int i = 0; i < 8; ++i) {
    if constexpr (OD == 0) static_cast<u16*>(out8)[i] = f2bf((float)Y[i] / alpha);
    else if constexpr (OD == 1) static_cast<u8*>(out8)[i] = hqn::f2e4m3_rn((float)Y[i] / alpha);
    else { const int q = Y[i] < -127 ? -127 : (Y[i] > 127 ? 127 : Y[i]); static_cast<int8_t*>(out8)[i] = (int8_t)q; }
  }
}
inline void decode_word_f(const u8 z[8], float inv_alpha, float out[8]) {
  int Y[8]; inverse_Y(z, Y);
  for (int i = 0; i < 8; ++i) out[i] = (float)Y[i] * inv_alpha;
}
inline void read8(BitReader& br, u32 k, u8 z[8]) {
  for (int p = 0; p < 8; ++p) { const u32 q = br.read_unary(), rem = br.read_bits((int)k); z[p] = (u8)(((q << k) | rem) & 0xFFu); }
}

// ───────────────────────────── encoder device helpers (encode_word_to_z)
inline int zigzag(int v) { return v >= 0 ? (v << 1) : ((-v << 1) - 1); }
inline void fix_parity_mod4(int u[8], const float w[8]) {
  int sum = 0; for (int i = 0; i < 8; ++i) sum += u[i];
  if ((sum & 3) == 0) return;
  int bj = 0; float ba = -1.f, bd = 0.f;
  for (int i = 0; i < 8; ++i) { const float d = (float)u[i] - w[i], ad = sycl::fabs(d); if (ad > ba) { ba = ad; bj = i; bd = d; } }
  int s = (int)(bd > 0.f) - (int)(bd < 0.f);
  if (s == 0) s = 1;
  u[bj] += -2 * s;
}
inline void encode_word_to_z(const u16* bf, int word_id, float alpha, int z[8]) {
  float w[8];
  const int base = word_id * 8;
  for (int i = 0; i < 8; ++i) w[i] = bf2f(f2bf(alpha * bf2f(bf[base + i])));
  int u0[8], u1[8];
  for (int i = 0; i < 8; ++i) { u0[i] = (int)sycl::rint(w[i] * 0.5f) * 2; u1[i] = (int)sycl::rint((w[i] - 1.0f) * 0.5f) * 2 + 1; }
  fix_parity_mod4(u0, w); fix_parity_mod4(u1, w);
  float d0 = 0.f, d1 = 0.f;
  for (int i = 0; i < 8; ++i) { const float e0 = (float)u0[i] - w[i], e1 = (float)u1[i] - w[i]; d0 = sycl::fma(e0, e0, d0); d1 = sycl::fma(e1, e1, d1); }
  const int* Y = (d0 <= d1) ? u0 : u1;
  const int c = Y[0] & 1;
  int s[8]; for (int i = 0; i < 8; ++i) s[i] = (Y[i] - c) >> 1;
  int p = 0; for (int i = 0; i < 7; ++i) p += s[i];
  p &= 1;
  const int T = (s[7] - p) >> 1;
  for (int i = 0; i < 7; ++i) z[i] = zigzag(s[i]);
  z[7] = 2 * zigzag(T) + c;
}
template <sycl::access::address_space AS, sycl::memory_scope MS>
inline void or_bit(u32* words, u32 pos) {
  sycl::atomic_ref<u32, sycl::memory_order::relaxed, MS, AS> a(words[pos >> 5]);
  a.fetch_or(1u << (31u - (pos & 31u)));
}

// ───────────────────────────── exclusive scan (replaces thrust::exclusive_scan)
inline void exclusive_scan_u32(sycl::queue& q, const u32* in, u32* out, size_t n) {
  constexpr size_t WG = 256;
  const size_t ng = (n + WG - 1) / WG;
  u32* sums = sycl::malloc_device<u32>(ng, q);
  q.parallel_for(sycl::nd_range<1>(ng * WG, WG), [=](sycl::nd_item<1> it) {
    const size_t i = it.get_global_id(0);
    const u32 v = i < n ? in[i] : 0u;
    const u32 ex = sycl::exclusive_scan_over_group(it.get_group(), v, sycl::plus<u32>());
    if (i < n) out[i] = ex;
    if (it.get_local_id(0) == WG - 1) sums[it.get_group(0)] = ex + v;
  });
  if (ng > 1) {
    u32* sums_ex = sycl::malloc_device<u32>(ng, q);
    exclusive_scan_u32(q, sums, sums_ex, ng);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] += sums_ex[i / WG]; });
    q.wait(); sycl::free(sums_ex, q);
  }
  q.wait(); sycl::free(sums, q);
}
}  // namespace hqs
