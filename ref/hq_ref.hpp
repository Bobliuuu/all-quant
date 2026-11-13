// hq_ref.hpp -- single-threaded CPU reference for every HyperQuant GPU kernel that
// is ported (upstream commit 227f3f6, files cuda/*.cu and integrations/llama/*.cu).
//
// The reference mirrors the CUDA arithmetic ORDER, including the FMA contractions
// nvcc emits by default (-fmad=true), so that integer outputs (bitstreams, symbols,
// int8, int32) and most float outputs are bit-exact.  Where a CUDA intrinsic is not
// correctly rounded (rsqrtf, expf) the corresponding test uses a tolerance.
//
// Upstream constraint surfaced while writing this: every decoder kernel decodes all
// symbols_per_stream symbols of every stream; if total_symbols is not a multiple of
// sps the last stream reads past the payload, runs into the zero padding and the
// unary reader never terminates (GPU hang).  All upstream callers satisfy sps | total.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "../common/hq_num.hpp"

#ifndef HQ_REF_FMA
#define HQ_REF_FMA 1  // mirror nvcc's default a*b+c -> fma contraction
#endif

namespace hqref {
using hqn::bf2f;
using hqn::f2bf_rn;

inline float madd(float a, float b, float c) {
#if HQ_REF_FMA
  return std::fmaf(a, b, c);
#else
  return a * b + c;
#endif
}
inline int zigzag(int v) { return v >= 0 ? (v << 1) : ((-v << 1) - 1); }
inline int inv_zigzag(uint32_t zz) { return (zz & 1u) ? -(int)((zz + 1u) >> 1) : (int)(zz >> 1); }

// ── E8int nearest point + structural strip (stage2_cuda_encoder.cu: encode_word_to_z)
inline void fix_parity_mod4(int u[8], const float w[8]) {
  int sum = 0;
  for (int i = 0; i < 8; ++i) sum += u[i];
  if ((sum & 3) == 0) return;
  int best_j = 0; float best_abs = -1.f, best_delta = 0.f;
  for (int i = 0; i < 8; ++i) {
    float d = (float)u[i] - w[i], ad = std::fabs(d);
    if (ad > best_abs) { best_abs = ad; best_j = i; best_delta = d; }
  }
  int s = (best_delta > 0.f) - (best_delta < 0.f);
  if (s == 0) s = 1;
  u[best_j] += -2 * s;
}
inline void e8_nearest(const float w[8], int Y[8]) {
  int u0[8], u1[8];
  for (int i = 0; i < 8; ++i) {
    u0[i] = (int)std::nearbyint(w[i] * 0.5f) * 2;
    u1[i] = (int)std::nearbyint((w[i] - 1.0f) * 0.5f) * 2 + 1;
  }
  fix_parity_mod4(u0, w); fix_parity_mod4(u1, w);
  float d0 = 0.f, d1 = 0.f;
  for (int i = 0; i < 8; ++i) {
    float e0 = (float)u0[i] - w[i], e1 = (float)u1[i] - w[i];
    d0 = madd(e0, e0, d0); d1 = madd(e1, e1, d1);
  }
  const int* s = (d0 <= d1) ? u0 : u1;
  for (int i = 0; i < 8; ++i) Y[i] = s[i];
}
inline void encode_word_to_z(const uint16_t* bf, float alpha, int z[8], int* Yout = nullptr) {
  float w[8];
  for (int i = 0; i < 8; ++i) w[i] = bf2f(f2bf_rn(alpha * bf2f(bf[i])));
  int Y[8]; e8_nearest(w, Y);
  if (Yout) for (int i = 0; i < 8; ++i) Yout[i] = Y[i];
  const int c = Y[0] & 1;
  int s[8];
  for (int i = 0; i < 8; ++i) s[i] = (Y[i] - c) >> 1;
  int p = 0; for (int i = 0; i < 7; ++i) p += s[i];
  p &= 1;
  const int T = (s[7] - p) >> 1;
  for (int i = 0; i < 7; ++i) z[i] = zigzag(s[i]);
  z[7] = 2 * zigzag(T) + c;
}

// ── Stage-2 Rice encoder (groups_per_thread = 1: one group = one 8-D word)
struct Encoded {
  std::vector<uint32_t> words, offsets;  // offsets: one per DECODER stream (bit offsets)
  std::vector<uint8_t> ks;
  uint32_t total_bits = 0;               // payload + 32 read-ahead pad bits (as upstream)
  std::vector<uint32_t> z;               // full (unmasked) symbols, for tests
};
inline void put_bit(std::vector<uint32_t>& w, uint32_t pos) { w[pos >> 5] |= 1u << (31u - (pos & 31u)); }

inline Encoded rice_encode(const std::vector<uint16_t>& bf, float alpha, int k, int sps) {
  if (bf.size() % 8 || sps % 8) throw std::runtime_error("bad sizes");
  const size_t nw = bf.size() / 8;
  Encoded E; E.z.resize(bf.size());
  std::vector<uint32_t> len(nw), off(nw);
  for (size_t g = 0; g < nw; ++g) {
    int z[8]; encode_word_to_z(&bf[g * 8], alpha, z);
    uint32_t L = 0;
    for (int i = 0; i < 8; ++i) { E.z[g * 8 + i] = (uint32_t)z[i]; L += ((uint32_t)z[i] >> k) + 1u + (uint32_t)k; }
    len[g] = L;
  }
  uint32_t run = 0;
  for (size_t g = 0; g < nw; ++g) { off[g] = run; run += len[g]; }
  const uint32_t payload = run;
  E.total_bits = payload + 32u;
  E.words.assign((E.total_bits + 31u) >> 5, 0u);
  uint32_t pos = 0;
  for (size_t i = 0; i < bf.size(); ++i) {
    uint32_t sym = E.z[i], q = sym >> k, rem = sym & ((1u << k) - 1u);
    pos += q; put_bit(E.words, pos); ++pos;
    for (int b = k - 1; b >= 0; --b) { if ((rem >> b) & 1u) put_bit(E.words, pos); ++pos; }
  }
  const size_t stride = (size_t)sps / 8, nds = (bf.size() + sps - 1) / sps;
  E.offsets.resize(nds);
  for (size_t d = 0; d < nds; ++d) E.offsets[d] = off[d * stride];
  E.ks.assign(nds, (uint8_t)k);
  return E;
}

// ── Bit reader (semantics of BitReader / MRBitReader: MSB-first, zeros past the end)
struct Reader {
  const uint32_t* w; size_t wc; uint64_t pos; size_t stride_words = 1; size_t start_word = 0;
  // contiguous
  Reader(const uint32_t* w_, size_t wc_, uint64_t bitoff) : w(w_), wc(wc_), pos(bitoff) {}
  uint32_t bit(uint64_t p) const {
    size_t wi = (size_t)(p >> 5);
    if (stride_words != 1) wi = start_word + wi * stride_words;
    return wi < wc ? (w[wi] >> (31u - (uint32_t)(p & 31u))) & 1u : 0u;
  }
  uint32_t unary(uint64_t guard = 1u << 20) {
    uint32_t q = 0;
    while (!bit(pos)) { ++pos; if (++q > guard) throw std::runtime_error("unary runaway"); }
    ++pos; return q;
  }
  uint32_t bits(int n) { uint32_t v = 0; for (int i = 0; i < n; ++i) v = (v << 1) | bit(pos++); return v; }
};

// decode_stage2_kernel: zigzag symbols masked to 8 bits
inline std::vector<uint8_t> decode_symbols(const std::vector<uint32_t>& words, const std::vector<uint32_t>& off,
                                           const std::vector<uint8_t>& ks, int sps, size_t total) {
  std::vector<uint8_t> out(total);
  for (size_t s = 0; s < off.size(); ++s) {
    Reader r(words.data(), words.size(), off[s]);
    int k = ks[s];
    for (int i = 0; i < sps; ++i) {
      uint32_t q = r.unary(), rem = r.bits(k);
      size_t o = s * (size_t)sps + i;
      if (o < total) out[o] = (uint8_t)(((q << k) | rem) & 0xFFu);
    }
  }
  return out;
}

inline void inverse_word_Y(const uint8_t z[8], int Y[8]) {
  int s[8];
  for (int i = 0; i < 7; ++i) s[i] = inv_zigzag(z[i]);
  const int c = (int)(z[7] & 1u), t = inv_zigzag((uint32_t)z[7] >> 1);
  int p = 0; for (int i = 0; i < 7; ++i) p += s[i];
  p &= 1;
  s[7] = t * 2 + p;
  for (int i = 0; i < 8; ++i) Y[i] = s[i] * 2 + c;
}
// inverse_e8int_word<OD>: 0=bf16 (Y/alpha), 1=fp8 e4m3 (Y/alpha), 2=int8 sat(Y)
inline void inverse_to(const std::vector<uint8_t>& z, float alpha, int od, std::vector<uint8_t>& out) {
  size_t n = z.size(); out.assign(n * (od == 0 ? 2 : 1), 0);
  for (size_t g = 0; g < n / 8; ++g) {
    int Y[8]; inverse_word_Y(&z[g * 8], Y);
    for (int i = 0; i < 8; ++i) {
      size_t e = g * 8 + i;
      if (od == 0) { uint16_t b = f2bf_rn((float)Y[i] / alpha); memcpy(&out[e * 2], &b, 2); }
      else if (od == 1) out[e] = hqn::f2e4m3_rn((float)Y[i] / alpha);
      else { int q = Y[i] < -127 ? -127 : (Y[i] > 127 ? 127 : Y[i]); out[e] = (uint8_t)(int8_t)q; }
    }
  }
}
// mr_decode_e8_word: float(Y) * inv_alpha
inline void decode_word_f(const uint8_t z[8], float inv_alpha, float out[8]) {
  int Y[8]; inverse_word_Y(z, Y);
  for (int i = 0; i < 8; ++i) out[i] = (float)Y[i] * inv_alpha;
}

// ── Interleaved layout (decode_interleaved_stage2_to_bf16_kernel input). Upstream ships
// only the decoder; this builds the layout from the symbol streams: 32 streams per
// group, words round-robin, each stream word-aligned in its own column.
struct Interleaved { std::vector<uint32_t> words, group_base; };
inline Interleaved build_interleaved(const std::vector<uint32_t>& zfull, int k, int sps) {
  size_t ns = zfull.size() / sps, ng = (ns + 31) / 32;
  Interleaved I; I.group_base.resize(ng);
  std::vector<std::vector<uint32_t>> col(ns);
  for (size_t s = 0; s < ns; ++s) {
    uint64_t bits = 0;
    for (int i = 0; i < sps; ++i) bits += (zfull[s * sps + i] >> k) + 1 + k;
    col[s].assign((size_t)((bits + 31) / 32) + 1, 0u);  // +1 word read-ahead pad
    uint32_t pos = 0;
    for (int i = 0; i < sps; ++i) {
      uint32_t sym = zfull[s * sps + i], q = sym >> k, rem = sym & ((1u << k) - 1u);
      pos += q; put_bit(col[s], pos); ++pos;
      for (int b = k - 1; b >= 0; --b) { if ((rem >> b) & 1u) put_bit(col[s], pos); ++pos; }
    }
  }
  size_t base = 0;
  for (size_t g = 0; g < ng; ++g) {
    size_t rows = 0;
    for (size_t l = 0; l < 32 && g * 32 + l < ns; ++l) rows = std::max(rows, col[g * 32 + l].size());
    I.group_base[g] = (uint32_t)base;
    I.words.resize(base + rows * 32, 0u);
    for (size_t l = 0; l < 32 && g * 32 + l < ns; ++l)
      for (size_t r = 0; r < col[g * 32 + l].size(); ++r) I.words[base + r * 32 + l] = col[g * 32 + l][r];
    base += rows * 32;
  }
  return I;
}

// ── RHT (rht_cuda.cu rht_lastdim_kernel<H>), exact butterfly order of the kernel.
// inv_sqrt is passed in: CUDA uses rsqrtf (approximate) -> exact for H = 256, 1024.
inline std::vector<uint16_t> rht_lastdim(const std::vector<uint16_t>& X, int rows, int K, int H,
                                         const std::vector<int8_t>& signs, float inv_sqrt) {
  std::vector<uint16_t> Y(X.size());
  std::vector<float> v(H);
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < K / H; ++b) {
      const size_t base = (size_t)r * K + (size_t)b * H;
      for (int e = 0; e < H; ++e) v[e] = bf2f(X[base + e]) * (float)signs[b * H + e];
      for (int len = 1; len < H; len <<= 1)  // element-bit stages in order 0..log2H-1
        for (int e = 0; e < H; ++e)
          if (!(e & len)) { float a = v[e], c = v[e + len]; v[e] = a + c; v[e + len] = a - c; }
      for (int e = 0; e < H; ++e) Y[base + e] = f2bf_rn(v[e] * inv_sqrt);
    }
  return Y;
}

// ── lattice_ext.cu small kernels
inline void rowmax_cast(const std::vector<uint16_t>& xp, int M, int K, float alpha,
                        std::vector<int8_t>& x8, std::vector<float>& deq) {
  x8.resize((size_t)M * K); deq.resize(M);
  for (int m = 0; m < M; ++m) {
    float amax = 0.f;
    for (int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(bf2f(xp[(size_t)m * K + k])));
    const float sc = amax > 0.f ? 127.0f / amax : 0.f;
    deq[m] = amax / (127.0f * alpha);
    for (int k = 0; k < K; ++k) {
      int q = (int)std::nearbyint(bf2f(xp[(size_t)m * K + k]) * sc);
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      x8[(size_t)m * K + k] = (int8_t)q;
    }
  }
}
inline std::vector<int32_t> int8_gemm_tn(const std::vector<int8_t>& A, const std::vector<int8_t>& B, int M, int N, int K) {
  std::vector<int32_t> C((size_t)M * N);  // col-major [M,N]
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k) s += (int32_t)A[(size_t)m * K + k] * (int32_t)B[(size_t)n * K + k];
      C[(size_t)n * M + m] = s;
    }
  return C;
}
inline std::vector<uint16_t> dequant(const std::vector<int32_t>& C, const std::vector<float>& deq, int M, int N) {
  std::vector<uint16_t> y((size_t)M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) y[(size_t)m * N + n] = f2bf_rn((float)C[(size_t)n * M + m] * deq[m]);
  return y;
}
inline void fp8_rowmax_cast(const std::vector<uint16_t>& xp, int M, int K, std::vector<uint8_t>& x8, std::vector<float>& sa) {
  x8.resize((size_t)M * K); sa.resize(M);
  for (int m = 0; m < M; ++m) {
    float amax = 0.f;
    for (int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(bf2f(xp[(size_t)m * K + k])));
    const float inv = amax > 0.f ? 448.f / amax : 0.f;
    sa[m] = amax / 448.f;
    for (int k = 0; k < K; ++k) {
      float v = bf2f(xp[(size_t)m * K + k]) * inv;
      v = std::fmax(-448.f, std::fmin(448.f, v));
      x8[(size_t)m * K + k] = hqn::f2e4m3_rn(v);
    }
  }
}

// ── fused_decode_gemv.cu
inline float stream_dot(const std::vector<uint32_t>& words, uint32_t off, int k, int sps, float inv_alpha,
                        const uint16_t* x) {
  Reader r(words.data(), words.size(), off);
  float acc = 0.f;
  for (int w = 0; w < sps / 8; ++w) {
    uint8_t z[8];
    for (int p = 0; p < 8; ++p) { uint32_t q = r.unary(), rem = r.bits(k); z[p] = (uint8_t)(((q << k) | rem) & 0xFFu); }
    float d[8]; decode_word_f(z, inv_alpha, d);
    for (int i = 0; i < 8; ++i) acc = madd(d[i], bf2f(x[w * 8 + i]), acc);
  }
  return acc;
}
inline std::vector<uint16_t> fused_gemv(const Encoded& E, int sps, float alpha, const std::vector<uint16_t>& x,
                                        const std::vector<uint16_t>& sigma, int N, int K) {
  const int spr = K / sps; const float inv_alpha = 1.0f / alpha;
  std::vector<uint16_t> y(N);
  for (int r = 0; r < N; ++r) {
    float total = 0.f;
    for (int s = 0; s < spr; ++s) total += stream_dot(E.words, E.offsets[r * spr + s], E.ks[r * spr + s], sps, inv_alpha, &x[s * sps]);
    y[r] = f2bf_rn(total * bf2f(sigma[r]));
  }
  return y;
}
inline std::vector<uint16_t> fused_gemm_ws(const Encoded& E, float alpha, const std::vector<uint16_t>& x,
                                           const std::vector<uint16_t>& sigma, int N, int K, int B) {
  const int sps = 512, spr = K / sps; const float inv_alpha = 1.0f / alpha;
  std::vector<uint16_t> y((size_t)B * N);
  std::vector<float> wv((size_t)spr * sps);
  for (int r = 0; r < N; ++r) {
    for (int s = 0; s < spr; ++s) {
      Reader rd(E.words.data(), E.words.size(), E.offsets[r * spr + s]);
      int k = E.ks[r * spr + s];
      for (int w = 0; w < sps / 8; ++w) {
        uint8_t z[8];
        for (int p = 0; p < 8; ++p) { uint32_t q = rd.unary(), rem = rd.bits(k); z[p] = (uint8_t)(((q << k) | rem) & 0xFFu); }
        decode_word_f(z, inv_alpha, &wv[(size_t)s * sps + w * 8]);
      }
    }
    for (int b = 0; b < B; ++b) {
      float acc = 0.f;
      for (int s = 0; s < spr; ++s) {
        float t = 0.f;
        for (int j = 0; j < sps; ++j) t = madd(wv[(size_t)s * sps + j], bf2f(x[(size_t)b * K + s * sps + j]), t);
        acc += t;
      }
      y[(size_t)b * N + r] = f2bf_rn(acc * bf2f(sigma[r]));
    }
  }
  return y;
}

// ── rice_fused_attention.cu (float reference following the kernel's structure;
// compared with tolerance because CUDA expf is not correctly rounded)
struct AttnIn {
  Encoded K, V;  // consolidated (chunk-major) bitstreams
  std::vector<uint16_t> norms_k, norms_v;            // [H, n_complete] bf16
  std::vector<uint16_t> pend_uk, pend_nk, pend_uv, pend_nv;  // [H, n_pend, D], [H, n_pend]
  std::vector<uint16_t> Q;                           // [H*G, D] bf16
  int H, G, D, n_complete, n_pend, ct, sps; float alpha;
};
inline float warp_sum32(const float* v) {  // __shfl_down_sync tree, value seen by lane 0
  float t[32]; for (int i = 0; i < 32; ++i) t[i] = v[i];
  for (int m = 16; m > 0; m >>= 1)
    for (int l = 0; l < 32; ++l) t[l] = t[l] + (l + m < 32 ? t[l + m] : t[l]);
  return t[0];
}
inline std::vector<float> fused_attn(const AttnIn& A) {
  const int H = A.H, G = A.G, D = A.D, T = A.n_complete + A.n_pend, nch = A.n_complete / A.ct;
  const float inv_D = 1.0f / std::sqrt((float)D), inv_alpha = 1.0f / A.alpha, sq = std::sqrt((float)D);
  std::vector<float> out((size_t)H * G * D);
  for (int h = 0; h < H; ++h) {
    std::vector<float> sc((size_t)T * G, -1e30f), so((size_t)G * D, 0.f), tile((size_t)std::max(A.ct, A.n_pend) * D);
    auto load_chunk = [&](const Encoded& E, const std::vector<uint16_t>& norms, int c) {
      int sid = c * H + h;
      Reader r(E.words.data(), E.words.size(), E.offsets[sid]);
      int k = E.ks[sid];
      for (int w = 0; w < A.sps / 8; ++w) {
        uint8_t z[8];
        for (int p = 0; p < 8; ++p) { uint32_t q = r.unary(), rem = r.bits(k); z[p] = (uint8_t)(((q << k) | rem) & 0xFFu); }
        float d[8]; decode_word_f(z, inv_alpha, d);
        int tok = w / (D / 8), d0 = (w % (D / 8)) * 8;
        float nf = bf2f(norms[(size_t)h * A.n_complete + c * A.ct + tok]) / sq;
        for (int i = 0; i < 8; ++i) tile[(size_t)tok * D + d0 + i] = d[i] * nf;
      }
    };
    auto load_pend = [&](const std::vector<uint16_t>& u, const std::vector<uint16_t>& n) {
      for (int t = 0; t < A.n_pend; ++t) {
        float nf = bf2f(n[(size_t)h * A.n_pend + t]) / sq;
        for (int d = 0; d < D; ++d) tile[(size_t)t * D + d] = bf2f(u[((size_t)h * A.n_pend + t) * D + d]) * nf;
      }
    };
    auto scores = [&](int base, int nt) {
      for (int q = 0; q < G; ++q)
        for (int t = 0; t < nt; ++t) {
          float lane[32];
          for (int l = 0; l < 32; ++l) {
            float p = 0.f;
            for (int d = l; d < D; d += 32) p = madd(bf2f(A.Q[((size_t)h * G + q) * D + d]), tile[(size_t)t * D + d], p);
            lane[l] = p;
          }
          sc[(size_t)(base + t) * G + q] = warp_sum32(lane) * inv_D;
        }
    };
    auto accum = [&](int base, int nt) {
      for (int q = 0; q < G; ++q)
        for (int t = 0; t < nt; ++t) {
          float w = sc[(size_t)(base + t) * G + q];
          for (int d = 0; d < D; ++d) so[(size_t)q * D + d] = madd(w, tile[(size_t)t * D + d], so[(size_t)q * D + d]);
        }
    };
    for (int c = 0; c < nch; ++c) { load_chunk(A.K, A.norms_k, c); scores(c * A.ct, A.ct); }
    if (A.n_pend > 0) { load_pend(A.pend_uk, A.pend_nk); scores(A.n_complete, A.n_pend); }
    for (int q = 0; q < G; ++q) {
      float m = -1e30f, l = 0.f;
      for (int t = 0; t < T; ++t) { float x = sc[(size_t)t * G + q], m2 = std::fmax(m, x); l = madd(l, std::exp(m - m2), std::exp(x - m2)); m = m2; }
      for (int t = 0; t < T; ++t) { float& s = sc[(size_t)t * G + q]; s = std::exp(s - m) / l; }
    }
    for (int c = 0; c < nch; ++c) { load_chunk(A.V, A.norms_v, c); accum(c * A.ct, A.ct); }
    if (A.n_pend > 0) { load_pend(A.pend_uv, A.pend_nv); accum(A.n_complete, A.n_pend); }
    for (int q = 0; q < G; ++q)
      for (int d = 0; d < D; ++d) out[((size_t)h * G + q) * D + d] = so[(size_t)q * D + d];
  }
  return out;
}

}  // namespace hqref
