// gen_vectors.cpp -- generate the shared test vectors from the CPU reference.
//   g++ -O2 -std=c++17 -ffp-contract=off ref/gen_vectors.cpp -o gen_vectors && ./gen_vectors vectors
// (-ffp-contract=off: the reference requests FMA explicitly via std::fmaf, nothing implicit)
#include <cstdio>
#include <cstring>
#include <string>
#include "../common/hq_vec.hpp"
#include "hq_ref.hpp"

using namespace hqref;
using hqn::SplitMix64;

static std::vector<uint16_t> gauss_bf16(size_t n, uint64_t seed, double scale = 1.0, double outlier_frac = 0.0,
                                        double outlier_scale = 1.0) {
  SplitMix64 r(seed); std::vector<uint16_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    double g = r.gaussian() * scale;
    if (outlier_frac > 0 && r.uniform() < outlier_frac) g *= outlier_scale;
    v[i] = f2bf_rn((float)g);
  }
  return v;
}

static void codec_case(const std::string& dir, const std::string& name, size_t n, uint64_t seed, float alpha, int k,
                       int sps, double of = 0, double os = 1) {
  auto x = gauss_bf16(n, seed, 1.0, of, os);
  Encoded E = rice_encode(x, alpha, k, sps);
  auto z = decode_symbols(E.words, E.offsets, E.ks, sps, n);
  std::vector<uint8_t> ob, o8, oi;
  inverse_to(z, alpha, 0, ob); inverse_to(z, alpha, 1, o8); inverse_to(z, alpha, 2, oi);
  uint32_t ovf = 0; for (auto s : E.z) ovf += s > 255;
  uint32_t sat = 0;
  for (size_t g = 0; g < n / 8; ++g) { int Y[8]; inverse_word_Y(&z[g * 8], Y); for (int i = 0; i < 8; ++i) sat += (Y[i] < -127 || Y[i] > 127); }
  std::vector<uint32_t> zf(E.z.begin(), E.z.end());
  Interleaved I = build_interleaved(zf, k, sps);
  hqv::File F;
  F.put("in_bf16", x); F.put1<float>("alpha", alpha); F.put1<uint32_t>("k", k); F.put1<uint32_t>("sps", sps);
  F.put("words", E.words); F.put("offsets", E.offsets); F.put("ks", E.ks); F.put1<uint32_t>("total_bits", E.total_bits);
  F.put("z", z); F.put("out_bf16", ob); F.put("out_fp8", o8); F.put("out_int8", oi);
  F.put("il_words", I.words); F.put("il_group_base", I.group_base);
  F.put1<uint32_t>("n_sym_over_255", ovf); F.put1<uint32_t>("n_int8_saturated", sat);
  F.save(dir + "/codec_" + name + ".hqv");
  printf("codec_%-16s n=%zu alpha=%.2f k=%d sps=%d bits/sym=%.3f z>255=%u sat=%u\n", name.c_str(), n, alpha, k, sps,
         (double)(E.total_bits - 32) / n, ovf, sat);
}

static void rht_case(const std::string& dir, int H, int rows, int K, uint64_t seed) {
  auto x = gauss_bf16((size_t)rows * K, seed, 3.0, 0.01, 20.0);
  SplitMix64 r(seed ^ 0xABCDEF); std::vector<int8_t> s(K);
  for (auto& v : s) v = (r.next() >> 63) ? 1 : -1;
  float inv = 1.0f / std::sqrt((float)H);
  auto y = rht_lastdim(x, rows, K, H, s, inv);
  hqv::File F; F.put("x", x); F.put("signs", s); F.put("y", y);
  F.put1<uint32_t>("H", H); F.put1<uint32_t>("rows", rows); F.put1<uint32_t>("K", K);
  F.save(dir + "/rht_" + std::to_string(H) + ".hqv");
  printf("rht_%d rows=%d K=%d\n", H, rows, K);
}

static void lx_case(const std::string& dir) {
  const int M = 5, K = 1024, N = 96; const float alpha = 13.5f;
  auto xp = gauss_bf16((size_t)M * K, 77, 2.0, 0.002, 15.0);
  std::vector<int8_t> x8; std::vector<float> deq; rowmax_cast(xp, M, K, alpha, x8, deq);
  auto w = gauss_bf16((size_t)N * K, 78);
  Encoded E = rice_encode(w, alpha, 2, 512);
  auto z = decode_symbols(E.words, E.offsets, E.ks, 512, w.size());
  std::vector<uint8_t> wi; inverse_to(z, alpha, 2, wi);
  std::vector<int8_t> W8(wi.begin(), wi.end());
  auto C = int8_gemm_tn(x8, W8, M, N, K);
  auto y = dequant(C, deq, M, N);
  std::vector<uint8_t> f8; std::vector<float> sa; fp8_rowmax_cast(xp, M, K, f8, sa);
  hqv::File F;
  F.put1<uint32_t>("M", M); F.put1<uint32_t>("K", K); F.put1<uint32_t>("N", N); F.put1<float>("alpha", alpha);
  F.put("xp", xp); F.put("x8", x8); F.put("deq", deq); F.put("w_words", E.words); F.put("w_offsets", E.offsets);
  F.put("w_ks", E.ks); F.put("W8", W8); F.put("C", C); F.put("y", y); F.put("fp8_x", f8); F.put("fp8_scale", sa);
  F.save(dir + "/lattice_int8.hqv");
  printf("lattice_int8 M=%d K=%d N=%d\n", M, K, N);
}

static void gemv_case(const std::string& dir, const std::string& name, int N, int K, int sps, int B, uint64_t seed) {
  const float alpha = 13.5f;
  auto w = gauss_bf16((size_t)N * K, seed);
  int kk = 3;  // RiceLinear typically picks k=3 at alpha 13.5
  Encoded E = rice_encode(w, alpha, kk, sps);
  auto x = gauss_bf16((size_t)std::max(B, 1) * K, seed + 1, 0.5);
  auto sg = gauss_bf16(N, seed + 2, 0.02);
  for (auto& s : sg) s = f2bf_rn(std::fabs(bf2f(s)) + 0.01f);
  hqv::File F;
  F.put1<uint32_t>("N", N); F.put1<uint32_t>("K", K); F.put1<uint32_t>("sps", sps); F.put1<uint32_t>("B", B);
  F.put1<float>("alpha", alpha); F.put("words", E.words); F.put("offsets", E.offsets); F.put("ks", E.ks);
  F.put("x", x); F.put("sigma", sg);
  if (B == 0) F.put("y", fused_gemv(E, sps, alpha, x, sg, N, K));
  else F.put("y", fused_gemm_ws(E, alpha, x, sg, N, K, B));
  F.save(dir + "/" + name + ".hqv");
  printf("%s N=%d K=%d sps=%d B=%d\n", name.c_str(), N, K, sps, B);
}

static void attn_case(const std::string& dir, const std::string& name, int H, int G, int n_complete, int n_pend,
                      uint64_t seed, float alpha = 13.5f) {
  const int D = 128, sps = 512, ct = sps / D;
  AttnIn A; A.H = H; A.G = G; A.D = D; A.n_complete = n_complete; A.n_pend = n_pend; A.ct = ct; A.sps = sps; A.alpha = alpha;
  SplitMix64 r(seed);
  auto unit_vecs = [&](int n) {  // normalized vectors with norm sqrt(D), bf16
    std::vector<uint16_t> u((size_t)n * D);
    for (int i = 0; i < n; ++i) {
      std::vector<double> g(D); double s = 0;
      for (auto& v : g) { v = r.gaussian(); s += v * v; }
      for (int d = 0; d < D; ++d) u[(size_t)i * D + d] = f2bf_rn((float)(g[d] / std::sqrt(s) * std::sqrt((double)D)));
    }
    return u;
  };
  auto norms = [&](int n) { std::vector<uint16_t> v(n); for (auto& x : v) x = f2bf_rn((float)(8.0 + 4.0 * r.uniform())); return v; };
  auto build = [&](Encoded& C) {  // consolidated chunk-major stream, as RiceKVCacheLayer._append_chunk
    C = Encoded{};
    for (int c = 0; c < n_complete / ct; ++c) {
      auto u = unit_vecs(H * ct);  // [H, ct, D]
      Encoded e = rice_encode(u, alpha, 2, sps);
      uint32_t shift = (uint32_t)C.words.size() * 32u;
      for (auto o : e.offsets) C.offsets.push_back(o + shift);
      C.words.insert(C.words.end(), e.words.begin(), e.words.end());
      C.ks.insert(C.ks.end(), e.ks.begin(), e.ks.end());
    }
    if (C.words.empty()) { C.words.push_back(0); C.offsets.push_back(0); C.ks.push_back(2); }
  };
  build(A.K); build(A.V);
  A.norms_k = norms(H * std::max(n_complete, 1)); A.norms_v = norms(H * std::max(n_complete, 1));
  A.pend_uk = unit_vecs(H * std::max(n_pend, 1)); A.pend_uv = unit_vecs(H * std::max(n_pend, 1));
  A.pend_nk = norms(H * std::max(n_pend, 1)); A.pend_nv = norms(H * std::max(n_pend, 1));
  A.Q.resize((size_t)H * G * D);
  for (auto& q : A.Q) q = f2bf_rn((float)(r.gaussian() * 0.3));
  auto out = fused_attn(A);
  hqv::File F;
  for (auto& p : {std::make_pair("H", H), std::make_pair("G", G), std::make_pair("D", D), std::make_pair("n_complete", n_complete),
                  std::make_pair("n_pend", n_pend), std::make_pair("ct", ct), std::make_pair("sps", sps)})
    F.put1<uint32_t>(p.first, (uint32_t)p.second);
  F.put1<float>("alpha", alpha);
  F.put("words_k", A.K.words); F.put("offsets_k", A.K.offsets); F.put("ks_k", A.K.ks); F.put("norms_k", A.norms_k);
  F.put("words_v", A.V.words); F.put("offsets_v", A.V.offsets); F.put("ks_v", A.V.ks); F.put("norms_v", A.norms_v);
  F.put("pend_uk", A.pend_uk); F.put("pend_nk", A.pend_nk); F.put("pend_uv", A.pend_uv); F.put("pend_nv", A.pend_nv);
  F.put("Q", A.Q); F.put("out", out);
  F.save(dir + "/" + name + ".hqv");
  printf("%s H=%d G=%d T=%d+%d\n", name.c_str(), H, G, n_complete, n_pend);
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "vectors";
  // codec: alpha 13.5 is RiceLinear's default; ~9.9/~6.0 bracket the 4/3 bps calibrated alphas
  codec_case(dir, "a13_k2_s512", 64 * 4096, 1, 13.5f, 2, 512);
  codec_case(dir, "a10_k2_s512", 64 * 4096, 2, 9.9f, 2, 512);
  codec_case(dir, "a6_k1_s512", 32 * 4096, 3, 6.0f, 1, 512);
  codec_case(dir, "a20_k3_s64", 16 * 4096, 4, 20.0f, 3, 64);
  codec_case(dir, "a13_k0_s256", 16 * 4096, 5, 13.5f, 0, 256);
  codec_case(dir, "heavy_a13_k2", 32 * 4096, 6, 13.5f, 2, 512, 0.002, 12.0);  // z>255 wrap + int8 saturation
  for (int H : {256, 512, 1024, 2048}) rht_case(dir, H, 6, 4096, 100 + H);
  lx_case(dir);
  gemv_case(dir, "gemv_n260_k4096", 260, 4096, 512, 0, 300);
  gemv_case(dir, "gemv_n130_k2048_s128", 130, 2048, 128, 0, 301);
  gemv_case(dir, "gemmws_b3", 260, 4096, 512, 3, 302);
  gemv_case(dir, "gemmws_b11", 100, 2048, 512, 11, 303);
  attn_case(dir, "attn_h8g4_t64", 8, 4, 64, 0, 400);
  attn_case(dir, "attn_h8g4_t300p3", 8, 4, 300, 3, 401);
  attn_case(dir, "attn_h8g4_t508p3", 8, 4, 508, 3, 402);
  attn_case(dir, "attn_h8g4_p2only", 8, 4, 0, 2, 403);
  attn_case(dir, "attn_h4g7_t128", 4, 7, 128, 1, 404);  // GQA 7 (Qwen2.5-7B): beyond the 128-thread block
  return 0;
}
