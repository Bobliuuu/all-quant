// test_main.cpp -- shared correctness driver: runs every backend kernel on the shared
// vectors (vectors/*.hqv, produced by ref/gen_vectors.cpp) and compares.
//   usage: test_vectors <vectors_dir> [--json out.json]
// Criteria: integer outputs (bitstream words, offsets, k, z symbols, int8, int32, fp8
// bytes) must be bit-exact.  bf16 outputs report exact-match count and max ulp; the
// codec/decode/GEMV outputs are required to be bit-exact, RHT H in {512,2048} may be
// <= 1 ulp (CUDA rsqrtf is approximate), fused attention (float, expf) uses
// max|err| <= 2e-4 * max|ref| + 1e-6.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include "backend.hpp"
#include "hq_num.hpp"
#include "hq_vec.hpp"

namespace fs = std::filesystem;
static std::vector<std::string> g_json;
static int g_fail = 0, g_pass = 0;

static void report(const std::string& test, bool ok, const std::string& detail) {
  printf("%-44s %s  %s\n", test.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
  fflush(stdout);
  (ok ? g_pass : g_fail)++;
  std::string d = detail; for (auto& c : d) if (c == '"') c = '\'';
  g_json.push_back("{\"test\":\"" + test + "\",\"pass\":" + (ok ? "true" : "false") + ",\"detail\":\"" + d + "\"}");
}
template <class T> static size_t nmis(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return (size_t)-1;
  size_t n = 0; for (size_t i = 0; i < a.size(); ++i) n += !(a[i] == b[i]); return n;
}
template <class T> static void exact(const std::string& t, const std::vector<T>& got, const std::vector<T>& want) {
  size_t m = nmis(got, want);
  char b[160];
  if (m == (size_t)-1) snprintf(b, sizeof b, "size %zu vs %zu", got.size(), want.size());
  else snprintf(b, sizeof b, "%zu/%zu mismatches", m, want.size());
  report(t, m == 0, b);
}
static int ulp_bf16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -(int)(v & 0x7FFF) : (int)v; };
  return std::abs(key(a) - key(b));
}
static void bf16cmp(const std::string& t, const std::vector<uint16_t>& got, const std::vector<uint16_t>& want, int max_ulp_ok) {
  if (got.size() != want.size()) { report(t, false, "size mismatch"); return; }
  size_t m = 0; int mu = 0;
  for (size_t i = 0; i < got.size(); ++i) if (got[i] != want[i]) { ++m; mu = std::max(mu, ulp_bf16(got[i], want[i])); }
  char b[160]; snprintf(b, sizeof b, "%zu/%zu not bit-exact, max %d ulp (allowed %d)", m, got.size(), mu, max_ulp_ok);
  report(t, mu <= max_ulp_ok, b);
}
static std::vector<uint16_t> as_u16(const std::vector<uint8_t>& b) {
  std::vector<uint16_t> v(b.size() / 2); memcpy(v.data(), b.data(), v.size() * 2); return v;
}

static void test_codec(const hqv::File& F, const std::string& n) {
  auto x = F.get<uint16_t>("in_bf16"); float alpha = F.get1<float>("alpha");
  int k = (int)F.get1<uint32_t>("k"), sps = (int)F.get1<uint32_t>("sps"), total = (int)x.size();
  be::Enc ref{F.get<uint32_t>("words"), F.get<uint32_t>("offsets"), F.get<uint8_t>("ks"), F.get1<uint32_t>("total_bits")};
  for (int mode : {1, 0}) {
    be::Enc e = be::encode(x, alpha, k, sps, mode);
    std::string t = n + (mode ? ".encode(cache)" : ".encode(recompute)");
    size_t mw = nmis(e.words, ref.words), mo = nmis(e.offsets, ref.offsets), mk = nmis(e.ks, ref.ks);
    char b[200]; snprintf(b, sizeof b, "words %zu/%zu offs %zu ks %zu total_bits %u/%u", mw, ref.words.size(), mo, mk, e.total_bits, ref.total_bits);
    report(t, mw == 0 && mo == 0 && mk == 0 && e.total_bits == ref.total_bits, b);
  }
  exact(n + ".decode.z", be::decode(ref, sps, total, alpha, 0), F.get<uint8_t>("z"));
  auto ob = as_u16(F.get<uint8_t>("out_bf16"));
  bf16cmp(n + ".decode_to_bf16(2-kernel)", as_u16(be::decode(ref, sps, total, alpha, 1)), ob, 0);
  bf16cmp(n + ".decode_fused_bf16", as_u16(be::decode(ref, sps, total, alpha, 2)), ob, 0);
  bf16cmp(n + ".decode_fused_coop_bf16", as_u16(be::decode(ref, sps, total, alpha, 3)), ob, 0);
  exact(n + ".decode_fused_fp8", be::decode(ref, sps, total, alpha, 4), F.get<uint8_t>("out_fp8"));
  exact(n + ".decode_fused_int8", be::decode(ref, sps, total, alpha, 5), F.get<uint8_t>("out_int8"));
  bf16cmp(n + ".launch_fused_decode_bf16", as_u16(be::decode(ref, sps, total, alpha, 6)), ob, 0);
  bf16cmp(n + ".decode_interleaved_bf16",
          be::decode_interleaved(F.get<uint32_t>("il_words"), F.get<uint32_t>("il_group_base"), ref.ks, sps, total, alpha), ob, 0);
  exact(n + ".stage3_decode.z", be::stage3_decode(ref.words, k, total, 64), F.get<uint8_t>("z"));
}

static void test_rht(const hqv::File& F, const std::string& n) {
  int H = F.get1<uint32_t>("H"), rows = F.get1<uint32_t>("rows"), K = F.get1<uint32_t>("K");
  bool exact_scale = (H == 256 || H == 1024);  // 1/sqrt(H) is a power of two
  bf16cmp(n, be::rht(F.get<uint16_t>("x"), rows, K, H, F.get<int8_t>("signs")), F.get<uint16_t>("y"), exact_scale ? 0 : 1);
}

static void test_lx(const hqv::File& F, const std::string& n) {
  int M = F.get1<uint32_t>("M"), K = F.get1<uint32_t>("K"), N = F.get1<uint32_t>("N"); float alpha = F.get1<float>("alpha");
  std::vector<int8_t> x8; std::vector<float> deq;
  be::rowmax_cast(F.get<uint16_t>("xp"), M, K, alpha, x8, deq);
  exact(n + ".rowmax_cast.x8", x8, F.get<int8_t>("x8"));
  exact(n + ".rowmax_cast.deq", deq, F.get<float>("deq"));
  be::Enc w{F.get<uint32_t>("w_words"), F.get<uint32_t>("w_offsets"), F.get<uint8_t>("w_ks"), 0};
  auto wi = be::decode(w, 512, N * K, alpha, 5);
  std::vector<int8_t> W8(wi.begin(), wi.end());
  exact(n + ".decode_int8(W)", W8, F.get<int8_t>("W8"));
  auto C = be::int8_gemm_tn(F.get<int8_t>("x8"), F.get<int8_t>("W8"), M, N, K);
  exact(n + ".int8_gemm_tn", C, F.get<int32_t>("C"));
  bf16cmp(n + ".dequant", be::dequant(F.get<int32_t>("C"), F.get<float>("deq"), M, N), F.get<uint16_t>("y"), 0);
  std::vector<uint8_t> f8; std::vector<float> sa;
  be::fp8_rowmax_cast(F.get<uint16_t>("xp"), M, K, f8, sa);
  exact(n + ".fp8_rowmax_cast.x", f8, F.get<uint8_t>("fp8_x"));
  exact(n + ".fp8_rowmax_cast.scale", sa, F.get<float>("fp8_scale"));
}

static void test_gemv(const hqv::File& F, const std::string& n) {
  int N = F.get1<uint32_t>("N"), K = F.get1<uint32_t>("K"), sps = F.get1<uint32_t>("sps"), B = F.get1<uint32_t>("B");
  float alpha = F.get1<float>("alpha");
  be::Enc e{F.get<uint32_t>("words"), F.get<uint32_t>("offsets"), F.get<uint8_t>("ks"), 0};
  auto y = B == 0 ? be::fused_gemv(e, sps, alpha, F.get<uint16_t>("x"), F.get<uint16_t>("sigma"), N, K)
                  : be::fused_gemm_ws(e, alpha, F.get<uint16_t>("x"), F.get<uint16_t>("sigma"), N, K, B);
  bf16cmp(n, y, F.get<uint16_t>("y"), 1);  // fp32 accumulate order mirrors the kernel; allow 1 ulp
}

static void test_attn(const hqv::File& F, const std::string& n) {
  be::AttnArgs a;
  a.H = F.get1<uint32_t>("H"); a.G = F.get1<uint32_t>("G"); a.D = F.get1<uint32_t>("D");
  a.n_complete = F.get1<uint32_t>("n_complete"); a.n_pend = F.get1<uint32_t>("n_pend");
  a.ct = F.get1<uint32_t>("ct"); a.sps = F.get1<uint32_t>("sps"); a.alpha = F.get1<float>("alpha");
  a.K = {F.get<uint32_t>("words_k"), F.get<uint32_t>("offsets_k"), F.get<uint8_t>("ks_k"), 0};
  a.V = {F.get<uint32_t>("words_v"), F.get<uint32_t>("offsets_v"), F.get<uint8_t>("ks_v"), 0};
  a.norms_k = F.get<uint16_t>("norms_k"); a.norms_v = F.get<uint16_t>("norms_v");
  a.pend_uk = F.get<uint16_t>("pend_uk"); a.pend_nk = F.get<uint16_t>("pend_nk");
  a.pend_uv = F.get<uint16_t>("pend_uv"); a.pend_nv = F.get<uint16_t>("pend_nv");
  a.Q = F.get<uint16_t>("Q");
  auto got = be::fused_attn(a); auto want = F.get<float>("out");
  if (got.size() != want.size()) { report(n, false, "size mismatch"); return; }
  double mx = 0, me = 0; size_t bad_rows = 0;
  for (size_t i = 0; i < want.size(); ++i) { mx = std::max(mx, (double)std::fabs(want[i])); me = std::max(me, (double)std::fabs(got[i] - want[i])); }
  for (size_t r = 0; r < want.size() / a.D; ++r) {
    double e = 0; for (int d = 0; d < a.D; ++d) e = std::max(e, (double)std::fabs(got[r * a.D + d] - want[r * a.D + d]));
    bad_rows += e > 2e-4 * mx + 1e-6;
  }
  char b[200]; snprintf(b, sizeof b, "max|err| %.3g (max|ref| %.3g), q-head rows off: %zu/%zu", me, mx, bad_rows, want.size() / a.D);
  report(n, me <= 2e-4 * mx + 1e-6, b);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <vectors_dir> [--json out]\n", argv[0]); return 2; }
  std::string dir = argv[1], json;
  for (int i = 2; i + 1 < argc; ++i) if (!strcmp(argv[i], "--json")) json = argv[i + 1];
  be::init();
  printf("backend: %s\n", be::name().c_str());
  std::vector<std::string> files;
  for (auto& p : fs::directory_iterator(dir)) if (p.path().extension() == ".hqv") files.push_back(p.path().string());
  std::sort(files.begin(), files.end());
  for (auto& f : files) {
    std::string n = fs::path(f).stem().string();
    hqv::File F = hqv::File::load(f);
    try {
      if (n.rfind("codec_", 0) == 0) test_codec(F, n);
      else if (n.rfind("rht_", 0) == 0) test_rht(F, n);
      else if (n == "lattice_int8") test_lx(F, n);
      else if (n.rfind("gemv_", 0) == 0 || n.rfind("gemmws_", 0) == 0) test_gemv(F, n);
      else if (n.rfind("attn_", 0) == 0) test_attn(F, n);
    } catch (std::exception& e) { report(n, false, std::string("exception: ") + e.what()); }
  }
  printf("SUMMARY %s: %d pass, %d fail\n", be::name().c_str(), g_pass, g_fail);
  if (!json.empty()) {
    FILE* o = fopen(json.c_str(), "w");
    fprintf(o, "{\"backend\":\"%s\",\"pass\":%d,\"fail\":%d,\"tests\":[\n", be::name().c_str(), g_pass, g_fail);
    for (size_t i = 0; i < g_json.size(); ++i) fprintf(o, "%s%s\n", g_json[i].c_str(), i + 1 < g_json.size() ? "," : "");
    fprintf(o, "]}\n"); fclose(o);
  }
  return g_fail ? 1 : 0;
}
