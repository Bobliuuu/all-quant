// bench_main.cpp -- shared microbenchmark driver (same shapes, same data on every GPU).
//   usage: bench <out.jsonl> [--quick] [--alpha 7.1] [--k 2]
// Data: Gaussian bf16 weights encoded once by the CPU reference (Llama-3.1-8B shapes),
// so every backend decodes byte-identical bitstreams.  Times are device-event times
// per call (median-free mean over `iters`), inputs resident on the device.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "backend.hpp"
#include "../ref/hq_ref.hpp"

static FILE* g_out = nullptr;
static void emit(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#include <cstdarg>
static void emit(const char* fmt, ...) {
  char b[1024]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
  printf("%s\n", b); fflush(stdout);
  if (g_out) { fprintf(g_out, "%s\n", b); fflush(g_out); }
}

static std::vector<uint16_t> gauss(size_t n, uint64_t seed) {
  hqn::SplitMix64 r(seed); std::vector<uint16_t> v(n);
  for (auto& x : v) x = hqn::f2bf_rn((float)r.gaussian());
  return v;
}
static be::Enc enc(size_t n, uint64_t seed, float alpha, int k, int sps) {
  auto E = hqref::rice_encode(gauss(n, seed), alpha, k, sps);
  return {E.words, E.offsets, E.ks, E.total_bits};
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: bench out.jsonl [--quick] [--alpha a] [--k k]\n"); return 2; }
  bool quick = false; float alpha = 7.1f; int k = 2;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--quick")) quick = true;
    else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) alpha = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--k") && i + 1 < argc) k = atoi(argv[++i]);
  }
  g_out = fopen(argv[1], "a");
  be::init();
  const std::string B = be::name();
  const int it = quick ? 5 : 50;
  emit("{\"bench\":\"copy\",\"backend\":\"%s\",\"bytes\":%d,\"gbps_rw\":%.1f}", B.c_str(), 1 << 30, be::time_copy_gbps(1u << 30, quick ? 3 : 20));

  // 1. stand-alone decode of one 14336x4096 matrix (Llama MLP up/gate), sps 512
  {
    const int N = 14336, K = 4096, sps = 512; const size_t n = (size_t)N * K;
    be::Enc e = enc(n, 11, alpha, k, sps);
    const double bits = (double)(e.total_bits - 32), bps = bits / n;
    const char* vn[] = {"symbols(z)", "2-kernel bf16", "fused bf16", "fused coop bf16", "fused fp8", "fused int8", "launcher bf16"};
    for (int v = 0; v <= 6; ++v) {
      double ms = be::time_decode(e, sps, (int)n, alpha, v, quick ? 3 : 20);
      int ob = (v == 0 || v == 4 || v == 5) ? 1 : 2;
      emit("{\"bench\":\"decode\",\"backend\":\"%s\",\"variant\":\"%s\",\"N\":%d,\"K\":%d,\"sps\":%d,\"alpha\":%.3f,\"k\":%d,"
           "\"bits_per_sym\":%.4f,\"ms\":%.4f,\"Gsym_s\":%.2f,\"bitstream_GBps\":%.2f,\"out_GBps\":%.2f}",
           B.c_str(), vn[v], N, K, sps, alpha, k, bps, ms, n / (ms * 1e6), bits / 8 / (ms * 1e6), (double)n * ob / (ms * 1e6));
    }
  }
  // 2. RHT on a 2048x4096 activation (prefill), H = 256 / 1024
  for (int H : {256, 1024}) {
    const int rows = 2048, K = 4096; double ms = be::time_rht(rows, K, H, it);
    emit("{\"bench\":\"rht\",\"backend\":\"%s\",\"rows\":%d,\"K\":%d,\"H\":%d,\"ms\":%.4f,\"GBps_rw\":%.1f}", B.c_str(), rows, K, H, ms,
         4.0 * rows * K / (ms * 1e6));
  }
  // 3. fused Rice-decode GEMV (batch 1) on Llama-3.1-8B layer shapes, sps 512
  struct Sh { const char* n; int N, K; } shapes[] = {{"q/o_proj", 4096, 4096}, {"k/v_proj", 1024, 4096}, {"gate/up_proj", 14336, 4096}, {"down_proj", 4096, 14336}};
  double layer_ms = 0;
  for (auto& s : shapes) {
    be::Enc e = enc((size_t)s.N * s.K, 21 + s.N + s.K, alpha, k, 512);
    double ms = be::time_gemv(e, 512, alpha, s.N, s.K, it);
    int mult = (!strcmp(s.n, "q/o_proj") || !strcmp(s.n, "k/v_proj") || !strcmp(s.n, "gate/up_proj")) ? 2 : 1;
    layer_ms += mult * ms;
    emit("{\"bench\":\"gemv\",\"backend\":\"%s\",\"shape\":\"%s\",\"N\":%d,\"K\":%d,\"ms\":%.4f,\"bitstream_GBps\":%.2f}", B.c_str(), s.n,
         s.N, s.K, ms, (e.total_bits / 8.0) / (ms * 1e6));
    if (!strcmp(s.n, "q/o_proj") || !strcmp(s.n, "gate/up_proj")) {
      for (int b : {4, 8}) {
        double m2 = be::time_gemm_ws(e, alpha, s.N, s.K, b, quick ? 3 : 10);
        emit("{\"bench\":\"gemm_ws\",\"backend\":\"%s\",\"shape\":\"%s\",\"N\":%d,\"K\":%d,\"B\":%d,\"ms\":%.4f}", B.c_str(), s.n, s.N, s.K, b, m2);
      }
    }
  }
  emit("{\"bench\":\"gemv_layer\",\"backend\":\"%s\",\"ms_per_layer\":%.4f,\"ms_per_token_32_layers\":%.3f}", B.c_str(), layer_ms, 32 * layer_ms);
  // 4. fused Rice attention, one Llama layer (8 KV heads x 4 q/kv, D 128), T = n_complete + 3 pending
  for (int T : {128, 256, 508}) {
    be::AttnArgs a; a.H = 8; a.G = 4; a.D = 128; a.n_complete = T; a.n_pend = 3; a.ct = 4; a.sps = 512; a.alpha = alpha;
    auto build = [&](be::Enc& C, uint64_t seed) {
      C = {};
      for (int c = 0; c < T / 4; ++c) {
        auto E = hqref::rice_encode(gauss((size_t)8 * 4 * 128, seed + c), alpha, k, 512);
        uint32_t sh = (uint32_t)C.words.size() * 32u;
        for (auto o : E.offsets) C.offsets.push_back(o + sh);
        C.words.insert(C.words.end(), E.words.begin(), E.words.end()); C.ks.insert(C.ks.end(), E.ks.begin(), E.ks.end());
      }
    };
    build(a.K, 1000); build(a.V, 2000);
    std::vector<uint16_t> nrm((size_t)8 * T, hqn::f2bf_rn(10.f)), pn(8 * 3, hqn::f2bf_rn(10.f));
    a.norms_k = nrm; a.norms_v = nrm; a.pend_nk = pn; a.pend_nv = pn;
    a.pend_uk = gauss((size_t)8 * 3 * 128, 5); a.pend_uv = gauss((size_t)8 * 3 * 128, 6); a.Q = gauss((size_t)32 * 128, 7);
    double ms = be::time_attn(a, it);
    emit("{\"bench\":\"attn\",\"backend\":\"%s\",\"H\":8,\"G\":4,\"T\":%d,\"us\":%.2f}", B.c_str(), T + 3, ms * 1e3);
  }
  // 5. int8 IMMA linear (lattice_linear_forward), 4096x4096, H 256
  {
    const int N = 4096, K = 4096; be::Enc e = enc((size_t)N * K, 31, alpha, k, 512);
    for (int M : {1, 16, 2048}) {
      double ms = be::time_int8_linear(e, alpha, M, N, K, 256, quick ? 3 : 10);
      emit("{\"bench\":\"int8_linear\",\"backend\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms\":%.4f}", B.c_str(), M, N, K, ms);
    }
  }
  if (g_out) fclose(g_out);
  return 0;
}
