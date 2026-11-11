// Benchmark: fused K-query vs K-independent baseline; copy-kernel bandwidth peak.
//   bench [--B 2,4,8] [--ctx 4096,16384] [--K 1,2,4,8] [--nsplit N|auto|sweep] [--wg W] [--tn T] [--sg S]
//         [--path portable|xmx] [--iters N] [--hq 32 --hkv 8 --d 128 --page 64] [--nobase] [--json out.jsonl]
#include "uq_decode.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <algorithm>
#include <string>
#include <vector>

static std::vector<int> parse_list(const char* s) {
  std::vector<int> v; std::string t(s); size_t p = 0;
  while (p < t.size()) { size_t e = t.find(',', p); if (e == std::string::npos) e = t.size(); v.push_back(std::atoi(t.substr(p, e - p).c_str())); p = e + 1; }
  return v;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// best-of timing: warmup, then `iters` back-to-back submissions on an in-order queue
template <class F> static double time_ms(sycl::queue& q, int iters, F f) {
  for (int i = 0; i < 3; i++) f();
  q.wait();
  double best = 1e30;
  for (int rep = 0; rep < 3; rep++) {
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) f();
    q.wait();
    best = std::fmin(best, (now_ms() - t0) / iters);
  }
  return best;
}

static double peak_gbs(sycl::queue& q) {
  const size_t n = (size_t)256 << 20;  // 256 MiB each way
  auto* a = sycl::malloc_device<uint32_t>(n / 4, q); auto* b = sycl::malloc_device<uint32_t>(n / 4, q);
  q.fill(a, 0x01010101u, n / 4).wait();
  auto run = [&]() {
    q.parallel_for(sycl::range<1>(n / 16), [=](sycl::id<1> i) {
      sycl::uint4 v = reinterpret_cast<const sycl::uint4*>(a)[i];
      reinterpret_cast<sycl::uint4*>(b)[i] = v;
    });
  };
  double ms = time_ms(q, 10, run);
  sycl::free(a, q); sycl::free(b, q);
  return 2.0 * n / (ms * 1e-3) / 1e9;  // read + write bytes
}
// random=true: incompressible data. Uniform-fill buffers are served 2.5-4x faster on Arc (memory compression), so the
// "uniform" peak is NOT the bandwidth a KV stream of random FP4 codes can see; random data is the honest ceiling.
static double peak_read_gbs(sycl::queue& q, bool random = false) {
  const size_t n = (size_t)(random ? 256 : 512) << 20;
  auto* a = sycl::malloc_device<uint32_t>(n / 4, q); auto* o = sycl::malloc_device<uint32_t>(1 << 16, q);
  if (random) {
    std::vector<uint32_t> h(n / 4); std::mt19937 r(7); for (auto& x : h) x = r();
    q.memcpy(a, h.data(), n).wait();
  } else q.fill(a, 1u, n / 4).wait();
  auto run = [&]() {
    q.parallel_for(sycl::range<1>(n / 16), [=](sycl::id<1> i) {
      sycl::uint4 v = reinterpret_cast<const sycl::uint4*>(a)[i];
      uint32_t x = v.x() ^ v.y() ^ v.z() ^ v.w();
      if (x == 0xdeadbeef) o[0] = x;  // never true; defeats DCE
    });
  };
  double ms = time_ms(q, 10, run);
  sycl::free(a, q); sycl::free(o, q);
  return (double)n / (ms * 1e-3) / 1e9;
}

int main(int argc, char** argv) {
  std::vector<int> Bs{2, 8}, ctxs{4096, 16384}, Ks{1, 2, 4, 8};
  int HQ = 32, HKV = 8, D = 128, PAGE = 64, iters = 10, nsplit = 0 /*auto*/; bool sweep = false, nobase = false, peakonly = false;
  uq::Cfg c0; std::string json;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i]; auto nx = [&]() { return argv[++i]; };
    if (a == "--B") Bs = parse_list(nx()); else if (a == "--ctx") ctxs = parse_list(nx()); else if (a == "--K") Ks = parse_list(nx());
    else if (a == "--hq") HQ = std::atoi(nx()); else if (a == "--hkv") HKV = std::atoi(nx()); else if (a == "--d") D = std::atoi(nx());
    else if (a == "--page") PAGE = std::atoi(nx()); else if (a == "--iters") iters = std::atoi(nx());
    else if (a == "--wg") c0.wg = std::atoi(nx()); else if (a == "--tn") c0.tn = std::atoi(nx()); else if (a == "--sg") c0.sg = std::atoi(nx());
    else if (a == "--path") { std::string pv = nx(); c0.path = pv == "best" ? 4 : pv == "esimd" ? 3 : pv == "xmx2" ? 2 : pv == "xmx"; }
    else if (a == "--nsplit") { std::string v = nx(); if (v == "sweep") sweep = true; else if (v != "auto") nsplit = std::atoi(v.c_str()); }
    else if (a == "--nobase") nobase = true; else if (a == "--peakonly") peakonly = true; else if (a == "--json") json = nx();
  }
  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  auto dev = q.get_device();
  std::string dname = dev.get_info<sycl::info::device::name>();
  int eus = (int)dev.get_info<sycl::info::device::max_compute_units>();
  std::printf("device: %s  compute_units=%d  local_mem=%zu KB\n", dname.c_str(), eus, dev.get_info<sycl::info::device::local_mem_size>() / 1024);
  double pcopy = peak_gbs(q), pread_u = peak_read_gbs(q), pread = peak_read_gbs(q, true);
  std::printf("peak copy (r+w, uniform data) %.1f GB/s   peak read uniform-data %.1f GB/s   peak read RANDOM data %.1f GB/s (used for %% of peak)\n", pcopy, pread_u, pread);
  FILE* jf = json.empty() ? nullptr : std::fopen(json.c_str(), "w");
  if (jf) std::fprintf(jf, "{\"kind\":\"peak\",\"device\":\"%s\",\"copy_gbs\":%.2f,\"read_uniform_gbs\":%.2f,\"read_gbs\":%.2f}\n", dname.c_str(), pcopy, pread_u, pread);
  if (peakonly) return 0;

  std::mt19937 rng(1);
  for (int B : Bs) for (int ctx : ctxs) {
    const int npg = (ctx + PAGE - 1) / PAGE, pages = B * npg;
    const size_t nkc = (size_t)pages * HKV * PAGE * (D / 2), nks = (size_t)pages * HKV * PAGE * (D / 32);
    std::vector<uint8_t> h(std::max(nkc, nks));
    auto fillrand = [&](uint8_t* d, size_t n, int lo, int hi) {
      for (size_t i = 0; i < n; i++) h[i] = (uint8_t)(lo + (int)(rng() % (hi - lo + 1)));
      q.memcpy(d, h.data(), n).wait();
    };
    auto *kc = sycl::malloc_device<uint8_t>(nkc, q), *vc = sycl::malloc_device<uint8_t>(nkc, q);
    auto *ks = sycl::malloc_device<uint8_t>(nks, q), *vs = sycl::malloc_device<uint8_t>(nks, q);
    fillrand(kc, nkc, 0, 255); fillrand(vc, nkc, 0, 255); fillrand(ks, nks, 120, 125); fillrand(vs, nks, 120, 125);
    std::vector<int> hbt(B * npg); std::vector<int> perm(pages); for (int i = 0; i < pages; i++) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng); for (int i = 0; i < B * npg; i++) hbt[i] = perm[i];
    auto* bt = sycl::malloc_device<int>(B * npg, q); q.memcpy(bt, hbt.data(), B * npg * 4).wait();
    for (int K : Ks) {
      std::vector<int> hsl(B, ctx);
      auto* sl = sycl::malloc_device<int>(B, q); q.memcpy(sl, hsl.data(), B * 4).wait();
      const int HQK = HQ;
      std::vector<uint16_t> hq((size_t)B * K * HQK * D);
      for (auto& x : hq) x = 0x3f80 + (rng() % 64) - 32;  // ~1.0 bf16
      auto* qd = sycl::malloc_device<uq::bf16>(hq.size(), q); q.memcpy(qd, hq.data(), hq.size() * 2).wait();
      auto* out = sycl::malloc_device<uq::bf16>(hq.size(), q);
      // baseline pseudo-sequences
      std::vector<int> hbt2((size_t)B * K * npg), hsl2((size_t)B * K);
      for (int b = 0; b < B; b++) for (int j = 0; j < K; j++) {
        for (int p = 0; p < npg; p++) hbt2[((size_t)b * K + j) * npg + p] = hbt[b * npg + p];
        hsl2[b * K + j] = ctx - K + j + 1;
      }
      auto* bt2 = sycl::malloc_device<int>(hbt2.size(), q); q.memcpy(bt2, hbt2.data(), hbt2.size() * 4).wait();
      auto* sl2 = sycl::malloc_device<int>(hsl2.size(), q); q.memcpy(sl2, hsl2.data(), hsl2.size() * 4).wait();

      const double uniq_bytes = 2.0 * B * (double)ctx * HKV * (D / 2 + D / 32);
      std::vector<int> nss;
      if (sweep) nss = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96}; else if (nsplit) nss = {nsplit};
      else { int want = std::max(1, 4 * eus / std::max(1, B * HKV)); int ns = 1; while (ns * 2 <= want && ns * 2 <= std::max(1, npg)) ns *= 2; nss = {ns}; }
      for (int ns : nss) {
        if (ns > npg) continue;
        uq::Cfg c = c0; c.nsplit = ns;
        if (PAGE % c.tn) c.tn = 16;
        uq::Shape sf{B, K, HQ, HKV, D, PAGE, npg, 1.f / std::sqrt((float)D)};
        auto* mo = sycl::malloc_device<float>(uq::mo_elems(sf, ns), q); auto* ml = sycl::malloc_device<float>(uq::ml_elems(sf, ns), q);
        uq::Args af{qd, kc, ks, vc, vs, bt, sl, mo, ml, out};
        double ms_f = time_ms(q, iters, [&]() { uq::decode(q, sf, c, af); });
        double ms_b = -1;
        if (!nobase && K > 1) {
          uq::Shape sb{B * K, 1, HQ, HKV, D, PAGE, npg, sf.sm_scale};
          auto* mo2 = sycl::malloc_device<float>(uq::mo_elems(sb, ns), q); auto* ml2 = sycl::malloc_device<float>(uq::ml_elems(sb, ns), q);
          uq::Args ab{qd, kc, ks, vc, vs, bt2, sl2, mo2, ml2, out};
          ms_b = time_ms(q, iters, [&]() { uq::decode(q, sb, c, ab); });
          sycl::free(mo2, q); sycl::free(ml2, q);
        }
        double gbs = uniq_bytes / (ms_f * 1e-3) / 1e9;
        std::printf("B=%d ctx=%d K=%d nsplit=%d wg=%d tn=%d sg=%d path=%d | fused %.3f ms  %.1f GB/s (%.0f%% of read peak)", B, ctx, K, ns, c.wg, c.tn, c.sg, c.path, ms_f, gbs, 100 * gbs / pread);
        if (ms_b > 0) std::printf(" | K-indep %.3f ms (x%.2f)", ms_b, ms_b / ms_f);
        std::printf("\n"); std::fflush(stdout);
        if (jf) std::fprintf(jf, "{\"kind\":\"run\",\"device\":\"%s\",\"B\":%d,\"ctx\":%d,\"K\":%d,\"Hq\":%d,\"Hkv\":%d,\"D\":%d,\"page\":%d,\"nsplit\":%d,\"wg\":%d,\"tn\":%d,\"sg\":%d,\"path\":%d,\"fused_ms\":%.4f,\"kindep_ms\":%.4f,\"fp4kv_gbs\":%.2f,\"pct_read_peak\":%.1f,\"pct_copy_peak\":%.1f}\n",
                              dname.c_str(), B, ctx, K, HQ, HKV, D, PAGE, ns, c.wg, c.tn, c.sg, c.path, ms_f, ms_b, gbs, 100 * gbs / pread, 100 * gbs / pcopy);
        sycl::free(mo, q); sycl::free(ml, q);
      }
      for (void* p : {(void*)sl, (void*)qd, (void*)out, (void*)bt2, (void*)sl2}) sycl::free(p, q);
    }
    for (void* p : {(void*)kc, (void*)vc, (void*)ks, (void*)vs, (void*)bt}) sycl::free(p, q);
  }
  if (jf) std::fclose(jf);
  return 0;
}
