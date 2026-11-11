// load-pattern probe: how fast can the paged FP4 KV (shuffled pages) be streamed with different access shapes?
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
using namespace sycl::ext::intel::esimd;
using u8 = uint8_t;
template <class F> static double timeit(sycl::queue& q, F f) {
  for (int i = 0; i < 3; i++) f(); q.wait();
  double best = 1e30;
  for (int r = 0; r < 5; r++) { auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 10; i++) f(); q.wait(); best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10); }
  return best;
}
int main(int argc, char** argv) {
  const int B = 8, ctx = 16384, HKV = 8, PAGE = 64, DH = 64;
  const int npg = ctx / PAGE, pages = B * npg;
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  const size_t nk = (size_t)pages * HKV * PAGE * DH;
  u8* kc = sycl::malloc_device<u8>(nk, q); u8* vc = sycl::malloc_device<u8>(nk, q);
  { std::vector<u8> h(nk); std::mt19937 r2(5); for (auto& x : h) x = (u8)r2(); q.memcpy(kc, h.data(), nk).wait(); for (auto& x : h) x = (u8)r2(); q.memcpy(vc, h.data(), nk).wait(); }
  std::vector<int> perm(pages); for (int i = 0; i < pages; i++) perm[i] = i; std::mt19937 rng(1); std::shuffle(perm.begin(), perm.end(), rng);
  int* bt = sycl::malloc_device<int>(pages, q); q.memcpy(bt, perm.data(), pages * 4).wait();
  uint32_t* out = sycl::malloc_device<uint32_t>(1 << 20, q);
  const double bytes = 2.0 * nk;
  auto report = [&](const char* n, double ms) { printf("%-44s %.3f ms  %.1f GB/s\n", n, ms, bytes / ms / 1e6); };
  // (c) SYCL work-group per (b, hk, page): 256 work-items x 16B x 2 (K and V) = 4KB each
  {
    auto run = [&]() { q.parallel_for(sycl::nd_range<1>((size_t)B * HKV * npg * 256, 256), [=](sycl::nd_item<1> it) {
      int g = it.get_group(0), t = it.get_local_id(0); int p = g % npg, hk = (g / npg) % HKV, b = g / (npg * HKV);
      size_t base = ((size_t)bt[b * npg + p] * HKV + hk) * PAGE * DH;
      sycl::uint4 a = reinterpret_cast<const sycl::uint4*>(kc + base)[t], c = reinterpret_cast<const sycl::uint4*>(vc + base)[t];
      uint32_t x = a.x() ^ a.y() ^ a.z() ^ a.w() ^ c.x() ^ c.y() ^ c.z() ^ c.w(); if (x == 0xdeadbeef) out[0] = x; }); };
    report("SYCL wg256 per (b,hk,page) uint4 K+V", timeit(q, run));
  }
  // (a) ESIMD thread per (b,hk,split): 4x256B K + 4x256B V per 16-token tile, tiles sequential
  for (int ns : {8, 32}) {
    const int pps = npg / ns;
    auto run = [&]() { q.parallel_for(sycl::nd_range<1>((size_t)B * HKV * ns, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      int g = it.get_global_id(0); int sp = g % ns, hk = (g / ns) % HKV, b = g / (ns * HKV);
      simd<uint32_t, 16> y = 0;
      for (int p = sp * pps; p < (sp + 1) * pps; p++) {
        size_t base = (((size_t)bt[b * npg + p] * HKV + hk) * PAGE) * 16;  // dwords
        for (int tile = 0; tile < 4; tile++) {
          simd<uint32_t, 256> a, c;
          for (int i = 0; i < 4; i++) { a.select<64, 1>(i * 64) = block_load<uint32_t, 64>((const uint32_t*)kc + base + tile * 256 + i * 64); c.select<64, 1>(i * 64) = block_load<uint32_t, 64>((const uint32_t*)vc + base + tile * 256 + i * 64); }
          simd<uint32_t, 256> z = a ^ c; for (int t = 0; t < 16; t++) y ^= z.select<16, 1>(t * 16);
        }
      }
      block_store<uint32_t, 16>(out + g * 16, y); }); };
    char nm[96]; snprintf(nm, 96, "ESIMD thread per (b,hk,split=%d) 256B blocks", ns); report(nm, timeit(q, run));
  }
  // (b) same but 1KB per message via 2D-free single big block_load<uint32_t, 128>? (512B)
  {
    const int ns = 8, pps = npg / ns;
    auto run = [&]() { q.parallel_for(sycl::nd_range<1>((size_t)B * HKV * ns, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      int g = it.get_global_id(0); int sp = g % ns, hk = (g / ns) % HKV, b = g / (ns * HKV);
      simd<uint32_t, 16> y = 0;
      for (int p = sp * pps; p < (sp + 1) * pps; p++) {
        size_t base = (((size_t)bt[b * npg + p] * HKV + hk) * PAGE) * 16;
        for (int tile = 0; tile < 4; tile++) {
          simd<uint32_t, 256> a, c;
          for (int i = 0; i < 2; i++) { a.select<128, 1>(i * 128) = block_load<uint32_t, 128>((const uint32_t*)kc + base + tile * 256 + i * 128); c.select<128, 1>(i * 128) = block_load<uint32_t, 128>((const uint32_t*)vc + base + tile * 256 + i * 128); }
          simd<uint32_t, 256> z = a ^ c; for (int t = 0; t < 16; t++) y ^= z.select<16, 1>(t * 16);
        }
      }
      block_store<uint32_t, 16>(out + g * 16, y); }); };
    report("ESIMD split=8 512B blocks", timeit(q, run));
  }
  // (d) ESIMD, a whole page-chunk (4KB K) per thread iteration, 8 x 512B loads issued back to back before use
  {
    const int ns = 8, pps = npg / ns;
    auto run = [&]() { q.parallel_for(sycl::nd_range<1>((size_t)B * HKV * ns, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      int g = it.get_global_id(0); int sp = g % ns, hk = (g / ns) % HKV, b = g / (ns * HKV);
      simd<uint32_t, 16> y = 0;
      for (int p = sp * pps; p < (sp + 1) * pps; p++) {
        size_t base = (((size_t)bt[b * npg + p] * HKV + hk) * PAGE) * 16;
        simd<uint32_t, 1024> a, c;
        for (int i = 0; i < 8; i++) { a.select<128, 1>(i * 128) = block_load<uint32_t, 128>((const uint32_t*)kc + base + i * 128); c.select<128, 1>(i * 128) = block_load<uint32_t, 128>((const uint32_t*)vc + base + i * 128); }
        simd<uint32_t, 1024> z = a ^ c; for (int t = 0; t < 64; t++) y ^= z.select<16, 1>(t * 16);
      }
      block_store<uint32_t, 16>(out + g * 16, y); }); };
    report("ESIMD split=8, 4KB chunk per iter (16 msgs)", timeit(q, run));
  }
}
