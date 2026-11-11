// micro-probe: ESIMD K path only (2D transposed block load -> register-LUT decode via iselect -> DPAS bf16 -> group-scaled S).
// Measures decode+MMA throughput and checks S against a CPU reference on a few tiles.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using u8 = uint8_t;
constexpr int D = 128, DH = 64, PAGE = 64, HKV = 8;

template <int RC>
struct KProbe {
  const uint16_t* q; const u8* kc; const u8* ks; const uint32_t* lutg; float* out; int pages_per_thread; int nthreads; float scl;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    const int t = (int)it.get_global_id(0);
    const int hk = t % HKV;
    simd<uint32_t, 256> lut;
    #pragma unroll
    for (int i = 0; i < 4; i++) lut.template select<64, 1>(i * 64) = block_load<uint32_t, 64>(lutg + i * 64);
    simd<uint32_t, RC * 8> A[8];
    #pragma unroll
    for (int kk = 0; kk < 8; kk++)
      #pragma unroll
      for (int r = 0; r < RC; r++)
        A[kk].template select<8, 1>(r * 8) = block_load<uint32_t, 8>(reinterpret_cast<const uint32_t*>(q) + r * (D / 2) + kk * 8);
    simd<float, RC * 16> total = 0.f;
    for (int pp = 0; pp < pages_per_thread; pp++) {
      const int page = t * pages_per_thread + pp;  // each thread owns its own pages
      const u8* kb = kc + ((size_t)page * PAGE) * DH;  // (head index folded into page for the probe)
      const u8* sb = ks + ((size_t)page * PAGE) * 4;
      for (int tile = 0; tile < PAGE / 16; tile++) {
#if defined(KP_LO1) || defined(KP_LO2) || defined(KP_LO3) || defined(KP_LO4) || defined(KP_LO5) || defined(KP_LO6)
        {
#if defined(KP_LO1)
          simd<uint32_t, 128> a = load_2d<uint32_t, 8, 16, 1, true, false>((const uint32_t*)kb, DH - 1, PAGE - 1, DH - 1, 0, tile * 16);
          simd<uint32_t, 128> b = load_2d<uint32_t, 8, 16, 1, true, false>((const uint32_t*)kb, DH - 1, PAGE - 1, DH - 1, 8, tile * 16);
#elif defined(KP_LO4)
          simd<uint32_t, 16> off = simd<uint32_t, 16>(0, 64);
          simd<uint32_t, 128> a = gather<uint32_t, 128, 8>((const uint32_t*)kb + tile * 256, off);
          simd<uint32_t, 128> b = gather<uint32_t, 128, 8>((const uint32_t*)kb + tile * 256 + 8, off);
#elif defined(KP_LO5)
          simd<uint32_t, 16> off = simd<uint32_t, 16>(0, 64);
          simd<uint32_t, 128> a, b;
          for (int i = 0; i < 2; i++) { a.template select<64,1>(i*64) = gather<uint32_t, 64, 4>((const uint32_t*)kb + tile * 256 + 4*i, off); b.template select<64,1>(i*64) = gather<uint32_t, 64, 4>((const uint32_t*)kb + tile * 256 + 8 + 4*i, off); }
#elif defined(KP_LO6)
          simd<uint32_t, 16> off = simd<uint32_t, 16>(0, 64);
          simd<uint32_t, 256> a16 = gather<uint32_t, 256, 16>((const uint32_t*)kb + tile * 256, off);
          simd<uint32_t, 128> a = a16.template select<128,1>(0), b = a16.template select<128,1>(128);
#elif defined(KP_LO2)
          simd<uint32_t, 128> a = load_2d<uint32_t, 8, 16, 1, false, false>((const uint32_t*)kb, DH - 1, PAGE - 1, DH - 1, 0, tile * 16);
          simd<uint32_t, 128> b = load_2d<uint32_t, 8, 16, 1, false, false>((const uint32_t*)kb, DH - 1, PAGE - 1, DH - 1, 8, tile * 16);
#else
          simd<uint32_t, 128> a, b;
          for (int i = 0; i < 2; i++) { a.template select<64,1>(i*64) = block_load<uint32_t, 64>((const uint32_t*)kb + tile * 256 + i * 64); b.template select<64,1>(i*64) = block_load<uint32_t, 64>((const uint32_t*)kb + tile * 256 + 128 + i * 64); }
#endif
          simd<uint32_t,128> x = a ^ b; simd<float,128> xf = x.template bit_cast_view<float>(); total += xf.template select<RC*16,1>(0);
        }
        continue;
#endif
        simd<uint32_t, 128> w0 = load_2d<uint32_t, 8, 16, 1, true, false>((const uint32_t*)kb, DH - 1, PAGE - 1, DH - 1, 0, tile * 16);
        simd<uint32_t, 128> w1 = load_2d<uint32_t, 8, 16, 1, true, false>((const uint32_t*)kb, DH - 1, PAGE - 1, DH - 1, 8, tile * 16);
        simd<uint32_t, 16> sd = block_load<uint32_t, 16>(reinterpret_cast<const uint32_t*>(sb) + tile * 16);
        simd<float, RC * 16> S = 0.f;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
          simd<float, RC * 16> acc = 0.f;
          #pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int kk = 2 * g + hh;
            simd<uint32_t, 128> B;
            #pragma unroll
            for (int h = 0; h < 2; h++) {
              const int j = 2 * kk + h;  // dword index 0..15
              simd<uint32_t, 16> wj = (j < 8) ? w0.template select<16, 1>(j * 16) : w1.template select<16, 1>((j - 8) * 16);
              simd<uint16_t, 32> v = wj.template bit_cast_view<uint16_t>();
              simd<uint16_t, 16> lo = v.template select<16, 2>(0), hi = v.template select<16, 2>(1);
              simd<uint16_t, 16> i0 = lo & 0xFF, i1 = lo >> 8, i2 = hi & 0xFF, i3 = hi >> 8;
#ifdef KP_NODEC
              B.template select<16, 1>((4 * h + 0) * 16) = wj; B.template select<16, 1>((4 * h + 1) * 16) = wj ^ 1; B.template select<16, 1>((4 * h + 2) * 16) = wj ^ 2; B.template select<16, 1>((4 * h + 3) * 16) = wj ^ 3;
#else
              B.template select<16, 1>((4 * h + 0) * 16) = lut.iselect(i0);
              B.template select<16, 1>((4 * h + 1) * 16) = lut.iselect(i1);
              B.template select<16, 1>((4 * h + 2) * 16) = lut.iselect(i2);
              B.template select<16, 1>((4 * h + 3) * 16) = lut.iselect(i3);
#endif
            }
#ifdef KP_NODPAS
            { simd<float,128> bf = B.template bit_cast_view<float>(); acc += bf.template select<RC*16,1>(0); }
#else
            acc = xmx::dpas<8, RC, float, float, uint32_t, uint32_t, xmx::dpas_argument_type::bf16, xmx::dpas_argument_type::bf16>(acc, B, A[kk]);
#endif
          }
          simd<uint32_t, 16> sbits = ((sd >> (8 * g)) & 0xFF) << 23;
          simd<float, 16> sc = sbits.template bit_cast_view<float>() * scl;
          #pragma unroll
          for (int r = 0; r < RC; r++) S.template select<16, 1>(r * 16) += acc.template select<16, 1>(r * 16) * sc;
        }
        if (pp == 0 && tile == 1 && t < 4) block_store<float, RC * 16>(out + nthreads * RC * 16 + t * RC * 16, S);  // debug: tile 1 of first page
        total += S;
      }
    }
    block_store<float, RC * 16>(out + t * RC * 16, total);
  }
};

static uint16_t f2bf(float f) { uint32_t b; memcpy(&b, &f, 4); return (uint16_t)((b + 0x7FFF + ((b >> 16) & 1)) >> 16); }
static float bf2f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f; }
static float fp4v(int c) { static const float m[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6}; return (c & 8 ? -1.f : 1.f) * m[c & 7]; }

int main(int argc, char** argv) {
  const int threads = argc > 1 ? atoi(argv[1]) : 768, pages_per = argc > 2 ? atoi(argv[2]) : 16;
  constexpr int RC = 4;
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  std::mt19937 rng(1);
  const size_t npages = (size_t)threads * pages_per;
  std::vector<u8> kc(npages * PAGE * DH), ks(npages * PAGE * 4);
  for (auto& x : kc) x = rng(); for (auto& x : ks) x = 124 + rng() % 4;
  std::vector<uint16_t> qh(RC * D); for (auto& x : qh) x = f2bf((float)((int)(rng() % 17) - 8) / 8.f);
  std::vector<uint32_t> lut(256);
  static const uint16_t mag[8] = {0, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0};
  for (int b = 0; b < 256; b++) { int lo = b & 15, hi = b >> 4; auto bits = [&](int c) { return (uint32_t)(mag[c & 7] | ((c & 8) << 12)); }; lut[b] = bits(lo) | (bits(hi) << 16); }
  u8 *dkc = sycl::malloc_device<u8>(kc.size(), q), *dks = sycl::malloc_device<u8>(ks.size(), q);
  uint16_t* dq = sycl::malloc_device<uint16_t>(qh.size(), q); uint32_t* dl = sycl::malloc_device<uint32_t>(256, q);
  float* dout = sycl::malloc_device<float>(threads * RC * 16 + 1024, q);
  q.memcpy(dkc, kc.data(), kc.size()); q.memcpy(dks, ks.data(), ks.size()); q.memcpy(dq, qh.data(), qh.size() * 2); q.memcpy(dl, lut.data(), 1024).wait();
  const float scl = 1.0f;
  auto run = [&]() { q.parallel_for(sycl::nd_range<1>(threads, 1), KProbe<RC>{dq, dkc, dks, dl, dout, pages_per, threads, scl}); };
  for (int i = 0; i < 3; i++) run(); q.wait();
  double best = 1e30;
  for (int rep = 0; rep < 5; rep++) { auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 10; i++) run(); q.wait(); best = std::fmin(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10); }
  double toks = (double)npages * PAGE, bytes = toks * (DH + 4);
  printf("threads=%d pages/thread=%d  %.3f ms   K tokens=%.0f  %.1f GB/s FP4-K read  (%.1f ns/token-per-thread-equiv)\n", threads, pages_per, best, toks, bytes / best / 1e6, best * 1e6 / toks * threads);
  std::vector<float> h(threads * RC * 16 + 1024);
  q.memcpy(h.data(), dout, h.size() * 4).wait();
  // check tile 1 of page 0 for threads 0..3 (page = t*pages_per)
  double maxerr = 0;
  for (int t = 0; t < 4; t++) {
    size_t page = (size_t)t * pages_per;
    for (int n = 0; n < 16; n++) for (int r = 0; r < RC; r++) {
      size_t tok = page * PAGE + 16 + n; double ref = 0;
      for (int d = 0; d < D; d++) {
        u8 byte = kc[tok * DH + d / 2]; int code = (d & 1) ? byte >> 4 : byte & 15;
        float scale = std::ldexp(1.f, (int)ks[tok * 4 + d / 32] - 127);
        ref += bf2f(qh[r * D + d]) * fp4v(code) * scale;
      }
      double got = h[threads * RC * 16 + t * RC * 16 + r * 16 + n];
      maxerr = std::fmax(maxerr, std::fabs(got - ref) / (1.0 + std::fabs(ref))); if (t == 0 && r < 2 && n < 6) printf("t0 r%d n%d got %g ref %g\n", r, n, got, ref);
    }
  }
  printf("max err %.3e\n", maxerr);
  return 0;
}
