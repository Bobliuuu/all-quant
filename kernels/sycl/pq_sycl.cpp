// SYCL port of the pq-repro fused decode attention (kernels/cuda/pq_attn.cuh).
// Same algorithm, same cache layout, same test vectors. Plain SYCL 2020 (no
// sub-group or joint_matrix extensions), so it runs on Intel GPUs, on NVIDIA/AMD
// through the DPC++ CUDA/HIP backends, and on CPU devices.
//   build: clang++ -fsycl -O3 -std=c++17 [-fsycl-targets=...] pq_sycl.cpp -o pq_sycl
//   pq_sycl check <vectors_dir>
//   pq_sycl bench <tables_dir> B T Hq Hkv bK bV [nsplit] [iters]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#define PQ_DEV inline
#include "../common/pq_ball_decode.h"

constexpr int D = 128, TT = 32, NT = 128, KSTR = D + 1;
enum { FMT_PQ = 0, FMT_BF16 = 1, FMT_FP8 = 2 };

struct PQArgs {
  const float* q;
  const uint8_t* kc; const uint16_t* ks; const uint32_t* ktab; int bK; int kL; long k_stream;
  const uint8_t* vc; const uint16_t* vs; const uint32_t* vtab; int bV; int vL; long v_stream;
  const uint16_t* kb; const uint16_t* vb;
  const uint8_t* k8; const uint8_t* v8; const float* k8s; const float* v8s;
  const float* signs;
  float* part_o; float* part_ml; float* out;
  int B, Hq, Hkv, T, nsplit, tokens_per_split, tab_smem;
};

static inline void bar(sycl::nd_item<1>& it) { sycl::group_barrier(it.get_group()); }

static inline void fwht128(float* x, sycl::nd_item<1>& it) {
  const int t = it.get_local_id(0);
  for (int h = 1; h < D; h <<= 1) {
    bar(it);
    float a = 0.f, b = 0.f;
    const bool lo = ((t & h) == 0);
    if (lo) { a = x[t]; b = x[t + h]; }
    bar(it);
    if (lo) { x[t] = a + b; x[t + h] = a - b; }
  }
  bar(it);
  x[t] *= 0.08838834764831845f;
  bar(it);
}

static inline void pq_tile(const uint8_t* stream, const uint16_t* sc, int t0, int n, int b, const PQTab& tab,
                           uint32_t* wbuf, float* tile, sycl::nd_item<1>& it) {
  const int tid = it.get_local_id(0);
  const long byte0 = (long)t0 * 2 * b;
  const int nw = (n * 2 * b + 3) / 4 + 2;
  const uint32_t* src = (const uint32_t*)(stream + byte0);
  for (int i = tid; i < nw; i += NT) wbuf[i] = src[i];
  bar(it);
  for (int gi = tid; gi < n * 16; gi += NT) {
    const int t = gi >> 4, g = gi & 15;
    const uint32_t idx = pq_get_bits(wbuf, (uint32_t)gi * b, b);
    int y[8];
    pq_ball_decode3(idx, tab, y);
    const float s = 0.5f * pq_half_to_float(sc[t0 + t]);
    float* dst = tile + t * KSTR + g * 8;
    for (int j = 0; j < 8; ++j) dst[j] = s * (float)y[j];
  }
  bar(it);
}

template <int FMT>
static inline void load_tile(const PQArgs& a, bool isK, int bh, int t0, int n, uint32_t* wbuf, const PQTab& tab,
                             float* tile, sycl::nd_item<1>& it) {
  const int tid = it.get_local_id(0);
  if constexpr (FMT == FMT_PQ) {
    if (isK) pq_tile(a.kc + (long)bh * a.k_stream, a.ks + (long)bh * a.T, t0, n, a.bK, tab, wbuf, tile, it);
    else pq_tile(a.vc + (long)bh * a.v_stream, a.vs + (long)bh * a.T, t0, n, a.bV, tab, wbuf, tile, it);
  } else if constexpr (FMT == FMT_BF16) {
    const uint32_t* p2 = (const uint32_t*)((isK ? a.kb : a.vb) + ((long)bh * a.T + t0) * D);
    for (int i = tid; i < n * D / 2; i += NT) {
      const uint32_t w = p2[i];
      const int e = 2 * i, t = e / D, d = e % D;
      tile[t * KSTR + d] = pq_bf16_to_float((uint16_t)(w & 0xFFFF));
      tile[t * KSTR + d + 1] = pq_bf16_to_float((uint16_t)(w >> 16));
    }
    bar(it);
  } else {
    const uint32_t* p4 = (const uint32_t*)((isK ? a.k8 : a.v8) + ((long)bh * a.T + t0) * D);
    const float* s = (isK ? a.k8s : a.v8s) + (long)bh * a.T + t0;
    for (int i = tid; i < n * D / 4; i += NT) {
      const uint32_t w = p4[i];
      const int e = 4 * i, t = e / D, d = e % D;
      const float st = s[t];
      for (int j = 0; j < 4; ++j) tile[t * KSTR + d + j] = st * pq_e4m3_to_float((uint8_t)(w >> (8 * j)));
    }
    bar(it);
  }
}

static size_t smem_floats(int G, int tab_words) {
  return G * D + TT * KSTR + G * TT + 3 * G + (TT * 16 + 4) + tab_words;  // u32 regions counted as 4-byte slots
}
static std::vector<uint32_t> build_soa(const std::vector<uint32_t>& rows, int b) {
  const int L = (int)rows.size() / 8;
  std::vector<uint32_t> o(pq_tab_words(L), 0);
  for (int i = 0; i < L; ++i) pq_tab_fill_row(o.data(), rows.data(), L, i);
  for (int j = 0; j < (1 << PQ_RADIX_BITS); ++j) pq_tab_fill_radix(o.data(), rows.data(), L, b, j);
  return o;
}

template <int FMT, int G> struct SplitK;
template <int FMT> struct ReduceK;

template <int FMT, int G>
void launch_split(sycl::queue& Q, PQArgs a, size_t slm_floats) {
  Q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> lm(sycl::range<1>(slm_floats), h);
    const size_t ngroups = (size_t)a.nsplit * a.Hkv * a.B;
    h.parallel_for<SplitK<FMT, G>>(sycl::nd_range<1>(ngroups * NT, NT), [=](sycl::nd_item<1> it) {
      float* smem = lm.get_multi_ptr<sycl::access::decorated::no>().get();
      float* qs = smem;
      float* tile = qs + G * D;
      float* sc = tile + TT * KSTR;
      float* mrow = sc + G * TT;
      float* lrow = mrow + G;
      float* alpha = lrow + G;
      uint32_t* wbuf = (uint32_t*)(alpha + G);
      uint32_t* ktab_s = wbuf + (TT * 16 + 4);
      uint32_t* vtab_s = ktab_s + pq_tab_words(a.kL);
      const int gid = it.get_group(0);
      const int split = gid % a.nsplit, hh = (gid / a.nsplit) % a.Hkv, b = gid / (a.nsplit * a.Hkv);
      const int tid = it.get_local_id(0);
      const int bh = b * a.Hkv + hh;
      const int tbeg = split * a.tokens_per_split;
      const int tend = sycl::min(a.T, tbeg + a.tokens_per_split);
      const uint32_t* kt = a.ktab;
      const uint32_t* vt = a.vtab;
      if (FMT == FMT_PQ && a.tab_smem) {
        const int kw = pq_tab_words(a.kL), vw = pq_tab_words(a.vL);
        for (int i = tid; i < kw; i += NT) ktab_s[i] = a.ktab[i];
        for (int i = tid; i < vw; i += NT) vtab_s[i] = a.vtab[i];
        kt = ktab_s;
        vt = vtab_s;
      }
      const PQTab ktab = pq_tab_view(kt, a.kL, a.bK);
      const PQTab vtab = pq_tab_view(vt, a.vL, a.bV);
      for (int g = 0; g < G; ++g) {
        const float* qp = a.q + ((long)b * a.Hq + hh * G + g) * D;
        qs[g * D + tid] = qp[tid] * (FMT == FMT_PQ ? a.signs[tid] : 1.f);
        if (FMT == FMT_PQ) fwht128(qs + g * D, it);
        bar(it);
        qs[g * D + tid] *= 0.08838834764831845f;
      }
      if (tid < G) { mrow[tid] = -1e30f; lrow[tid] = 0.f; }
      float acc[G];
      for (int g = 0; g < G; ++g) acc[g] = 0.f;
      bar(it);
      for (int t0 = tbeg; t0 < tend; t0 += TT) {
        const int n = sycl::min(TT, tend - t0);
        load_tile<FMT>(a, true, bh, t0, n, wbuf, ktab, tile, it);
        for (int i = tid; i < G * TT; i += NT) {
          const int g = i / TT, t = i % TT;
          float s = -1e30f;
          if (t < n) {
            const float* kr = tile + t * KSTR;
            const float* qr = qs + g * D;
            float a2 = 0.f;
            for (int d = 0; d < D; ++d) a2 += qr[d] * kr[d];
            s = a2;
          }
          sc[i] = s;
        }
        bar(it);
        constexpr int NJ = (G * TT + NT - 1) / NT;
        float pv[NJ];
        for (int j = 0; j < NJ; ++j) {
          const int ii = tid + j * NT;
          if (ii < G * TT) {
            const int g = ii / TT;
            float mx = mrow[g];
            for (int t = 0; t < n; ++t) mx = sycl::fmax(mx, sc[g * TT + t]);
            pv[j] = ((ii % TT) < n) ? sycl::exp(sc[ii] - mx) : 0.f;
            if ((ii % TT) == 0) alpha[g] = mx;
          }
        }
        bar(it);
        for (int j = 0; j < NJ; ++j) {
          const int ii = tid + j * NT;
          if (ii < G * TT) sc[ii] = pv[j];
        }
        bar(it);
        if (tid < G) {
          float l = 0.f;
          for (int t = 0; t < n; ++t) l += sc[tid * TT + t];
          const float mx = alpha[tid];
          const float al = sycl::exp(mrow[tid] - mx);
          lrow[tid] = lrow[tid] * al + l;
          mrow[tid] = mx;
          alpha[tid] = al;
        }
        bar(it);
        load_tile<FMT>(a, false, bh, t0, n, wbuf, vtab, tile, it);
        for (int g = 0; g < G; ++g) {
          float s = acc[g] * alpha[g];
          const float* pr = sc + g * TT;
          for (int t = 0; t < n; ++t) s += pr[t] * tile[t * KSTR + tid];
          acc[g] = s;
        }
        bar(it);
      }
      for (int g = 0; g < G; ++g) {
        const long pi = ((long)b * a.Hq + hh * G + g) * a.nsplit + split;
        a.part_o[pi * D + tid] = acc[g];
        if (tid == 0) { a.part_ml[pi * 2] = mrow[g]; a.part_ml[pi * 2 + 1] = lrow[g]; }
      }
    });
  });
  Q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> xs(sycl::range<1>(D), h);
    h.parallel_for<ReduceK<FMT * 100 + G>>(sycl::nd_range<1>((size_t)a.B * a.Hq * NT, NT), [=](sycl::nd_item<1> it) {
      float* x = xs.get_multi_ptr<sycl::access::decorated::no>().get();
      const int bh = it.get_group(0), tid = it.get_local_id(0);
      float M = -1e30f;
      for (int s = 0; s < a.nsplit; ++s) M = sycl::fmax(M, a.part_ml[((long)bh * a.nsplit + s) * 2]);
      float num = 0.f, den = 0.f;
      for (int s = 0; s < a.nsplit; ++s) {
        const long pi = (long)bh * a.nsplit + s;
        const float l = a.part_ml[pi * 2 + 1];
        if (l == 0.f) continue;
        const float w = sycl::exp(a.part_ml[pi * 2] - M);
        num += w * a.part_o[pi * D + tid];
        den += w * l;
      }
      x[tid] = num / den;
      bar(it);
      if (FMT == FMT_PQ) { fwht128(x, it); x[tid] *= a.signs[tid]; }
      a.out[(long)bh * D + tid] = x[tid];
    });
  });
}

static size_t g_slm_max = 64 * 1024;

template <int FMT> void launch(sycl::queue& Q, PQArgs a) {
  const int G = a.Hq / a.Hkv;
  int tabL = FMT == FMT_PQ ? pq_tab_words(a.kL) + pq_tab_words(a.vL) : 0;
  size_t fl = smem_floats(G, tabL);
  if (fl * 4 > g_slm_max) { a.tab_smem = 0; fl = smem_floats(G, 0); } else a.tab_smem = (FMT == FMT_PQ);
  switch (G) {
    case 1: launch_split<FMT, 1>(Q, a, fl); break;
    case 4: launch_split<FMT, 4>(Q, a, fl); break;
    case 8: launch_split<FMT, 8>(Q, a, fl); break;
    case 16: launch_split<FMT, 16>(Q, a, fl); break;
    case 32: launch_split<FMT, 32>(Q, a, fl); break;
    default: fprintf(stderr, "unsupported G=%d\n", G); exit(4);
  }
}
static void run(sycl::queue& Q, int fmt, const PQArgs& a) {
  if (fmt == FMT_PQ) launch<FMT_PQ>(Q, a); else if (fmt == FMT_BF16) launch<FMT_BF16>(Q, a); else launch<FMT_FP8>(Q, a);
}
static void set_split(PQArgs& a, int nsplit) {
  int per = (a.T + nsplit - 1) / nsplit;
  per = (per + TT - 1) / TT * TT;
  a.tokens_per_split = per;
  a.nsplit = (a.T + per - 1) / per;
}

template <class T> std::vector<T> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(3); }
  size_t n = f.tellg(); f.seekg(0);
  std::vector<T> v(n / sizeof(T)); f.read((char*)v.data(), n); return v;
}
template <class T> T* up(sycl::queue& Q, const std::vector<T>& v) {
  T* d = sycl::malloc_device<T>(v.size() + 16, Q);
  Q.memcpy(d, v.data(), v.size() * sizeof(T)).wait();
  return d;
}

static std::string devname(sycl::queue& Q) { return Q.get_device().get_info<sycl::info::device::name>(); }

int check(sycl::queue& Q, const std::string& dir) {
  std::ifstream cf(dir + "/cases.txt");
  std::string name;
  int fails = 0, n = 0;
  while (cf >> name) {
    const std::string d = dir + "/" + name + "/";
    std::ifstream mf(d + "meta.txt");
    int B, Hq, Hkv, T, DD, bK, bV, kL, vL; long ksb, vsb;
    mf >> B >> Hq >> Hkv >> T >> DD >> bK >> bV >> ksb >> vsb >> kL >> vL;
    PQArgs a{};
    a.B = B; a.Hq = Hq; a.Hkv = Hkv; a.T = T; a.bK = bK; a.bV = bV; a.kL = kL; a.vL = vL; a.k_stream = ksb; a.v_stream = vsb;
    a.q = up(Q, rd<float>(d + "q.f32"));
    a.kc = up(Q, rd<uint8_t>(d + "k_codes.u8")); a.vc = up(Q, rd<uint8_t>(d + "v_codes.u8"));
    a.ks = up(Q, rd<uint16_t>(d + "k_scale.f16")); a.vs = up(Q, rd<uint16_t>(d + "v_scale.f16"));
    a.ktab = up(Q, build_soa(rd<uint32_t>(d + "k_tab.u32"), bK)); a.vtab = up(Q, build_soa(rd<uint32_t>(d + "v_tab.u32"), bV));
    a.kb = up(Q, rd<uint16_t>(d + "k_bf16.u16")); a.vb = up(Q, rd<uint16_t>(d + "v_bf16.u16"));
    a.k8 = up(Q, rd<uint8_t>(d + "k_fp8.u8")); a.v8 = up(Q, rd<uint8_t>(d + "v_fp8.u8"));
    a.k8s = up(Q, rd<float>(d + "k_fp8s.f32")); a.v8s = up(Q, rd<float>(d + "v_fp8s.f32"));
    a.signs = up(Q, rd<float>(d + "signs.f32"));
    const size_t no = (size_t)B * Hq * D;
    a.out = sycl::malloc_device<float>(no, Q);
    a.part_o = sycl::malloc_device<float>(no * 64, Q);
    a.part_ml = sycl::malloc_device<float>((size_t)B * Hq * 64 * 2, Q);
    const char* fn[3] = {"pq", "bf16", "fp8"};
    const char* rf[3] = {"out_ref.f32", "out_bf16.f32", "out_fp8.f32"};
    for (int fmt = 0; fmt < 3; ++fmt) {
      auto ref = rd<float>(d + rf[fmt]);
      for (int ns : {1, 3, 8, 64}) {
        set_split(a, ns);
        Q.memset(a.out, 0, no * 4).wait();
        run(Q, fmt, a);
        Q.wait_and_throw();
        std::vector<float> o(no);
        Q.memcpy(o.data(), a.out, no * 4).wait();
        double me = 0, rn = 0, dn = 0;
        for (size_t i = 0; i < no; ++i) { me = fmax(me, fabs(o[i] - ref[i])); dn += (o[i] - ref[i]) * (o[i] - ref[i]); rn += ref[i] * ref[i]; }
        const double rel = sqrt(dn / rn);
        const bool ok = rel < 1e-4 && std::isfinite(me);
        fails += !ok; ++n;
        printf("{\"backend\":\"sycl\",\"device\":\"%s\",\"case\":\"%s\",\"fmt\":\"%s\",\"nsplit\":%d,\"max_abs_err\":%.3e,\"rel_l2_err\":%.3e,\"pass\":%s}\n",
               devname(Q).c_str(), name.c_str(), fn[fmt], a.nsplit, me, rel, ok ? "true" : "false");
      }
    }
  }
  printf("SUMMARY sycl %s: %d/%d pass\n", devname(Q).c_str(), n - fails, n);
  return fails ? 1 : 0;
}

int bench(sycl::queue& Q, int argc, char** argv) {
  const std::string tdir = argv[2];
  const int B = atoi(argv[3]), T = atoi(argv[4]), Hq = atoi(argv[5]), Hkv = atoi(argv[6]), bK = atoi(argv[7]), bV = atoi(argv[8]);
  const int nsplit_arg = argc > 9 ? atoi(argv[9]) : 0;
  const int iters = argc > 10 ? atoi(argv[10]) : 30;
  const char* only = std::getenv("PQ_FMT");  // optional: run only "pq", "bf16" or "fp8"
  std::mt19937_64 rng(42);
  PQArgs a{};
  a.B = B; a.Hq = Hq; a.Hkv = Hkv; a.T = T; a.bK = bK; a.bV = bV;
  auto kt = rd<uint32_t>(tdir + "/ball_" + std::to_string(bK) + ".u32");
  auto vt = rd<uint32_t>(tdir + "/ball_" + std::to_string(bV) + ".u32");
  a.kL = kt.size() / 8; a.vL = vt.size() / 8;
  a.ktab = up(Q, build_soa(kt, bK)); a.vtab = up(Q, build_soa(vt, bV));
  a.k_stream = ((long)T * 2 * bK + 8 + 15) / 16 * 16; a.v_stream = ((long)T * 2 * bV + 8 + 15) / 16 * 16;
  const size_t nbh = (size_t)B * Hkv;
  std::vector<uint8_t> kc(nbh * a.k_stream), vc(nbh * a.v_stream);
  for (auto& x : kc) x = rng() & 255; for (auto& x : vc) x = rng() & 255;
  std::vector<uint16_t> sc(nbh * T, 0x2C00);
  std::vector<float> q((size_t)B * Hq * D), sg(D);
  std::normal_distribution<float> nd;
  for (auto& x : q) x = nd(rng);
  for (auto& x : sg) x = (rng() & 1) ? 1.f : -1.f;
  std::vector<uint16_t> bf(nbh * T * D);
  for (auto& x : bf) x = 0x3C00 + (rng() & 0x3FF);
  std::vector<uint8_t> f8(nbh * T * D);
  for (auto& x : f8) { x = rng() & 255; if ((x & 0x7F) == 0x7F) x = 0x30; }
  std::vector<float> f8s(nbh * T, 0.01f);
  a.q = up(Q, q); a.kc = up(Q, kc); a.vc = up(Q, vc); a.ks = up(Q, sc); a.vs = up(Q, sc); a.signs = up(Q, sg);
  a.kb = up(Q, bf); a.vb = up(Q, bf); a.k8 = up(Q, f8); a.v8 = up(Q, f8); a.k8s = up(Q, f8s); a.v8s = up(Q, f8s);
  const size_t no = (size_t)B * Hq * D;
  a.out = sycl::malloc_device<float>(no, Q);
  a.part_o = sycl::malloc_device<float>(no * 1024, Q);
  a.part_ml = sycl::malloc_device<float>((size_t)B * Hq * 1024 * 2, Q);
  const char* fn[3] = {"pq", "bf16", "fp8"};
  for (int fmt = 0; fmt < 3; ++fmt) {
    if (only && strcmp(only, fn[fmt])) continue;
    double kv_bytes;
    if (fmt == FMT_PQ) kv_bytes = (double)nbh * T * (2.0 * bK + 2 + 2.0 * bV + 2);
    else if (fmt == FMT_BF16) kv_bytes = (double)nbh * T * D * 2 * 2;
    else kv_bytes = (double)nbh * T * (D + 4) * 2;
    std::vector<int> splits;
    if (nsplit_arg > 0) splits = {nsplit_arg}; else splits = {8, 16, 32, 64, 128, 256};
    double best = 1e30; int bests = 0;
    for (int ns : splits) {
      set_split(a, ns);
      for (int w = 0; w < 2; ++w) run(Q, fmt, a);
      Q.wait_and_throw();
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; ++i) run(Q, fmt, a);
      Q.wait_and_throw();
      const double us = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count() / iters;
      if (us < best) { best = us; bests = a.nsplit; }
    }
    printf("{\"backend\":\"sycl\",\"device\":\"%s\",\"fmt\":\"%s\",\"B\":%d,\"T\":%d,\"Hq\":%d,\"Hkv\":%d,\"bK\":%d,\"bV\":%d,"
           "\"nsplit\":%d,\"us\":%.2f,\"kv_bytes\":%.0f,\"bytes_per_token_head\":%.1f,\"eff_GBps\":%.1f}\n",
           devname(Q).c_str(), fn[fmt], B, T, Hq, Hkv, fmt == 0 ? bK : 0, fmt == 0 ? bV : 0, bests, best, kv_bytes,
           kv_bytes / ((double)nbh * T), kv_bytes / (best * 1e3));
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: pq_sycl check <dir> | bench <tables> B T Hq Hkv bK bV [nsplit] [iters]\n"); return 1; }
  sycl::queue Q{sycl::default_selector_v, sycl::property::queue::in_order()};
  g_slm_max = Q.get_device().get_info<sycl::info::device::local_mem_size>();
  fprintf(stderr, "device: %s, local mem %zu\n", devname(Q).c_str(), g_slm_max);
  if (!strcmp(argv[1], "check")) return check(Q, argv[2]);
  return bench(Q, argc, argv);
}
