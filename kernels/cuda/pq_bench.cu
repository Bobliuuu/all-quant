// Host driver for the pq-repro decode-attention kernels: correctness on the shared
// test vectors and timing on synthetic caches. Builds with nvcc or hipcc:
//   nvcc  -O3 -std=c++17 -arch=sm_90 pq_bench.cu -o pq_bench
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 pq_bench.cu -o pq_bench
// usage:
//   pq_bench check <vectors_dir>
//   pq_bench bench <tables_dir> B T Hq Hkv bK bV [nsplit] [iters]   (prints one JSON line per format)
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaSuccess hipSuccess
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define PQ_BACKEND "hip"
#else
#include <cuda_runtime.h>
#define PQ_BACKEND "cuda"
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "pq_attn.cuh"

#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "ERR %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

template <class T> std::vector<T> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(3); }
  size_t n = f.tellg(); f.seekg(0);
  std::vector<T> v(n / sizeof(T)); f.read((char*)v.data(), n); return v;
}
template <class T> T* up(const std::vector<T>& v) {
  T* d; CK(cudaMalloc(&d, v.size() * sizeof(T) + 16)); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}

static size_t g_smem_max = 64 * 1024;

template <int FMT, int G> void launch_g(PQArgs a) {
  const int tabL = (FMT == FMT_PQ) ? (pq_tab_words(a.kL) + pq_tab_words(a.vL)) : 0;
  size_t sm = pq_smem_bytes(G, tabL);
  static const bool force_global = getenv("PQ_TAB_GLOBAL") && atoi(getenv("PQ_TAB_GLOBAL"));
  if (sm > g_smem_max || force_global) { a.tab_smem = 0; sm = pq_smem_bytes(G, 0); } else a.tab_smem = (FMT == FMT_PQ);
  CK(cudaFuncSetAttribute((const void*)pq_attn_split<FMT, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  dim3 grid(a.nsplit, a.Hkv, a.B);
  pq_attn_split<FMT, G><<<grid, PQ_NT, sm>>>(a);
  pq_reduce<FMT><<<a.B * a.Hq, PQ_NT>>>(a);
}
template <int FMT> void launch(PQArgs a) {
  const int G = a.Hq / a.Hkv;
  switch (G) {
    case 1: launch_g<FMT, 1>(a); break;
    case 4: launch_g<FMT, 4>(a); break;
    case 8: launch_g<FMT, 8>(a); break;
    case 16: launch_g<FMT, 16>(a); break;
    case 32: launch_g<FMT, 32>(a); break;
    default: fprintf(stderr, "unsupported G=%d\n", G); exit(4);
  }
}
void run(int fmt, PQArgs a) {
  if (fmt == FMT_PQ) launch<FMT_PQ>(a); else if (fmt == FMT_BF16) launch<FMT_BF16>(a); else launch<FMT_FP8>(a);
}
void set_split(PQArgs& a, int nsplit) {
  int per = (a.T + nsplit - 1) / nsplit;
  per = (per + PQ_TT - 1) / PQ_TT * PQ_TT;
  a.tokens_per_split = per;
  a.nsplit = (a.T + per - 1) / per;
}

int check(const std::string& dir) {
  std::ifstream cf(dir + "/cases.txt");
  std::string name;
  int fails = 0, n = 0;
  while (cf >> name) {
    const std::string d = dir + "/" + name + "/";
    std::ifstream mf(d + "meta.txt");
    int B, Hq, Hkv, T, D, bK, bV, kL, vL; long ksb, vsb;
    mf >> B >> Hq >> Hkv >> T >> D >> bK >> bV >> ksb >> vsb >> kL >> vL;
    PQArgs a{};
    a.B = B; a.Hq = Hq; a.Hkv = Hkv; a.T = T; a.bK = bK; a.bV = bV; a.kL = kL; a.vL = vL; a.k_stream = ksb; a.v_stream = vsb;
    a.q = up(rd<float>(d + "q.f32"));
    a.kc = up(rd<uint8_t>(d + "k_codes.u8")); a.vc = up(rd<uint8_t>(d + "v_codes.u8"));
    a.ks = up(rd<uint16_t>(d + "k_scale.f16")); a.vs = up(rd<uint16_t>(d + "v_scale.f16"));
    a.ktab = up(pq_build_soa(rd<uint32_t>(d + "k_tab.u32"), bK)); a.vtab = up(pq_build_soa(rd<uint32_t>(d + "v_tab.u32"), bV));
    a.kb = up(rd<uint16_t>(d + "k_bf16.u16")); a.vb = up(rd<uint16_t>(d + "v_bf16.u16"));
    a.k8 = up(rd<uint8_t>(d + "k_fp8.u8")); a.v8 = up(rd<uint8_t>(d + "v_fp8.u8"));
    a.k8s = up(rd<float>(d + "k_fp8s.f32")); a.v8s = up(rd<float>(d + "v_fp8s.f32"));
    a.signs = up(rd<float>(d + "signs.f32"));
    const size_t no = (size_t)B * Hq * D;
    CK(cudaMalloc(&a.out, no * 4));
    CK(cudaMalloc(&a.part_o, no * 64 * 4)); CK(cudaMalloc(&a.part_ml, (size_t)B * Hq * 64 * 2 * 4));
    const char* fn[3] = {"pq", "bf16", "fp8"};
    const char* rf[3] = {"out_ref.f32", "out_bf16.f32", "out_fp8.f32"};
    for (int fmt = 0; fmt < 3; ++fmt) {
      auto ref = rd<float>(d + rf[fmt]);
      for (int ns : {1, 3, 8, 64}) {
        set_split(a, ns);
        CK(cudaMemset(a.out, 0, no * 4));
        run(fmt, a);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> o(no);
        CK(cudaMemcpy(o.data(), a.out, no * 4, cudaMemcpyDeviceToHost));
        double me = 0, rn = 0, dn = 0;
        for (size_t i = 0; i < no; ++i) { me = fmax(me, fabs(o[i] - ref[i])); dn += (o[i] - ref[i]) * (o[i] - ref[i]); rn += ref[i] * ref[i]; }
        const double rel = sqrt(dn / rn);
        const bool ok = rel < 1e-4 && std::isfinite(me);
        fails += !ok; ++n;
        printf("{\"backend\":\"%s\",\"case\":\"%s\",\"fmt\":\"%s\",\"nsplit\":%d,\"max_abs_err\":%.3e,\"rel_l2_err\":%.3e,\"pass\":%s}\n",
               PQ_BACKEND, name.c_str(), fn[fmt], a.nsplit, me, rel, ok ? "true" : "false");
      }
    }
  }
  printf("SUMMARY %s: %d/%d pass\n", PQ_BACKEND, n - fails, n);
  return fails ? 1 : 0;
}

int bench(int argc, char** argv) {
  const std::string tdir = argv[2];
  const int B = atoi(argv[3]), T = atoi(argv[4]), Hq = atoi(argv[5]), Hkv = atoi(argv[6]), bK = atoi(argv[7]), bV = atoi(argv[8]);
  const int nsplit_arg = argc > 9 ? atoi(argv[9]) : 0;
  const int iters = argc > 10 ? atoi(argv[10]) : 50;
  const int D = PQ_D;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  std::mt19937_64 rng(42);
  PQArgs a{};
  a.B = B; a.Hq = Hq; a.Hkv = Hkv; a.T = T; a.bK = bK; a.bV = bV;
  auto kt = rd<uint32_t>(tdir + "/ball_" + std::to_string(bK) + ".u32");
  auto vt = rd<uint32_t>(tdir + "/ball_" + std::to_string(bV) + ".u32");
  a.kL = kt.size() / 8; a.vL = vt.size() / 8;
  a.ktab = up(pq_build_soa(kt, bK)); a.vtab = up(pq_build_soa(vt, bV));
  a.k_stream = ((long)T * 2 * bK + 8 + 15) / 16 * 16; a.v_stream = ((long)T * 2 * bV + 8 + 15) / 16 * 16;
  const size_t nbh = (size_t)B * Hkv;
  std::vector<uint8_t> kc(nbh * a.k_stream), vc(nbh * a.v_stream);
  for (auto& x : kc) x = rng() & 255; for (auto& x : vc) x = rng() & 255;  // every b-bit index is a valid codeword
  std::vector<uint16_t> sc(nbh * T, 0x2C00);  // fp16 0.0625
  std::vector<float> q((size_t)B * Hq * D), sg(D);
  std::normal_distribution<float> nd;
  for (auto& x : q) x = nd(rng);
  for (auto& x : sg) x = (rng() & 1) ? 1.f : -1.f;
  std::vector<uint16_t> bf(nbh * T * D);
  for (auto& x : bf) x = 0x3C00 + (rng() & 0x3FF);  // finite bf16 values
  std::vector<uint8_t> f8(nbh * T * D);
  for (auto& x : f8) { x = rng() & 255; if ((x & 0x7F) == 0x7F) x = 0x30; }
  std::vector<float> f8s(nbh * T, 0.01f);
  a.q = up(q); a.kc = up(kc); a.vc = up(vc); a.ks = up(sc); a.vs = up(sc); a.signs = up(sg);
  a.kb = up(bf); a.vb = up(bf); a.k8 = up(f8); a.v8 = up(f8); a.k8s = up(f8s); a.v8s = up(f8s);
  const size_t no = (size_t)B * Hq * D;
  CK(cudaMalloc(&a.out, no * 4));
  CK(cudaMalloc(&a.part_o, no * 1024 * 4)); CK(cudaMalloc(&a.part_ml, (size_t)B * Hq * 1024 * 2 * 4));
  const char* fn[3] = {"pq", "bf16", "fp8"};
  for (int fmt = 0; fmt < 3; ++fmt) {
    double kv_bytes;
    if (fmt == FMT_PQ) kv_bytes = (double)nbh * T * (2.0 * bK + 2 + 2.0 * bV + 2);
    else if (fmt == FMT_BF16) kv_bytes = (double)nbh * T * D * 2 * 2;
    else kv_bytes = (double)nbh * T * (D + 4) * 2;
    std::vector<int> splits;
    if (nsplit_arg > 0) splits = {nsplit_arg}; else splits = {8, 16, 32, 64, 128, 256};
    double best = 1e30; int bests = 0;
    for (int ns : splits) {
      set_split(a, ns);
      if (a.nsplit > 1024) continue;
      for (int w = 0; w < 3; ++w) run(fmt, a);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      cudaEventRecord(e0);
      for (int i = 0; i < iters; ++i) run(fmt, a);
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      const double us = 1000.0 * ms / iters;
      if (us < best) { best = us; bests = a.nsplit; }
    }
    printf("{\"backend\":\"%s\",\"gpu\":\"%s\",\"fmt\":\"%s\",\"B\":%d,\"T\":%d,\"Hq\":%d,\"Hkv\":%d,\"bK\":%d,\"bV\":%d,"
           "\"nsplit\":%d,\"us\":%.2f,\"kv_bytes\":%.0f,\"bytes_per_token_head\":%.1f,\"eff_GBps\":%.1f}\n",
           PQ_BACKEND, prop.name, fn[fmt], B, T, Hq, Hkv, fmt == 0 ? bK : 0, fmt == 0 ? bV : 0, bests, best, kv_bytes,
           kv_bytes / ((double)nbh * T), kv_bytes / (best * 1e3));
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: pq_bench check <dir> | bench <tables> B T Hq Hkv bK bV [nsplit] [iters]\n"); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
#ifdef __HIPCC__
  g_smem_max = 64 * 1024;
#else
  g_smem_max = prop.sharedMemPerBlockOptin;
#endif
  if (!strcmp(argv[1], "check")) return check(argv[2]);
  return bench(argc, argv);
}
