// B300 dev bench for variant 4 (sm_100a/103a tcgen05).  Includes uq_decode.cu directly so variant 4 is dispatched here
// without touching the main dispatcher.  -DUQ_SM90_ONLY strips the legacy kernels (fast compile; no legacy baseline);
// correctness is checked against the CPU reference vectors (--vectors DIR) and, in full builds, against legacy variant 1.
// build: nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -I cuda -o bench_sm100 bench/cuda2/bench_sm100.cu
#include "../../cuda/uq_decode.cu"
#include "../../cuda/uq_decode_sm100.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <iterator>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(2);} } while (0)
static cudaError_t L(const UQParams& p, int nw, int st) { return p.variant == 4 ? dispatch_sm100(p, nw, st, 0) : uq_decode_launch(p, nw, st, 0); }
__global__ void fill_rand(uint8_t* p, size_t n, uint32_t seed, int lo, int hi) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = hi < 0 ? (uint8_t)x : (uint8_t)(lo + (x % (uint32_t)(hi - lo + 1)));
  }
}
__device__ static float bf2f_d(uint16_t b) { uint32_t u = (uint32_t)b << 16; return __uint_as_float(u); }

__device__ float e2m1(int c) { const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f}; return (c & 8 ? -1.f : 1.f) * t[c & 7]; }
__device__ float deq(const uint8_t* codes, const uint8_t* sc, size_t row, int d) {
  uint8_t byte = codes[row * 64 + d / 2]; int c = (d & 1) ? (byte >> 4) : (byte & 15);
  return e2m1(c) * exp2f((int)sc[row * 4 + d / 32] - 127);
}
__global__ void ref_scores(UQParams p, float* scr, int ctx_max) {
  int row = blockIdx.x; int h = row % p.HQ, j = (row / p.HQ) % p.K, b = row / (p.HQ * p.K); int hk = h / (p.HQ / p.HKV);
  int len = p.seq_lens[b] - p.K + j + 1; const uint16_t* q = (const uint16_t*)p.q + (size_t)row * 128;
  for (int key = threadIdx.x; key < len; key += blockDim.x) {
    size_t page = p.block_table[b * p.max_pages + key / 64]; size_t r = (page * p.HKV + hk) * 64 + key % 64; float acc = 0;
    for (int d = 0; d < 128; d++) acc += bf2f_d(q[d]) * deq(p.kc, p.ks, r, d);
    scr[(size_t)row * ctx_max + key] = acc * p.sm_scale;
  }
}
__global__ void ref_out(UQParams p, const float* scr, int ctx_max, float* out) {
  int row = blockIdx.x; int h = row % p.HQ, j = (row / p.HQ) % p.K, b = row / (p.HQ * p.K); int hk = h / (p.HQ / p.HKV);
  int len = p.seq_lens[b] - p.K + j + 1; int d = threadIdx.x; const float* sc = scr + (size_t)row * ctx_max;
  float m = -1e30f; for (int k = 0; k < len; k++) m = fmaxf(m, sc[k]);
  float l = 0, o = 0;
  for (int k = 0; k < len; k++) { float e = expf(sc[k] - m); l += e; size_t page = p.block_table[b * p.max_pages + k / 64]; o += e * deq(p.vc, p.vs, (page * p.HKV + hk) * 64 + k % 64, d); }
  out[(size_t)row * 128 + d] = o / l;
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static std::vector<char> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary); if (!f) { printf("missing %s\n", p.c_str()); exit(2); }
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void* up(const std::vector<char>& h) { void* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); return d; }

static int run_vectors(const std::string& dir) {
  std::ifstream mf(dir + "/manifest.txt"); std::string line; int fails = 0, runs = 0; double worstx = 0;
  while (std::getline(mf, line)) {
    std::istringstream is(line); std::string tag, name; int K, B, HQ, HKV, D, PAGE, pages, mp; is >> tag >> name >> K >> B >> HQ >> HKV >> D >> PAGE >> pages >> mp;
    if (D != 128 || PAGE != 64) continue;
    auto kc = rd(dir + "/" + name + "_kc.bin"), ks = rd(dir + "/" + name + "_ks.bin"), vc = rd(dir + "/" + name + "_vc.bin"), vs = rd(dir + "/" + name + "_vs.bin"),
         bt = rd(dir + "/" + name + "_bt.bin"), sl = rd(dir + "/" + name + "_sl.bin"), q = rd(dir + "/" + name + "_K" + std::to_string(K) + "_q.bin"),
         ref = rd(dir + "/" + name + "_K" + std::to_string(K) + "_ref.bin");
    UQParams p{}; p.q = up(q); p.kc = (uint8_t*)up(kc); p.ks = (uint8_t*)up(ks); p.vc = (uint8_t*)up(vc); p.vs = (uint8_t*)up(vs); p.block_table = (int32_t*)up(bt); p.seq_lens = (int32_t*)up(sl);
    p.B = B; p.K = K; p.HQ = HQ; p.HKV = HKV; p.D = D; p.PAGE = PAGE; p.max_pages = mp; p.sm_scale = 1.f / sqrtf((float)D); p.variant = 4;
    size_t on = (size_t)B * K * HQ * D; CK(cudaMalloc(&p.out, on * 2)); const float* rf = (const float*)ref.data();
    for (int st : {2, 3, 4}) for (int ns : {1, 2, 3, 8}) {
      p.nsplit = ns; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
      CK(cudaMemset(p.out, 0xFF, on * 2)); cudaError_t e = L(p, 0, st);
      if (e == cudaErrorInvalidValue) { cudaGetLastError(); cudaFree(mo); cudaFree(ml); continue; }
      CK(e); CK(cudaDeviceSynchronize());
      std::vector<uint16_t> h(on); CK(cudaMemcpy(h.data(), p.out, on * 2, cudaMemcpyDeviceToHost));
      double mx = 0, mxx = 0; for (size_t i = 0; i < on; i++) { double d = fabs((double)bf2f(h[i]) - rf[i]); if (std::isnan(d)) d = 1e9; double ex = d - fabs((double)rf[i]) / 64; mx = std::max(mx, d); mxx = std::max(mxx, ex); }
      runs++; worstx = std::max(worstx, mxx);
      if (mxx > 1e-2) { fails++; printf("FAIL %s K=%d st=%d ns=%d maxerr=%.3e excess=%.3e\n", name.c_str(), K, st, ns, mx, mxx); }
      cudaFree(mo); cudaFree(ml);
    }
    printf("case %-10s K=%d done\n", name.c_str(), K);
    cudaFree((void*)p.q); cudaFree((void*)p.kc); cudaFree((void*)p.ks); cudaFree((void*)p.vc); cudaFree((void*)p.vs); cudaFree((void*)p.block_table); cudaFree((void*)p.seq_lens); cudaFree(p.out);
  }
  printf("RESULT variant4 vectors runs=%d fails=%d worst_err_beyond_bf16_rounding=%.3e\n", runs, fails, worstx);
  return fails ? 1 : 0;
}

int main(int argc, char** argv) {
  int B = 8, ctx = 32768, HQ = 64, HKV = 8, D = 128, PAGE = 64; std::vector<int> Ks = {1, 2, 4, 8}; int iters = 50; bool legacy = true, ragged = true, refmode = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--vectors") return run_vectors(argv[++i]);
    if (a == "--batch") B = atoi(argv[++i]); else if (a == "--ctx") ctx = atoi(argv[++i]); else if (a == "--iters") iters = atoi(argv[++i]);
    else if (a == "--ref") refmode = true; else if (a == "--nolegacy") legacy = false; else if (a == "--noragged") ragged = false;
    else if (a == "--ks") { Ks.clear(); while (i + 1 < argc && argv[i + 1][0] != '-') Ks.push_back(atoi(argv[++i])); }
  }
#ifdef UQ_SM90_ONLY
  legacy = false;
#endif
  int mp = (ctx + PAGE - 1) / PAGE, npg = B * mp;
  size_t kb = (size_t)npg * HKV * PAGE * (D / 2), sb = (size_t)npg * HKV * PAGE * (D / 32);
  uint8_t *kc, *vc, *ks, *vs; CK(cudaMalloc(&kc, kb)); CK(cudaMalloc(&vc, kb)); CK(cudaMalloc(&ks, sb)); CK(cudaMalloc(&vs, sb));
  fill_rand<<<1024, 256>>>(kc, kb, 1, 0, -1); fill_rand<<<1024, 256>>>(vc, kb, 2, 0, -1);
  fill_rand<<<1024, 256>>>(ks, sb, 3, 124, 130); fill_rand<<<1024, 256>>>(vs, sb, 4, 124, 130);
  std::vector<int32_t> bt(B * mp), sl(B, ctx); std::mt19937 rng(0); std::vector<int32_t> perm(npg); for (int i = 0; i < npg; i++) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng); for (int i = 0; i < B * mp; i++) bt[i] = perm[i];
  if (ragged) for (int b = 0; b < B; b++) sl[b] = ctx - (b % 3) * 37;
  int32_t *dbt, *dsl; CK(cudaMalloc(&dbt, bt.size() * 4)); CK(cudaMalloc(&dsl, B * 4));
  CK(cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsl, sl.data(), B * 4, cudaMemcpyHostToDevice));
  double kvbytes = 0; for (int b = 0; b < B; b++) kvbytes += (double)sl[b] * HKV * (D / 2 + D / 32) * 2;
  size_t pn = (size_t)1 << 30; uint8_t *x, *y; CK(cudaMalloc(&x, pn)); CK(cudaMalloc(&y, pn));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 3; i++) cudaMemcpyAsync(y, x, pn, cudaMemcpyDeviceToDevice);
  cudaEventRecord(e0); for (int i = 0; i < 20; i++) cudaMemcpyAsync(y, x, pn, cudaMemcpyDeviceToDevice); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float tm; cudaEventElapsedTime(&tm, e0, e1); double peak = 2.0 * pn / (tm / 20 * 1e-3) / 1e9; printf("HBM copy peak %.0f GB/s\n", peak);
  cudaFree(x); cudaFree(y);
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0); printf("device %s SMs %d\n", pr.name, pr.multiProcessorCount);
  for (int K : Ks) {
    UQParams p{}; p.kc = kc; p.ks = ks; p.vc = vc; p.vs = vs; p.block_table = dbt; p.seq_lens = dsl;
    p.B = B; p.K = K; p.HQ = HQ; p.HKV = HKV; p.D = D; p.PAGE = PAGE; p.max_pages = mp; p.sm_scale = 1.f / sqrtf((float)D);
    size_t qn = (size_t)B * K * HQ * D; void* q; CK(cudaMalloc(&q, qn * 2)); fill_rand<<<256, 256>>>((uint8_t*)q, qn * 2, 5, 0x30, 0x3f);
    p.q = q; CK(cudaMalloc(&p.out, qn * 2)); void* ref; CK(cudaMalloc(&ref, qn * 2));
    if (refmode) {
      float *scr, *rout; CK(cudaMalloc(&scr, (size_t)B * K * HQ * ctx * 4)); CK(cudaMalloc(&rout, qn * 4));
      p.nsplit = 1; ref_scores<<<B * K * HQ, 256>>>(p, scr, ctx); ref_out<<<B * K * HQ, 128>>>(p, scr, ctx, rout); CK(cudaDeviceSynchronize());
      std::vector<float> rf(qn); CK(cudaMemcpy(rf.data(), rout, qn * 4, cudaMemcpyDeviceToHost)); double rmax = 0; for (float v : rf) rmax = std::max(rmax, (double)fabs(v));
      for (int var : {1, 4}) for (int ns : {2, 9, 16}) {
        p.variant = var; p.nsplit = ns; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
        CK(cudaMemset(p.out, 0xFF, qn * 2)); cudaError_t e = var == 4 ? L(p, 0, 3) : (uq_block_m(K, HQ, HKV) >= 128 ? L(p, 8, 2) : L(p, 4, 2));
        if (e != cudaSuccess) { cudaGetLastError(); printf("  K=%d var%d ns=%d launch err\n", K, var, ns); cudaFree(mo); cudaFree(ml); continue; }
        CK(cudaDeviceSynchronize()); std::vector<uint16_t> h(qn); CK(cudaMemcpy(h.data(), p.out, qn * 2, cudaMemcpyDeviceToHost));
        double mxabs = 0, mxex = 0; for (size_t i = 0; i < qn; i++) { double d = fabs((double)bf2f(h[i]) - rf[i]); if (std::isnan(d)) d = 1e9; mxabs = std::max(mxabs, d); mxex = std::max(mxex, d - fabs((double)rf[i]) / 64); }
        printf("  REF K=%d var%d ns=%d: max abs err %.3e, excess over 2^-6 rel %.3e (max |ref| %.3f)\n", K, var, ns, mxabs, mxex, rmax); cudaFree(mo); cudaFree(ml);
      }
      cudaFree(scr); cudaFree(rout); cudaFree(q); cudaFree(p.out); cudaFree(ref); continue;
    }
    auto timeit = [&](int var, int mb, int nw, int st, int ns, double& ms) -> bool {
      p.variant = var; p.mb = mb; p.nsplit = ns; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
      bool ok = L(p, nw, st) == cudaSuccess;
      if (ok) { for (int i = 0; i < 3; i++) L(p, nw, st);
        cudaEventRecord(e0); for (int i = 0; i < iters; i++) L(p, nw, st); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float t; cudaEventElapsedTime(&t, e0, e1); ms = t / iters; if (cudaGetLastError() != cudaSuccess) ok = false; } else cudaGetLastError();
      cudaFree(mo); cudaFree(ml); return ok;
    };
    double legms = 1e9;
    if (legacy) {
      int bm = uq_block_m(K, HQ, HKV);
      for (int var : {1, 2}) for (int mb : {1, 2}) for (int nw : {2, 4, 8}) for (int st : {2, 3, 4}) for (int ns : {4, 8, 9, 12, 16, 24}) {
        if (var == 1 && mb != 1) continue; if (var == 1 && bm == 128 && nw != 8) continue;
        double ms; if (timeit(var, var == 2 ? mb : 0, nw, st, ns, ms) && ms < legms) { legms = ms; p.variant = var; }
      }
      p.variant = 1; p.nsplit = 16; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
      if (L(p, 8, 2) != cudaSuccess) L(p, 4, 2); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref, p.out, qn * 2, cudaMemcpyDeviceToDevice)); cudaFree(mo); cudaFree(ml);
      printf("K=%d legacy best %.4f ms\n", K, legms);
    }
    double best = 1e9; int bst = 0, bns = 0, nmis = 0;
    for (int st : {2, 3, 4}) for (int ns : {2, 3, 4, 5, 6, 8, 9, 12, 16, 24}) {
      double ms; if (!timeit(4, 0, 0, st, ns, ms)) continue;
      if (legacy) {
        p.variant = 4; p.nsplit = ns; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
        CK(cudaMemset(p.out, 0xFF, qn * 2)); L(p, 0, st); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> h(qn), r(qn); CK(cudaMemcpy(h.data(), p.out, qn * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), ref, qn * 2, cudaMemcpyDeviceToHost));
        double mx = 0; for (size_t i = 0; i < qn; i++) { double d = fabs((double)bf2f(h[i]) - bf2f(r[i])); if (std::isnan(d)) d = 1e9; mx = std::max(mx, d - fabs((double)bf2f(r[i])) / 32); }   // 2^-5 relative: random peaky softmax data, legacy and variant 4 each sit ~1-2 bf16 ulp from the fp32 reference (see --ref)
        cudaFree(mo); cudaFree(ml);
        if (mx > 3e-2) { if (nmis++ < 4) printf("  MISMATCH var4 st=%d ns=%d maxdiff=%.3e\n", st, ns, mx); continue; }
      }
      if (ms < best) { best = ms; bst = st; bns = ns; }
    }
    printf("K=%d sm100 best %.4f ms (st=%d ns=%d) %.0f GB/s %.1f%% peak\n", K, best, bst, bns, kvbytes / best / 1e6, 100 * kvbytes / best / 1e6 / peak);
    if (legacy && best < 1e8) printf("K=%d speedup sm100/legacy %.2fx\n", K, legms / best);
    fflush(stdout); cudaFree(q); cudaFree(p.out); cudaFree(ref);
  }
}
