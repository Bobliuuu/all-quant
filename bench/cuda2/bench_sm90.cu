// cuda2 dev bench: variant 3 (sm_90a wgmma) vs best of variants 1/2, with output cross-check.  Same shape/params as cuda/bench_standalone.cu.
// build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -DUQ_ENABLE_SM90 -I cuda -o bench_sm90 bench/cuda2/bench_sm90.cu cuda/uq_decode.cu
#include "uq_decode.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(2);} } while (0)
__global__ void fill_rand(uint8_t* p, size_t n, uint32_t seed, int lo, int hi) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = hi < 0 ? (uint8_t)x : (uint8_t)(lo + (x % (uint32_t)(hi - lo + 1)));
  }
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
int main(int argc, char** argv) {
  int B = 8, ctx = 32768, HQ = 64, HKV = 8, D = 128, PAGE = 64; std::vector<int> Ks = {1, 2, 4, 8}; int iters = 50; bool only3 = false, ragged = true, smoke = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--batch") B = atoi(argv[++i]); else if (a == "--ctx") ctx = atoi(argv[++i]); else if (a == "--iters") iters = atoi(argv[++i]);
    else if (a == "--smoke") smoke = true; else if (a == "--only3") only3 = true; else if (a == "--noragged") ragged = false;
    else if (a == "--ks") { Ks.clear(); while (i + 1 < argc && argv[i + 1][0] != '-') Ks.push_back(atoi(argv[++i])); }
  }
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
    p.q = q; CK(cudaMalloc(&p.out, qn * 2));
    void* ref; CK(cudaMalloc(&ref, qn * 2));
    struct R { int var, mb, nw, st, ns; double ms; };
    auto timeit = [&](int var, int mb, int nw, int st, int ns, double& ms) -> bool {
      p.variant = var; p.mb = mb; p.nsplit = ns; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
      cudaError_t e = uq_decode_launch(p, nw, st, 0);
      bool ok = e == cudaSuccess;
      if (ok) { for (int i = 0; i < 3; i++) uq_decode_launch(p, nw, st, 0);
        cudaEventRecord(e0); for (int i = 0; i < iters; i++) uq_decode_launch(p, nw, st, 0); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float t; cudaEventElapsedTime(&t, e0, e1); ms = t / iters; if (cudaGetLastError() != cudaSuccess) ok = false; }
      else cudaGetLastError();
      cudaFree(mo); cudaFree(ml); return ok;
    };
    if (smoke) {
      float *mo, *ml; p.nsplit = 16; p.variant = 1; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
      CK(uq_decode_launch(p, 4, 2, 0)); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref, p.out, qn * 2, cudaMemcpyDeviceToDevice));
      for (int mw : {1, 2}) { p.variant = 3; p.nsplit = 8; CK(cudaMemset(p.out, 0xFF, qn * 2));
        cudaError_t e = uq_decode_launch(p, mw, 2, 0); printf("K=%d mw=%d launch %s\n", K, mw, cudaGetErrorString(e)); fflush(stdout);
        e = cudaDeviceSynchronize(); printf("  sync %s\n", cudaGetErrorString(e)); fflush(stdout); if (e) exit(3);
        std::vector<uint16_t> h(qn), r(qn); CK(cudaMemcpy(h.data(), p.out, qn * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), ref, qn * 2, cudaMemcpyDeviceToHost));
        double mx = 0, mxr = 0; for (size_t i = 0; i < qn; i++) { double d = fabs((double)bf2f(h[i]) - bf2f(r[i])); if (std::isnan(d)) d = 1e9; if (d > mx) mx = d; mxr = std::max(mxr, (double)fabs(bf2f(r[i]))); }
        printf("  maxdiff vs legacy %.3e (max |ref| %.3f); sample out %f %f ref %f %f\n", mx, mxr, bf2f(h[0]), bf2f(h[1]), bf2f(r[0]), bf2f(r[1])); }
      continue;
    }
    R b12{0, 0, 0, 0, 0, 1e9};
    if (!only3) {
      int bm = uq_block_m(K, HQ, HKV);
      for (int var : {1, 2}) for (int mb : {1, 2}) for (int nw : {2, 4, 8}) for (int st : {2, 3, 4}) for (int ns : {4, 8, 9, 12, 16, 20, 24, 32}) {
        if (var == 1 && mb != 1) continue; if (var == 1 && bm == 128 && nw != 8) continue;
        double ms; if (timeit(var, var == 2 ? mb : 0, nw, st, ns, ms) && ms < b12.ms) b12 = {var, mb, nw, st, ns, ms};
      }
      p.variant = b12.var; p.mb = b12.mb; p.nsplit = b12.ns; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
      uq_decode_launch(p, b12.nw, b12.st, 0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref, p.out, qn * 2, cudaMemcpyDeviceToDevice)); cudaFree(mo); cudaFree(ml);
      printf("K=%d legacy best %.4f ms (var=%d mb=%d nw=%d st=%d ns=%d) %.0f GB/s %.1f%% peak\n", K, b12.ms, b12.var, b12.mb, b12.nw, b12.st, b12.ns, kvbytes / b12.ms / 1e6, 100 * kvbytes / b12.ms / 1e6 / peak);
    }
    R b3{3, 0, 0, 0, 0, 1e9};
    int nmis = 0;
    for (int mw : {1, 2}) for (int st : {2, 3, 4}) for (int ns : {2, 3, 4, 5, 6, 8, 9, 12, 16, 20, 24, 32}) {
      double ms; if (!timeit(3, 0, mw, st, ns, ms)) continue;
      if (!only3) {
        p.variant = 3; p.nsplit = ns; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
        CK(cudaMemset(p.out, 0xFF, qn * 2)); uq_decode_launch(p, mw, st, 0); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> h(qn), r(qn); CK(cudaMemcpy(h.data(), p.out, qn * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), ref, qn * 2, cudaMemcpyDeviceToHost));
        double mx = 0; for (size_t i = 0; i < qn; i++) { double d = fabs((double)bf2f(h[i]) - bf2f(r[i])); if (std::isnan(d)) d = 1e9; if (d > mx) mx = d; }
        cudaFree(mo); cudaFree(ml);
        if (mx > 2e-3) { if (nmis++ < 4) printf("  MISMATCH var3 mw=%d st=%d ns=%d maxdiff=%.3e\n", mw, st, ns, mx); continue; }
      }
      if (ms < b3.ms) b3 = {3, 0, mw, st, ns, ms};
    }
    printf("K=%d sm90 best %.4f ms (mw=%d st=%d ns=%d) %.0f GB/s %.1f%% peak\n", K, b3.ms, b3.nw, b3.st, b3.ns, kvbytes / b3.ms / 1e6, 100 * kvbytes / b3.ms / 1e6 / peak);
    if (!only3 && b3.ms < 1e8) printf("K=%d speedup sm90/legacy %.2fx\n", K, b12.ms / b3.ms);
    fflush(stdout); cudaFree(q); cudaFree(p.out); cudaFree(ref);
  }
}
