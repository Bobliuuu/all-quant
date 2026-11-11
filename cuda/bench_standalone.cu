// Standalone (no torch) benchmark + tuner for uq_decode.  usage: bench_standalone [--batch 8 --ctx 32768 --hq 64 --hkv 8 --d 128 --page 64 --ks 1 2 4 8 --samepage]
#include "uq_decode.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(2);} } while (0)
__global__ void fill_rand(uint8_t* p, size_t n, uint32_t seed, int lo, int hi) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = hi < 0 ? (uint8_t)x : (uint8_t)(lo + (x % (uint32_t)(hi - lo + 1)));
  }
}
#ifdef UQ_PROF
void uq_prof_read(unsigned long long*); void uq_prof_reset();
#endif
int main(int argc, char** argv) {
  int B = 8, ctx = 32768, HQ = 64, HKV = 8, D = 128, PAGE = 64; std::vector<int> Ks = {1, 2, 4, 8}; bool samepage = false; int iters = 30; bool tune = true; int variant = 0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--batch") B = atoi(argv[++i]); else if (a == "--ctx") ctx = atoi(argv[++i]); else if (a == "--hq") HQ = atoi(argv[++i]);
    else if (a == "--hkv") HKV = atoi(argv[++i]); else if (a == "--d") D = atoi(argv[++i]); else if (a == "--page") PAGE = atoi(argv[++i]);
    else if (a == "--samepage") samepage = true; else if (a == "--iters") iters = atoi(argv[++i]); else if (a == "--variant") variant = atoi(argv[++i]); else if (a == "--notune") tune = false;
    else if (a == "--ks") { Ks.clear(); while (i + 1 < argc && argv[i + 1][0] != '-') Ks.push_back(atoi(argv[++i])); }
  }
  int mp = (ctx + PAGE - 1) / PAGE, npg = B * mp;
  size_t kb = (size_t)npg * HKV * PAGE * (D / 2), sb = (size_t)npg * HKV * PAGE * (D / 32);
  uint8_t *kc, *vc, *ks, *vs; CK(cudaMalloc(&kc, kb)); CK(cudaMalloc(&vc, kb)); CK(cudaMalloc(&ks, sb)); CK(cudaMalloc(&vs, sb));
  fill_rand<<<1024, 256>>>(kc, kb, 1, 0, -1); fill_rand<<<1024, 256>>>(vc, kb, 2, 0, -1);
  fill_rand<<<1024, 256>>>(ks, sb, 3, 124, 130); fill_rand<<<1024, 256>>>(vs, sb, 4, 124, 130);
  std::vector<int32_t> bt(B * mp), sl(B, ctx); std::mt19937 rng(0); std::vector<int32_t> perm(npg); for (int i = 0; i < npg; i++) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng); for (int i = 0; i < B * mp; i++) bt[i] = samepage ? 0 : perm[i];
  int32_t *dbt, *dsl; CK(cudaMalloc(&dbt, bt.size() * 4)); CK(cudaMalloc(&dsl, B * 4));
  CK(cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsl, sl.data(), B * 4, cudaMemcpyHostToDevice));
  double kvbytes = (double)B * ctx * HKV * (D / 2 + D / 32) * 2;
  // HBM peak
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
    void* q; size_t qn = (size_t)B * K * HQ * D; CK(cudaMalloc(&q, qn * 2)); fill_rand<<<256, 256>>>((uint8_t*)q, qn * 2, 5, 0x30, 0x3f);
    p.q = q; CK(cudaMalloc(&p.out, qn * 2));
    int bm = uq_block_m(K, HQ, HKV);
    std::vector<int> nws = variant == 2 ? std::vector<int>{2, 4, 8, 16} : bm == 16 ? std::vector<int>{1, 2, 4, 8} : bm == 32 ? std::vector<int>{2, 4, 8} : bm == 64 ? std::vector<int>{4, 8} : std::vector<int>{8};
    std::vector<int> sts = {2, 3, 4}, nss = {4, 8, 12, 16, 24, 32, 48, 64, 96};
    if (!tune) { nws = {0}; sts = {3}; nss = {16}; }
    double best = 1e9; int bw = 0, bs = 0, bn = 0, bmb = 0;
    std::vector<int> mbs = variant == 2 ? std::vector<int>{1, 2} : std::vector<int>{0};
    p.variant = variant;
    for (int mbv : mbs) for (int nw : nws) for (int st : sts) for (int ns : nss) {
      p.nsplit = ns; p.mb = mbv; float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
      cudaError_t e = uq_decode_launch(p, nw, st, 0);
      if (e != cudaSuccess) { cudaGetLastError(); cudaFree(mo); cudaFree(ml); continue; }
      for (int i = 0; i < 3; i++) uq_decode_launch(p, nw, st, 0);
      cudaEventRecord(e0); for (int i = 0; i < iters; i++) uq_decode_launch(p, nw, st, 0); cudaEventRecord(e1); cudaEventSynchronize(e1);
      float t; cudaEventElapsedTime(&t, e0, e1); t /= iters; CK(cudaGetLastError());
      if (t < best) { best = t; bw = nw; bs = st; bn = ns; bmb = mbv; }
      cudaFree(mo); cudaFree(ml);
    }
    printf("K=%d best %.4f ms  (mb=%d nw=%d st=%d ns=%d)  %.0f GB/s  %.1f%% of copy peak\n", K, best, bmb, bw, bs, bn, kvbytes / best / 1e6, 100 * kvbytes / best / 1e6 / peak);
#ifdef UQ_PROF
    { p.variant = 2; p.mb = bmb; p.nsplit = bn; float *mo, *ml; cudaMalloc(&mo, uq_mo_elems(p) * 4); cudaMalloc(&ml, uq_ml_elems(p) * 4); p.mo = mo; p.ml = ml;
      uq_prof_reset(); uq_decode_launch(p, bw, bs, 0); cudaDeviceSynchronize(); unsigned long long pr[8]; uq_prof_read(pr);
      double it = (double)pr[6] / 1.0; printf("  PROF K=%d cycles/iter (warp0 lane0): wait=%.0f kdeq+S=%.0f softmax=%.0f pv=%.0f loopovh=%.0f  iters/warp=%.1f\n", K,
             pr[0] / it, pr[1] / it, pr[2] / it, pr[3] / it, pr[5] / it, it / pr[7]); }
#endif
    fflush(stdout); cudaFree(q); cudaFree(p.out);
  }
  return 0;
}
