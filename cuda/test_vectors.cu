// Standalone harness: reads cuda/vectors/*, runs uq_decode_launch over a config sweep, checks max abs err.
// usage: test_vectors <vectors_dir> [tol=1e-3]
#include "uq_decode.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

static std::vector<char> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary); if (!f) { printf("missing %s\n", p.c_str()); exit(2); }
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void* up(const std::vector<char>& h) { void* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); return d; }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "vectors";
  double tol = argc > 2 ? atof(argv[2]) : 1e-3;
  std::ifstream mf(dir + "/manifest.txt");
  std::string line; int fails = 0, runs = 0; double worst = 0, worst_ex = 0;
  while (std::getline(mf, line)) {
    std::istringstream is(line); std::string tag, name; int K, B, HQ, HKV, D, PAGE, pages, mp;
    is >> tag >> name >> K >> B >> HQ >> HKV >> D >> PAGE >> pages >> mp;
    auto kc = rd(dir + "/" + name + "_kc.bin"), ks = rd(dir + "/" + name + "_ks.bin"), vc = rd(dir + "/" + name + "_vc.bin"),
         vs = rd(dir + "/" + name + "_vs.bin"), bt = rd(dir + "/" + name + "_bt.bin"), sl = rd(dir + "/" + name + "_sl.bin"),
         q = rd(dir + "/" + name + "_K" + std::to_string(K) + "_q.bin"), ref = rd(dir + "/" + name + "_K" + std::to_string(K) + "_ref.bin");
    UQParams p{}; p.q = up(q); p.kc = (uint8_t*)up(kc); p.ks = (uint8_t*)up(ks); p.vc = (uint8_t*)up(vc); p.vs = (uint8_t*)up(vs);
    p.block_table = (int32_t*)up(bt); p.seq_lens = (int32_t*)up(sl);
    p.B = B; p.K = K; p.HQ = HQ; p.HKV = HKV; p.D = D; p.PAGE = PAGE; p.max_pages = mp; p.sm_scale = 1.f / sqrtf((float)D);
    size_t on = (size_t)B * K * HQ * D; CK(cudaMalloc(&p.out, on * 2));
    const float* rf = (const float*)ref.data();
    int nws[] = {0, 1, 2, 4, 8, 16}, sts[] = {0, 2, 4}, nss[] = {1, 3, 8, 16}, mbs[] = {0, 1, 2}, vars[] = {
#ifndef UQ_SM90_ONLY
      1, 2,
#endif
#ifdef UQ_ENABLE_SM90
      3
#endif
    };
    for (int var : vars) for (int mbv : mbs) for (int nw : nws) for (int st : sts) for (int ns : nss) {
      if (var == 1 && mbv) continue;
      p.nsplit = ns; p.variant = var; p.mb = mbv;
      float *mo, *ml; CK(cudaMalloc(&mo, uq_mo_elems(p) * 4)); CK(cudaMalloc(&ml, uq_ml_elems(p) * 4)); p.mo = mo; p.ml = ml;
      CK(cudaMemset(p.out, 0xFF, on * 2));
      cudaError_t e = uq_decode_launch(p, nw, st, 0);
      if (e == cudaErrorInvalidValue) { cudaGetLastError(); cudaFree(mo); cudaFree(ml); continue; }  // unsupported combo
      CK(e); CK(cudaDeviceSynchronize());
      std::vector<uint16_t> h(on); CK(cudaMemcpy(h.data(), p.out, on * 2, cudaMemcpyDeviceToHost));
      // raw max abs err, and err in excess of the bf16 output rounding (bf16 P and output rounding: allow 2^-6 relative)
      double mx = 0, mxx = 0;
      for (size_t i = 0; i < on; i++) {
        double d = fabs((double)bf2f(h[i]) - rf[i]);
        if (std::isnan(d)) d = 1e9;
        double ex = d - fabs((double)rf[i]) * (1.0 / 64);
        if (d > mx) mx = d; if (ex > mxx) mxx = ex;
      }
      runs++; worst = std::max(worst, mx); worst_ex = std::max(worst_ex, mxx);
      if (mxx > tol) { fails++; printf("FAIL %s K=%d var=%d mb=%d nw=%d st=%d ns=%d maxerr=%.3e excess=%.3e\n", name.c_str(), K, var, mbv, nw, st, ns, mx, mxx); }
      cudaFree(mo); cudaFree(ml);
    }
    printf("case %-10s K=%d done\n", name.c_str(), K);
    cudaFree((void*)p.q); cudaFree((void*)p.kc); cudaFree((void*)p.ks); cudaFree((void*)p.vc); cudaFree((void*)p.vs);
    cudaFree((void*)p.block_table); cudaFree((void*)p.seq_lens); cudaFree(p.out);
  }
  printf("RESULT runs=%d fails=%d worst_max_abs_err=%.3e worst_err_beyond_bf16_rounding=%.3e tol=%.1e\n", runs, fails, worst, worst_ex, tol);
  return fails ? 1 : 0;
}
