// Standalone correctness harness: runs every case in <vecdir>/manifest.txt through the
// SYCL kernels and compares with the fp32 reference stored in the vectors.
//   test_vectors <vecdir> [--path portable|xmx] [--nsplit N] [--wg W] [--tn T] [--sg S]
#include "uq_decode.hpp"
#include "uq_common.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <vecdir> [opts]\n", argv[0]); return 2; }
  std::string dir = argv[1];
  uq::Cfg cfg0; int path = 0; bool sweep = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::atoi(argv[++i]); };
    if (a == "--nsplit") cfg0.nsplit = nxt(); else if (a == "--wg") cfg0.wg = nxt();
    else if (a == "--tn") cfg0.tn = nxt(); else if (a == "--sg") cfg0.sg = nxt();
    else if (a == "--path") path = std::string(argv[++i]) == "xmx"; else if (a == "--sweep") sweep = true;
  }
  cfg0.path = path;
  sycl::queue q{sycl::default_selector_v, sycl::property::queue::enable_profiling{}};
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  std::ifstream mf(dir + "/manifest.txt");
  std::string name; int B, K, HQ, HKV, D, PAGE, MP; double sm;
  int fails = 0, runs = 0; double worst = 0;
  while (mf >> name >> B >> K >> HQ >> HKV >> D >> PAGE >> MP >> sm) {
    auto ld = [&](const char* n) { return uqc::read_file(dir + "/" + name + "." + n + ".bin"); };
    auto vq = ld("q"), vkc = ld("kc"), vks = ld("ks"), vvc = ld("vc"), vvs = ld("vs"), vbt = ld("bt"), vsl = ld("sl"), vref = ld("ref");
    uq::Shape s{B, K, HQ, HKV, D, PAGE, MP, (float)sm};
    std::vector<int> nsplits = sweep ? std::vector<int>{1, 2, 3, 8} : std::vector<int>{cfg0.nsplit};
    for (int ns : nsplits) {
      uq::Cfg c = cfg0; c.nsplit = ns;
      if (PAGE % c.tn) c.tn = PAGE < 32 ? PAGE : 16;
      auto* dq = uqc::dev_copy<uq::bf16>(q, vq); auto* dkc = uqc::dev_copy<uq::u8>(q, vkc);
      auto* dks = uqc::dev_copy<uq::u8>(q, vks); auto* dvc = uqc::dev_copy<uq::u8>(q, vvc);
      auto* dvs = uqc::dev_copy<uq::u8>(q, vvs);
      auto* dbt = uqc::dev_copy<int>(q, vbt); auto* dsl = uqc::dev_copy<int>(q, vsl);
      auto* mo = sycl::malloc_device<float>(uq::mo_elems(s, ns), q);
      auto* ml = sycl::malloc_device<float>(uq::ml_elems(s, ns), q);
      size_t nout = (size_t)B * K * HQ * D;
      auto* out = sycl::malloc_device<uq::bf16>(nout, q);
      uq::Args a{dq, dkc, dks, dvc, dvs, dbt, dsl, mo, ml, out};
      uq::decode(q, s, c, a).wait();
      std::vector<uq::bf16> hout(nout);
      q.memcpy(hout.data(), out, nout * sizeof(uq::bf16)).wait();
      const float* ref = (const float*)vref.data();
      double maxabs = 0, maxtol = 0, wo = 0, wr = 0; size_t wi = 0; bool nan = false;
      for (size_t i = 0; i < nout; i++) {
        double o = (float)hout[i], r = ref[i], e = std::fabs(o - r);
        if (o != o) nan = true;
        maxabs = std::fmax(maxabs, e);
        // pass if within 1e-3 abs OR within one bf16 ulp (2^-7 relative) of the fp32 reference: output is bf16 and P is rounded to bf16 before PV as in the Triton kernel
        { double x = e - (1e-3 + std::fabs(r) * (1.0 / 128)); if (x > maxtol) { maxtol = x; wi = i; wo = o; wr = r; } }
      }
      bool ok = !nan && maxabs <= 1e-3 + 0 || (!nan && maxtol <= 0);
      std::printf("%-22s nsplit=%d wg=%d tn=%d sg=%d  max_abs_err=%.3e  %s\n", name.c_str(), ns, c.wg, c.tn, c.sg, maxabs,
                  ok ? "PASS" : "FAIL");
      if (!ok) std::printf("   worst elem idx=%zu (b=%zu,j=%zu) out=%g ref=%g\n", wi, wi / ((size_t)K * HQ * D), (wi / ((size_t)HQ * D)) % K, wo, wr);
      worst = std::fmax(worst, maxabs); runs++; fails += !ok;
      for (void* p : {(void*)dq, (void*)dkc, (void*)dks, (void*)dvc, (void*)dvs, (void*)dbt, (void*)dsl, (void*)mo, (void*)ml, (void*)out}) sycl::free(p, q);
    }
  }
  std::printf("SUMMARY runs=%d fails=%d worst_max_abs_err=%.3e\n", runs, fails, worst);
  return fails ? 1 : 0;
}
