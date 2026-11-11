#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <cmath>
using namespace sycl::ext::intel::esimd;
int main() {
  sycl::queue q{sycl::gpu_selector_v};
  const int N = 1 << 16;
  float* x = sycl::malloc_shared<float>(N, q); float* y = sycl::malloc_shared<float>(N, q); float* z = sycl::malloc_shared<float>(N, q);
  for (int i = 0; i < N; i++) x[i] = -30.f * i / N;
  q.parallel_for(sycl::nd_range<1>(N / 16, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    int o = it.get_global_id(0) * 16;
    simd<float, 16> v = block_load<float, 16>(x + o);
    block_store<float, 16>(y + o, exp2(v));
    block_store<float, 16>(z + o, exp(v * 0.6931471805599453f));
  }).wait();
  double m1 = 0, m2 = 0;
  for (int i = 0; i < N; i++) { double r = std::exp2((double)x[i]); m1 = std::fmax(m1, std::fabs(y[i] - r) / r); m2 = std::fmax(m2, std::fabs(z[i] - r) / r); }
  printf("exp2 max rel err %.3e   exp(x*ln2) %.3e\n", m1, m2);
}
