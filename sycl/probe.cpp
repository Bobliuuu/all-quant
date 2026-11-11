// Device capability probe: sub-group sizes, SLM, and the XMX (joint_matrix) shapes the runtime reports.
#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
  for (auto& p : sycl::platform::get_platforms()) for (auto& d : p.get_devices()) {
    std::printf("== %s | %s | driver %s\n", d.get_info<sycl::info::device::name>().c_str(), p.get_info<sycl::info::platform::name>().c_str(), d.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("  compute_units=%u max_wg=%zu local_mem=%zu KB global_mem=%zu MB\n", d.get_info<sycl::info::device::max_compute_units>(), d.get_info<sycl::info::device::max_work_group_size>(), d.get_info<sycl::info::device::local_mem_size>() / 1024, d.get_info<sycl::info::device::global_mem_size>() >> 20);
    std::printf("  sub_group_sizes:"); for (auto s : d.get_info<sycl::info::device::sub_group_sizes>()) std::printf(" %zu", (size_t)s); std::printf("\n");
    std::printf("  aspects: fp16=%d fp64=%d atomic64=%d usm_device=%d\n", d.has(sycl::aspect::fp16), d.has(sycl::aspect::fp64), d.has(sycl::aspect::atomic64), d.has(sycl::aspect::usm_device_allocations));
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
    if (true) {
      std::printf("  joint_matrix combinations (reported by runtime):\n");
      using namespace sycl::ext::oneapi::experimental::matrix;
      for (auto& c : d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
        std::printf("    M=%zu N=%zu K=%zu  A=%d B=%d C=%d D=%d  (max M/N/K %zu/%zu/%zu)\n", c.msize, c.nsize, c.ksize, (int)c.atype, (int)c.btype, (int)c.ctype, (int)c.dtype, c.max_msize, c.max_nsize, c.max_ksize);
    } else std::printf("  no ext_oneapi_matrix aspect\n");
#endif
  }
}
