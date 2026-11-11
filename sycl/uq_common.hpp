#pragma once
#include <sycl/sycl.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
namespace uqc {
inline std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
template <class T> T* dev_copy(sycl::queue& q, const std::vector<char>& v) {
  T* p = (T*)sycl::malloc_device(v.size() ? v.size() : 1, q);
  q.memcpy(p, v.data(), v.size()).wait();
  return p;
}
}  // namespace uqc
