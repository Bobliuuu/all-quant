// CPU check of kernels/common/pq_ball_decode.h against points dumped by Python.
// usage: test_decode_cpu <table.u32> <idx.u32> <pts.i32>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "../kernels/common/pq_ball_decode.h"
template <class T> std::vector<T> rd(const char* p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate); size_t n = f.tellg(); f.seekg(0);
  std::vector<T> v(n / sizeof(T)); f.read((char*)v.data(), n); return v;
}
int main(int, char** argv) {
  auto tab = rd<uint32_t>(argv[1]); auto idx = rd<uint32_t>(argv[2]); auto pts = rd<int32_t>(argv[3]);
  int L = tab.size() / 8, bad = 0;
  int b = 0; while ((1ull << b) <= (uint64_t)tab[(L - 1) * 8]) ++b;  // b from argv instead:
  b = atoi(argv[4]);
  std::vector<uint32_t> soa(pq_tab_words(L));
  for (int i = 0; i < L; ++i) pq_tab_fill_row(soa.data(), tab.data(), L, i);
  for (int j = 0; j < (1 << PQ_RADIX_BITS); ++j) pq_tab_fill_radix(soa.data(), tab.data(), L, b, j);
  PQTab TV = pq_tab_view(soa.data(), L, b);
  for (size_t i = 0; i < idx.size(); ++i) {
    int y[8], y2[8], y3[8]; pq_ball_decode(idx[i], tab.data(), L, y); pq_ball_decode2(idx[i], tab.data(), L, y2);
    if (pq_max_nd(tab.data(), L) <= 5) pq_ball_decode3<5>(idx[i], TV, y3); else pq_ball_decode3<8>(idx[i], TV, y3);
    for (int j = 0; j < 8; ++j) if (y[j] != pts[i * 8 + j] || y2[j] != y[j] || y3[j] != y[j]) { if (bad < 5) printf("mismatch idx %u\n", idx[i]); ++bad; break; }
  }
  // fp16/e4m3 helpers spot checks
  printf("half(0x3C00)=%g half(0x0001)=%g e4m3(0x38)=%g e4m3(0x01)=%g bf16(0x3F80)=%g\n", pq_half_to_float(0x3C00),
         pq_half_to_float(1), pq_e4m3_to_float(0x38), pq_e4m3_to_float(1), pq_bf16_to_float(0x3F80));
  printf("%s: %zu indices, %d mismatches\n", argv[1], idx.size(), bad);
  return bad != 0;
}
