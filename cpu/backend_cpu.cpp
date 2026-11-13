// backend_cpu.cpp -- the CPU reference (ref/hq_ref.hpp) exposed through the backend
// interface, to validate the shared driver and the vectors themselves.
#include "../common/backend.hpp"
#include "../ref/hq_ref.hpp"
#include <cstring>
namespace be {
std::string name() { return "cpu-ref"; }
void init() {}
Enc encode(const std::vector<u16>& x, float a, int k, int sps, int) {
  auto E = hqref::rice_encode(x, a, k, sps); return {E.words, E.offsets, E.ks, E.total_bits};
}
std::vector<u8> decode(const Enc& e, int sps, int total, float alpha, int v) {
  auto z = hqref::decode_symbols(e.words, e.offsets, e.ks, sps, total);
  if (v == 0) return z;
  std::vector<u8> o; hqref::inverse_to(z, alpha, v == 4 ? 1 : (v == 5 ? 2 : 0), o); return o;
}
std::vector<u16> decode_interleaved(const std::vector<u32>& w, const std::vector<u32>& gb, const std::vector<u8>& ks,
                                    int sps, int total, float alpha) {
  std::vector<u8> z(total);
  for (int s = 0; s < total / sps; ++s) {
    hqref::Reader r(w.data(), w.size(), 0); r.stride_words = 32; r.start_word = gb[s / 32] + (s % 32);
    int k = ks[s];
    for (int i = 0; i < sps; ++i) { u32 q = r.unary(), rem = r.bits(k); z[(size_t)s * sps + i] = (u8)(((q << k) | rem) & 0xFF); }
  }
  std::vector<u8> o; hqref::inverse_to(z, alpha, 0, o);
  std::vector<u16> r(o.size() / 2); memcpy(r.data(), o.data(), o.size()); return r;
}
std::vector<u8> stage3_decode(const std::vector<u32>& w, int k, int total, int) {
  std::vector<u8> ks(1, (u8)k); std::vector<u32> off(1, 0);
  return hqref::decode_symbols(w, off, ks, total, total);
}
std::vector<u16> rht(const std::vector<u16>& x, int rows, int K, int H, const std::vector<i8>& s) {
  return hqref::rht_lastdim(x, rows, K, H, s, 1.0f / std::sqrt((float)H));
}
void rowmax_cast(const std::vector<u16>& xp, int M, int K, float a, std::vector<i8>& x8, std::vector<float>& d) { hqref::rowmax_cast(xp, M, K, a, x8, d); }
std::vector<i32> int8_gemm_tn(const std::vector<i8>& A, const std::vector<i8>& B, int M, int N, int K) { return hqref::int8_gemm_tn(A, B, M, N, K); }
std::vector<u16> dequant(const std::vector<i32>& C, const std::vector<float>& d, int M, int N) { return hqref::dequant(C, d, M, N); }
void fp8_rowmax_cast(const std::vector<u16>& xp, int M, int K, std::vector<u8>& x8, std::vector<float>& sa) { hqref::fp8_rowmax_cast(xp, M, K, x8, sa); }
static hqref::Encoded toE(const Enc& e) { hqref::Encoded E; E.words = e.words; E.offsets = e.offsets; E.ks = e.ks; return E; }
std::vector<u16> fused_gemv(const Enc& e, int sps, float a, const std::vector<u16>& x, const std::vector<u16>& s, int N, int K) { return hqref::fused_gemv(toE(e), sps, a, x, s, N, K); }
std::vector<u16> fused_gemm_ws(const Enc& e, float a, const std::vector<u16>& x, const std::vector<u16>& s, int N, int K, int B) { return hqref::fused_gemm_ws(toE(e), a, x, s, N, K, B); }
std::vector<float> fused_attn(const AttnArgs& a) {
  hqref::AttnIn A; A.K = toE(a.K); A.V = toE(a.V); A.norms_k = a.norms_k; A.norms_v = a.norms_v;
  A.pend_uk = a.pend_uk; A.pend_nk = a.pend_nk; A.pend_uv = a.pend_uv; A.pend_nv = a.pend_nv; A.Q = a.Q;
  A.H = a.H; A.G = a.G; A.D = a.D; A.n_complete = a.n_complete; A.n_pend = a.n_pend; A.ct = a.ct; A.sps = a.sps; A.alpha = a.alpha;
  return hqref::fused_attn(A);
}
double time_decode(const Enc&, int, int, float, int, int) { return -1; }
double time_rht(int, int, int, int) { return -1; }
double time_gemv(const Enc&, int, float, int, int, int) { return -1; }
double time_gemm_ws(const Enc&, float, int, int, int, int) { return -1; }
double time_attn(const AttnArgs&, int) { return -1; }
double time_int8_linear(const Enc&, float, int, int, int, int, int) { return -1; }
double time_copy_gbps(size_t, int) { return -1; }
}  // namespace be
