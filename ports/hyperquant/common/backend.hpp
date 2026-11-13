// backend.hpp -- the host-level interface every backend (CUDA reference = upstream
// kernels, HIP port, SYCL port) implements.  The shared drivers test_main.cpp and
// bench_main.cpp are compiled once per backend against this interface, so the
// correctness checks and benchmark shapes are identical on every GPU.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace be {
using u8 = uint8_t; using u16 = uint16_t; using u32 = uint32_t; using i8 = int8_t; using i32 = int32_t;

std::string name();     // backend + device name
void init();

// ---- Stage-2 Rice codec (stage2_cuda_encoder.cu / stage2_cuda_decoder.cu)
struct Enc { std::vector<u32> words, offsets; std::vector<u8> ks; u32 total_bits = 0; };
// cache_mode: 0 = kRecompute, 1 = kB2U8Cache (upstream default)
Enc encode(const std::vector<u16>& bf16, float alpha, int k, int sps, int cache_mode);
// variant: 0 decode() z symbols (u8)       1 decode_to_bf16 (two kernels)
//          2 decode_fused_to_bf16          3 decode_fused_coop_to_bf16
//          4 decode_fused_to(fp8 e4m3)     5 decode_fused_to(int8)
//          6 metalrice_launch_fused_decode_bf16 (integration launcher)
std::vector<u8> decode(const Enc& e, int sps, int total, float alpha, int variant);
std::vector<u16> decode_interleaved(const std::vector<u32>& il_words, const std::vector<u32>& group_base,
                                    const std::vector<u8>& ks, int sps, int total, float alpha);
// Stage-3 self-synchronising decoder: one contiguous Rice stream, single k, no offsets
std::vector<u8> stage3_decode(const std::vector<u32>& words, int k, int total, int subsequence_words);

// ---- RHT (rht_cuda.cu): y = (1/sqrt(H)) FWHT_H(x * signs) per H-block of each row
std::vector<u16> rht(const std::vector<u16>& x, int rows, int K, int H, const std::vector<i8>& signs);

// ---- lattice_ext.cu (int8 IMMA path)
void rowmax_cast(const std::vector<u16>& xp, int M, int K, float alpha, std::vector<i8>& x8, std::vector<float>& deq);
std::vector<i32> int8_gemm_tn(const std::vector<i8>& A, const std::vector<i8>& B, int M, int N, int K);
std::vector<u16> dequant(const std::vector<i32>& C, const std::vector<float>& deq, int M, int N);
void fp8_rowmax_cast(const std::vector<u16>& xp, int M, int K, std::vector<u8>& x8, std::vector<float>& sa);

// ---- fused_decode_gemv.cu
std::vector<u16> fused_gemv(const Enc& e, int sps, float alpha, const std::vector<u16>& x,
                            const std::vector<u16>& sigma, int N, int K);
std::vector<u16> fused_gemm_ws(const Enc& e, float alpha, const std::vector<u16>& x,
                               const std::vector<u16>& sigma, int N, int K, int B);

// ---- rice_fused_attention.cu
struct AttnArgs {
  Enc K, V; std::vector<u16> norms_k, norms_v, pend_uk, pend_nk, pend_uv, pend_nv, Q;
  int H, G, D, n_complete, n_pend, ct, sps; float alpha;
};
std::vector<float> fused_attn(const AttnArgs& a);

// ---- timing (device-resident inputs prepared outside the timed region; ms per call)
double time_decode(const Enc& e, int sps, int total, float alpha, int variant, int iters);
double time_rht(int rows, int K, int H, int iters);
double time_gemv(const Enc& e, int sps, float alpha, int N, int K, int iters);
double time_gemm_ws(const Enc& e, float alpha, int N, int K, int B, int iters);
double time_attn(const AttnArgs& a, int iters);
// int8 linear forward of one layer, as lattice_linear_forward (rht -> rowmax cast ->
// decode W to int8 -> int8 GEMM -> dequant); returns ms per forward
double time_int8_linear(const Enc& w, float alpha, int M, int N, int K, int H, int iters);
double time_copy_gbps(size_t bytes, int iters);  // device-memory copy bandwidth (GB/s, read+write)
}  // namespace be
