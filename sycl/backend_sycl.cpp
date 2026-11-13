// backend_sycl.cpp -- SYCL port backend (kernels: hq_sycl_kernels.hpp + below).
// Build (DPC++ / oneAPI clang): clang++ -fsycl -O3 -std=c++17 -ffp-contract=fast
//   -fsycl-targets=<nvptx64-nvidia-cuda | amdgcn-amd-amdhsa | spir64_gen | spir64> ...
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../common/backend.hpp"
#include "hq_sycl_kernels.hpp"

using namespace hqs;
namespace {
sycl::queue* g_q = nullptr;
std::string g_name;
int g_sg = 32;      // sub-group size used by the RHT
int g_sg_attn = 32; // >= 32 required by the attention kernel's 32-lane logical warps
sycl::queue& Q() { return *g_q; }

template <class T> struct DB {
  T* p = nullptr; size_t n = 0;
  explicit DB(size_t n_) : n(n_) { p = sycl::malloc_device<T>(std::max<size_t>(n, 1), Q()); Q().memset(p, 0, std::max<size_t>(n, 1) * sizeof(T)).wait(); }
  explicit DB(const std::vector<T>& v) : DB(v.size()) { if (n) Q().memcpy(p, v.data(), n * sizeof(T)).wait(); }
  ~DB() { if (p) sycl::free(p, Q()); }
  DB(const DB&) = delete; DB& operator=(const DB&) = delete;
  std::vector<T> get() const { std::vector<T> v(n); if (n) Q().memcpy(v.data(), p, n * sizeof(T)).wait(); return v; }
};
struct Timer {
  std::chrono::steady_clock::time_point t;
  void start() { Q().wait(); t = std::chrono::steady_clock::now(); }
  double stop_ms() { Q().wait(); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }
};
size_t rup(size_t a, size_t b) { return (a + b - 1) / b * b; }

// ───────────────────────────── encoder (Stage2CudaEncoder::encode)
struct EncDev {
  int num_words, num_streams, sps; float alpha; u32 k; int cache_mode;
  DB<u16> bf; DB<u32> lengths, offsets, ovf; DB<u64> zc;
  EncDev(const std::vector<be::u16>& x, float a, int k_, int sps_, int cm)
      : num_words((int)(x.size() / 8)), num_streams((int)(x.size() / 8)), sps(sps_), alpha(a), k((u32)k_), cache_mode(cm),
        bf(x), lengths(x.size() / 8), offsets(x.size() / 8), ovf(1), zc(cm ? x.size() / 8 : 1) {}
  be::Enc run(bool copy = true) {
    const int ns = num_streams, nw = num_words; const float al = alpha; const u32 rk = k;
    const u16* B = bf.p; u32* L = lengths.p; u32* O = offsets.p; u32* F = ovf.p; u64* Z = zc.p;
    const bool cache = cache_mode != 0;
    Q().memset(F, 0, 4);
    Q().parallel_for(sycl::nd_range<1>(rup(ns, 256), 256), [=](sycl::nd_item<1> it) {
      const int s = (int)it.get_global_id(0);
      if (s >= ns) return;
      int z[8]; encode_word_to_z(B, s, al, z);
      u32 len = 0; u64 packed = 0; bool of = false;
      for (int i = 0; i < 8; ++i) { const u32 sym = (u32)z[i]; len += (sym >> rk) + 1u + rk; if (sym > 0xFFu) of = true; else packed |= (u64)sym << (i * 8); }
      if (cache) { Z[s] = packed; if (of) sycl::atomic_ref<u32, sycl::memory_order::relaxed, sycl::memory_scope::device>(*F).fetch_or(1u); }
      L[s] = len;
    });
    exclusive_scan_u32(Q(), L, O, ns);
    u32 last_len, last_off, of = 0;
    Q().memcpy(&last_len, L + ns - 1, 4); Q().memcpy(&last_off, O + ns - 1, 4); Q().memcpy(&of, F, 4); Q().wait();
    const u32 payload = last_off + last_len, total_bits = payload + 32u;
    const int wc = (int)((total_bits + 31u) >> 5);
    DB<u32> W((size_t)wc);
    u32* Wp = W.p;
    const bool use_cache = cache && of == 0;
    Q().submit([&](sycl::handler& h) {
      sycl::local_accessor<u32, 1> sw(4096, h);
      sycl::local_accessor<u32, 1> meta(3, h);
      h.parallel_for(sycl::nd_range<1>(rup(ns, 256), 256), [=](sycl::nd_item<1> it) {
        const int lid = (int)it.get_local_id(0), bd = 256;
        const int block_start = (int)it.get_group(0) * bd;
        const int block_end = sycl::min(ns, block_start + bd), active_threads = block_end - block_start;
        const u32 gs = O[block_start], ge = block_end < ns ? O[block_end] : payload;
        const u32 block_words = (ge - gs + 31u) >> 5;
        const bool use_smem = block_words <= 4096u;
        const int s = block_start + lid;
        const bool active = lid < active_threads;
        auto emit = [&](u32* dst, u32 bit_pos, auto orbit) {
          u64 packed = 0; int z[8];
          if (use_cache) packed = Z[s]; else encode_word_to_z(B, s, al, z);
          for (int i = 0; i < 8; ++i) {
            const u32 sym = use_cache ? (u32)((packed >> (i * 8)) & 0xFFull) : (u32)z[i];
            const u32 q = sym >> rk, rem = sym & ((1u << rk) - 1u);
            bit_pos += q; orbit(dst, bit_pos); ++bit_pos;
            for (int b = (int)rk - 1; b >= 0; --b) { if ((rem >> b) & 1u) orbit(dst, bit_pos); ++bit_pos; }
          }
        };
        if (use_smem) {
          for (u32 i = lid; i < block_words; i += bd) sw[i] = 0u;
          sycl::group_barrier(it.get_group());
          if (active) emit(&sw[0], O[s] - gs, [](u32* d, u32 p) { or_bit<sycl::access::address_space::local_space, sycl::memory_scope::work_group>(d, p); });
          sycl::group_barrier(it.get_group());
          const u32 wb = gs >> 5, sh = gs & 31u;
          for (u32 i = lid; i < block_words; i += bd) {
            const u32 v = sw[i];
            if (v == 0u) continue;
            using AR = sycl::atomic_ref<u32, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
            if (sh == 0u) AR(Wp[wb + i]).fetch_or(v);
            else { AR(Wp[wb + i]).fetch_or(v >> sh); AR(Wp[wb + i + 1]).fetch_or(v << (32u - sh)); }
          }
          return;
        }
        if (!active) return;
        emit(Wp, O[s], [](u32* d, u32 p) { or_bit<sycl::access::address_space::global_space, sycl::memory_scope::device>(d, p); });
      });
    });
    Q().wait();
    be::Enc e; e.total_bits = total_bits;
    if (!copy) return e;
    e.words = W.get();
    const int stride = sps / 8, nds = (nw * 8 + sps - 1) / sps;
    auto offs = offsets.get();
    e.offsets.resize(nds);
    for (int d = 0; d < nds; ++d) e.offsets[d] = offs[(size_t)d * stride];
    e.ks.assign(nds, (u8)k);
    return e;
  }
};

// ───────────────────────────── decoders (Stage2CudaDecoder)
struct DecDev {
  int ns, sps, total, nw, wc; DB<u32> w, o; DB<u8> k; DB<u8> z; DB<u16> bf; DB<u8> b8;
  DecDev(const be::Enc& e, int sps_, int total_)
      : ns((int)e.offsets.size()), sps(sps_), total(total_), nw(total_ / 8), wc((int)e.words.size()), w(e.words), o(e.offsets), k(e.ks),
        z((size_t)total_), bf((size_t)total_), b8((size_t)total_) {}
  void symbols() {
    const u32* W = w.p; const u32* O = o.p; const u8* K = k.p; u8* out = z.p; const int ns_ = ns, S = sps, tot = total, wc_ = wc;
    Q().parallel_for(sycl::nd_range<1>(rup(ns_, 256), 256), [=](sycl::nd_item<1> it) {
      const int s = (int)it.get_global_id(0); if (s >= ns_) return;
      BitReader br(W, wc_, O[s]); const u32 kk = K[s]; const int base = s * S;
      for (int i = 0; i < S; ++i) {
        const u32 q = br.read_unary(), r = br.read_bits((int)kk);
        if (base + i < tot) out[base + i] = (u8)(((q << kk) | r) & 0xFFu);
      }
    });
  }
  void inverse_bf16(float alpha) {
    const u8* Z = z.p; u16* out = bf.p; const int n = nw;
    Q().parallel_for(sycl::nd_range<1>(rup(n, 256), 256), [=](sycl::nd_item<1> it) {
      const int g = (int)it.get_global_id(0); if (g >= n) return;
      u8 zz[8]; for (int i = 0; i < 8; ++i) zz[i] = Z[g * 8 + i];
      inverse_word<0>(zz, alpha, out + (size_t)g * 8);
    });
  }
  template <int OD> void fused(float alpha) {
    const u32* W = w.p; const u32* O = o.p; const u8* K = k.p; const int ns_ = ns, S = sps, n = nw, wc_ = wc;
    void* base = OD == 0 ? (void*)bf.p : (void*)b8.p;
    Q().parallel_for(sycl::nd_range<1>(rup(ns_, 256), 256), [=](sycl::nd_item<1> it) {
      const int s = (int)it.get_global_id(0); if (s >= ns_) return;
      BitReader br(W, wc_, O[s]); const u32 kk = K[s]; const int bw = s * S / 8;
      for (int w8 = 0; w8 < S / 8; ++w8) {
        u8 zz[8]; read8(br, kk, zz);
        const int g = bw + w8;
        if (g < n) inverse_word<OD>(zz, alpha, (char*)base + (size_t)g * 8 * (OD == 0 ? 2 : 1));
      }
    });
  }
  // decode_fused_coop_stage2_to_bf16_kernel: each 32-stream group stages its contiguous
  // bitstream region in local memory.  Restructured: no early return before the barrier.
  template <int CAP> void coop_launch(float alpha, int wpb) {
    const u32* W = w.p; const u32* O = o.p; const u8* K = k.p; u16* out = bf.p; const int ns_ = ns, S = sps, n = nw, wc_ = wc;
    const int total_warps = (ns_ + 31) / 32, blocks = (total_warps + wpb - 1) / wpb;
    Q().submit([&](sycl::handler& h) {
      sycl::local_accessor<u32, 1> sm((size_t)wpb * CAP, h);
      h.parallel_for(sycl::nd_range<1>((size_t)blocks * wpb * 32, (size_t)wpb * 32), [=](sycl::nd_item<1> it) {
        const int lid = (int)it.get_local_id(0), lane = lid & 31, wib = lid >> 5;
        const int gw = (int)(it.get_global_id(0) >> 5), wbase = gw * 32;
        const bool warp_ok = wbase < ns_;
        int start_word = 0, region = 0; bool use_smem = false;
        if (warp_ok) {
          start_word = (int)(O[wbase] >> 5);
          const int nf = wbase + 32;
          const int end_word = nf < ns_ ? (int)((O[nf] + 31u) >> 5) : wc_;
          region = end_word - start_word; use_smem = region <= CAP;
          if (use_smem) for (int i = lane; i < region; i += 32) sm[wib * CAP + i] = BitReader::load(W, wc_, start_word + i);
        }
        sycl::group_barrier(it.get_group());
        const int s = wbase + lane;
        if (!warp_ok || s >= ns_) return;
        const u32 kk = K[s]; const int bw = s * S / 8;
        BitReader br = use_smem ? BitReader(&sm[wib * CAP], region, O[s] - (u32)start_word * 32u) : BitReader(W, wc_, O[s]);
        for (int w8 = 0; w8 < S / 8; ++w8) {
          u8 zz[8]; read8(br, kk, zz);
          const int g = bw + w8; if (g < n) inverse_word<0>(zz, alpha, out + (size_t)g * 8);
        }
      });
    });
  }
  void coop(float alpha) {  // same cap/occupancy policy as the CUDA host code
    const long long avg = ((long long)wc + ns - 1) / ns, est = (32LL * avg) * 5 / 4 + 64;
    if (est > 3072) { fused<0>(alpha); return; }
    if (est > 1536) coop_launch<3072>(alpha, 2);
    else if (est > 768) coop_launch<1536>(alpha, 4);
    else coop_launch<768>(alpha, 8);
  }
};

std::vector<be::u8> run_decode(DecDev& d, float alpha, int v) {
  switch (v) {
    case 0: d.symbols(); Q().wait(); return d.z.get();
    case 1: d.symbols(); d.inverse_bf16(alpha); break;
    case 2: case 6: d.fused<0>(alpha); break;
    case 3: d.coop(alpha); break;
    case 4: d.fused<1>(alpha); Q().wait(); return d.b8.get();
    case 5: d.fused<2>(alpha); Q().wait(); return d.b8.get();
  }
  Q().wait();
  auto b = d.bf.get(); std::vector<be::u8> r(b.size() * 2); memcpy(r.data(), b.data(), r.size()); return r;
}

// ───────────────────────────── RHT (rht_lastdim_kernel<H>), sub-group size SG
template <int H, int SG>
void rht_launch(const u16* X, const int8_t* S, u16* Y, int rows, int K, float inv) {
  constexpr int SLOTS = H / SG, WPB = 8;
  const int bpr = K / H, tot = rows * bpr, grid = (tot + WPB - 1) / WPB;
  Q().parallel_for(sycl::nd_range<1>((size_t)grid * WPB * SG, WPB * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
    auto sg = it.get_sub_group();
    const int wg = (int)it.get_group(0) * WPB + (int)(it.get_local_id(0) / SG);
    if (wg >= tot) return;  // whole sub-group exits together
    const int lane = (int)sg.get_local_linear_id();
    const long base = (long)wg * H; const int sb = (wg % bpr) * H;
    float reg[SLOTS];
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) { const int e = s * SG + lane; reg[s] = bf2f(X[base + e]) * (float)S[sb + e]; }
#pragma unroll
    for (int len = 1; len < SG; len <<= 1) {
      const bool low = (lane & len) == 0;
#pragma unroll
      for (int s = 0; s < SLOTS; ++s) { const float p = sycl::permute_group_by_xor(sg, reg[s], len); reg[s] = low ? (reg[s] + p) : (p - reg[s]); }
    }
#pragma unroll
    for (int ls = 1; ls < SLOTS; ls <<= 1)
#pragma unroll
      for (int s = 0; s < SLOTS; ++s)
        if ((s & ls) == 0) { const float a = reg[s], b = reg[s + ls]; reg[s] = a + b; reg[s + ls] = a - b; }
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) { const int e = s * SG + lane; Y[base + e] = f2bf(reg[s] * inv); }
  });
}
template <int SG> void rht_dispatch_sg(const u16* X, const int8_t* S, u16* Y, int rows, int K, int H) {
  const float inv = 1.0f / std::sqrt((float)H);  // host, correctly rounded (CUDA: rsqrtf)
  switch (H) {
    case 256: rht_launch<256, SG>(X, S, Y, rows, K, inv); break;
    case 512: rht_launch<512, SG>(X, S, Y, rows, K, inv); break;
    case 1024: rht_launch<1024, SG>(X, S, Y, rows, K, inv); break;
    case 2048: rht_launch<2048, SG>(X, S, Y, rows, K, inv); break;
    default: throw std::runtime_error("bad H");
  }
}
void rht_dispatch(const u16* X, const int8_t* S, u16* Y, int rows, int K, int H) {
  if (g_sg == 16) rht_dispatch_sg<16>(X, S, Y, rows, K, H);
  else if (g_sg == 64) rht_dispatch_sg<64>(X, S, Y, rows, K, H);
  else rht_dispatch_sg<32>(X, S, Y, rows, K, H);
}

// ───────────────────────────── lattice_ext kernels
void rowmax_launch(const u16* X, int8_t* X8, float* D, int M, int K, float alpha) {
  Q().parallel_for(sycl::nd_range<1>((size_t)M * 256, 256), [=](sycl::nd_item<1> it) {
    const int m = (int)it.get_group(0), t = (int)it.get_local_id(0); const long base = (long)m * K;
    float local = 0.f;
    for (int k = t; k < K; k += 256) local = sycl::fmax(local, sycl::fabs(bf2f(X[base + k])));
    const float amax = sycl::reduce_over_group(it.get_group(), local, sycl::maximum<float>());
    const float sc = amax > 0.f ? 127.0f / amax : 0.f;
    if (t == 0) D[m] = amax / (127.0f * alpha);
    for (int k = t; k < K; k += 256) {
      int q = (int)sycl::rint(bf2f(X[base + k]) * sc);
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      X8[base + k] = (int8_t)q;
    }
  });
}
void dequant_launch(const int32_t* C, const float* D, u16* Y, int M, int N) {
  const long tot = (long)M * N;
  Q().parallel_for(sycl::range<1>(tot), [=](sycl::id<1> id) {
    const long i = id[0]; const int m = (int)(i / N), n = (int)(i % N);
    Y[i] = f2bf((float)C[(long)n * M + m] * D[m]);
  });
}
// int8 TN GEMM (cuBLASLt / hipBLASLt stand-in): C[n*M+m] = sum_k A[m,k] B[n,k].
// Portable SLM-tiled kernel (no joint_matrix / XMX): correct, not tuned.
void int8_gemm_launch(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
  constexpr int T = 16, KT = 64;
  Q().submit([&](sycl::handler& h) {
    sycl::local_accessor<int8_t, 2> sa(sycl::range<2>(T, KT), h), sb(sycl::range<2>(T, KT), h);
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(rup(M, T), rup(N, T)), sycl::range<2>(T, T)), [=](sycl::nd_item<2> it) {
      const int ly = (int)it.get_local_id(0), lx = (int)it.get_local_id(1);
      const int m = (int)it.get_global_id(0), n = (int)it.get_global_id(1);
      const int m0 = (int)it.get_group(0) * T, n0 = (int)it.get_group(1) * T;
      int acc = 0;
      for (int k0 = 0; k0 < K; k0 += KT) {
        for (int i = lx; i < KT; i += T) {
          const int kk = k0 + i;
          sa[ly][i] = (m0 + ly < M && kk < K) ? A[(long)(m0 + ly) * K + kk] : 0;
          sb[ly][i] = (n0 + ly < N && kk < K) ? B[(long)(n0 + ly) * K + kk] : 0;
        }
        sycl::group_barrier(it.get_group());
        for (int i = 0; i < KT; ++i) acc += (int)sa[ly][i] * (int)sb[lx][i];
        sycl::group_barrier(it.get_group());
      }
      if (m < M && n < N) C[(long)n * M + m] = acc;
    });
  });
}
void fp8_rowmax_launch(const u16* X, u8* X8, float* S, int M, int K) {
  Q().parallel_for(sycl::nd_range<1>((size_t)M * 256, 256), [=](sycl::nd_item<1> it) {
    const int m = (int)it.get_group(0), t = (int)it.get_local_id(0); const long base = (long)m * K;
    float local = 0.f;
    for (int k = t; k < K; k += 256) local = sycl::fmax(local, sycl::fabs(bf2f(X[base + k])));
    const float amax = sycl::reduce_over_group(it.get_group(), local, sycl::maximum<float>());
    const float inv = amax > 0.f ? 448.f / amax : 0.f;
    if (t == 0) S[m] = amax / 448.f;
    for (int k = t; k < K; k += 256) { float v = bf2f(X[base + k]) * inv; v = sycl::fmax(-448.f, sycl::fmin(448.f, v)); X8[base + k] = hqn::f2e4m3_rn(v); }
  });
}

// ───────────────────────────── fused GEMV / warp-specialised GEMM
void gemv_launch(const u32* W, int wc, const u32* O, const u8* Kk, int ns, int sps, float alpha, const u16* X, const u16* SG,
                 u16* Y, int N, int K) {
  constexpr int BT = 256;
  const int spr = K / sps, rpb = BT / spr, grid = (N + rpb - 1) / rpb;
  const float inv_alpha = 1.0f / alpha;
  Q().submit([&](sycl::handler& h) {
    sycl::local_accessor<u16, 1> sx((size_t)K, h);
    sycl::local_accessor<float, 1> sacc(BT, h);
    h.parallel_for(sycl::nd_range<1>((size_t)grid * BT, BT), [=](sycl::nd_item<1> it) {
      const int t = (int)it.get_local_id(0);
      for (int i = t; i < K; i += BT) sx[i] = X[i];
      sycl::group_barrier(it.get_group());
      const int active = rpb * spr, lr = t / spr, sir = t % spr, gr = (int)it.get_group(0) * rpb + lr;
      float acc = 0.f;
      if (t < active && gr < N) {
        const int gs = gr * spr + sir;
        if (gs < ns) {
          BitReader br(W, wc, O[gs]); const u32 kk = Kk[gs];
          const int bx = sir * sps;
          for (int w8 = 0; w8 < sps / 8; ++w8) {
            u8 z[8]; read8(br, kk, z); float d[8]; decode_word_f(z, inv_alpha, d);
            for (int i = 0; i < 8; ++i) acc = sycl::fma(d[i], bf2f(sx[bx + w8 * 8 + i]), acc);
          }
        }
      }
      sacc[t] = acc;
      sycl::group_barrier(it.get_group());
      if (sir == 0 && t < active && gr < N) {
        float tot = 0.f; const int b = lr * spr;
        for (int s = 0; s < spr; ++s) tot += sacc[b + s];
        Y[gr] = f2bf(tot * bf2f(SG[gr]));
      }
    });
  });
}
void gemm_ws_launch(const u32* W, int wc, const u32* O, const u8* Kk, int ns, float alpha, const u16* X, const u16* SG, u16* Y,
                    int N, int K, int B) {
  constexpr int TR = 4, TB = 8, SPS = 512, BT = 64;
  const int spr = K / SPS;
  const float inv_alpha = 1.0f / alpha;
  Q().submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> sw(2 * TR * SPS, h);
    sycl::local_accessor<u16, 1> sx(TB * SPS, h);
    sycl::local_accessor<float, 1> sacc(TB * TR, h);
    const size_t gx = (size_t)(N + TR - 1) / TR, gy = (size_t)(B + TB - 1) / TB;
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(gy, gx * BT), sycl::range<2>(1, BT)), [=](sycl::nd_item<2> it) {
      const int t = (int)it.get_local_id(1), warp = t >> 5, lane = t & 31;
      const int br0 = (int)it.get_group(1) * TR, bb0 = (int)it.get_group(0) * TB;
      if (warp == 1) sacc[(lane / TR) * TR + lane % TR] = 0.f;
      auto produce = [&](int stage, int sk) {
        const int gr = br0 + lane;
        if (lane < TR && gr < N) {
          const int gs = gr * spr + sk;
          if (gs < ns) {
            BitReader br(W, wc, O[gs]); const u32 kk = Kk[gs];
            for (int w8 = 0; w8 < SPS / 8; ++w8) {
              u8 z[8]; read8(br, kk, z); float d[8]; decode_word_f(z, inv_alpha, d);
              for (int i = 0; i < 8; ++i) sw[(stage * TR + lane) * SPS + w8 * 8 + i] = d[i];
            }
          } else for (int j = 0; j < SPS; ++j) sw[(stage * TR + lane) * SPS + j] = 0.f;
        } else if (lane < TR) for (int j = 0; j < SPS; ++j) sw[(stage * TR + lane) * SPS + j] = 0.f;
      };
      if (warp == 0) produce(0, 0);
      sycl::group_barrier(it.get_group());
      int stage = 0;
      for (int sk = 0; sk < spr; ++sk) {
        const int bx = sk * SPS, ns_ = 1 - stage;
        if (warp == 0 && sk + 1 < spr) produce(ns_, sk + 1);
        if (warp == 1)
          for (int b = 0; b < TB; ++b) {
            const int gb = bb0 + b;
            for (int j = lane; j < SPS; j += 32) { const int gj = bx + j; sx[b * SPS + j] = (gb < B && gj < K) ? X[(long)gb * K + gj] : (u16)0; }
          }
        sycl::group_barrier(it.get_group());
        if (warp == 1) {
          const int r = lane % TR, b = lane / TR; float s = 0.f;
          for (int j = 0; j < SPS; ++j) s = sycl::fma(sw[(stage * TR + r) * SPS + j], bf2f(sx[b * SPS + j]), s);
          sacc[b * TR + r] += s;
        }
        sycl::group_barrier(it.get_group());
        stage = 1 - stage;
      }
      if (warp == 1) {
        const int r = lane % TR, b = lane / TR, gr = br0 + r, gb = bb0 + b;
        if (gr < N && gb < B) Y[(long)gb * N + gr] = f2bf(sacc[b * TR + r] * bf2f(SG[gr]));
      }
    });
  });
}

// ───────────────────────────── fused Rice attention (rice_fused_attn_decode_kernel)
struct AttnDev {
  DB<u32> wk, ok, wv, ov; DB<u8> kk, kv; DB<u16> nk, nv, puk, pnk, puv, pnv, q; DB<float> out;
  explicit AttnDev(const be::AttnArgs& a) : wk(a.K.words), ok(a.K.offsets), wv(a.V.words), ov(a.V.offsets), kk(a.K.ks), kv(a.V.ks),
    nk(a.norms_k), nv(a.norms_v), puk(a.pend_uk), pnk(a.pend_nk), puv(a.pend_uv), pnv(a.pend_nv), q(a.Q), out((size_t)a.H * a.G * a.D) {}
  template <int SG> void launch_sg(const be::AttnArgs& a) {
    constexpr int ATTN_BLOCK = 128, T_MAX = 512, CT_MAX = 8, D_MAX = 128, Q_MAX = 8;
    if (a.n_complete + a.n_pend > T_MAX || a.G > Q_MAX || a.ct > CT_MAX || a.D != D_MAX) return;  // as upstream launcher
    const u32 *WK = wk.p, *OK = ok.p, *WV = wv.p, *OV = ov.p; const u8 *KK = kk.p, *KV = kv.p;
    const u16 *NK = nk.p, *NV = nv.p, *PUK = puk.p, *PNK = pnk.p, *PUV = puv.p, *PNV = pnv.p, *QP = q.p; float* OUT = out.p;
    const int wck = (int)wk.n, wcv = (int)wv.n, H = a.H, G = a.G, D = a.D, nc = a.n_complete, np = a.n_pend, ct = a.ct, sps = a.sps;
    const bool has_p = np > 0;
    const float inv_D = 1.0f / std::sqrt((float)D), sq = std::sqrt((float)D), inv_alpha = 1.0f / a.alpha;
    Q().submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> sK(CT_MAX * D_MAX, h), sQ(Q_MAX * D_MAX, h), ssc(T_MAX * Q_MAX, h), sout(Q_MAX * D_MAX, h);
      h.parallel_for(sycl::nd_range<1>((size_t)H * ATTN_BLOCK, ATTN_BLOCK), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto g = it.get_group(); auto sg = it.get_sub_group();
        const int tid = (int)it.get_local_id(0), kvh = (int)it.get_group(0), T = nc + np, nch = nc / ct;
        const int qh = tid >> 5, lane = tid & 31;
        for (int qq = 0; qq < G; ++qq) for (int d = tid; d < D; d += ATTN_BLOCK) sQ[qq * D + d] = bf2f(QP[(kvh * G + qq) * D + d]);
        for (int i = tid; i < T * G; i += ATTN_BLOCK) ssc[i] = -1e30f;
        for (int i = tid; i < G * D; i += ATTN_BLOCK) sout[i] = 0.f;
        sycl::group_barrier(g);
        auto warp_sum = [&](float v) {  // __shfl_down_sync tree over a 32-lane logical warp (value valid in lane 0)
          for (int m = 16; m > 0; m >>= 1) v += sycl::shift_group_left(sg, v, m);
          return v;
        };
        auto scores = [&](int base_t, int nt) {
          const bool on = qh < G;
          for (int t = 0; t < nt; ++t) {
            float p = 0.f;
            if (on) for (int d = lane; d < D; d += 32) p = sycl::fma(sQ[qh * D + d], sK[t * D + d], p);
            p = warp_sum(p);  // all lanes of the sub-group participate
            if (on && lane == 0) ssc[(base_t + t) * G + qh] = p * inv_D;
          }
        };
        auto accum = [&](int base_t, int nt) {
          if (qh >= G) return;
          for (int t = 0; t < nt; ++t) { const float w = ssc[(base_t + t) * G + qh]; for (int d = lane; d < D; d += 32) sout[qh * D + d] = sycl::fma(w, sK[t * D + d], sout[qh * D + d]); }
        };
        auto load_chunk = [&](const u32* Wd, int wc, const u32* Od, const u8* Kd, const u16* Nd, int c) {
          if (tid != 0) return;
          const int sid = c * H + kvh; BitReader br(Wd, wc, Od[sid]); const u32 kq = Kd[sid];
          const int wppt = D / 8;
          for (int w8 = 0; w8 < sps / 8; ++w8) {
            u8 z[8]; read8(br, kq, z); float dec[8]; decode_word_f(z, inv_alpha, dec);
            const int tok = w8 / wppt, d0 = (w8 % wppt) * 8;
            const float nf = bf2f(Nd[kvh * nc + c * ct + tok]) / sq;
            for (int i = 0; i < 8; ++i) sK[tok * D + d0 + i] = dec[i] * nf;
          }
        };
        auto load_pend = [&](const u16* U, const u16* Nn) {
          if (tid != 0) return;
          for (int t = 0; t < np; ++t) { const float nf = bf2f(Nn[kvh * np + t]) / sq; for (int d = 0; d < D; ++d) sK[t * D + d] = bf2f(U[(kvh * np + t) * D + d]) * nf; }
        };
        for (int c = 0; c < nch; ++c) { load_chunk(WK, wck, OK, KK, NK, c); sycl::group_barrier(g); scores(c * ct, ct); sycl::group_barrier(g); }
        if (has_p) { load_pend(PUK, PNK); sycl::group_barrier(g); scores(nc, np); sycl::group_barrier(g); }
        if (qh < G && lane == 0) {
          float m = -1e30f, l = 0.f;
          for (int t = 0; t < T; ++t) { const float x = ssc[t * G + qh], m2 = sycl::fmax(m, x); l = sycl::fma(l, sycl::exp(m - m2), sycl::exp(x - m2)); m = m2; }
          for (int t = 0; t < T; ++t) ssc[t * G + qh] = sycl::exp(ssc[t * G + qh] - m) / l;
        }
        sycl::group_barrier(g);
        for (int c = 0; c < nch; ++c) { load_chunk(WV, wcv, OV, KV, NV, c); sycl::group_barrier(g); accum(c * ct, ct); sycl::group_barrier(g); }
        if (has_p) { load_pend(PUV, PNV); sycl::group_barrier(g); accum(nc, np); sycl::group_barrier(g); }
        for (int qq = 0; qq < G; ++qq) for (int d = tid; d < D; d += ATTN_BLOCK) OUT[(kvh * G + qq) * D + d] = sout[qq * D + d];
      });
    });
  }
  void launch(const be::AttnArgs& a) { if (g_sg_attn == 64) launch_sg<64>(a); else launch_sg<32>(a); }
};

// ───────────────────────────── Stage-3 self-synchronising decoder (Stage3CudaDecoder)
inline u32 rbit(const u32* w, int wc, u32 p) { const u32 wi = p >> 5; return wi >= (u32)wc ? 0u : (w[wi] >> (31u - (p & 31u))) & 1u; }
inline bool rsym(const u32* w, int wc, u32& p, int k, u32 end, u32& sym) {
  u32 q = 0;
  while (p < end && rbit(w, wc, p) == 0u) { ++p; ++q; }
  if (p >= end) return false;
  ++p;
  if (p + (u32)k > end) return false;
  u32 v = 0; for (int i = 0; i < k; ++i) { v = (v << 1) | rbit(w, wc, p); ++p; }
  sym = (q << k) | v; return true;
}
}  // namespace

namespace be {
std::string name() { return g_name; }
void init() {
  g_q = new sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order());
  auto d = g_q->get_device();
  auto sgs = d.get_info<sycl::info::device::sub_group_sizes>();
  auto has = [&](size_t s) { return std::find(sgs.begin(), sgs.end(), s) != sgs.end(); };
  g_sg = has(32) ? 32 : (has(16) ? 16 : 64);
  g_sg_attn = has(32) ? 32 : 64;
  std::string sgl; for (auto s : sgs) sgl += std::to_string(s) + "/";
  g_name = "sycl:" + d.get_platform().get_info<sycl::info::platform::name>() + ":" + d.get_info<sycl::info::device::name>() +
           " sg{" + sgl + "} rht_sg=" + std::to_string(g_sg);
}
Enc encode(const std::vector<u16>& x, float alpha, int k, int sps, int cm) { EncDev e(x, alpha, k, sps, cm); return e.run(); }
std::vector<u8> decode(const Enc& e, int sps, int total, float alpha, int v) { DecDev d(e, sps, total); return run_decode(d, alpha, v); }

std::vector<u16> decode_interleaved(const std::vector<u32>& w, const std::vector<u32>& gb, const std::vector<u8>& ks, int sps, int total, float alpha) {
  DB<u32> W(w), G(gb); DB<u8> K(ks); DB<u16> out((size_t)total);
  const u32* Wp = W.p; const u32* Gp = G.p; const u8* Kp = K.p; u16* Op = out.p; const int ns = total / sps, wc = (int)w.size(), n = total / 8;
  Q().parallel_for(sycl::nd_range<1>(rup(ns, 256), 256), [=](sycl::nd_item<1> it) {
    const int s = (int)it.get_global_id(0); if (s >= ns) return;
    BitReader br(Wp, wc, (int)Gp[s >> 5] + (s & 31), 0, 32); const u32 kk = Kp[s]; const int bw = s * sps / 8;
    for (int w8 = 0; w8 < sps / 8; ++w8) { u8 z[8]; read8(br, kk, z); if (bw + w8 < n) inverse_word<0>(z, alpha, Op + (size_t)(bw + w8) * 8); }
  });
  Q().wait(); return out.get();
}

std::vector<u8> stage3_decode(const std::vector<u32>& words, int k, int total, int sub) {
  // host chunk plan (build_chunk_plan_host), then the phase kernels as upstream
  const int wc = (int)words.size(), nsub = (wc + sub - 1) / sub;
  const u32 total_bits = (u32)wc * 32u;
  std::vector<u32> cs(nsub, 0), cc(nsub, 0), co(nsub, 0);
  u32 p = 0; int dec = 0;
  for (int s = 0; s < nsub; ++s) {
    cs[s] = p; const u32 hard = (u32)std::min<long>(wc, (long)(s + 1) * sub) * 32u;
    while (dec < total && p < total_bits) { if (p >= hard) break; u32 sym; if (!rsym(words.data(), wc, p, k, total_bits, sym)) throw std::runtime_error("plan"); ++cc[s]; ++dec; }
  }
  while (dec < total) { u32 sym; if (!rsym(words.data(), wc, p, k, total_bits, sym)) throw std::runtime_error("plan2"); ++cc.back(); ++dec; }
  const u32 payload = p; u32 run = 0;
  for (int s = 0; s < nsub; ++s) { co[s] = run; run += cc[s]; }
  DB<u32> W(words), CS(cs), A(cs), Bv(cs), CO(co), SE((size_t)nsub), SC((size_t)nsub), CH(1); DB<u8> out((size_t)total);
  const u32* Wp = W.p; u32 *SEp = SE.p, *SCp = SC.p, *COp = CO.p; u8* Op = out.p;
  auto populate = [&](const u32* starts) {
    Q().parallel_for(sycl::range<1>(nsub), [=](sycl::id<1> id) {
      const int s = (int)id[0]; u32 bp = starts[s]; const u32 he = (u32)sycl::min(wc, (s + 1) * sub) * 32u; u32 c = 0, sym;
      while (bp < he) { if (!rsym(Wp, wc, bp, k, he, sym)) break; ++c; }
      SEp[s] = bp; SCp[s] = c;
    });
  };
  u32* in = A.p; u32* outp = Bv.p; int* chg = (int*)CH.p;
  populate(CS.p);
  for (int iter = 0; iter < 12; ++iter) {
    Q().memset(chg, 0, 4);
    const u32* ip = in; u32* op = outp;
    Q().parallel_for(sycl::range<1>(nsub), [=](sycl::id<1> id) {
      const int s = (int)id[0]; const u32 ns_ = s > 0 ? SEp[s - 1] : 0u; op[s] = ns_;
      if (ns_ != ip[s]) sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(*chg).store(1);
    });
    int changed = 0; Q().memcpy(&changed, chg, 4).wait();
    std::swap(in, outp);
    if (!changed) break;
    populate(in);
  }
  { const u32* st = in;
    Q().parallel_for(sycl::range<1>(nsub), [=](sycl::id<1> id) {
      const int s = (int)id[0]; u32 sb = st[s], eb = s + 1 < nsub ? st[s + 1] : payload;
      if (sb > eb) sb = eb;
      if (eb > payload) eb = payload;
      u32 bp = sb, c = 0, sym; while (bp < eb) { if (!rsym(Wp, wc, bp, k, eb, sym)) break; ++c; }
      SEp[s] = bp; SCp[s] = c;
    }); }
  exclusive_scan_u32(Q(), SCp, COp, nsub);
  u32 lo, lc; Q().memcpy(&lo, COp + nsub - 1, 4); Q().memcpy(&lc, SCp + nsub - 1, 4); Q().wait();
  if (lo + lc != (u32)total) {  // validated fallback, as upstream
    Q().memcpy(COp, co.data(), nsub * 4).wait();
    const u32* csp = CS.p;
    Q().parallel_for(sycl::range<1>(nsub), [=](sycl::id<1> id) {
      const int s = (int)id[0]; u32 bp = csp[s]; const u32 eb = s + 1 < nsub ? csp[s + 1] : payload;
      u32 op = COp[s]; const u32 nop = s + 1 < nsub ? COp[s + 1] : (u32)total; u32 sym;
      while (bp < eb && op < nop) { if (!rsym(Wp, wc, bp, k, eb, sym)) break; Op[op++] = (u8)(sym & 0xFFu); }
    });
  } else {
    const u32* st = in;
    Q().parallel_for(sycl::range<1>(nsub), [=](sycl::id<1> id) {
      const int s = (int)id[0]; u32 bp = st[s]; const u32 eb = SEp[s]; u32 op = COp[s]; const u32 nop = op + SCp[s]; u32 sym;
      while (bp < eb && op < nop) { if (!rsym(Wp, wc, bp, k, eb, sym)) break; Op[op++] = (u8)(sym & 0xFFu); }
    });
  }
  Q().wait(); return out.get();
}

std::vector<u16> rht(const std::vector<u16>& x, int rows, int K, int H, const std::vector<i8>& s) {
  DB<u16> X(x), Y(x.size()); DB<i8> S(s); rht_dispatch(X.p, S.p, Y.p, rows, K, H); Q().wait(); return Y.get();
}
void rowmax_cast(const std::vector<u16>& xp, int M, int K, float alpha, std::vector<i8>& x8, std::vector<float>& deq) {
  DB<u16> X(xp); DB<i8> Q8((size_t)M * K); DB<float> D(M); rowmax_launch(X.p, Q8.p, D.p, M, K, alpha); Q().wait(); x8 = Q8.get(); deq = D.get();
}
std::vector<i32> int8_gemm_tn(const std::vector<i8>& A, const std::vector<i8>& B, int M, int N, int K) {
  DB<i8> a(A), b(B); DB<i32> c((size_t)M * N); int8_gemm_launch(a.p, b.p, c.p, M, N, K); Q().wait(); return c.get();
}
std::vector<u16> dequant(const std::vector<i32>& C, const std::vector<float>& deq, int M, int N) {
  DB<i32> c(C); DB<float> d(deq); DB<u16> y((size_t)M * N); dequant_launch(c.p, d.p, y.p, M, N); Q().wait(); return y.get();
}
void fp8_rowmax_cast(const std::vector<u16>& xp, int M, int K, std::vector<u8>& x8, std::vector<float>& sa) {
  DB<u16> X(xp); DB<u8> Q8((size_t)M * K); DB<float> S(M); fp8_rowmax_launch(X.p, Q8.p, S.p, M, K); Q().wait(); x8 = Q8.get(); sa = S.get();
}
std::vector<u16> fused_gemv(const Enc& e, int sps, float alpha, const std::vector<u16>& x, const std::vector<u16>& sg, int N, int K) {
  DB<u32> w(e.words), o(e.offsets); DB<u8> k(e.ks); DB<u16> X(x), S(sg), Y((size_t)N);
  gemv_launch(w.p, (int)w.n, o.p, k.p, (int)o.n, sps, alpha, X.p, S.p, Y.p, N, K); Q().wait(); return Y.get();
}
std::vector<u16> fused_gemm_ws(const Enc& e, float alpha, const std::vector<u16>& x, const std::vector<u16>& sg, int N, int K, int B) {
  DB<u32> w(e.words), o(e.offsets); DB<u8> k(e.ks); DB<u16> X(x), S(sg), Y((size_t)B * N);
  gemm_ws_launch(w.p, (int)w.n, o.p, k.p, (int)o.n, alpha, X.p, S.p, Y.p, N, K, B); Q().wait(); return Y.get();
}
std::vector<float> fused_attn(const AttnArgs& a) { AttnDev d(a); d.launch(a); Q().wait(); return d.out.get(); }

// ───────────────────────────── timing (host wall clock around in-order queue; waits excluded per call)
double time_decode(const Enc& e, int sps, int total, float alpha, int v, int iters) {
  DecDev d(e, sps, total);
  auto f = [&] {
    switch (v) { case 0: d.symbols(); break; case 1: d.symbols(); d.inverse_bf16(alpha); break; case 2: case 6: d.fused<0>(alpha); break;
                 case 3: d.coop(alpha); break; case 4: d.fused<1>(alpha); break; case 5: d.fused<2>(alpha); break; }
  };
  f(); Timer t; t.start(); for (int i = 0; i < iters; ++i) f(); return t.stop_ms() / iters;
}
double time_rht(int rows, int K, int H, int iters) {
  DB<u16> X((size_t)rows * K), Y((size_t)rows * K); std::vector<i8> s(K, 1); DB<i8> S(s);
  rht_dispatch(X.p, S.p, Y.p, rows, K, H); Timer t; t.start();
  for (int i = 0; i < iters; ++i) rht_dispatch(X.p, S.p, Y.p, rows, K, H);
  return t.stop_ms() / iters;
}
double time_gemv(const Enc& e, int sps, float alpha, int N, int K, int iters) {
  DB<u32> w(e.words), o(e.offsets); DB<u8> k(e.ks); DB<u16> X((size_t)K), S((size_t)N), Y((size_t)N);
  auto f = [&] { gemv_launch(w.p, (int)w.n, o.p, k.p, (int)o.n, sps, alpha, X.p, S.p, Y.p, N, K); };
  f(); f(); Timer t; t.start(); for (int i = 0; i < iters; ++i) f(); return t.stop_ms() / iters;
}
double time_gemm_ws(const Enc& e, float alpha, int N, int K, int B, int iters) {
  DB<u32> w(e.words), o(e.offsets); DB<u8> k(e.ks); DB<u16> X((size_t)B * K), S((size_t)N), Y((size_t)B * N);
  auto f = [&] { gemm_ws_launch(w.p, (int)w.n, o.p, k.p, (int)o.n, alpha, X.p, S.p, Y.p, N, K, B); };
  f(); Timer t; t.start(); for (int i = 0; i < iters; ++i) f(); return t.stop_ms() / iters;
}
double time_attn(const AttnArgs& a, int iters) {
  AttnDev d(a); d.launch(a); d.launch(a); Timer t; t.start(); for (int i = 0; i < iters; ++i) d.launch(a); return t.stop_ms() / iters;
}
double time_int8_linear(const Enc& e, float alpha, int M, int N, int K, int H, int iters) {
  DecDev dec(e, 512, N * K);
  DB<u16> X((size_t)M * K), XP((size_t)M * K), Y((size_t)M * N); std::vector<i8> s(K, 1); DB<i8> S(s);
  DB<i8> X8((size_t)M * K); DB<float> D(M); DB<i32> C((size_t)M * N);
  auto f = [&] {
    rht_dispatch(X.p, S.p, XP.p, M, K, H); rowmax_launch(XP.p, X8.p, D.p, M, K, alpha);
    dec.fused<2>(alpha); int8_gemm_launch(X8.p, (const int8_t*)dec.b8.p, C.p, M, N, K); dequant_launch(C.p, D.p, Y.p, M, N);
  };
  f(); Timer t; t.start(); for (int i = 0; i < iters; ++i) f(); return t.stop_ms() / iters;
}
double time_copy_gbps(size_t bytes, int iters) {
  DB<u8> a(bytes), b(bytes); Q().memcpy(b.p, a.p, bytes).wait(); Timer t; t.start();
  for (int i = 0; i < iters; ++i) Q().memcpy(b.p, a.p, bytes);
  double ms = t.stop_ms() / iters; return 2.0 * bytes / (ms * 1e6);
}
}  // namespace be
