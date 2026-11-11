// UltraQuant SYCL port: fused multi-token (K-query) FP4/UE8M0 paged decode attention.
// Layouts (identical to uqmt/attn.py):
//   kc,vc uint8 [pages,Hkv,PAGE,D/2] (low nibble = even element)
//   ks,vs uint8 [pages,Hkv,PAGE,D/32] UE8M0, value 2^(s-127)
//   q bf16 [B,K,Hq,D]; bt int32 [B,max_pages]; sl int32 [B] (includes the K spec tokens)
//   query j sees sl-K+j+1 keys.  Split-KV (flash-decoding) kernel + reduce kernel.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <stdexcept>

// default: fp32 P in PV (most accurate). -DUQ_P_BF16 rounds P to bf16 like the Triton kernel / the XMX path.
#ifdef UQ_P_BF16
#define UQ_PROUND(x) ((float)bf16(x))
#else
#define UQ_PROUND(x) (x)
#endif

namespace uq {
using bf16 = sycl::ext::oneapi::bfloat16;
using u8 = uint8_t;

struct Shape {
  int B, K, HQ, HKV, D, PAGE, max_pages;
  float sm_scale;
  int G() const { return HQ / HKV; }
  int M() const { return K * (HQ / HKV); }  // query rows per (b, kv-head) tile
};

struct Cfg {
  int nsplit = 8;   // KV splits (any >=1)
  int wg = 128;     // work-group size
  int tn = 32;      // tokens per SLM tile (divides PAGE)
  int sg = 16;      // sub-group size (16 or 32)
  int path = 0;     // 0 portable, 1 XMX (joint_matrix)
};

struct Args {
  const bf16* q; const u8 *kc, *ks, *vc, *vs; const int *bt, *sl;
  float *mo, *ml; bf16* out;
};

inline size_t mo_elems(const Shape& s, int nsplit) { return (size_t)s.B * s.HKV * nsplit * s.M() * s.D; }
inline size_t ml_elems(const Shape& s, int nsplit) { return (size_t)s.B * s.HKV * nsplit * s.M(); }

// ---- FP4 (e2m1) decode ------------------------------------------------------------
// Bit-trick used by the Triton kernel: bf16 bits = sign<<15 | code3<<6, times 2^126.
// That route needs denormal support for code 1 (0.5), which GPUs may flush, so the
// default uses the 8-entry value table (exact) and the bit trick is validated separately
// (UQ_BITTRICK=1 builds it in).
SYCL_EXTERNAL inline float fp4_val(unsigned c) {
#ifdef UQ_BITTRICK
  unsigned bits = ((c & 8u) << 12) | ((c & 7u) << 6);  // bf16 bits
  return sycl::bit_cast<float>(bits << 16) * 8.507059173023462e37f;
#else
  const float mag = 0.5f * (float)(c & 7u);  // codes 0..3 -> 0,.5,1,1.5
  // 0,.5,1,1.5 | 2,3,4,6
  const float hi = (c & 3u) == 0 ? 2.f : (c & 3u) == 1 ? 3.f : (c & 3u) == 2 ? 4.f : 6.f;
  const float v = (c & 4u) ? hi : mag;
  return (c & 8u) ? -v : v;
#endif
}
SYCL_EXTERNAL inline float ue8m0(unsigned s) { return sycl::bit_cast<float>(s << 23); }  // 2^(s-127), s in 1..254

// ---- portable split kernel -------------------------------------------------------
template <int D, int TN, int T, int SGS>
struct SplitKernel {
  Shape s; Args a; int nsplit, pps;
  sycl::local_accessor<float, 1> qs, kv, ps, st;  // st: m[M], l[M], alpha[M]

  [[sycl::reqd_sub_group_size(SGS)]] void operator()(sycl::nd_item<1> it) const {
    constexpr int DH = D / 2, NG = D / 32, KST = D + 1;  // padded row stride: conflict-free SLM
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int MAXACC = (16 * 8 * D + T - 1) / T;      // supports M up to 128 rows
    const int grp = (int)it.get_group(0), t = (int)it.get_local_id(0);
    const int sp = grp % nsplit, hk = (grp / nsplit) % s.HKV, b = grp / (nsplit * s.HKV);
    const int K = s.K, G = s.HQ / s.HKV, M = K * G, HKV = s.HKV, PAGE = s.PAGE;
    const int seq_len = a.sl[b];
    const int n_pages = (seq_len + PAGE - 1) / PAGE;
    auto sg = it.get_sub_group();
    const int lane = (int)sg.get_local_linear_id(), sgid = (int)sg.get_group_linear_id();
    constexpr int NSG = T / SGS;

    float* Q = &qs[0]; float* KV = &kv[0]; float* P = &ps[0];
    float* Mx = &st[0]; float* Ls = &st[M]; float* Al = &st[2 * M];
    for (int i = t; i < M * D; i += T) {
      const int r = i / D, d = i % D, j = r / G, g = r % G;
      Q[i] = (float)a.q[((size_t)(b * K + j) * s.HQ + hk * G + g) * D + d] * (s.sm_scale * LOG2E);
    }
    for (int r = t; r < M; r += T) { Mx[r] = -INFINITY; Ls[r] = 0.f; }
    float acc[MAXACC];
#pragma unroll
    for (int i = 0; i < MAXACC; i++) acc[i] = 0.f;

    const int p_lo = sp * pps, p_hi = sycl::min(p_lo + pps, n_pages);
    for (int p = p_lo; p < p_hi; p++) {
      const size_t cbase = ((size_t)a.bt[b * s.max_pages + p] * HKV + hk) * PAGE;
      for (int tile = 0; tile < PAGE / TN; tile++) {
        const int tok0 = p * PAGE + tile * TN;
        if (tok0 >= seq_len) break;  // uniform
        sycl::group_barrier(it.get_group());
        // dequantize K tile into SLM
        for (int i = t; i < TN * DH; i += T) {
          const int n = i / DH, e = i % DH;
          const size_t row = cbase + tile * TN + n;
          const unsigned byte = a.kc[row * DH + e];
          const float sc = ue8m0(a.ks[row * NG + (2 * e) / 32]);
          KV[n * KST + 2 * e] = fp4_val(byte & 0xFu) * sc;
          KV[n * KST + 2 * e + 1] = fp4_val(byte >> 4) * sc;
        }
        sycl::group_barrier(it.get_group());
        // scores S[r][n] = q[r] . k[n]  (log2 domain, already scaled)
        for (int idx = t; idx < M * TN; idx += T) {
          const int r = idx / TN, n = idx % TN, j = r / G;
          const float* qr = Q + r * D; const float* kr = KV + n * KST;
          float sacc = 0.f;
#pragma unroll 8
          for (int d = 0; d < D; d++) sacc += qr[d] * kr[d];
          const int limit = seq_len - K + j + 1;
          P[idx] = (tok0 + n < limit) ? sacc : -INFINITY;
        }
        sycl::group_barrier(it.get_group());
        // online softmax: one sub-group per row
        for (int r = sgid; r < M; r += NSG) {
          float mx = -INFINITY;
          for (int n = lane; n < TN; n += SGS) mx = sycl::fmax(mx, P[r * TN + n]);
          mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
          const float m_old = Mx[r];
          const float m_new = sycl::fmax(m_old, mx);
          const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
          const float alpha = sycl::exp2(m_old - m_safe);
          float sum = 0.f;
          for (int n = lane; n < TN; n += SGS) {
            const float pr = sycl::exp2(P[r * TN + n] - m_safe);
            P[r * TN + n] = pr; sum += pr;
          }
          sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
          if (lane == 0) { Ls[r] = Ls[r] * alpha + sum; Mx[r] = m_new; Al[r] = alpha; }
        }
        sycl::group_barrier(it.get_group());
        // dequantize V tile (same SLM buffer; K no longer needed)
        for (int i = t; i < TN * DH; i += T) {
          const int n = i / DH, e = i % DH;
          const size_t row = cbase + tile * TN + n;
          const unsigned byte = a.vc[row * DH + e];
          const float sc = ue8m0(a.vs[row * NG + (2 * e) / 32]);
          KV[n * KST + 2 * e] = fp4_val(byte & 0xFu) * sc;
          KV[n * KST + 2 * e + 1] = fp4_val(byte >> 4) * sc;
        }
        sycl::group_barrier(it.get_group());
        // acc[r][d] = acc*alpha + sum_n bf16(P[r][n]) * V[n][d]
#pragma unroll
        for (int i = 0; i < MAXACC; i++) {
          const int idx = t + i * T;
          if (idx < M * D) {
            const int r = idx / D, d = idx % D;
            float x = acc[i] * Al[r];
            const float* pr = P + r * TN;
#pragma unroll 4
            for (int n = 0; n < TN; n++) x += UQ_PROUND(pr[n]) * KV[n * KST + d];
            acc[i] = x;
          }
        }
      }
    }
    sycl::group_barrier(it.get_group());
    const size_t base = ((size_t)(b * HKV + hk) * nsplit + sp) * M;
#pragma unroll
    for (int i = 0; i < MAXACC; i++) {
      const int idx = t + i * T;
      if (idx < M * D) {
        const int r = idx / D, d = idx % D;
        a.mo[(base + r) * D + d] = acc[i] / sycl::fmax(Ls[r], 1e-30f);
      }
    }
    for (int r = t; r < M; r += T)
      a.ml[base + r] = Ls[r] > 0.f ? Mx[r] + sycl::log2(sycl::fmax(Ls[r], 1e-30f)) : -INFINITY;
  }
};

// ---- reduce kernel -----------------------------------------------------------------
template <int D>
struct ReduceKernel {
  Shape s; Args a; int nsplit;
  void operator()(sycl::nd_item<1> it) const {
    const int grp = (int)it.get_group(0), d = (int)it.get_local_id(0);
    const int M = s.M(), G = s.G();
    const int r = grp % M, hk = (grp / M) % s.HKV, b = grp / (M * s.HKV);
    const size_t base = (size_t)(b * s.HKV + hk) * nsplit;
    float mx = -INFINITY;
    for (int i = 0; i < nsplit; i++) mx = sycl::fmax(mx, a.ml[(base + i) * M + r]);
    if (mx == -INFINITY) mx = 0.f;
    float num = 0.f, den = 0.f;
    for (int i = 0; i < nsplit; i++) {
      const float w = sycl::exp2(a.ml[(base + i) * M + r] - mx);
      num += w * a.mo[((base + i) * M + r) * D + d]; den += w;
    }
    const int j = r / G, g = r % G;
    a.out[((size_t)(b * s.K + j) * s.HQ + hk * G + g) * D + d] = bf16(num / sycl::fmax(den, 1e-30f));
  }
};

// ---- launchers --------------------------------------------------------------------
template <int D, int TN, int T, int SGS>
sycl::event launch_split(sycl::queue& q, const Shape& s, const Args& a, int nsplit) {
  const int M = s.M();
  const int pps = (s.max_pages + nsplit - 1) / nsplit;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> qs(M * D, h), kv(TN * (D + 1), h), ps(M * TN, h), st(3 * M, h);
    h.parallel_for(sycl::nd_range<1>((size_t)s.B * s.HKV * nsplit * T, T),
                   SplitKernel<D, TN, T, SGS>{s, a, nsplit, pps, qs, kv, ps, st});
  });
}

template <int D>
sycl::event launch_reduce(sycl::queue& q, const Shape& s, const Args& a, int nsplit, sycl::event dep) {
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for(sycl::nd_range<1>((size_t)s.B * s.HKV * s.M() * D, D), ReduceKernel<D>{s, a, nsplit});
  });
}

template <int D, int TN, int T, int SGS>
sycl::event run_cfg(sycl::queue& q, const Shape& s, const Args& a, int nsplit) {
  return launch_reduce<D>(q, s, a, nsplit, launch_split<D, TN, T, SGS>(q, s, a, nsplit));
}

template <int D>
sycl::event run_D(sycl::queue& q, const Shape& s, const Cfg& c, const Args& a) {
  const int key = c.wg * 100000 + c.tn * 100 + c.sg;
#define UQ_CASE(T_, TN_, SG_) if (key == (T_) * 100000 + (TN_) * 100 + (SG_)) return run_cfg<D, TN_, T_, SG_>(q, s, a, c.nsplit);
  UQ_CASE(128, 32, 16) UQ_CASE(128, 32, 32) UQ_CASE(128, 16, 16) UQ_CASE(128, 16, 32)
  UQ_CASE(64, 32, 16) UQ_CASE(64, 32, 32) UQ_CASE(64, 16, 16)
  UQ_CASE(256, 32, 16) UQ_CASE(256, 32, 32) UQ_CASE(256, 16, 16) UQ_CASE(256, 16, 32)
  UQ_CASE(128, 64, 16) UQ_CASE(128, 64, 32) UQ_CASE(256, 64, 16) UQ_CASE(256, 64, 32)
#undef UQ_CASE
  throw std::runtime_error("unsupported (wg,tn,sg) config");
}

// Fused: all K queries x G heads share each KV page read.
inline sycl::event decode(sycl::queue& q, const Shape& s, const Cfg& c, const Args& a) {
  if (s.D == 128) return run_D<128>(q, s, c, a);
  if (s.D == 64) return run_D<64>(q, s, c, a);
  throw std::runtime_error("head_dim must be 64 or 128");
}
}  // namespace uq
