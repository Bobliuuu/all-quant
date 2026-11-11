// XMX (DPAS) path: QK^T and PV on joint_matrix bf16 (8x16x16 tiles, fp32 accumulate).
// Included from uq_decode.hpp. FP4 is dequantised on the fly into bf16 SLM tiles in the VNNI
// layout the B operand wants (exact: e2m1 * 2^k is representable in bf16).
#pragma once
namespace uq {
namespace jm = sycl::ext::oneapi::experimental::matrix;

// e2m1 code -> bf16 bits with the UE8M0 scale folded in as an exponent add (exact, no float math).
// code3 -> 0,.5,1,1.5,2,3,4,6 = bits {0, 0x3F00, then 0x3F00 + (code3<<6) for code3>=2}; scale 2^(s-127) adds (s-127)<<7.
SYCL_EXTERNAL inline uint32_t fp4_bf16bits(unsigned c, int off) {
  const unsigned c3 = c & 7u;
  const int mag = c3 == 0 ? 0 : (0x3F00 + off + (c3 > 1 ? (int)(c3 << 6) : 0));
  return (uint32_t)mag | ((c & 8u) << 12);
}
SYCL_EXTERNAL inline uint32_t fp4_pair(unsigned byte, int off) {  // low nibble -> low 16 bits
  return fp4_bf16bits(byte & 0xFu, off) | (fp4_bf16bits(byte >> 4, off) << 16);
}

template <int D, int TN, int T>
struct SplitKernelXMX {
  Shape s; Args a; int nsplit, pps;
  sycl::local_accessor<bf16, 1> qb, kvb, pb;   // Q [Mpad][D]; K/V tile [D*TN] VNNI; P [Mpad][TN]
  sycl::local_accessor<float, 1> sb, ob, st;   // S [Mpad][TN]; O [Mpad][D]; stats m,l,alpha
  sycl::local_accessor<sycl::uint2, 1> lut;    // 256-entry byte -> (bf16 pair bits, nonzero mask), exponent pre-biased by -32

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    constexpr int DH = D / 2, NG = D / 32, SGS = 16, NSG = T / SGS, NT = TN / 16, DT = D / 16;
    constexpr float LOG2E = 1.4426950408889634f;
    const int grp = (int)it.get_group(0), t = (int)it.get_local_id(0);
    const int sp = grp % nsplit, hk = (grp / nsplit) % s.HKV, b = grp / (nsplit * s.HKV);
    const int K = s.K, G = s.HQ / s.HKV, M = K * G, HKV = s.HKV, PAGE = s.PAGE;
    const int MT = (M + 7) / 8, Mpad = MT * 8;
    const int seq_len = a.sl[b];
    const int n_pages = (seq_len + PAGE - 1) / PAGE;
    auto sg = it.get_sub_group();
    const int lane = (int)sg.get_local_linear_id(), sgid = (int)sg.get_group_linear_id();
    const float scl = s.sm_scale * LOG2E;

    bf16* Qb = &qb[0]; bf16* KV = &kvb[0]; bf16* Pb = &pb[0];
    float* S = &sb[0]; float* O = &ob[0];
    float* Mx = &st[0]; float* Ls = &st[M]; float* Al = &st[2 * M];
    auto lp = [](bf16* p) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p); };
    auto lpf = [](float* p) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p); };

    for (int i = t; i < 256; i += T) {
      const unsigned lo = (unsigned)i & 0xF, hi = (unsigned)i >> 4;
      const int bias = -(32 << 7);
      lut[i] = sycl::uint2(fp4_bf16bits(lo, bias) | (fp4_bf16bits(hi, bias) << 16),
                           ((lo & 7) ? 0xFFFFu : 0u) | ((hi & 7) ? 0xFFFF0000u : 0u));
    }
    for (int i = t; i < Mpad * D; i += T) {
      const int r = i / D, d = i % D, j = r / G, g = r % G;
      Qb[i] = r < M ? a.q[((size_t)(b * K + j) * s.HQ + hk * G + g) * D + d] : bf16(0.f);
      O[i] = 0.f;
    }
    for (int i = t; i < Mpad * TN; i += T) Pb[i] = bf16(0.f);
    for (int r = t; r < M; r += T) { Mx[r] = -INFINITY; Ls[r] = 0.f; }

    const int p_lo = sp * pps, p_hi = sycl::min(p_lo + pps, n_pages);
    for (int p = p_lo; p < p_hi; p++) {
      const size_t cbase = ((size_t)a.bt[b * s.max_pages + p] * HKV + hk) * PAGE;
      for (int tile = 0; tile < PAGE / TN; tile++) {
        const int tok0 = p * PAGE + tile * TN;
        if (tok0 >= seq_len) break;  // uniform
        sycl::group_barrier(it.get_group());
        // K tile -> VNNI [D/2][TN] of 32-bit (bf16 pair along d); n fastest so SLM writes are conflict free
        {
          uint32_t* KV32 = reinterpret_cast<uint32_t*>(KV);
          for (int i = t; i < TN * (DH / 4); i += T) {
            const int e4 = i / TN, n = i % TN;
            const size_t row = cbase + tile * TN + n;
            const uint32_t w = *reinterpret_cast<const uint32_t*>(a.kc + row * DH + e4 * 4);
            const uint32_t off = (uint32_t)((sycl::max((int)a.ks[row * NG + (e4 * 8) / 32], 95) - 95) << 7);
            const uint32_t offp = off | (off << 16);
#pragma unroll
            for (int q = 0; q < 4; q++) { const sycl::uint2 l = lut[(w >> (8 * q)) & 0xFFu]; KV32[(e4 * 4 + q) * TN + n] = (l.x() + offp) & l.y(); }
          }
        }
        sycl::group_barrier(it.get_group());
        // S = Q K^T
        for (int id = sgid; id < MT * NT; id += NSG) {
          const int mt = id / NT, nt = id % NT;
          jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> mc;
          jm::joint_matrix_fill(sg, mc, 0.f);
          for (int kk = 0; kk < DT; kk++) {
            jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major> ma;
            jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, 16, jm::layout::ext_intel_packed> mb;
            jm::joint_matrix_load(sg, ma, lp(Qb + mt * 8 * D + kk * 16), D);
            jm::joint_matrix_load(sg, mb, lp(KV + (kk * 8) * (TN * 2) + nt * 32), TN * 2);
            jm::joint_matrix_mad(sg, mc, ma, mb, mc);
          }
          jm::joint_matrix_store(sg, mc, lpf(S + mt * 8 * TN + nt * 16), TN, jm::layout::row_major);
        }
        sycl::group_barrier(it.get_group());
        // online softmax (one sub-group per row), P -> bf16, rescale O rows
        for (int r = sgid; r < M; r += NSG) {
          const int limit = seq_len - K + r / G + 1;
          float mx = -INFINITY;
          for (int n = lane; n < TN; n += SGS) {
            const float x = (tok0 + n < limit) ? S[r * TN + n] * scl : -INFINITY;
            S[r * TN + n] = x; mx = sycl::fmax(mx, x);
          }
          mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
          const float m_old = Mx[r], m_new = sycl::fmax(m_old, mx);
          const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
          const float alpha = sycl::exp2(m_old - m_safe);
          float sum = 0.f;
          for (int n = lane; n < TN; n += SGS) {
            const float pr = sycl::exp2(S[r * TN + n] - m_safe);
            Pb[r * TN + n] = bf16(pr); sum += pr;
          }
          sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
          for (int d = lane; d < D; d += SGS) O[r * D + d] *= alpha;
          if (lane == 0) { Ls[r] = Ls[r] * alpha + sum; Mx[r] = m_new; }
        }
        // V tile -> VNNI over tokens [TN/2][D] of 32-bit (bf16 pair along tokens); 8 words per thread, 2x16B stores
        {
          uint32_t* KV32 = reinterpret_cast<uint32_t*>(KV);
          for (int i = t; i < (TN / 2) * (DH / 4); i += T) {
            const int tp = i / (DH / 4), e4 = i % (DH / 4);
            const size_t r0 = cbase + tile * TN + 2 * tp, r1 = r0 + 1;
            const uint32_t w0 = *reinterpret_cast<const uint32_t*>(a.vc + r0 * DH + e4 * 4);
            const uint32_t w1 = *reinterpret_cast<const uint32_t*>(a.vc + r1 * DH + e4 * 4);
            const uint32_t f0 = (uint32_t)((sycl::max((int)a.vs[r0 * NG + (e4 * 8) / 32], 95) - 95) << 7);
            const uint32_t f1 = (uint32_t)((sycl::max((int)a.vs[r1 * NG + (e4 * 8) / 32], 95) - 95) << 7);
            const uint32_t p0 = f0 | (f0 << 16), p1 = f1 | (f1 << 16);
            uint32_t o[8];
#pragma unroll
            for (int q = 0; q < 4; q++) {
              const sycl::uint2 l0 = lut[(w0 >> (8 * q)) & 0xFFu], l1 = lut[(w1 >> (8 * q)) & 0xFFu];
              const uint32_t x0 = (l0.x() + p0) & l0.y(), x1 = (l1.x() + p1) & l1.y();
              o[2 * q] = (x0 & 0xFFFFu) | (x1 << 16);
              o[2 * q + 1] = (x0 >> 16) | (x1 & 0xFFFF0000u);
            }
            sycl::uint4* dst = reinterpret_cast<sycl::uint4*>(KV32 + tp * D + e4 * 8);
            dst[0] = sycl::uint4(o[0], o[1], o[2], o[3]);
            dst[1] = sycl::uint4(o[4], o[5], o[6], o[7]);
          }
        }
        sycl::group_barrier(it.get_group());
        // O += P V
        for (int id = sgid; id < MT * DT; id += NSG) {
          const int mt = id / DT, dt = id % DT;
          jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> mc;
          jm::joint_matrix_load(sg, mc, lpf(O + mt * 8 * D + dt * 16), D, jm::layout::row_major);
          for (int kk = 0; kk < NT; kk++) {
            jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major> ma;
            jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, 16, jm::layout::ext_intel_packed> mb;
            jm::joint_matrix_load(sg, ma, lp(Pb + mt * 8 * TN + kk * 16), TN);
            jm::joint_matrix_load(sg, mb, lp(KV + (kk * 8) * (D * 2) + dt * 32), D * 2);
            jm::joint_matrix_mad(sg, mc, ma, mb, mc);
          }
          jm::joint_matrix_store(sg, mc, lpf(O + mt * 8 * D + dt * 16), D, jm::layout::row_major);
        }
      }
    }
    sycl::group_barrier(it.get_group());
    const size_t base = ((size_t)(b * HKV + hk) * nsplit + sp) * M;
    for (int i = t; i < M * D; i += T) {
      const int r = i / D, d = i % D;
      a.mo[(base + r) * D + d] = O[i] / sycl::fmax(Ls[r], 1e-30f);
    }
    for (int r = t; r < M; r += T)
      a.ml[base + r] = Ls[r] > 0.f ? Mx[r] + sycl::log2(sycl::fmax(Ls[r], 1e-30f)) : -INFINITY;
  }
};

template <int D, int TN, int T>
sycl::event launch_split_xmx(sycl::queue& q, const Shape& s, const Args& a, int nsplit) {
  const int M = s.M(), Mpad = (M + 7) / 8 * 8;
  const int pps = (s.max_pages + nsplit - 1) / nsplit;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<bf16, 1> qb(Mpad * D, h), kvb(D * TN, h), pb(Mpad * TN, h);
    sycl::local_accessor<float, 1> sb(Mpad * TN, h), ob(Mpad * D, h), st(3 * M, h);
    sycl::local_accessor<sycl::uint2, 1> lut(256, h);
    h.parallel_for(sycl::nd_range<1>((size_t)s.B * s.HKV * nsplit * T, T),
                   SplitKernelXMX<D, TN, T>{s, a, nsplit, pps, qb, kvb, pb, sb, ob, st, lut});
  });
}

template <int D, int TN, int T>
sycl::event run_xmx_cfg(sycl::queue& q, const Shape& s, const Args& a, int nsplit) {
  return launch_reduce<D>(q, s, a, nsplit, launch_split_xmx<D, TN, T>(q, s, a, nsplit));
}

template <int D>
sycl::event run_xmx(sycl::queue& q, const Shape& s, const Cfg& c, const Args& a) {
#define UQ_XCASE(T_, TN_) if (c.wg == (T_) && c.tn == (TN_)) return run_xmx_cfg<D, TN_, T_>(q, s, a, c.nsplit);
  UQ_XCASE(128, 32) UQ_XCASE(128, 64) UQ_XCASE(128, 16) UQ_XCASE(64, 32) UQ_XCASE(64, 64) UQ_XCASE(256, 32) UQ_XCASE(256, 64)
  UQ_XCASE(32, 32) UQ_XCASE(32, 64) UQ_XCASE(64, 16) UQ_XCASE(256, 16) UQ_XCASE(512, 32) UQ_XCASE(512, 64)
#undef UQ_XCASE
  throw std::runtime_error("unsupported xmx (wg,tn) config");
}
}  // namespace uq
