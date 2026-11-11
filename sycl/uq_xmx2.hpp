// XMX v2: sub-group-private pipelines. Every sub-group owns its own KV token stream (its own split), its own SLM tile
// region and its own online-softmax state; there is NO work-group barrier in the hot loop (only one barrier after the
// shared Q tile is staged). O accumulators live in joint_matrix registers across tiles (rescaled with joint_matrix_apply),
// M rows = K*G stacked (MT = ceil(M/8) DPAS row tiles). Included from uq_decode.hpp after uq_xmx.hpp.
#pragma once
namespace uq {
namespace jm2 = sycl::ext::oneapi::experimental::matrix;

template <int D, int TN, int NSG, int MT>
struct SplitKernelX2 {
  Shape s; Args a; int nsplit, pps, nwsp;
  sycl::local_accessor<bf16, 1> qb, kvb, pb;   // Q [MT*8][D] (shared); KV NSG x [D*TN]; P NSG x [MT*8][TN]
  sycl::local_accessor<float, 1> sb, st;       // S NSG x [MT*8][TN]; st NSG x [MT*8] alpha / l
  sycl::local_accessor<sycl::uint2, 1> lut;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    constexpr int DH = D / 2, NG = D / 32, NT = TN / 16, DT = D / 16, MP = MT * 8, T = NSG * 16;
    constexpr float LOG2E = 1.4426950408889634f;
    const int grp = (int)it.get_group(0), t = (int)it.get_local_id(0);
    auto sg = it.get_sub_group();
    const int lane = (int)sg.get_local_linear_id(), sgid = (int)sg.get_group_linear_id();
    const int wsp = grp % nwsp, hk = (grp / nwsp) % s.HKV, b = grp / (nwsp * s.HKV);
    const int sp = wsp * NSG + sgid;
    const int K = s.K, G = s.HQ / s.HKV, M = K * G, HKV = s.HKV, PAGE = s.PAGE;
    const int seq_len = a.sl[b];
    const int n_pages = (seq_len + PAGE - 1) / PAGE;
    const float scl = s.sm_scale * LOG2E;

    bf16* Qb = &qb[0];
    bf16* KV = &kvb[0] + sgid * (D * TN);
    bf16* Pb = &pb[0] + sgid * (MP * TN);
    float* S = &sb[0] + sgid * (MP * TN);
    float* Al = &st[0] + sgid * (2 * MP);
    float* Ls = Al + MP;
    auto lp = [](bf16* p) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p); };
    auto lpf = [](float* p) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p); };
    uint32_t* KV32 = reinterpret_cast<uint32_t*>(KV);

    for (int i = t; i < 256; i += T) {
      const unsigned lo = (unsigned)i & 0xF, hi = (unsigned)i >> 4;
      const int bias = -(32 << 7);
      lut[i] = sycl::uint2(fp4_bf16bits(lo, bias) | (fp4_bf16bits(hi, bias) << 16),
                           ((lo & 7) ? 0xFFFFu : 0u) | ((hi & 7) ? 0xFFFF0000u : 0u));
    }
    for (int i = t; i < MP * D; i += T) {
      const int r = i / D, d = i % D, j = r / G, g = r % G;
      Qb[i] = r < M ? a.q[((size_t)(b * K + j) * s.HQ + hk * G + g) * D + d] : bf16(0.f);
    }
    for (int i = lane; i < MP * TN; i += 16) Pb[i] = bf16(0.f);
    sycl::group_barrier(it.get_group());
    if (sp >= nsplit) return;

    jm2::joint_matrix<sycl::sub_group, float, jm2::use::accumulator, 8, 16> O[MT][DT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int dt = 0; dt < DT; dt++) jm2::joint_matrix_fill(sg, O[mt][dt], 0.f);
    float m_run[MT], l_run[MT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) { m_run[mt] = -INFINITY; l_run[mt] = 0.f; }
    const int lr = lane >> 1, hf = lane & 1;

    const int p_lo = sp * pps, p_hi = sycl::min(p_lo + pps, n_pages);
    for (int p = p_lo; p < p_hi; p++) {
      const size_t cbase = ((size_t)a.bt[b * s.max_pages + p] * HKV + hk) * PAGE;
      for (int tile = 0; tile < PAGE / TN; tile++) {
        const int tok0 = p * PAGE + tile * TN;
        if (tok0 >= seq_len) break;  // uniform per sub-group
        // ---- K tile -> VNNI [D/2][TN] (lane = token) ----
        sycl::group_barrier(sg);
#ifndef X2_NOKD
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          const int n = nt * 16 + lane;
          const size_t row = cbase + tile * TN + n;
#pragma unroll
          for (int u = 0; u < DH / 16; u++) {
            const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(a.kc + row * DH + u * 16);
            const uint32_t off = (uint32_t)((sycl::max((int)a.ks[row * NG + u], 95) - 95) << 7);
            const uint32_t offp = off | (off << 16);
            const uint32_t w4[4] = {v.x(), v.y(), v.z(), v.w()};
#pragma unroll
            for (int j = 0; j < 16; j++) {
              const sycl::uint2 l = lut[(w4[j >> 2] >> (8 * (j & 3))) & 0xFFu];
              KV32[(u * 16 + j) * TN + n] = (l.x() + offp) & l.y();
            }
          }
        }
#endif
        sycl::group_barrier(sg);
        // ---- S = Q K^T ----
#ifndef X2_NOQK
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++) {
            jm2::joint_matrix<sycl::sub_group, float, jm2::use::accumulator, 8, 16> mc;
            jm2::joint_matrix_fill(sg, mc, 0.f);
#pragma unroll
            for (int kk = 0; kk < DT; kk++) {
              jm2::joint_matrix<sycl::sub_group, bf16, jm2::use::a, 8, 16, jm2::layout::row_major> ma;
              jm2::joint_matrix<sycl::sub_group, bf16, jm2::use::b, 16, 16, jm2::layout::ext_intel_packed> mb;
              jm2::joint_matrix_load(sg, ma, lp(Qb + mt * 8 * D + kk * 16), D);
              jm2::joint_matrix_load(sg, mb, lp(KV + (kk * 8) * (TN * 2) + nt * 32), TN * 2);
              jm2::joint_matrix_mad(sg, mc, ma, mb, mc);
            }
            jm2::joint_matrix_store(sg, mc, lpf(S + mt * 8 * TN + nt * 16), TN, jm2::layout::row_major);
          }
#endif
        sycl::group_barrier(sg);
#ifndef X2_NOSM
        // ---- online softmax: lane pair (lr, hf) owns row mt*8+lr, half hf of the TN columns ----
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          const int row = mt * 8 + lr;
          const int limit = seq_len - K + row / G + 1;
          float x[TN / 2];
          float mx = -INFINITY;
#pragma unroll
          for (int i = 0; i < TN / 2; i++) {
            const int n = hf * (TN / 2) + i;
            x[i] = (row < M && tok0 + n < limit) ? S[row * TN + n] * scl : -INFINITY;
            mx = sycl::fmax(mx, x[i]);
          }
          mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 1));
          const float m_new = sycl::fmax(m_run[mt], mx);
          const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
          const float alpha = sycl::exp2(m_run[mt] - m_safe);
          float sum = 0.f;
#pragma unroll
          for (int i = 0; i < TN / 2; i++) {
            const float pr = sycl::exp2(x[i] - m_safe);
            Pb[row * TN + hf * (TN / 2) + i] = bf16(pr); sum += pr;
          }
          sum += sycl::permute_group_by_xor(sg, sum, 1);
          l_run[mt] = l_run[mt] * alpha + sum; m_run[mt] = m_new;
          if (hf == 0) Al[row] = alpha;
        }
#endif
#ifndef X2_NOVD
        // ---- V tile -> VNNI over tokens [TN/2][D] ----
        for (int i = lane; i < (TN / 2) * (DH / 4); i += 16) {
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
#endif
        sycl::group_barrier(sg);
#ifndef X2_NOPV
        // ---- O = O*alpha + P V ----
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
          for (int dt = 0; dt < DT; dt++) {
            sycl::ext::intel::experimental::matrix::joint_matrix_apply(sg, O[mt][dt], [&](float& e, size_t r, size_t) { e *= Al[mt * 8 + r]; });
#pragma unroll
            for (int kk = 0; kk < NT; kk++) {
              jm2::joint_matrix<sycl::sub_group, bf16, jm2::use::a, 8, 16, jm2::layout::row_major> ma;
              jm2::joint_matrix<sycl::sub_group, bf16, jm2::use::b, 16, 16, jm2::layout::ext_intel_packed> mb;
              jm2::joint_matrix_load(sg, ma, lp(Pb + mt * 8 * TN + kk * 16), TN);
              jm2::joint_matrix_load(sg, mb, lp(KV + (kk * 8) * (D * 2) + dt * 32), D * 2);
              jm2::joint_matrix_mad(sg, O[mt][dt], ma, mb, O[mt][dt]);
            }
          }
#endif
      }
    }
    // ---- epilogue ----
    const size_t base = ((size_t)(b * HKV + hk) * nsplit + sp) * M;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      const int row = mt * 8 + lr;
      if (hf == 0) {
        Ls[row] = l_run[mt];
        if (row < M) a.ml[base + row] = l_run[mt] > 0.f ? m_run[mt] + sycl::log2(sycl::fmax(l_run[mt], 1e-30f)) : -INFINITY;
      }
    }
    sycl::group_barrier(sg);
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int dt = 0; dt < DT; dt++)
        sycl::ext::intel::experimental::matrix::joint_matrix_apply(sg, O[mt][dt], [&](float& e, size_t r, size_t c) {
          const int row = mt * 8 + (int)r;
          if (row < M) a.mo[(base + row) * D + dt * 16 + c] = e / sycl::fmax(Ls[row], 1e-30f);
        });
  }
};

template <int D, int TN, int NSG, int MT>
sycl::event launch_split_x2(sycl::queue& q, const Shape& s, const Args& a, int nsplit) {
  constexpr int MP = MT * 8;
  const int pps = (s.max_pages + nsplit - 1) / nsplit;
  const int nwsp = (nsplit + NSG - 1) / NSG;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<bf16, 1> qb(MP * D, h), kvb(NSG * D * TN, h), pb(NSG * MP * TN, h);
    sycl::local_accessor<float, 1> sb(NSG * MP * TN, h), st(NSG * 2 * MP, h);
    sycl::local_accessor<sycl::uint2, 1> lut(256, h);
    h.parallel_for(sycl::nd_range<1>((size_t)s.B * s.HKV * nwsp * NSG * 16, NSG * 16),
                   SplitKernelX2<D, TN, NSG, MT>{s, a, nsplit, pps, nwsp, qb, kvb, pb, sb, st, lut});
  });
}

template <int D, int TN, int NSG, int MT>
sycl::event run_x2_cfg(sycl::queue& q, const Shape& s, const Args& a, int nsplit) {
  return launch_reduce<D>(q, s, a, nsplit, launch_split_x2<D, TN, NSG, MT>(q, s, a, nsplit));
}

template <int D, int MT>
sycl::event run_x2_mt(sycl::queue& q, const Shape& s, const Cfg& c, const Args& a) {
  const int nsg = c.wg / 16;
#define UQ_X2(N_, T_) if (nsg == (N_) && c.tn == (T_)) return run_x2_cfg<D, T_, N_, MT>(q, s, a, c.nsplit);
  UQ_X2(1, 16) UQ_X2(2, 16) UQ_X2(4, 16) UQ_X2(8, 16) UQ_X2(1, 32) UQ_X2(2, 32) UQ_X2(4, 32) UQ_X2(8, 32) UQ_X2(16, 16) UQ_X2(16, 32)
#undef UQ_X2
  throw std::runtime_error("unsupported xmx2 (wg,tn) config");
}

template <int D>
sycl::event run_xmx2(sycl::queue& q, const Shape& s, const Cfg& c, const Args& a) {
  const int mt = (s.M() + 7) / 8;
  if (mt == 1) return run_x2_mt<D, 1>(q, s, c, a);
  if (mt == 2) return run_x2_mt<D, 2>(q, s, c, a);
  if (mt <= 4) return run_x2_mt<D, 4>(q, s, c, a);
  Cfg c1 = c; c1.wg = 256; c1.tn = c.tn;  // big M: fall back to the barrier-based path
  return run_xmx<D>(q, s, c1, a);
}
}  // namespace uq
