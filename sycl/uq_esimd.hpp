// ESIMD/DPAS path (Intel Xe2/Xe3): explicit-register kernel, one hardware thread per (batch, kv-head, split).
//  * KV bytes are read with plain 256B block loads in "lane = dword" layout (full cache lines, no 2D/gather loads).
//  * FP4 -> bf16 is a 256-entry byte LUT that lives in 16 GRFs and is indexed with register-indirect addressing
//    (simd::iselect); the UE8M0 block scale is folded in as a 16-bit exponent add on the looked-up words (zero codes decode
//    to +-2^-100 so the add can never create garbage; scale byte clamped to [105,250]); K, V and P never touch SLM except
//    the tiny P transpose and the shared Q^T operand.
//  * S^T = K Q^T on DPAS (tokens = M rows, queries = N lanes), online softmax is lane-parallel over queries (no cross-lane
//    reductions), O = P V on DPAS with V decoded straight into the VNNI B operand (pairs of tokens per 32-bit word).
//  * No work-group barrier in the hot loop (one barrier after staging Q^T).
// Supports D=128, PAGE % 16 == 0, K*G <= 8 (RC = K*G); other shapes fall back to the joint_matrix path.
#pragma once
#include <sycl/ext/intel/esimd.hpp>
namespace uq {
namespace esx = sycl::ext::intel::esimd;
namespace xmxe = sycl::ext::intel::esimd::xmx;

namespace eu {
using esx::simd; using esx::simd_mask;
constexpr unsigned EPS_BITS = 0x0D80u;            // 2^-100 as bf16 bits (exponent field 27)
inline simd<uint32_t, 256> build_lut() {            // byte -> (bf16(lo code) | bf16(hi code) << 16), unscaled, zero -> +-2^-100
  simd<uint16_t, 256> idx(0, 1);
  simd<uint16_t, 256> lo = idx & 15, hi = idx >> 4;
  auto f = [](simd<uint16_t, 256> c) {
    simd<uint16_t, 256> m = c & 7;
    simd<uint16_t, 256> mag = (m << 6) + 0x3F00;
    mag.merge(simd<uint16_t, 256>(0x3F00), m == 1);
    mag.merge(simd<uint16_t, 256>(EPS_BITS), m == 0);
    return simd<uint16_t, 256>(mag | ((c & 8) << 12));
  };
  simd<uint32_t, 256> r = esx::convert<uint32_t>(f(lo)) | (esx::convert<uint32_t>(f(hi)) << 16);
  return r;
}
// scale-exponent offset table SG[g*16 + t] (uint16) for 16 tokens whose 4 group scale bytes are packed in sd[t]
inline simd<uint16_t, 64> scale_table(simd<uint32_t, 16> sd) {
  simd<uint16_t, 64> sg;
#pragma unroll
  for (int g = 0; g < 4; g++) {
    simd<uint32_t, 16> s = (sd >> (8 * g)) & 0xFFu;
    s = esx::max(s, simd<uint32_t, 16>(105u)); s = esx::min(s, simd<uint32_t, 16>(250u));
    simd<uint32_t, 16> o = (s << 7) - (127u << 7);
    sg.template select<16, 1>(g * 16) = esx::convert<uint16_t>(o & 0xFFFFu);
  }
  return sg;
}
// per-16-bit-lane add (no carry between the two halves of a 32-bit word)
inline simd<uint32_t, 16> add16(simd<uint32_t, 16> a, simd<uint32_t, 16> b) {
  simd<uint16_t, 32> r = a.template bit_cast_view<uint16_t>() + b.template bit_cast_view<uint16_t>();
  return r.template bit_cast_view<uint32_t>();
}
}  // namespace eu

template <int RC, int MT, int NTH>
struct EsimdKernel {
  Shape s; Args a; int nsplit, pps, nwsp;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    using namespace eu;
    constexpr int D = 128, DH = 64, SLM_Q = 4096, SLM_P = 512;
    esx::slm_init<SLM_Q + NTH * SLM_P>();
    const int grp = (int)it.get_group(0), tid = (int)it.get_local_id(0);
    const int nqg = (s.M() + 15) / 16;
    const int qg = grp % nqg, wsp = (grp / nqg) % nwsp, hk = (grp / (nqg * nwsp)) % s.HKV, b = grp / (nqg * nwsp * s.HKV);
    const int sp = wsp * NTH + tid;
    const int K = s.K, G = s.HQ / s.HKV, MT_ = s.M(), HKV = s.HKV, PAGE = s.PAGE;
    const int rq0 = qg * 16, M = (MT_ - rq0 < 16 ? MT_ - rq0 : 16);   // query rows handled by this thread (<= 16)
    const int seq_len = a.sl[b];
    const int n_pages = (seq_len + PAGE - 1) / PAGE;
    const float scl = s.sm_scale * 1.4426950408889634f;

    // ---- stage Q^T (B operands of S^T = K Q^T) in SLM: B_kk[kp][r] = Q_row_r dword (32*(kk&1) + 4*kp + (kk>>1)); thread tid does kk = tid.. ----
    {
      simd<uint32_t, 16> rowoff; simd_mask<16> rm;
#pragma unroll
      for (int r = 0; r < 16; r++) {
        const int j = (rq0 + r) / G, g = (rq0 + r) % G;
        rowoff[r] = r < M ? (uint32_t)(((size_t)(b * K + j) * s.HQ + hk * G + g) * (D / 2) * 4) : 0u;
        rm[r] = r < M;
      }
      const uint8_t* qb = reinterpret_cast<const uint8_t*>(a.q);
      for (int kk = tid; kk < 8; kk += NTH) {
#pragma unroll
        for (int kp = 0; kp < 8; kp++) {
          const int dw = 32 * (kk & 1) + 4 * kp + (kk >> 1);
          simd<uint32_t, 16> v = esx::gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(qb) + dw, rowoff, rm, simd<uint32_t, 16>(0u));
          esx::slm_block_store<uint32_t, 16>(kk * 512 + kp * 64, v);
        }
      }
    }
    esx::barrier();
    if (sp >= nsplit) return;

    simd<uint32_t, 256> lut = build_lut();
    simd<float, RC * 16> O[MT][8];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
      for (int j = 0; j < 8; j++) O[m][j] = 0.f;
    simd<float, 16> m_run(-INFINITY), l_run(0.f);
    simd<uint16_t, 16> lane_i(0, 1);
    simd<uint16_t, 16> gbase = (lane_i >> 2) * 16;          // lane c -> scale group c/4, table stride 16
    simd<int, 16> limit;
#pragma unroll
    for (int r = 0; r < 16; r++) limit[r] = r < M ? seq_len - K + (rq0 + r) / G + 1 : 0x7fffffff;
    simd<uint32_t, RC * 8> gofs[MT];                        // A_P[r][kp] <- SLM P image [kp][r]
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
      for (int r = 0; r < RC; r++)
#pragma unroll
        for (int kp = 0; kp < 8; kp++) gofs[m][r * 8 + kp] = (uint32_t)(SLM_Q + tid * SLM_P + (kp * 16 + m * RC + r) * 4);

    const uint32_t* kc32 = reinterpret_cast<const uint32_t*>(a.kc);
    const uint32_t* vc32 = reinterpret_cast<const uint32_t*>(a.vc);
    const uint32_t* ks32 = reinterpret_cast<const uint32_t*>(a.ks);
    const uint32_t* vs32 = reinterpret_cast<const uint32_t*>(a.vs);
    const int p_lo = sp * pps, p_hi = (p_lo + pps < n_pages ? p_lo + pps : n_pages);
    const int tpp = PAGE / 16;
    const int ntiles = (p_hi > p_lo ? (p_hi - p_lo) : 0) * tpp;
#if defined(UQ_PD1)
#define UQ_ESIMD_PD 1
#elif defined(UQ_PD2)
#define UQ_ESIMD_PD 2
#elif defined(UQ_PD4)
#define UQ_ESIMD_PD 4
#elif defined(UQ_PD6)
#define UQ_ESIMD_PD 6
#elif defined(UQ_PD8)
#define UQ_ESIMD_PD 8
#else
#define UQ_ESIMD_PD 3
#endif
    const auto PFP = sycl::ext::oneapi::experimental::properties{esx::cache_hint_L1<esx::cache_hint::cached>, esx::cache_hint_L2<esx::cache_hint::cached>};
    auto row_of = [&](int it) SYCL_ESIMD_FUNCTION -> size_t {
      const int pg = p_lo + it / tpp;
      return ((size_t)a.bt[b * s.max_pages + pg] * HKV + hk) * PAGE + (it % tpp) * 16;
    };
    auto prefetch_tile = [&](int it) SYCL_ESIMD_FUNCTION {
      const size_t rr = row_of(it);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        esx::prefetch<uint32_t, 64>(kc32 + rr * 16 + i * 64, PFP);
        esx::prefetch<uint32_t, 64>(vc32 + rr * 16 + i * 64, PFP);
      }
      esx::prefetch<uint32_t, 16>(ks32 + rr, PFP);
      esx::prefetch<uint32_t, 16>(vs32 + rr, PFP);
    };
#pragma unroll
    for (int d = 0; d < UQ_ESIMD_PD; d++) if (d < ntiles) prefetch_tile(d);
    for (int it = 0; it < ntiles; it++) {
      {
        if (it + UQ_ESIMD_PD < ntiles) prefetch_tile(it + UQ_ESIMD_PD);
        const int p = p_lo + it / tpp, tile = it % tpp;
        const int tok0 = p * PAGE + tile * 16;
        if (tok0 >= seq_len) break;
        const size_t row0 = row_of(it);
        // ---- raw loads ----
        simd<uint32_t, 256> Kr;
#ifdef E_NOLOAD
        Kr = simd<uint32_t, 256>(0x12345678u + (uint32_t)tile, 0x01010101u);
        simd<uint16_t, 64> SGK = scale_table(simd<uint32_t, 16>(0x7E7E7E7Eu));
#else
#pragma unroll
        for (int i = 0; i < 4; i++) {
          Kr.template select<64, 1>(i * 64) = esx::block_load<uint32_t, 64>(kc32 + row0 * 16 + i * 64);
        }
        simd<uint16_t, 64> SGK = scale_table(esx::block_load<uint32_t, 16>(ks32 + row0));
#endif
#ifdef E_LOADONLY
        {
          simd<uint32_t, 256> Vq;
#pragma unroll
          for (int i = 0; i < 4; i++) Vq.template select<64, 1>(i * 64) = esx::block_load<uint32_t, 64>(vc32 + row0 * 16 + i * 64);
          simd<uint32_t, 16> sv = esx::block_load<uint32_t, 16>(vs32 + row0);
          simd<uint32_t, 256> z = Kr ^ Vq; simd<uint32_t, 16> y = z.template select<16, 1>(0);
#pragma unroll
          for (int t = 1; t < 16; t++) y ^= z.template select<16, 1>(t * 16);
          y ^= sv ^ simd<uint32_t, 16>(SGK.template bit_cast_view<uint32_t>().template select<16, 1>(0));
          simd<float, RC * 16> yf = y.template bit_cast_view<float>().template replicate_w<RC, 16>(0);
          O[0] += yf;
        }
        continue;
#endif
        // ---- S^T = K Q^T (2 x 8 tokens) ----
        simd<float, 128> acc0 = 0.f, acc1 = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          simd<uint32_t, 64> Aa[2][2];                         // [token half][hp]: A operand of kk = 2i + hp for tokens 8h..8h+7
#pragma unroll
          for (int t = 0; t < 16; t++) {
            const int h = t / 8, tt = t % 8;
            simd<uint32_t, 16> R = Kr.template select<16, 1>(t * 16);
            simd<uint16_t, 32> v = R.template bit_cast_view<uint16_t>();
            simd<uint16_t, 16> half = (i < 2) ? simd<uint16_t, 16>(v.template select<16, 2>(0)) : simd<uint16_t, 16>(v.template select<16, 2>(1));
            simd<uint16_t, 16> idx = (i & 1) ? (half >> 8) : (half & 0xFF);
#ifdef E_NOKD
            simd<uint32_t, 16> g = R;
#else
            simd<uint32_t, 16> g = lut.iselect(idx);
#endif
            simd<uint16_t, 32> g16 = g.template bit_cast_view<uint16_t>();
            simd<uint16_t, 16> o16 = SGK.template replicate_vs_w_hs<4, 16, 4, 0>(t);   // lane w -> scale offset of group w/4
#pragma unroll
            for (int hp = 0; hp < 2; hp++) {
              simd<uint16_t, 128> A16 = Aa[h][hp].template bit_cast_view<uint16_t>();
              A16.template select<8, 2>(tt * 16) = g16.template select<8, 2>(hp * 16) + o16.template select<8, 1>(hp * 8);
              A16.template select<8, 2>(tt * 16 + 1) = g16.template select<8, 2>(hp * 16 + 1) + o16.template select<8, 1>(hp * 8);
              Aa[h][hp] = A16.template bit_cast_view<uint32_t>();
            }
          }
#pragma unroll
          for (int hp = 0; hp < 2; hp++) {
            const int kk = 2 * i + hp;
            simd<uint32_t, 128> B = esx::slm_block_load<uint32_t, 128>(kk * 512);
#ifdef E_NOQK
            { simd<float, 64> af0 = Aa[0][hp].template bit_cast_view<float>(), af1 = Aa[1][hp].template bit_cast_view<float>(); simd<float, 128> bf = B.template bit_cast_view<float>(); acc0.template select<64, 1>(0) += af0 + bf.template select<64, 1>(0); acc1.template select<64, 1>(0) += af1 + bf.template select<64, 1>(64); }
#else
            acc0 = xmxe::dpas<8, 8, float, float, uint32_t, uint32_t, xmxe::dpas_argument_type::bf16, xmxe::dpas_argument_type::bf16>(acc0, B, Aa[0][hp]);
            acc1 = xmxe::dpas<8, 8, float, float, uint32_t, uint32_t, xmxe::dpas_argument_type::bf16, xmxe::dpas_argument_type::bf16>(acc1, B, Aa[1][hp]);
#endif
          }
        }
        // ---- online softmax over the 16 tokens (rows of S^T), lane = query ----
        simd<float, 16> x[16];
#pragma unroll
        for (int t = 0; t < 16; t++) {
          x[t] = (t < 8) ? simd<float, 16>(acc0.template select<16, 1>(t * 16)) : simd<float, 16>(acc1.template select<16, 1>((t - 8) * 16));
        }
        if (tok0 + 16 > seq_len - K + 1) {   // tail tile: causal / length mask
#pragma unroll
          for (int t = 0; t < 16; t++) x[t].merge(simd<float, 16>(-INFINITY), simd<int, 16>(tok0 + t) >= limit);
        }
        simd<float, 16> mx = x[0];
#pragma unroll
        for (int t = 1; t < 16; t++) mx = esx::max(mx, x[t]);
        simd<float, 16> m_new = esx::max(m_run, mx);
        simd<float, 16> m_safe = m_new;
        m_safe.merge(simd<float, 16>(0.f), m_new == simd<float, 16>(-INFINITY));
        simd<float, 16> alpha = esx::exp2(simd<float, 16>((m_run - m_safe) * scl));
        simd<float, 16> msc = m_safe * scl, sum = 0.f;
        simd<uint32_t, 16> pk[8];
#pragma unroll
        for (int t = 0; t < 16; t += 2) {
#ifdef E_NOSM
          simd<float, 16> p0 = simd<float, 16>(x[t] * scl - msc), p1 = simd<float, 16>(x[t + 1] * scl - msc);
#else
          simd<float, 16> p0 = esx::exp2(simd<float, 16>(x[t] * scl - msc)), p1 = esx::exp2(simd<float, 16>(x[t + 1] * scl - msc));
#endif
          sum += p0 + p1;
          simd<uint32_t, 16> b0 = p0.template bit_cast_view<uint32_t>(), b1 = p1.template bit_cast_view<uint32_t>();
          b0 += 0x7FFFu + ((b0 >> 16) & 1u); b1 += 0x7FFFu + ((b1 >> 16) & 1u);
          pk[t / 2] = (b0 >> 16) | (b1 & 0xFFFF0000u);
        }
        l_run = l_run * alpha + sum; m_run = m_new;
#ifdef E_NOSLMP
        simd<uint32_t, RC * 8> AP[MT];
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
          for (int kp = 0; kp < 8; kp++) AP[m].template select<RC, 1>(0) = pk[kp].template select<RC, 1>(0) + (uint32_t)kp;
#else
#pragma unroll
        for (int kp = 0; kp < 8; kp++) esx::slm_block_store<uint32_t, 16>(SLM_Q + tid * SLM_P + kp * 64, pk[kp]);
        simd<uint32_t, RC * 8> AP[MT];
#pragma unroll
        for (int m = 0; m < MT; m++) AP[m] = esx::slm_gather<uint32_t, RC * 8>(gofs[m]);
#endif
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
          for (int j = 0; j < 8; j++)
#pragma unroll
            for (int r = 0; r < RC; r++) { const float ar = alpha[m * RC + r]; O[m][j].template select<16, 1>(r * 16) *= ar; }
        // ---- V decode (pairs of tokens per word) and O += P V ----
        simd<uint32_t, 256> Vr;
#ifdef E_NOLOAD
        Vr = Kr + 0x11111111u;
        simd<uint16_t, 64> SGV = SGK;
#else
#pragma unroll
        for (int i = 0; i < 4; i++) Vr.template select<64, 1>(i * 64) = esx::block_load<uint32_t, 64>(vc32 + row0 * 16 + i * 64);
        simd<uint16_t, 64> SGV = scale_table(esx::block_load<uint32_t, 16>(vs32 + row0));
#endif
        simd<uint32_t, 16> xs[8], ys[8], offp[8];
#pragma unroll
        for (int pr = 0; pr < 8; pr++) {
          simd<uint32_t, 16> R0 = Vr.template select<16, 1>(2 * pr * 16), R1 = Vr.template select<16, 1>((2 * pr + 1) * 16);
          xs[pr] = (R0 & 0x0F0F0F0Fu) | ((R1 & 0x0F0F0F0Fu) << 4);
          ys[pr] = ((R0 >> 4) & 0x0F0F0F0Fu) | (R1 & 0xF0F0F0F0u);
          simd<uint16_t, 32> o2;
          o2.template select<16, 2>(0) = SGV.template replicate_vs_w_hs<4, 16, 4, 0>(2 * pr);
          o2.template select<16, 2>(1) = SGV.template replicate_vs_w_hs<4, 16, 4, 0>(2 * pr + 1);
          offp[pr] = o2.template bit_cast_view<uint32_t>();
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
          simd<uint32_t, 128> B;
#pragma unroll
          for (int pr = 0; pr < 8; pr++) {
            simd<uint32_t, 16> src = (j & 1) ? ys[pr] : xs[pr];
            simd<uint16_t, 32> v = src.template bit_cast_view<uint16_t>();
            simd<uint16_t, 16> half = (j >> 1) < 2 ? simd<uint16_t, 16>(v.template select<16, 2>(0)) : simd<uint16_t, 16>(v.template select<16, 2>(1));
            simd<uint16_t, 16> idx = ((j >> 1) & 1) ? (half >> 8) : (half & 0xFF);
            #ifdef E_NOVD
            B.template select<16, 1>(pr * 16) = src + offp[pr];
#else
            {
              simd<uint32_t, 16> g = lut.iselect(idx);
              simd<uint16_t, 32> g16 = g.template bit_cast_view<uint16_t>(), o16 = offp[pr].template bit_cast_view<uint16_t>();
              simd<uint16_t, 32> r16 = g16 + o16;
              B.template select<16, 1>(pr * 16) = r16.template bit_cast_view<uint32_t>();
            }
#endif
          }
#pragma unroll
          for (int m = 0; m < MT; m++) {
#ifdef E_NOPV
            O[m][j] += simd<float, RC * 16>(B.template bit_cast_view<float>().template select<RC * 16, 1>(0));
#else
            O[m][j] = xmxe::dpas<8, RC, float, float, uint32_t, uint32_t, xmxe::dpas_argument_type::bf16, xmxe::dpas_argument_type::bf16>(O[m][j], B, AP[m]);
#endif
          }
        }
      }
    }
    // ---- epilogue: O_j[r][c] is d = 8c + j ----
    const size_t base = ((size_t)(b * HKV + hk) * nsplit + sp) * MT_ + rq0;
#pragma unroll
    for (int rr = 0; rr < RC * MT; rr++) {
      const int r = rr;
      if (r < M) {
        const float l = l_run[r];
        const float inv = 1.f / (l > 1e-30f ? l : 1e-30f);
        simd<uint32_t, 16> offs(0, 32);   // c*8 floats in bytes
#pragma unroll
        for (int j = 0; j < 8; j++) {
          simd<float, 16> v = O[r / RC][j].template select<16, 1>((r % RC) * 16) * inv;
          esx::scatter<float, 16>(a.mo + (base + r) * D + j, offs, v);
        }
        { const float mr = m_run[r]; float lse = -INFINITY; if (l > 0.f) { simd<float, 1> lv(l > 1e-30f ? l : 1e-30f); lse = mr * scl + (float)esx::log2(lv)[0]; } a.ml[base + r] = lse; }
      }
    }
  }
};

template <int RC, int MT, int NTH>
sycl::event launch_split_esimd(sycl::queue& q, const Shape& s, const Args& a, int nsplit) {
  const int pps = (s.max_pages + nsplit - 1) / nsplit;
  const int nwsp = (nsplit + NTH - 1) / NTH;
  return q.submit([&](sycl::handler& h) {
        if constexpr (MT == 1) {
      h.parallel_for(sycl::nd_range<1>((size_t)s.B * s.HKV * nwsp * ((s.M() + 15) / 16) * NTH, NTH), EsimdKernel<RC, MT, NTH>{s, a, nsplit, pps, nwsp});
    } else {   // O accumulators need >128 GRFs
      h.parallel_for(sycl::nd_range<1>((size_t)s.B * s.HKV * nwsp * ((s.M() + 15) / 16) * NTH, NTH),
                     sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, EsimdKernel<RC, MT, NTH>{s, a, nsplit, pps, nwsp});
    }
  });
}
template <int RC, int MT, int NTH>
sycl::event run_esimd_cfg(sycl::queue& q, const Shape& s, const Args& a, int nsplit) {
  return launch_reduce<128>(q, s, a, nsplit, launch_split_esimd<RC, MT, NTH>(q, s, a, nsplit));
}
inline bool esimd_supported(const Shape& s) { return s.D == 128 && s.PAGE % 16 == 0 && s.M() <= 64 && (s.HQ % s.HKV) == 0; }
inline sycl::event run_esimd(sycl::queue& q, const Shape& s, const Cfg& c, const Args& a) {
  const int nth = c.wg / 16 > 0 ? c.wg / 16 : 1;   // threads per work-group (cfg.wg = 16*threads, like the other paths)
  const int M = s.M();
#define UQ_E(RC_, MT_, N_) if (nth == (N_) && (M <= 16 ? M : 16) <= (RC_) * (MT_)) return run_esimd_cfg<RC_, MT_, N_>(q, s, a, c.nsplit);
  UQ_E(4, 1, 2) UQ_E(4, 1, 4) UQ_E(4, 1, 8) UQ_E(8, 1, 2) UQ_E(8, 1, 4) UQ_E(8, 1, 8) UQ_E(8, 2, 2) UQ_E(8, 2, 4) UQ_E(8, 2, 8)
#undef UQ_E
  throw std::runtime_error("unsupported esimd config");
}
}  // namespace uq
