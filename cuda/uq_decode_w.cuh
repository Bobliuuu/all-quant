// Design W: warp-independent split-KV kernel.  Included from uq_decode.cu.
// Each warp owns a key slice (KPW keys) of every page and has its OWN cp.async ring: no CTA-wide barriers
// in the main loop.  K is dequantised straight into the mma B-fragment registers (no smem round trip, no
// ldmatrix) using a permuted reduction order: thread t of a quad owns the 16 bytes (= one UE8M0 group) of
// its key row, the A operand (Q) is gathered with the same permutation.  V is dequantised into a warp
// private bf16 tile (column-permuted, undone in the epilogue) and read with ldmatrix.trans.
// Each warp may cover MB 16-row blocks of the query tile (K fragments are reused across them).

// pair i of 32-bit word w = bf16x2(e_i, e_{i+4}) * 2^-126  (sign/magnitude bit placement)
DEV uint32_t pair_raw(uint32_t w, int i) {
  uint32_t sg = w << (12 - 4 * i);
  uint32_t mg = (6 - 4 * i >= 0) ? (w << (6 - 4 * i)) : (w >> (4 * i - 6));
  return (sg & 0x80008000u) | (mg & 0x01C001C0u);
}
DEV void deq_word(uint32_t w, uint32_t s, uint32_t* o) {   // o[4]
  #pragma unroll
  for (int i = 0; i < 4; i++) o[i] = pair_raw(w, i);
  if (s <= 128u) {
    uint32_t f = (s + 126u) << 7; f |= f << 16;
    #pragma unroll
    for (int i = 0; i < 4; i++) o[i] = mul_bf16x2(o[i], f);
  } else {
    uint32_t f1 = 0x7E807E80u, f2 = s << 7; f2 |= f2 << 16;
    #pragma unroll
    for (int i = 0; i < 4; i++) o[i] = mul_bf16x2(mul_bf16x2(o[i], f1), f2);
  }
}

template <int D, int PAGE, int BM, int MB, int NW, int ST>
struct CfgW {
  static constexpr int DH = D / 2, NG = D / 32, LD = D + 8, BPT = DH / 4, WPT = BPT / 4, STEPS = D / 16;
  static constexpr int NWM = BM / (16 * MB);
  static constexpr int NWK = (NWM > 0 && NW / NWM > 0) ? NW / NWM : 1;
  // PS (page-split): warps of a row group take distinct pages (KPW = PAGE <= 64); else key-split a 128-key page in halves
  static constexpr bool PS = PAGE <= 64;
  static constexpr int KPW = PS ? PAGE : 64;
  static constexpr int NT = KPW / 8;
  static constexpr int SLICE = 2 * KPW * (DH + NG);
  static constexpr int VT = 0;   // V is dequantised straight into B fragments (no smem tile)
  static constexpr int CPT = D / 8, VWC = CPT / 8 > 0 ? CPT / 8 : 1;   // d-columns / 32-bit words per thread
  static constexpr int WSM = ST * SLICE + VT;
  static constexpr int CS = D + 4;
  static constexpr int COMB = NWK * BM * CS * 4 + 2 * NWK * BM * 4;
  static constexpr int SMEM = (NW * WSM > COMB) ? NW * WSM : COMB;
  static constexpr bool OK = (BM % (16 * MB) == 0) && NWM >= 1 && (NW % NWM == 0) && (PS || NWK == PAGE / 64) &&
                             (KPW % 16 == 0) && (KPW >= 16) && (D == 64 || D == 128) && (MB <= 2 || NW <= 4) && SMEM <= 227 * 1024;
};

#ifdef UQ_PROF
__device__ unsigned long long g_prof[8];
#define PROF_T(i) do { if (lane == 0 && warp == 0) { long long _n = clock64(); pacc[i] += _n - _tp; _tp = _n; } } while (0)
#else
#define PROF_T(i) do {} while (0)
#endif
template <int D, int PAGE, int BM, int MB, int NW, int ST>
__global__ void __launch_bounds__(NW * 32)
uq_split_w(const UQParams a, int G, int gp_shift) {
  using C = CfgW<D, PAGE, BM, MB, NW, ST>;
  constexpr int DH = C::DH, NG = C::NG, LD = C::LD, BPT = C::BPT, WPT = C::WPT, STEPS = C::STEPS;
  constexpr int NWM = C::NWM, NWK = C::NWK, KPW = C::KPW, NT = C::NT;
  constexpr int T = NW * 32;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp % NWM, wk = warp / NWM;
  const int g = lane >> 2, t = lane & 3;
  const int b = blockIdx.x, hk = blockIdx.y, sp = blockIdx.z;
  const int K = a.K, HKV = a.HKV, HQ = a.HQ, nsplit = a.nsplit;
  const int seq_len = a.seq_lens[b];
  const int n_pages = (seq_len + PAGE - 1) / PAGE;
  const int pps = (n_pages + nsplit - 1) / nsplit;
  const int p_lo = sp * pps;
  const int p_hi = min(p_lo + pps, n_pages);
  const size_t rowbase = ((size_t)(b * HKV + hk) * nsplit + sp) * BM;
  if (p_lo >= p_hi) {
    for (int r = tid; r < BM; r += T) a.ml[rowbase + r] = -INFINITY;
    return;
  }
  uint8_t* wsm = smem + (size_t)warp * C::WSM;
  const int32_t* bt = a.block_table + (size_t)b * a.max_pages;
  const int gm = (1 << gp_shift) - 1;
  auto load_slice = [&](size_t page, int s) {
    uint8_t* dst = wsm + s * C::SLICE;
    size_t cb = (page * HKV + hk) * PAGE + (C::PS ? 0 : wk * KPW);
    const uint8_t* src[4] = {a.kc + cb * DH, a.ks + cb * NG, a.vc + cb * DH, a.vs + cb * NG};
    constexpr int sz[4] = {KPW * DH, KPW * NG, KPW * DH, KPW * NG};
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      for (int c = lane; c < sz[u] / 16; c += 32) cp_async16(smem_u32(dst + c * 16), src[u] + c * 16);
      dst += sz[u];
    }
  };
  const int nwp = p_hi - p_lo;
  const int nit = C::PS ? (nwp - wk + NWK - 1) / NWK : nwp;   // iterations of this warp (may be <= 0)
  auto page_of = [&](int i) { return p_lo + (C::PS ? wk + i * NWK : i); };   // page of iteration i
  #pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < nit) load_slice((size_t)bt[page_of(s)], s);
    cp_commit();
  }
  int pg_next = (ST - 1 < nit) ? bt[page_of(ST - 1)] : 0;
  // ---- Q fragments with the permuted reduction order (global loads, one time)
  uint32_t qa[MB][STEPS][4];
  {
    const uint16_t* q = (const uint16_t*)a.q;
    #pragma unroll
    for (int mb = 0; mb < MB; mb++) {
      #pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        int r = (wm * MB + mb) * 16 + g + h2 * 8;
        int j = r >> gp_shift, gg = r & gm;
        bool ok = j < K && gg < G;
        const uint16_t* qr = q + ((size_t)(b * K + j) * HQ + hk * G + gg) * D;
        #pragma unroll
        for (int ss = 0; ss < STEPS; ss++) {
          int d0 = t * 2 * BPT + 8 * (ss >> 1) + 2 * (ss & 1);
          uint32_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;
          if (ok) { v0 = qr[d0]; v1 = qr[d0 + 4]; v2 = qr[d0 + 1]; v3 = qr[d0 + 5]; }
          qa[mb][ss][h2 == 0 ? 0 : 1] = v0 | (v1 << 16);
          qa[mb][ss][h2 == 0 ? 2 : 3] = v2 | (v3 << 16);
        }
      }
    }
  }
  int lim[MB][2];
  #pragma unroll
  for (int mb = 0; mb < MB; mb++)
    #pragma unroll
    for (int h2 = 0; h2 < 2; h2++) {
      int r = (wm * MB + mb) * 16 + g + h2 * 8;
      int j = r >> gp_shift, gg = r & gm;
      lim[mb][h2] = (j < K && gg < G) ? seq_len - K + j + 1 : 0;
    }
  const float sc = a.sm_scale * 1.4426950408889634f;
  float mrow[MB][2], lrow[MB][2];
  float acc[MB][D / 8][4];
  #pragma unroll
  for (int mb = 0; mb < MB; mb++) {
    mrow[mb][0] = mrow[mb][1] = -INFINITY; lrow[mb][0] = lrow[mb][1] = 0.f;
    #pragma unroll
    for (int i = 0; i < D / 8; i++) acc[mb][i][0] = acc[mb][i][1] = acc[mb][i][2] = acc[mb][i][3] = 0.f;
  }
#ifdef UQ_PROF
  long long pacc[6] = {0, 0, 0, 0, 0, 0}; long long _tp = clock64();
#endif
  for (int it = 0; it < nit; it++) {
    const int p = page_of(it);
    PROF_T(5);
    cp_wait<ST - 2>();
    __syncwarp();
    { if (it + ST - 1 < nit) load_slice((size_t)pg_next, (it + ST - 1) % ST); cp_commit();
      if (it + ST < nit) pg_next = bt[page_of(it + ST)]; }
    PROF_T(0);
#if UQ_ABL & 4
    if (it + 1 < nit) continue;   // load-only ablation: skip all compute
#endif
    const uint8_t* st = wsm + (it % ST) * C::SLICE;
    const uint8_t* kr = st; const uint8_t* ksc = st + KPW * DH;
    const uint8_t* vr = ksc + KPW * NG; const uint8_t* vsc = vr + KPW * DH;
    // ---- S = Q K^T, K dequantised straight into B fragments
    float s[MB][NT][4], s2[MB][NT][4];
    #pragma unroll
    for (int mb = 0; mb < MB; mb++)
      #pragma unroll
      for (int i = 0; i < NT; i++) { s[mb][i][0] = s[mb][i][1] = s[mb][i][2] = s[mb][i][3] = 0.f; s2[mb][i][0] = s2[mb][i][1] = s2[mb][i][2] = s2[mb][i][3] = 0.f; }
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      int key = nt * 8 + g;
      uint32_t w[WPT];
      if constexpr (WPT == 4) { uint4 raw = *(const uint4*)(kr + key * DH + t * BPT); w[0] = raw.x; w[1] = raw.y; w[2] = raw.z; w[3] = raw.w; }
      else { uint2 raw = *(const uint2*)(kr + key * DH + t * BPT); w[0] = raw.x; w[1] = raw.y; }
      uint32_t ks_ = ksc[key * NG + (t * BPT) / 16];
      uint32_t R[4 * WPT];
      #pragma unroll
      for (int j = 0; j < WPT; j++) deq_word(w[j], ks_, R + 4 * j);
      #pragma unroll
      for (int ss = 0; ss < STEPS; ss += 2)
        #pragma unroll
        for (int mb = 0; mb < MB; mb++) {
          mma16816(s[mb][nt], qa[mb][ss], R[2 * ss], R[2 * ss + 1]);
          mma16816(s2[mb][nt], qa[mb][ss + 1], R[2 * ss + 2], R[2 * ss + 3]);   // independent chain (ILP)
        }
    }
    #pragma unroll
    for (int mb = 0; mb < MB; mb++)
      #pragma unroll
      for (int nt = 0; nt < NT; nt++) { s[mb][nt][0] += s2[mb][nt][0]; s[mb][nt][1] += s2[mb][nt][1]; s[mb][nt][2] += s2[mb][nt][2]; s[mb][nt][3] += s2[mb][nt][3]; }
    PROF_T(1);
    // ---- mask + online softmax (per m-block)
    #pragma unroll
    for (int mb = 0; mb < MB; mb++) {
      float mx0 = -INFINITY, mx1 = -INFINITY;
      #pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        int pos = p * PAGE + (C::PS ? 0 : wk * KPW) + nt * 8 + 2 * t;
        s[mb][nt][0] = pos < lim[mb][0] ? s[mb][nt][0] * sc : -INFINITY;
        s[mb][nt][1] = pos + 1 < lim[mb][0] ? s[mb][nt][1] * sc : -INFINITY;
        s[mb][nt][2] = pos < lim[mb][1] ? s[mb][nt][2] * sc : -INFINITY;
        s[mb][nt][3] = pos + 1 < lim[mb][1] ? s[mb][nt][3] * sc : -INFINITY;
        mx0 = fmaxf(mx0, fmaxf(s[mb][nt][0], s[mb][nt][1]));
        mx1 = fmaxf(mx1, fmaxf(s[mb][nt][2], s[mb][nt][3]));
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      float mn0 = fmaxf(mrow[mb][0], mx0), mn1 = fmaxf(mrow[mb][1], mx1);
      float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
      float al0 = exp2f(mrow[mb][0] - ms0), al1 = exp2f(mrow[mb][1] - ms1);
      mrow[mb][0] = mn0; mrow[mb][1] = mn1;
      float ps0 = 0.f, ps1 = 0.f;
      #pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        s[mb][nt][0] = exp2f(s[mb][nt][0] - ms0); s[mb][nt][1] = exp2f(s[mb][nt][1] - ms0);
        s[mb][nt][2] = exp2f(s[mb][nt][2] - ms1); s[mb][nt][3] = exp2f(s[mb][nt][3] - ms1);
        ps0 += s[mb][nt][0] + s[mb][nt][1]; ps1 += s[mb][nt][2] + s[mb][nt][3];
      }
      lrow[mb][0] = lrow[mb][0] * al0 + ps0; lrow[mb][1] = lrow[mb][1] * al1 + ps1;
      #pragma unroll
      for (int i = 0; i < D / 8; i++) { acc[mb][i][0] *= al0; acc[mb][i][1] *= al0; acc[mb][i][2] *= al1; acc[mb][i][3] *= al1; }
    }
    PROF_T(2);
    // ---- O += P V.  V is dequantised straight from the raw stage into B fragments: thread g owns the CPT
    // columns d = CPT*g + i (i = n-tile index); keys of a fragment are (2t,2t+1) / (2t+8,2t+9) of the 16-key chunk.
    #pragma unroll
    for (int kk = 0; kk < KPW / 16; kk++) {
      uint32_t pa[MB][4];
      #pragma unroll
      for (int mb = 0; mb < MB; mb++) {
        pa[mb][0] = pack_bf16(s[mb][2 * kk][0], s[mb][2 * kk][1]); pa[mb][1] = pack_bf16(s[mb][2 * kk][2], s[mb][2 * kk][3]);
        pa[mb][2] = pack_bf16(s[mb][2 * kk + 1][0], s[mb][2 * kk + 1][1]); pa[mb][3] = pack_bf16(s[mb][2 * kk + 1][2], s[mb][2 * kk + 1][3]);
      }
      const int gi = (C::CPT * g) / 32;
      const int kA = kk * 16 + 2 * t;
      uint32_t wa[2][C::VWC], wb[2][C::VWC], sa[2], sb[2];   // [0]: keys kA,kA+1  [1]: keys kA+8,kA+9
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const uint8_t* ra = vr + (kA + 8 * hh) * DH + g * (C::CPT / 2);
        const uint8_t* rb = ra + DH;
        if constexpr (C::VWC == 2) { uint2 x = *(const uint2*)ra, y = *(const uint2*)rb; wa[hh][0] = x.x; wa[hh][1] = x.y; wb[hh][0] = y.x; wb[hh][1] = y.y; }
        else { wa[hh][0] = *(const uint32_t*)ra; wb[hh][0] = *(const uint32_t*)rb; }
        sa[hh] = vsc[(kA + 8 * hh) * NG + gi]; sb[hh] = vsc[(kA + 8 * hh + 1) * NG + gi];
      }
      uint32_t Bv[2][C::CPT];
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        uint32_t fa = (min(sa[hh], 128u) + 126u) << 7; fa |= ((min(sb[hh], 128u) + 126u) << 7) << 16;
        uint32_t fb = (max(sa[hh], 128u) - 1u) << 7;   fb |= ((max(sb[hh], 128u) - 1u) << 7) << 16;
        #pragma unroll
        for (int w = 0; w < C::VWC; w++) {
          uint32_t clo = __byte_perm(wa[hh][w], wb[hh][w], 0x5410), chi = __byte_perm(wa[hh][w], wb[hh][w], 0x7632);
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            Bv[hh][8 * w + j] = mul_bf16x2(mul_bf16x2(pair_raw(clo, j), fa), fb);
            Bv[hh][8 * w + 4 + j] = mul_bf16x2(mul_bf16x2(pair_raw(chi, j), fa), fb);
          }
        }
      }
      #pragma unroll
      for (int i = 0; i < C::CPT; i++)
        #pragma unroll
        for (int mb = 0; mb < MB; mb++) mma16816(acc[mb][i], pa[mb], Bv[0][i], Bv[1][i]);
    }
  }
#ifdef UQ_PROF
  if (lane == 0 && warp == 0) { for (int i = 0; i < 6; i++) atomicAdd(&g_prof[i], (unsigned long long)pacc[i]); atomicAdd(&g_prof[6], (unsigned long long)nit); atomicAdd(&g_prof[7], 1ull); }
#endif
  // ---- finalize: quad-reduce l, cross-warp (key-slice) combine through smem
  __syncthreads();
  float* cb = (float*)smem;
  float* cm = cb + NWK * BM * C::CS;
  float* cl = cm + NWK * BM;
  #pragma unroll
  for (int mb = 0; mb < MB; mb++) {
    float l0 = lrow[mb][0], l1 = lrow[mb][1];
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    int r0 = (wm * MB + mb) * 16 + g, r1 = r0 + 8;
    #pragma unroll
    for (int i = 0; i < D / 8; i++) {
      // tile i, MMA column n (= owning thread g) holds d = CPT*n + i
      int d0 = C::CPT * (2 * t) + i, d1 = C::CPT * (2 * t + 1) + i;
      float* row0 = cb + ((size_t)wk * BM + r0) * C::CS; float* row1 = cb + ((size_t)wk * BM + r1) * C::CS;
      row0[d0] = acc[mb][i][0]; row0[d1] = acc[mb][i][1]; row1[d0] = acc[mb][i][2]; row1[d1] = acc[mb][i][3];
    }
    if (t == 0) {
      cm[wk * BM + r0] = mrow[mb][0]; cm[wk * BM + r1] = mrow[mb][1]; cl[wk * BM + r0] = l0; cl[wk * BM + r1] = l1;
    }
  }
  __syncthreads();
  float* mo = a.mo + rowbase * D;
  for (int c = tid; c < BM * (D / 4); c += T) {
    int r = c / (D / 4), d4 = c % (D / 4);
    int j = r >> gp_shift, gg = r & gm;
    if (!(j < K && gg < G)) { if (d4 == 0) a.ml[rowbase + r] = -INFINITY; continue; }
    float M = -INFINITY;
    #pragma unroll
    for (int w = 0; w < NWK; w++) M = fmaxf(M, cm[w * BM + r]);
    float Ms = M == -INFINITY ? 0.f : M;
    float L = 0.f; float4 o = make_float4(0, 0, 0, 0);
    #pragma unroll
    for (int w = 0; w < NWK; w++) {
      float wt = exp2f(cm[w * BM + r] - Ms);
      L += wt * cl[w * BM + r];
      float4 v = *(const float4*)(cb + ((size_t)w * BM + r) * C::CS + d4 * 4);
      o.x += wt * v.x; o.y += wt * v.y; o.z += wt * v.z; o.w += wt * v.w;
    }
    float inv = L > 0.f ? 1.f / L : 0.f;
    o.x *= inv; o.y *= inv; o.z *= inv; o.w *= inv;
    *(float4*)(mo + (size_t)r * D + d4 * 4) = o;
    if (d4 == 0) a.ml[rowbase + r] = L > 0.f ? Ms + log2f(L) : -INFINITY;
  }
}
