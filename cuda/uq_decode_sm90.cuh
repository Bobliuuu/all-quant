// Variant 3: Hopper (sm_90a) warp-specialised wgmma kernel.  Included from uq_decode.cu (UQ_ENABLE_SM90).
// D=128, PAGE=64, query tile M=64 rows (K*pow2(G) <= 64).
//   * 1 elected thread of the dequant warpgroups issues cp.async.bulk (1D TMA) copies of the four contiguous
//     per-(page,head) FP4 slices (K codes, K scales, V codes, V scales) into a raw ring (mbarrier complete_tx).
//   * 2 dequant warpgroups (256 threads) expand FP4/UE8M0 -> bf16 into 128B-swizzled K / V tiles (the layout
//     wgmma reads), fence.proxy.async, arrive on deq_full.
//   * MW MMA warpgroups (each owns alternate pages, own online-softmax state): S = Q K^T with wgmma m64n64k16
//     (A = Q smem, B = K tile K-major), online softmax in registers, O += P V with wgmma m64n128k16
//     (A = P registers, B = V tile MN-major via the transpose bit), partial O merged through smem at the end.
// Scratch contract identical to the other variants (mo/ml [B,Hkv,nsplit,BM,D] / [..,BM]), then uq_reduce_kernel.
#include "uq_wgmma_ops.cuh"
#ifdef UQ90_PROF
__device__ unsigned long long g_prof90[16];
#define PT(i) do { long long _n = clock64(); pacc[i] += _n - _tp; _tp = _n; } while (0)
#else
#define PT(i) do {} while (0)
#endif
#ifndef UQ90_ABL
#define UQ90_ABL 0   // perf ablation (not correct): 1 = dequant WGs skip the dequant, 2 = MMA WG skips wgmma+softmax
#endif

DEV void mbar_init(uint32_t a, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(cnt)); }
DEV void mbar_wait(uint32_t a, uint32_t par) {
  uint32_t ok = 0;
  for (long long n = 0; n < (1ll << 26); n++) {   // bounded spin: trap instead of hanging (debug aid, ~seconds)
    asm volatile("{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.u32 %0, 1, 0, P1;\n}\n" : "=r"(ok) : "r"(a), "r"(par) : "memory");
    if (ok) return;
  }
#ifdef UQ90_DEBUG
  printf("uq_sm90: mbarrier timeout blk(%d,%d,%d) tid %d bar %u par %u\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, a & 0xFFFF, par);
#endif
  __trap();
}
DEV void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory"); }
DEV void mbar_expect(uint32_t a, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory"); }
DEV void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
DEV float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
#ifdef UQ90_PROF
void uq_sm90_prof_read(unsigned long long* o) { cudaMemcpyFromSymbol(o, g_prof90, 16 * 8); }
void uq_sm90_prof_reset() { unsigned long long z[16] = {0}; cudaMemcpyToSymbol(g_prof90, z, sizeof(z)); }
#endif
DEV void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
DEV void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEV void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DEV void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
DEV void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
DEV void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
// shared-memory matrix descriptor, 128B swizzle
DEV uint64_t sdesc(uint32_t addr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo_bytes >> 4) << 16) | ((uint64_t)(sbo_bytes >> 4) << 32) | (1ull << 62);
}

template <int MW, int DS, int RS, int ND>
struct Cfg90 {
  static constexpr int QB = 0, STG = 32768, RAWB = 8704;   // raw = K codes 4096, K sc 256, V codes 4096, V sc 256
  static constexpr int OFF_Q = 0, OFF_DEQ = QB, OFF_RAW = OFF_DEQ + DS * STG, OFF_LUT = (OFF_RAW + RS * RAWB + 127) & ~127, OFF_CB = OFF_LUT + 32768;
  static constexpr int CBB = MW > 1 ? 128 * 16 * 16 + 128 * 16 : 0;
  static constexpr int OFF_BAR = (OFF_CB + CBB + 127) & ~127;
  static constexpr int SMEM = OFF_BAR + 512 + 1024;
  static constexpr int NT = 128 * (MW + ND + 1);   // + 1 producer warpgroup (only 1 lane works)
  static constexpr int REG_PR = 24;
  static constexpr int NDQ = 128 * ND;
  static constexpr int REG_DQ = 56;
  // registers per thread at launch: each SM sub-partition holds 16K regs and gets ceil(warps/4) warps
  static constexpr int WPP = (NT / 32 + 3) / 4;
  static constexpr int REG_L = (16384 / 32 / WPP / 8) * 8 > 255 ? 248 : (16384 / 32 / WPP / 8) * 8;
  static constexpr int REG_MMA_RAW = ((NT * REG_L - NDQ * REG_DQ - 128 * REG_PR) / (128 * MW) / 8) * 8;
  static constexpr int REG_MMA = REG_MMA_RAW > 232 ? 232 : REG_MMA_RAW;
  static constexpr bool OK = SMEM <= 227 * 1024 && (MW == 1 || MW == 2);
};

template <int MW, int DS, int RS, int ND>
__global__ void __launch_bounds__(128 * (MW + ND + 1), 1)
uq_sm90_kernel(const UQParams a, int G, int gp_shift, int bme) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = Cfg90<MW, DS, RS, ND>;
  constexpr int D = 128, PAGE = 64, DH = 64, NG = 4;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  const uint32_t sbase0 = smem_u32(smem_raw);
  const uint32_t sb = (sbase0 + 1023u) & ~1023u;
  uint8_t* gb = smem_raw + (sb - sbase0);
  const int tid = threadIdx.x, wg = tid >> 7;
  const int b = blockIdx.x, hk = blockIdx.y, sp = blockIdx.z;
  const int K = a.K, HKV = a.HKV, HQ = a.HQ, nsplit = a.nsplit;
  const int seq_len = a.seq_lens[b];
  const int n_pages = (seq_len + PAGE - 1) / PAGE;
  const int pps = (n_pages + nsplit - 1) / nsplit;
  const int p_lo = sp * pps, p_hi = min(p_lo + pps, n_pages);
  const size_t rowbase = ((size_t)(b * HKV + hk) * nsplit + sp) * bme;
  if (p_lo >= p_hi) {
    for (int r = tid; r < bme; r += C::NT) a.ml[rowbase + r] = -INFINITY;
    return;
  }
  const int npg = p_hi - p_lo;
  const uint32_t sLut = sb + C::OFF_LUT;
  const uint32_t sQ = sb + C::OFF_Q, sDeq = sb + C::OFF_DEQ, sRaw = sb + C::OFF_RAW, sBar = sb + C::OFF_BAR;
  auto bar_full = [&](int i) { return sBar + 8 * i; };                  // raw full [RS]
  auto bar_dfull = [&](int i) { return sBar + 8 * (RS + i); };          // deq full [DS]
  auto bar_dfree = [&](int i) { return sBar + 8 * (RS + DS + i); };     // deq free [DS]
  auto bar_rfree = [&](int i) { return sBar + 8 * (RS + 2 * DS + i); };   // raw stage free [RS]
  {   // 32 conflict-free replicas of the byte -> bf16x2 (two e2m1 nibbles, true values) table
    const uint16_t mags[8] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0};
    for (int e = tid; e < 8192; e += C::NT) {
      int bt_ = e >> 5, n0 = bt_ & 15, n1 = bt_ >> 4;
      uint32_t v0 = mags[n0 & 7] | ((n0 & 8) << 12), v1 = mags[n1 & 7] | ((n1 & 8) << 12);
      *(uint32_t*)(gb + C::OFF_LUT + e * 4) = v0 | (v1 << 16);
    }
  }
  if (tid == 0) {
    for (int i = 0; i < RS; i++) mbar_init(bar_full(i), 1);
    for (int i = 0; i < DS; i++) { mbar_init(bar_dfull(i), C::NDQ); mbar_init(bar_dfree(i), 4); }
    for (int i = 0; i < RS; i++) mbar_init(bar_rfree(i), C::NDQ);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int32_t* bt = a.block_table + (size_t)b * a.max_pages;

  if (wg >= MW + ND) {
    // ============================ producer warpgroup (one lane issues the 1D bulk copies) ============================
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(C::REG_PR));
    if (tid == 128 * (MW + ND)) {
#ifdef UQ90_DEBUG
      if (blockIdx.x == 3 && blockIdx.y == 0 && blockIdx.z == 0) printf("producer start sBar %u npg %d RS %d\n", sBar & 0xFFFF, npg, RS);
#endif
      for (int it = 0; it < npg; it++) {
        const int rs = it % RS;
        mbar_wait(bar_rfree(rs), ((it / RS) & 1) ^ 1);
        size_t page = (size_t)bt[p_lo + it];
        size_t cb = (page * HKV + hk) * PAGE;
        uint32_t d = sRaw + rs * C::RAWB;
        mbar_expect(bar_full(rs), C::RAWB);
        bulk_g2s(d, a.kc + cb * DH, 4096, bar_full(rs));
        bulk_g2s(d + 4096, a.ks + cb * NG, 256, bar_full(rs));
        bulk_g2s(d + 4352, a.vc + cb * DH, 4096, bar_full(rs));
        bulk_g2s(d + 8448, a.vs + cb * NG, 256, bar_full(rs));
      }
    }
    return;
  }
  if (wg >= MW) {
    // ============================ dequant warpgroups ============================
    const int t = tid - 128 * MW;   // 0..NDQ-1
    const uint32_t lb = sLut + (tid & 31) * 4;
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(C::REG_DQ));
#ifdef UQ90_PROF
    long long pacc[8] = {0}, _tp = clock64();
#endif
    constexpr int JM = (512 + C::NDQ - 1) / C::NDQ;
    for (int it = 0; it < npg; it++) {
      const int rs = it % RS, ds = it % DS;
      mbar_wait(bar_full(rs), (it / RS) & 1);
      PT(0);
      const uint32_t raw = sRaw + rs * C::RAWB;
      uint4 rv[JM]; uint32_t scv[JM];
      #pragma unroll
      for (int j = 0; j < JM; j++) {   // chunk c: tensor w (0 = K, 1 = V), 16B group u
        const int c = t + j * C::NDQ;
        const int w = (c >> 8) & 1, u = c & 255;
        const int key = (u & 7) | ((u >> 5) << 3), g = (u >> 3) & 3;
        uint32_t ca = raw + (w ? 4352 : 0) + key * 64 + g * 16;
        uint32_t sa = raw + (w ? 4352 : 0) + 4096 + key * 4 + g;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(rv[j].x), "=r"(rv[j].y), "=r"(rv[j].z), "=r"(rv[j].w) : "r"(ca));
        asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(scv[j]) : "r"(sa));
      }
      mbar_arrive(bar_rfree(rs));       // raw stage fully read by this thread (arrive has release semantics for the loads)
      mbar_wait(bar_dfree(ds), ((it / DS) & 1) ^ 1);
      PT(1);
      #pragma unroll
      for (int j = 0; j < (UQ90_ABL & 1 ? 0 : JM); j++) {
        const int c = t + j * C::NDQ;
        if (c >= 512) break;
        const int w = c >> 8, u = c & 255;
        const int key = (u & 7) | ((u >> 5) << 3), g = (u >> 3) & 3;   // quarter-warp = 8 distinct keys, same group: conflict-free swizzled stores
        uint32_t o[4][4];
        uint32_t wv[4] = {rv[j].x, rv[j].y, rv[j].z, rv[j].w};
        const uint32_t f = (scv[j] << 7) * 0x10001u;   // bf16x2 of 2^(sc-127)
        #pragma unroll
        for (int k = 0; k < 4; k++) {
          #pragma unroll
          for (int jj = 0; jj < 4; jj++) {
            uint32_t idx = __byte_perm(wv[k], 0u, 0x4440u | jj);   // byte jj, zero-extended
            uint32_t addr = idx * 128u + lb;
            uint32_t v;
            asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr));
            o[k][jj] = mul_bf16x2(v, f);
          }
        }
        const uint32_t tile = sDeq + ds * C::STG + w * 16384;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          int C16 = g * 4 + i;
          uint32_t dst = tile + (C16 >> 3) * 8192 + key * 128 + (((C16 & 7) ^ (key & 7)) << 4);
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(o[i][0]), "r"(o[i][1]), "r"(o[i][2]), "r"(o[i][3]) : "memory");
        }
      }
      PT(2);
      fence_async();
      mbar_arrive(bar_dfull(ds));
      PT(3);
    }
#ifdef UQ90_PROF
    if (t == 0) { for (int i = 0; i < 5; i++) atomicAdd(&g_prof90[i], (unsigned long long)pacc[i]); atomicAdd(&g_prof90[15], (unsigned long long)npg); }
#endif
    return;
  }

  // ============================ MMA / softmax warpgroups ============================
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(C::REG_MMA));
  const int tw = tid & 127, w4 = tw >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3;
  const int r0 = 16 * w4 + gq, r1 = r0 + 8;
  const int gm = (1 << gp_shift) - 1;
  auto lim_of = [&](int r) { int j = r >> gp_shift, gg = r & gm; return (j < K && gg < G) ? seq_len - K + j + 1 : 0; };
  const int lim0 = lim_of(r0), lim1 = lim_of(r1);
  const float sc = a.sm_scale * 1.4426950408889634f;
  const bool warp_active = 16 * w4 < K * (1 << gp_shift);
  const int min_lim = seq_len - K + 1;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  float acc[64];
  #pragma unroll
  for (int i = 0; i < 64; i++) acc[i] = 0.f;
  uint32_t qa[8][4];
  {
    const uint16_t* q = (const uint16_t*)a.q;
    #pragma unroll
    for (int h2 = 0; h2 < 2; h2++) {
      int r = r0 + 8 * h2, j = r >> gp_shift, gg = r & gm;
      bool ok = j < K && gg < G;
      const uint32_t* qr = (const uint32_t*)(q + ((size_t)(b * K + j) * HQ + hk * G + gg) * D);
      #pragma unroll
      for (int ksx = 0; ksx < 8; ksx++) {
        qa[ksx][h2] = ok ? qr[ksx * 8 + tq] : 0u;          // k = 16*ksx + 2*tq, +1
        qa[ksx][2 + h2] = ok ? qr[ksx * 8 + 4 + tq] : 0u;  // k + 8
      }
    }
  }
#ifdef UQ90_PROF
  long long pacc[8] = {0}, _tp = clock64();
#endif
  for (int it = wg; it < npg; it += MW) {
    const int ds = it % DS;
    PT(6);
    mbar_wait(bar_dfull(ds), (it / DS) & 1);
    PT(0);
    const uint32_t kt = sDeq + ds * C::STG, vt = kt + 16384;
    if (UQ90_ABL & 2) { if (lane == 0) mbar_arrive(bar_dfree(ds)); continue; }
    float s[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) s[i] = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) reg_fence(s[i]);
    wg_fence();
    #pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      uint32_t off = (ks >> 2) * 8192 + (ks & 3) * 32;
      if (!(UQ90_ABL & 8)) wgmma_s_rs(s, qa[ks], sdesc(kt + off, 16, 1024), ks > 0);
    }
    wg_commit();
    wg_wait0();
    PT(1);
    #pragma unroll
    for (int i = 0; i < 32; i++) reg_fence(s[i]);
    // ---- mask + online softmax (warps whose 16 rows are all padding skip it; P stays 0)
    const int p = p_lo + it;
    uint32_t pa[4][4];
    if (UQ90_ABL & 4) {
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        pa[kk][0] = pack_bf16(s[8 * kk + 0], s[8 * kk + 1]); pa[kk][1] = pack_bf16(s[8 * kk + 2], s[8 * kk + 3]);
        pa[kk][2] = pack_bf16(s[8 * kk + 4], s[8 * kk + 5]); pa[kk][3] = pack_bf16(s[8 * kk + 6], s[8 * kk + 7]);
      }
    } else if (warp_active) {
      float mx0 = -INFINITY, mx1 = -INFINITY;
      if ((p + 1) * PAGE > min_lim) {   // only the last page(s) contain masked keys
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
          int pos = p * PAGE + nt * 8 + 2 * tq;
          s[4 * nt + 0] = pos < lim0 ? s[4 * nt + 0] : -INFINITY;
          s[4 * nt + 1] = pos + 1 < lim0 ? s[4 * nt + 1] : -INFINITY;
          s[4 * nt + 2] = pos < lim1 ? s[4 * nt + 2] : -INFINITY;
          s[4 * nt + 3] = pos + 1 < lim1 ? s[4 * nt + 3] : -INFINITY;
        }
      }
      #pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        mx0 = fmaxf(mx0, fmaxf(s[4 * nt], s[4 * nt + 1]));
        mx1 = fmaxf(mx1, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      // m0/m1 hold the running max of the RAW scores (scaled on use): p = 2^((s - m) * sc).  Lazy rescale (FA4 style):
      // the max is only moved (and O/l rescaled) when some row grows by more than 2^8 in the exponent, so p <= 256.
      bool upd = ((mx0 - m0) * sc > 8.f) | ((mx1 - m1) * sc > 8.f);
      if (__any_sync(0xffffffffu, upd)) {
        float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
        float al0 = ex2f((m0 - ms0) * sc), al1 = ex2f((m1 - ms1) * sc);
        m0 = mn0; m1 = mn1;
        l0 *= al0; l1 *= al1;
        #pragma unroll
        for (int i = 0; i < 16; i++) { acc[4 * i] *= al0; acc[4 * i + 1] *= al0; acc[4 * i + 2] *= al1; acc[4 * i + 3] *= al1; }
      }
      const float b0 = m0 == -INFINITY ? 0.f : -m0 * sc, b1 = m1 == -INFINITY ? 0.f : -m1 * sc;
      float ps0 = 0.f, ps1 = 0.f;
      #pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        s[4 * nt + 0] = ex2f(fmaf(s[4 * nt + 0], sc, b0)); s[4 * nt + 1] = ex2f(fmaf(s[4 * nt + 1], sc, b0));
        s[4 * nt + 2] = ex2f(fmaf(s[4 * nt + 2], sc, b1)); s[4 * nt + 3] = ex2f(fmaf(s[4 * nt + 3], sc, b1));
        ps0 += s[4 * nt] + s[4 * nt + 1]; ps1 += s[4 * nt + 2] + s[4 * nt + 3];
      }
      l0 += ps0; l1 += ps1;
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        pa[kk][0] = pack_bf16(s[8 * kk + 0], s[8 * kk + 1]); pa[kk][1] = pack_bf16(s[8 * kk + 2], s[8 * kk + 3]);
        pa[kk][2] = pack_bf16(s[8 * kk + 4], s[8 * kk + 5]); pa[kk][3] = pack_bf16(s[8 * kk + 6], s[8 * kk + 7]);
      }
    } else {
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) { pa[kk][0] = pa[kk][1] = pa[kk][2] = pa[kk][3] = 0u; }
    }
    PT(2);
    #pragma unroll
    for (int i = 0; i < 64; i++) reg_fence(acc[i]);
    wg_fence();
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) if (!(UQ90_ABL & 8)) wgmma_pv(acc, pa[kk], sdesc(vt + kk * 2048, 8192, 1024));
    wg_commit();
    wg_wait0();
    #pragma unroll
    for (int i = 0; i < 64; i++) reg_fence(acc[i]);
    PT(3);
    if (lane == 0) mbar_arrive(bar_dfree(ds));
  }
#ifdef UQ90_PROF
  if (tw == 0) { for (int i = 0; i < 4; i++) atomicAdd(&g_prof90[8 + i], (unsigned long long)pacc[i]); atomicAdd(&g_prof90[12], (unsigned long long)((npg - wg + MW - 1) / MW)); atomicAdd(&g_prof90[13], (unsigned long long)pacc[6]); }
#endif
  // ---- finalize
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  if constexpr (MW == 2) {
    float* cbf = (float*)(gb + C::OFF_CB);   // [16][128 thr][4] floats + [128][4] (m0,m1,l0,l1)
    if (wg == 1) {
      #pragma unroll
      for (int i = 0; i < 16; i++) *(float4*)(cbf + (i * 128 + tw) * 4) = make_float4(acc[4 * i], acc[4 * i + 1], acc[4 * i + 2], acc[4 * i + 3]);
      *(float4*)(cbf + 16 * 128 * 4 + tw * 4) = make_float4(m0, m1, l0, l1);
    }
    named_bar(2, 256);
    if (wg == 1) return;
    float4 o = *(float4*)(cbf + 16 * 128 * 4 + tw * 4);
    float M0 = fmaxf(m0, o.x), M1 = fmaxf(m1, o.y);
    float Ms0 = M0 == -INFINITY ? 0.f : M0, Ms1 = M1 == -INFINITY ? 0.f : M1;
    float wa0 = ex2f((m0 - Ms0) * sc), wb0 = ex2f((o.x - Ms0) * sc), wa1 = ex2f((m1 - Ms1) * sc), wb1 = ex2f((o.y - Ms1) * sc);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      float4 v = *(float4*)(cbf + (i * 128 + tw) * 4);
      acc[4 * i] = acc[4 * i] * wa0 + v.x * wb0; acc[4 * i + 1] = acc[4 * i + 1] * wa0 + v.y * wb0;
      acc[4 * i + 2] = acc[4 * i + 2] * wa1 + v.z * wb1; acc[4 * i + 3] = acc[4 * i + 3] * wa1 + v.w * wb1;
    }
    l0 = l0 * wa0 + o.z * wb0; l1 = l1 * wa1 + o.w * wb1;
    m0 = M0; m1 = M1;
  }
  auto row_ok = [&](int r) { return (r >> gp_shift) < K && (r & gm) < G; };
  float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  float* mo = a.mo + rowbase * D;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    int col = i * 8 + 2 * tq;
    if (r0 < bme && row_ok(r0)) *(float2*)(mo + (size_t)r0 * D + col) = make_float2(acc[4 * i] * i0, acc[4 * i + 1] * i0);
    if (r1 < bme && row_ok(r1)) *(float2*)(mo + (size_t)r1 * D + col) = make_float2(acc[4 * i + 2] * i1, acc[4 * i + 3] * i1);
  }
  if (tq == 0) {
    if (r0 < bme) a.ml[rowbase + r0] = (l0 > 0.f && row_ok(r0)) ? m0 * sc + log2f(l0) : -INFINITY;
    if (r1 < bme) a.ml[rowbase + r1] = (l1 > 0.f && row_ok(r1)) ? m1 * sc + log2f(l1) : -INFINITY;
  }
#endif
}

template <int MW, int DS, int RS, int ND>
static cudaError_t launch_sm90(const UQParams& p, cudaStream_t s) {
  using C = Cfg90<MW, DS, RS, ND>;
  if constexpr (!C::OK) return cudaErrorInvalidValue; else {
    int bm = uq_block_m(p.K, p.HQ, p.HKV);
    if (p.D != 128 || p.PAGE != 64 || bm > 64) return cudaErrorInvalidValue;
    int dev = 0, maj = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
    if (maj != 9) return cudaErrorInvalidValue;
    auto kern = uq_sm90_kernel<MW, DS, RS, ND>;
    static bool init = false;
    if (!init) {
      cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
      if (e != cudaSuccess) { cudaGetLastError(); return e; }
      init = true;
    }
    int G = p.HQ / p.HKV, gp = pow2ceil(G), shift = 0;
    while ((1 << shift) < gp) shift++;
    kern<<<dim3(p.B, p.HKV, p.nsplit), C::NT, C::SMEM, s>>>(p, G, shift, bm);
    int rows = p.B * p.HKV * p.K * G;
    uq_reduce_kernel<128><<<(rows + 3) / 4, 128, 0, s>>>(p, G, bm, shift);
    return cudaGetLastError();
  }
}

// mw = # MMA warpgroups (1,2); stages = dequant stages; nd = # dequant warpgroups (2,3) encoded in p.mb (0 -> 2)
#define SM90_CASE(MW_, DS_, RS_, ND_) if (mw == MW_ && stages == (RS_ == 4 ? DS_ : DS_ + 10 * (RS_ / 2 - 1)) && nd == ND_) return launch_sm90<MW_, DS_, RS_, ND_>(p, s);
// stages code: DS (raw ring 4), or DS + 10*(RS/2-1) for deeper raw rings (RS=6 -> +20, RS=8 -> +30, RS=12 -> +50)
static cudaError_t dispatch_sm90(const UQParams& p, int mw, int stages, cudaStream_t s) {
  if (mw == 0) mw = 2;
  if (stages == 0) stages = 3;
  int nd = p.mb == 0 ? 2 : p.mb;
  SM90_CASE(1, 2, 4, 2) SM90_CASE(1, 3, 4, 2) SM90_CASE(1, 2, 4, 3) SM90_CASE(1, 3, 4, 3)
  SM90_CASE(1, 2, 8, 2) SM90_CASE(1, 3, 8, 2) SM90_CASE(1, 2, 8, 3) SM90_CASE(1, 3, 8, 3)
  SM90_CASE(2, 2, 4, 2) SM90_CASE(2, 3, 4, 2) SM90_CASE(2, 2, 8, 2) SM90_CASE(2, 3, 8, 2)
  return cudaErrorInvalidValue;
}
