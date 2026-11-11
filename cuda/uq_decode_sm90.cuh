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

DEV void mbar_init(uint32_t a, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(cnt)); }
DEV void mbar_wait(uint32_t a, uint32_t par) {
  uint32_t ok = 0;
  for (long long n = 0; n < (1ll << 26); n++) {   // bounded spin: trap instead of hanging (debug aid, ~seconds)
    asm volatile("{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.u32 %0, 1, 0, P1;\n}\n" : "=r"(ok) : "r"(a), "r"(par) : "memory");
    if (ok) return;
  }
  __trap();
}
DEV void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory"); }
DEV void mbar_expect(uint32_t a, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory"); }
DEV void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
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

template <int MW, int DS, int RS>
struct Cfg90 {
  static constexpr int QB = 16384, STG = 32768, RAWB = 8704;   // raw = K codes 4096, K sc 256, V codes 4096, V sc 256
  static constexpr int OFF_Q = 0, OFF_DEQ = QB, OFF_RAW = OFF_DEQ + DS * STG, OFF_CB = OFF_RAW + RS * RAWB;
  static constexpr int CBB = MW > 1 ? 128 * 16 * 16 + 128 * 16 : 0;
  static constexpr int OFF_BAR = (OFF_CB + CBB + 127) & ~127;
  static constexpr int SMEM = OFF_BAR + 128 + 1024;
  static constexpr int NT = 128 * MW + 256;
  static constexpr bool OK = SMEM <= 227 * 1024 && (MW == 1 || MW == 2);
};

template <int MW, int DS, int RS>
__global__ void __launch_bounds__(128 * MW + 256, 1)
uq_sm90_kernel(const UQParams a, int G, int gp_shift, int bme) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = Cfg90<MW, DS, RS>;
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
  const uint32_t sQ = sb + C::OFF_Q, sDeq = sb + C::OFF_DEQ, sRaw = sb + C::OFF_RAW, sBar = sb + C::OFF_BAR;
  auto bar_full = [&](int i) { return sBar + 8 * i; };                  // raw full [RS]
  auto bar_dfull = [&](int i) { return sBar + 8 * (RS + i); };          // deq full [DS]
  auto bar_dfree = [&](int i) { return sBar + 8 * (RS + DS + i); };     // deq free [DS]
  const uint32_t bar_q = sBar + 8 * (RS + 2 * DS);
  if (tid == 0) {
    for (int i = 0; i < RS; i++) mbar_init(bar_full(i), 1);
    for (int i = 0; i < DS; i++) { mbar_init(bar_dfull(i), 256); mbar_init(bar_dfree(i), 4); }
    mbar_init(bar_q, 256);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int32_t* bt = a.block_table + (size_t)b * a.max_pages;

  if (wg >= MW) {
    // ============================ dequant / producer warpgroups ============================
    const int t = tid - 128 * MW;   // 0..255
    auto issue = [&](int it) {
      int s = it % RS;
      size_t page = (size_t)bt[p_lo + it];
      size_t cb = (page * HKV + hk) * PAGE;
      uint32_t d = sRaw + s * C::RAWB;
      mbar_expect(bar_full(s), C::RAWB);
      bulk_g2s(d, a.kc + cb * DH, 4096, bar_full(s));
      bulk_g2s(d + 4096, a.ks + cb * NG, 256, bar_full(s));
      bulk_g2s(d + 4352, a.vc + cb * DH, 4096, bar_full(s));
      bulk_g2s(d + 8448, a.vs + cb * NG, 256, bar_full(s));
    };
    if (t == 0) for (int i = 0; i < RS && i < npg; i++) issue(i);
    // Q tile -> swizzled smem (rows j*gp+g, zero padded to 64)
    {
      const uint16_t* q = (const uint16_t*)a.q;
      for (int c = t; c < 64 * 16; c += 256) {
        int r = c >> 4, cc = c & 15;
        int j = r >> gp_shift, gg = r & ((1 << gp_shift) - 1);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (j < K && gg < G) v = *(const uint4*)(q + ((size_t)(b * K + j) * HQ + hk * G + gg) * D + cc * 8);
        uint32_t dst = sQ + (cc >> 3) * 8192 + r * 128 + (((cc & 7) ^ (r & 7)) << 4);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
      }
      fence_async();
      mbar_arrive(bar_q);
    }
    const int key = t >> 2, g = t & 3;
    for (int it = 0; it < npg; it++) {
      const int rs = it % RS, ds = it % DS;
      mbar_wait(bar_full(rs), (it / RS) & 1);
      mbar_wait(bar_dfree(ds), ((it / DS) & 1) ^ 1);
      const uint32_t raw = sRaw + rs * C::RAWB;
      #pragma unroll
      for (int w = 0; w < 2; w++) {   // 0 = K, 1 = V
        uint32_t ca = raw + (w ? 4352 : 0) + key * 64 + g * 16;
        uint32_t sa = raw + (w ? 4352 : 0) + 4096 + key * 4 + g;
        uint4 rv; uint32_t sc;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(rv.x), "=r"(rv.y), "=r"(rv.z), "=r"(rv.w) : "r"(ca));
        asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(sc) : "r"(sa));
        uint4 o[4];
        dequant_chunk(rv, sc, o);
        const uint32_t tile = sDeq + ds * C::STG + w * 16384;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          int C16 = g * 4 + i;
          uint32_t dst = tile + (C16 >> 3) * 8192 + key * 128 + (((C16 & 7) ^ (key & 7)) << 4);
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(o[i].x), "r"(o[i].y), "r"(o[i].z), "r"(o[i].w) : "memory");
        }
      }
      fence_async();
      mbar_arrive(bar_dfull(ds));
      named_bar(1, 256);                       // all dequant threads done with raw stage rs
      if (t == 0 && it + RS < npg) issue(it + RS);
    }
    return;
  }

  // ============================ MMA / softmax warpgroups ============================
  const int tw = tid & 127, w4 = tw >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3;
  const int r0 = 16 * w4 + gq, r1 = r0 + 8;
  const int gm = (1 << gp_shift) - 1;
  auto lim_of = [&](int r) { int j = r >> gp_shift, gg = r & gm; return (j < K && gg < G) ? seq_len - K + j + 1 : 0; };
  const int lim0 = lim_of(r0), lim1 = lim_of(r1);
  const float sc = a.sm_scale * 1.4426950408889634f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  float acc[64];
  #pragma unroll
  for (int i = 0; i < 64; i++) acc[i] = 0.f;
  mbar_wait(bar_q, 0);
  for (int it = wg; it < npg; it += MW) {
    const int ds = it % DS;
    mbar_wait(bar_dfull(ds), (it / DS) & 1);
    const uint32_t kt = sDeq + ds * C::STG, vt = kt + 16384;
    float s[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) s[i] = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) reg_fence(s[i]);
    wg_fence();
    #pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      uint32_t off = (ks >> 2) * 8192 + (ks & 3) * 32;
      wgmma_s(s, sdesc(sQ + off, 16, 1024), sdesc(kt + off, 16, 1024), ks > 0);
    }
    wg_commit();
    wg_wait0();
    #pragma unroll
    for (int i = 0; i < 32; i++) reg_fence(s[i]);
    // ---- mask + online softmax
    const int p = p_lo + it;
    float mx0 = -INFINITY, mx1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      int pos = p * PAGE + nt * 8 + 2 * tq;
      s[4 * nt + 0] = pos < lim0 ? s[4 * nt + 0] * sc : -INFINITY;
      s[4 * nt + 1] = pos + 1 < lim0 ? s[4 * nt + 1] * sc : -INFINITY;
      s[4 * nt + 2] = pos < lim1 ? s[4 * nt + 2] * sc : -INFINITY;
      s[4 * nt + 3] = pos + 1 < lim1 ? s[4 * nt + 3] * sc : -INFINITY;
      mx0 = fmaxf(mx0, fmaxf(s[4 * nt], s[4 * nt + 1]));
      mx1 = fmaxf(mx1, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
    float al0 = exp2f(m0 - ms0), al1 = exp2f(m1 - ms1);
    m0 = mn0; m1 = mn1;
    float ps0 = 0.f, ps1 = 0.f;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[4 * nt + 0] = exp2f(s[4 * nt + 0] - ms0); s[4 * nt + 1] = exp2f(s[4 * nt + 1] - ms0);
      s[4 * nt + 2] = exp2f(s[4 * nt + 2] - ms1); s[4 * nt + 3] = exp2f(s[4 * nt + 3] - ms1);
      ps0 += s[4 * nt] + s[4 * nt + 1]; ps1 += s[4 * nt + 2] + s[4 * nt + 3];
    }
    l0 = l0 * al0 + ps0; l1 = l1 * al1 + ps1;
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[4 * i] *= al0; acc[4 * i + 1] *= al0; acc[4 * i + 2] *= al1; acc[4 * i + 3] *= al1; }
    uint32_t pa[4][4];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      pa[kk][0] = pack_bf16(s[8 * kk + 0], s[8 * kk + 1]); pa[kk][1] = pack_bf16(s[8 * kk + 2], s[8 * kk + 3]);
      pa[kk][2] = pack_bf16(s[8 * kk + 4], s[8 * kk + 5]); pa[kk][3] = pack_bf16(s[8 * kk + 6], s[8 * kk + 7]);
    }
    #pragma unroll
    for (int i = 0; i < 64; i++) reg_fence(acc[i]);
    wg_fence();
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) wgmma_pv(acc, pa[kk], sdesc(vt + kk * 2048, 8192, 1024));
    wg_commit();
    wg_wait0();
    #pragma unroll
    for (int i = 0; i < 64; i++) reg_fence(acc[i]);
    if (lane == 0) mbar_arrive(bar_dfree(ds));
  }
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
    float wa0 = exp2f(m0 - Ms0), wb0 = exp2f(o.x - Ms0), wa1 = exp2f(m1 - Ms1), wb1 = exp2f(o.y - Ms1);
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
    if (r0 < bme) a.ml[rowbase + r0] = l0 > 0.f ? m0 + log2f(l0) : -INFINITY;
    if (r1 < bme) a.ml[rowbase + r1] = l1 > 0.f ? m1 + log2f(l1) : -INFINITY;
  }
#endif
}

template <int MW, int DS, int RS>
static cudaError_t launch_sm90(const UQParams& p, cudaStream_t s) {
  using C = Cfg90<MW, DS, RS>;
  if constexpr (!C::OK) return cudaErrorInvalidValue; else {
    int bm = uq_block_m(p.K, p.HQ, p.HKV);
    if (p.D != 128 || p.PAGE != 64 || bm > 64) return cudaErrorInvalidValue;
    int dev = 0, maj = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
    if (maj != 9) return cudaErrorInvalidValue;
    auto kern = uq_sm90_kernel<MW, DS, RS>;
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

// mw: 1/2 MMA warpgroups; stages (deq stages) 2..3; raw stages fixed by table.
static cudaError_t dispatch_sm90(const UQParams& p, int mw, int stages, cudaStream_t s) {
  if (mw == 0) mw = 2;
  if (stages == 0) stages = 3;
  if (mw == 1) { if (stages == 2) return launch_sm90<1, 2, 4>(p, s); if (stages == 3) return launch_sm90<1, 3, 4>(p, s); if (stages == 4) return launch_sm90<1, 4, 6>(p, s); }
  if (mw == 2) { if (stages == 2) return launch_sm90<2, 2, 4>(p, s); if (stages == 3) return launch_sm90<2, 3, 4>(p, s); if (stages == 4) return launch_sm90<2, 4, 6>(p, s); }
  return cudaErrorInvalidValue;
}
