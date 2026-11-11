// Variant 4: Blackwell (sm_100a / sm_103a) tcgen05 kernel, bf16 (kind::f16) UMMA, accumulators in TMEM.
// D=128, PAGE=64, query tile M=128 rows (K*pow2(G) <= 128; rows >= K*G zero padded).
// Same architecture as variant 3 (uq_decode_sm90.cuh) but with the Blackwell machinery:
//   * 1 elected thread issues cp.async.bulk (1D TMA) copies of the raw FP4 slices (mbarrier complete_tx),
//   * 2 dequant warpgroups expand FP4/UE8M0 -> bf16 into 128B-swizzled K/V tiles (permuted pair order, see deq_word),
//   * 1 MMA warp (one thread) issues tcgen05.mma: S = Q K^T (A = Q smem K-major, B = K tile, D = TMEM, 64 cols,
//     double-buffered) and O += P V (A = P smem K-major written by the softmax warps, B = V tile MN-major, D = TMEM 128 cols),
//   * 4 softmax warps (warp%4 == TMEM lane quadrant, thread = row): tcgen05.ld S, online softmax entirely in-thread,
//     P -> bf16 smem, conditional rescale of O in TMEM (tcgen05.ld/st), final O via tcgen05.ld.
// Included after uq_decode.cu's helpers (deq_word, pack_bf16, uq_reduce_kernel, pow2ceil).  Raw PTX only, no CUTLASS needed.
#define S1_DEV __device__ __forceinline__
#if defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL)
#define S1_ON 1
#endif

S1_DEV void s1_mbar_init(uint32_t a, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(cnt)); }
S1_DEV void s1_mbar_wait(uint32_t a, uint32_t par) {
  uint32_t ok = 0;
  for (long long n = 0; n < (1ll << 27); n++) {
    asm volatile("{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.u32 %0, 1, 0, P1;\n}\n" : "=r"(ok) : "r"(a), "r"(par) : "memory");
    if (ok) return;
  }
  __trap();
}
S1_DEV void s1_mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory"); }
S1_DEV void s1_mbar_expect(uint32_t a, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory"); }
S1_DEV void s1_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
S1_DEV void s1_named_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
S1_DEV void s1_fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
S1_DEV void s1_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
S1_DEV void s1_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
S1_DEV uint64_t s1_sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {   // 128B swizzle, version 1 (Blackwell)
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 46) | (2ull << 61);
}
// instruction descriptor: bf16 x bf16 -> f32, M, N, a/b major (0 = K-major, 1 = MN-major)
S1_DEV constexpr uint32_t s1_idesc(int M, int N, int amaj, int bmaj) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)amaj << 15) | ((uint32_t)bmaj << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
S1_DEV void s1_mma(uint32_t d_tmem, uint64_t da, uint64_t db, uint32_t idesc, int accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               ::"r"(d_tmem), "l"(da), "l"(db), "r"(idesc), "r"(accum) : "memory");
}
S1_DEV void s1_commit(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory"); }
S1_DEV void s1_ld32(uint32_t taddr, float* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]), "=f"(r[8]), "=f"(r[9]), "=f"(r[10]), "=f"(r[11]),
                 "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]), "=f"(r[16]), "=f"(r[17]), "=f"(r[18]), "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]),
                 "=f"(r[23]), "=f"(r[24]), "=f"(r[25]), "=f"(r[26]), "=f"(r[27]), "=f"(r[28]), "=f"(r[29]), "=f"(r[30]), "=f"(r[31])
               : "r"(taddr) : "memory");
}
S1_DEV void s1_st32(uint32_t taddr, const float* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};\n"
               :: "r"(taddr), "f"(r[0]), "f"(r[1]), "f"(r[2]), "f"(r[3]), "f"(r[4]), "f"(r[5]), "f"(r[6]), "f"(r[7]), "f"(r[8]), "f"(r[9]), "f"(r[10]), "f"(r[11]),
                 "f"(r[12]), "f"(r[13]), "f"(r[14]), "f"(r[15]), "f"(r[16]), "f"(r[17]), "f"(r[18]), "f"(r[19]), "f"(r[20]), "f"(r[21]), "f"(r[22]),
                 "f"(r[23]), "f"(r[24]), "f"(r[25]), "f"(r[26]), "f"(r[27]), "f"(r[28]), "f"(r[29]), "f"(r[30]), "f"(r[31]) : "memory");
}
S1_DEV void s1_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
S1_DEV void s1_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }

template <int DS, int RS>
struct Cfg100 {
  static constexpr int QB = 32768, STG = 32768, PB = 16384, RAWB = 8704;
  static constexpr int OFF_Q = 0, OFF_P = QB, OFF_DEQ = QB + PB, OFF_RAW = OFF_DEQ + DS * STG;
  static constexpr int OFF_BAR = (OFF_RAW + RS * RAWB + 127) & ~127;
  static constexpr int SMEM = OFF_BAR + 256 + 1024;
  static constexpr int NT = 128 + 256 + 32;   // softmax WG, 2 dequant WGs, MMA warp
  static constexpr bool OK = SMEM <= 227 * 1024;
};

template <int DS, int RS>
__global__ void __launch_bounds__(416, 1)
uq_sm100_kernel(const UQParams a, int G, int gp_shift, int bme) {
#ifdef S1_ON
  using C = Cfg100<DS, RS>;
  constexpr int D = 128, PAGE = 64, DH = 64, NG = 4;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  const uint32_t sbase0 = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sb = (sbase0 + 1023u) & ~1023u;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
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
  const uint32_t sQ = sb + C::OFF_Q, sP = sb + C::OFF_P, sDeq = sb + C::OFF_DEQ, sRaw = sb + C::OFF_RAW, sBar = sb + C::OFF_BAR;
  auto bar_full = [&](int i) { return sBar + 8 * i; };
  auto bar_dfull = [&](int i) { return sBar + 8 * (RS + i); };
  auto bar_dfree = [&](int i) { return sBar + 8 * (RS + DS + i); };
  const uint32_t bar_q = sBar + 8 * (RS + 2 * DS);
  const uint32_t bar_s0 = bar_q + 8, bar_s1 = bar_q + 16, bar_p = bar_q + 24, bar_pv = bar_q + 32, tmem_slot = bar_q + 40;
  if (tid == 0) {
    for (int i = 0; i < RS; i++) s1_mbar_init(bar_full(i), 1);
    for (int i = 0; i < DS; i++) { s1_mbar_init(bar_dfull(i), 256); s1_mbar_init(bar_dfree(i), 1); }
    s1_mbar_init(bar_q, 256); s1_mbar_init(bar_s0, 1); s1_mbar_init(bar_s1, 1); s1_mbar_init(bar_p, 128); s1_mbar_init(bar_pv, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (wid == 12) {   // TMEM: S0 [0,64), S1 [64,128), O [128,256)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;\n" ::"r"(tmem_slot) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  s1_fence_before();
  __syncthreads();
  s1_fence_after();
  uint32_t tbase;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(tbase) : "r"(tmem_slot));
  const int32_t* bt = a.block_table + (size_t)b * a.max_pages;

  if (wid >= 4 && wid < 12) {
    // ============================ dequant / producer ============================
    const int t = tid - 128;   // 0..255
    auto issue = [&](int it) {
      int s = it % RS;
      size_t page = (size_t)bt[p_lo + it];
      size_t cb = (page * HKV + hk) * PAGE;
      uint32_t d = sRaw + s * C::RAWB;
      s1_mbar_expect(bar_full(s), C::RAWB);
      s1_bulk(d, a.kc + cb * DH, 4096, bar_full(s));
      s1_bulk(d + 4096, a.ks + cb * NG, 256, bar_full(s));
      s1_bulk(d + 4352, a.vc + cb * DH, 4096, bar_full(s));
      s1_bulk(d + 8448, a.vs + cb * NG, 256, bar_full(s));
    };
    if (t == 0) for (int i = 0; i < RS && i < npg; i++) issue(i);
    {
      const uint16_t* q = (const uint16_t*)a.q;
      for (int c = t; c < 128 * 16; c += 256) {
        int r = c >> 4, cc = c & 15;
        int j = r >> gp_shift, gg = r & ((1 << gp_shift) - 1);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (j < K && gg < G) v = *(const uint4*)(q + ((size_t)(b * K + j) * HQ + hk * G + gg) * D + cc * 8);
        uint32_t dst = sQ + (cc >> 3) * 16384 + r * 128 + (((cc & 7) ^ (r & 7)) << 4);
        uint32_t p0 = __byte_perm(v.x, v.z, 0x5410), p1 = __byte_perm(v.x, v.z, 0x7632), p2 = __byte_perm(v.y, v.w, 0x5410), p3 = __byte_perm(v.y, v.w, 0x7632);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(p0), "r"(p1), "r"(p2), "r"(p3) : "memory");
      }
      s1_fence_async();
      s1_mbar_arrive(bar_q);
    }
    const int key = (t & 7) | ((t >> 5) << 3), g = (t >> 3) & 3;
    for (int it = 0; it < npg; it++) {
      const int rs = it % RS, ds = it % DS;
      s1_mbar_wait(bar_full(rs), (it / RS) & 1);
      s1_mbar_wait(bar_dfree(ds), ((it / DS) & 1) ^ 1);
      const uint32_t raw = sRaw + rs * C::RAWB;
      #pragma unroll
      for (int w = 0; w < 2; w++) {
        uint32_t ca = raw + (w ? 4352 : 0) + key * 64 + g * 16;
        uint32_t sa = raw + (w ? 4352 : 0) + 4096 + key * 4 + g;
        uint4 rv; uint32_t sc;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(rv.x), "=r"(rv.y), "=r"(rv.z), "=r"(rv.w) : "r"(ca));
        asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(sc) : "r"(sa));
        uint32_t o[4][4];
        uint32_t wv[4] = {rv.x, rv.y, rv.z, rv.w};
        #pragma unroll
        for (int k = 0; k < 4; k++) deq_word(wv[k], sc, o[k]);
        const uint32_t tile = sDeq + ds * C::STG + w * 16384;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          int C16 = g * 4 + i;
          uint32_t dst = tile + (C16 >> 3) * 8192 + key * 128 + (((C16 & 7) ^ (key & 7)) << 4);
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(o[i][0]), "r"(o[i][1]), "r"(o[i][2]), "r"(o[i][3]) : "memory");
        }
      }
      s1_fence_async();
      s1_mbar_arrive(bar_dfull(ds));
      s1_named_bar(1, 256);
      if (t == 0 && it + RS < npg) issue(it + RS);
    }
    return;
  }

  if (wid == 12) {
    // ============================ MMA issuer (one thread) ============================
    if (lane == 0) {
      const uint32_t id_s = s1_idesc(128, 64, 0, 0), id_o = s1_idesc(128, 128, 0, 1);
      s1_mbar_wait(bar_q, 0);
      auto issue_s = [&](int it) {
        const int ds = it % DS;
        s1_mbar_wait(bar_dfull(ds), (it / DS) & 1);
        s1_fence_after();
        const uint32_t kt = sDeq + ds * C::STG;
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
          uint32_t offa = (ks >> 2) * 16384 + (ks & 3) * 32, offb = (ks >> 2) * 8192 + (ks & 3) * 32;
          s1_mma(tbase + (it & 1) * 64, s1_sdesc(sQ + offa, 16, 1024), s1_sdesc(kt + offb, 16, 1024), id_s, ks > 0);
        }
        s1_commit((it & 1) ? bar_s1 : bar_s0);
      };
      issue_s(0);
      for (int it = 0; it < npg; it++) {
        if (it + 1 < npg) issue_s(it + 1);   // S(it+1) overlaps softmax(it); its TMEM buffer was freed by softmax(it-1)
        s1_mbar_wait(bar_p, it & 1);
        s1_fence_after();
        const int ds = it % DS;
        const uint32_t vt = sDeq + ds * C::STG + 16384;
        #pragma unroll
        for (int kk = 0; kk < 4; kk++)
          s1_mma(tbase + 128, s1_sdesc(sP + kk * 32, 16, 1024), s1_sdesc(vt + kk * 2048, 8192, 1024), id_o, (it > 0) | (kk > 0));
        s1_commit(bar_pv);
        s1_commit(bar_dfree(ds));
      }
    }
    __syncwarp();
    s1_named_bar(3, 160);
    s1_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;\n" ::"r"(tbase) : "memory");
    return;
  }

  // ============================ softmax / epilogue warps (wid 0..3) ============================
  const int row = tid;   // 0..127 = TMEM lane
  const uint32_t tlane = tbase + ((uint32_t)(wid * 32) << 16);
  const int gm = (1 << gp_shift) - 1;
  const int jj = row >> gp_shift, gg = row & gm;
  const int lim = (jj < K && gg < G) ? seq_len - K + jj + 1 : 0;
  const float sc = a.sm_scale * 1.4426950408889634f;
  float m = -INFINITY, l = 0.f;
  for (int it = 0; it < npg; it++) {
    const int sbuf = it & 1;
    s1_mbar_wait(sbuf ? bar_s1 : bar_s0, (it >> 1) & 1);
    s1_fence_after();
    float s[64];
    s1_ld32(tlane + sbuf * 64, s);
    s1_ld32(tlane + sbuf * 64 + 32, s + 32);
    s1_wait_ld();
    const int pos0 = (p_lo + it) * PAGE;
    float mx = -INFINITY;
    #pragma unroll
    for (int j = 0; j < 64; j++) { s[j] = (pos0 + j < lim) ? s[j] * sc : -INFINITY; mx = fmaxf(mx, s[j]); }
    float mn = fmaxf(m, mx), ms = mn == -INFINITY ? 0.f : mn;
    float al = exp2f(m - ms);
    m = mn;
    float ps = 0.f;
    #pragma unroll
    for (int j = 0; j < 64; j++) { s[j] = exp2f(s[j] - ms); ps += s[j]; }
    l = l * al + ps;
    if (it > 0) s1_mbar_wait(bar_pv, (it - 1) & 1);   // previous PV done: P smem free, O stable
    #pragma unroll
    for (int c = 0; c < 8; c++) {
      uint32_t w0 = pack_bf16(s[8 * c + 0], s[8 * c + 1]), w1 = pack_bf16(s[8 * c + 2], s[8 * c + 3]);
      uint32_t w2 = pack_bf16(s[8 * c + 4], s[8 * c + 5]), w3 = pack_bf16(s[8 * c + 6], s[8 * c + 7]);
      uint32_t dst = sP + row * 128 + ((c ^ (row & 7)) << 4);
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(w0), "r"(w1), "r"(w2), "r"(w3) : "memory");
    }
    if (it > 0) {
      s1_fence_after();
      if (__any_sync(0xffffffffu, al != 1.f)) {
        #pragma unroll
        for (int q4 = 0; q4 < 4; q4++) {
          float o[32];
          s1_ld32(tlane + 128 + 32 * q4, o);
          s1_wait_ld();
          #pragma unroll
          for (int j = 0; j < 32; j++) o[j] *= al;
          s1_st32(tlane + 128 + 32 * q4, o);
        }
        s1_wait_st();
      }
    }
    s1_fence_before();
    s1_fence_async();
    s1_mbar_arrive(bar_p);
  }
  // ---- epilogue
  s1_mbar_wait(bar_pv, (npg - 1) & 1);
  s1_fence_after();
  const bool ok_row = (jj < K && gg < G) && row < bme;
  const float inv = l > 0.f ? 1.f / l : 0.f;
  float* mo = a.mo + (rowbase + row) * D;
  #pragma unroll
  for (int q4 = 0; q4 < 4; q4++) {
    float o[32];
    s1_ld32(tlane + 128 + 32 * q4, o);
    s1_wait_ld();
    if (ok_row) {
      #pragma unroll
      for (int p = 0; p < 32; p++) {   // stored position p in 8-block b holds d = 8b + (p even ? p/2 : p/2 + 3)  (e_i, e_{i+4} pairs)
        int blk = (32 * q4 + p) >> 3, pp = p & 7;
        int d = 8 * blk + (pp >> 1) + ((pp & 1) ? 4 : 0);
        mo[d] = o[p] * inv;
      }
    }
  }
  if (row < bme) a.ml[rowbase + row] = l > 0.f ? m + log2f(l) : -INFINITY;
  s1_fence_before();
  s1_named_bar(3, 160);
#endif
}

template <int DS, int RS>
static cudaError_t launch_sm100(const UQParams& p, cudaStream_t s) {
  using C = Cfg100<DS, RS>;
  if constexpr (!C::OK) return cudaErrorInvalidValue; else {
    int bm = uq_block_m(p.K, p.HQ, p.HKV);
    if (p.D != 128 || p.PAGE != 64 || bm > 128) return cudaErrorInvalidValue;
    int dev = 0, maj = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
    if (maj != 10) return cudaErrorInvalidValue;
    auto kern = uq_sm100_kernel<DS, RS>;
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

// stages = dequant stages (2..4); nw unused.
static cudaError_t dispatch_sm100(const UQParams& p, int nw, int stages, cudaStream_t s) {
  (void)nw;
  if (stages == 0) stages = 3;
  if (stages == 2) return launch_sm100<2, 4>(p, s);
  if (stages == 3) return launch_sm100<3, 4>(p, s);
  if (stages == 4) return launch_sm100<4, 6>(p, s);
  return cudaErrorInvalidValue;
}
