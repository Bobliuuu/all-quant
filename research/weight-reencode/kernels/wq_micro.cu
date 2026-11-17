// Hopper loop-overhead micro-benchmark for the int4 fused GEMV (wq_gemv.cu C_INT4).
// Every warp streams int4 tiles from an L2-resident buffer (so DRAM never limits) and runs one of:
//   MODE 0  load only (uint4 per lane per 32 weights, XOR-folded so nothing is dead-code eliminated)
//   MODE 1  load + int4 dequant (LOP3 | HSUB2 | HFMA2 per pair), XOR-folded
//   MODE 2  load + dequant + 4 x mma.sync m16n8k16 with activation fragments held in registers
//           (no activation loads, no per-tile address math, no param loads)
// Compared with the full k_fixed<C_INT4> loop (activation-fragment LDGs, param LDGs, addressing, reduction)
// this splits int4's ~7.9 issued lane-instructions per weight into load / dequant / mma / everything else.
// A wgmma-fed pipeline keeps the dequant (A from registers) but drops the per-warp mma.sync issue (one wgmma per
// warpgroup per 64x8x16) and the activation-fragment loads (B from shared memory via descriptor).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

typedef __nv_bfloat162 bf162;
__device__ __forceinline__ uint32_t as_u32(bf162 v) { return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ bf162 as_b2(uint32_t v) { return *reinterpret_cast<bf162*>(&v); }
__device__ __forceinline__ uint4 ldg_nc4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int MODE>
__global__ void __launch_bounds__(256) k_micro(const uint4* w, int mask, int iters, uint32_t sig2, uint32_t mu2,
                                              uint32_t xb0, uint32_t xb1, float* out) {
  const int lane = threadIdx.x & 31;
  const int gw = (blockIdx.x * 8 + (threadIdx.x >> 5));
  uint32_t fold = 0;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  uint4 cur = ldg_nc4(w + ((size_t)(gw & mask) * 32 + lane));
#pragma unroll 1
  for (int i = 0; i < iters; i++) {
    uint4 nxt = ldg_nc4(w + ((size_t)((gw + i + 1) & mask) * 32 + lane));
    const uint32_t r[4] = {cur.x, cur.y, cur.z, cur.w};
    if (MODE == 0) {
      fold ^= r[0] ^ r[1] ^ r[2] ^ r[3];
    } else {
#pragma unroll
      for (int s = 0; s < 4; s++) {
        uint32_t a[4];
#pragma unroll
        for (int q = 0; q < 4; q++) {
          uint32_t v = ((r[s] >> (4 * q)) & 0x000f000fu) | 0x43004300u;
          v = as_u32(__hsub2(as_b2(v), as_b2(0x43004300u)));
          a[q] = as_u32(__hfma2(as_b2(v), as_b2(sig2), as_b2(mu2)));
        }
        if (MODE == 1) fold ^= a[0] ^ a[1] ^ a[2] ^ a[3];
        else mma16816(acc, a, xb0, xb1);
      }
    }
    cur = nxt;
  }
  if (fold == 0x12345678u || acc[0] + acc[1] + acc[2] + acc[3] == 1.2345f) out[gw * 32 + lane] = (float)fold + acc[0];
}

void micro(int mode, torch::Tensor w, int mask, int blocks, int iters, torch::Tensor out) {
  cudaStream_t st = c10::cuda::getCurrentCUDAStream();
  const uint4* p = reinterpret_cast<const uint4*>(w.data_ptr());
  const uint32_t sig2 = 0x3c003c00u, mu2 = 0x00000000u, xb = 0x3f803f80u;  // bf16 pairs
  if (mode == 0) k_micro<0><<<blocks, 256, 0, st>>>(p, mask, iters, sig2, mu2, xb, xb, out.data_ptr<float>());
  else if (mode == 1) k_micro<1><<<blocks, 256, 0, st>>>(p, mask, iters, sig2, mu2, xb, xb, out.data_ptr<float>());
  else k_micro<2><<<blocks, 256, 0, st>>>(p, mask, iters, sig2, mu2, xb, xb, out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("micro", &micro); }
