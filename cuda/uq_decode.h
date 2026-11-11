// UltraQuant native CUDA: fused multi-token (K-query) FP4/UE8M0 paged decode attention.
// Same op and cache layouts as uqmt/attn.py (Triton reference):
//   kc/vc  uint8 [pages, Hkv, PAGE, D/2]   low nibble = even element (e2m1)
//   ks/vs  uint8 [pages, Hkv, PAGE, D/32]  UE8M0, value 2^(s-127)
//   q      bf16  [B, K, Hq, D]   block_table int32 [B, max_pages]
//   seq_lens int32 [B] including the K spec tokens; query j sees seq_len-K+j+1 keys.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct UQParams {
  const void* q;           // bf16 [B,K,Hq,D]
  const uint8_t* kc;
  const uint8_t* ks;
  const uint8_t* vc;
  const uint8_t* vs;
  const int32_t* block_table;
  const int32_t* seq_lens;
  float* mo;               // fp32 [B,Hkv,nsplit,BLOCK_M,D]  scratch
  float* ml;               // fp32 [B,Hkv,nsplit,BLOCK_M]    scratch (log2 sum-exp)
  void* out;               // bf16 [B,K,Hq,D]
  int B, K, HQ, HKV, D, PAGE, max_pages;
  int nsplit;              // any value 1..128
  float sm_scale;
  int variant;             // 0 auto, 1 = smem-dequant design, 2 = warp-independent design W
  int mb;                  // design W: 16-row m-blocks per warp (0 = auto); variant 3: # dequant warpgroups (0 = 2)
  // cuda2 (appended; zero-init = UE8M0/32 format as before)
  int fmt;                 // 0 = UE8M0 per 32 (UltraQuant), 1 = NVFP4: E4M3 scale per 16 (ks/vs [.., D/16]) + per-tensor fp32 k_ts/v_ts
  float k_ts, v_ts;        // fmt==1: k_ts is folded into sm_scale, v_ts multiplies the output (set both, 1.0 = none)
};

// Rows per CTA tile: max(16, pow2(K * pow2(G))).  Scratch sizes depend on it.
int uq_block_m(int K, int HQ, int HKV);
size_t uq_mo_elems(const UQParams& p);   // fp32 elements of mo
size_t uq_ml_elems(const UQParams& p);   // fp32 elements of ml

// nwarps: 0 = auto (4, or 8 for BLOCK_M=128); stages: 0 = auto (3).
// Returns cudaErrorInvalidValue for unsupported (D, PAGE, K*G) combos.
cudaError_t uq_decode_launch(const UQParams& p, int nwarps, int stages, cudaStream_t s);
// CUDA-graph friendly entry: no allocation, no sync, no host reads of device memory.  Workspace = fp32 mo/ml scratch (uq_workspace_bytes(p)).
size_t uq_workspace_bytes(const UQParams& p);
cudaError_t uq_decode_launch_ws(UQParams p, void* workspace, int nwarps, int stages, cudaStream_t s);
int uq_has_sm90();   // 1 if this build contains the sm_90a wgmma variant (variant 3) and the current device is sm_90
