# Porting log: HyperQuant GPU kernels to ROCm (HIP) and SYCL

Upstream: github.com/moonmath-ai/HyperQuant @ 227f3f6 (vendored unmodified in `upstream/`).
Reference: the upstream CUDA kernels, driven unmodified by `cuda/backend_cuda.cu`.
Method (same as ~/code/ultraquant-more): one CPU reference (`ref/hq_ref.hpp`) that mirrors the
CUDA arithmetic order, one set of shared vectors (`vectors/*.hqv`, made by `ref/gen_vectors.cpp`),
one test driver and one bench driver (`common/test_main.cpp`, `common/bench_main.cpp`) compiled
against each backend through the same interface (`common/backend.hpp`).  Every number in the
results tables is parsed from `results/`.

## Kernels ported

| upstream file | kernels | HIP (`hip/`) | SYCL (`sycl/`) |
|---|---|---|---|
| cuda/stage2_cuda_encoder.cu | compute_stream_lengths(_and_cache_u8), write_streams(_from_cache_u8), thrust scan | hipify + rocThrust | hand port, own work-group scan |
| cuda/stage2_cuda_decoder.cu | decode_stage2 (z), inverse_e8int (bf16), fused (bf16/fp8/int8), fused coop (smem staging), interleaved layout, integration launcher | hipify + fp8 fix + 32-lane `__syncwarp` | hand port |
| cuda/stage3_cuda_decoder.cu | self-synchronising decoder (host chunk plan + 4 phases) | hipify | hand port |
| cuda/rht_cuda.cu | warp-shuffle FWHT, H in {256,512,1024,2048} | hipify (`__shfl_xor(..,32)`) | sub-group-size-generic (16/32/64) |
| cuda/lattice_ext.cu | rowmax_cast (int8), dequant, fp8_rowmax_cast, cuBLASLt int8 TN GEMM | hipBLASLt (+ dot4 fallback) | SLM-tiled int8 GEMM (not tuned) |
| integrations/llama/fused_decode_gemv.cu | rice_fused_gemv (B=1), rice_fused_gemm_ws (warp-specialised, B<=8 tile) | hipify | hand port |
| integrations/llama/rice_fused_attention.cu | Flash-Rice two-pass decode attention (T<=512) | hipify (`__shfl_down(..,32)`) | hand port, 32-lane logical warps |

HIP = `tools/hipify_hq.sh` (mechanical sed, re-runnable) + `tools/hip_fixups.py` (every hand fix
listed with its reason) + `hip/hq_hip_compat.hpp`.  The HIP test backend is generated from the CUDA
one (`tools/make_hip_backend.py`), so both drive their kernels identically.

## Correctness criteria (identical for every backend)
* bit-exact: bitstream words, stream offsets, k table, total_bits, z symbols, int8/int32/fp8 bytes,
  all bf16 decode outputs, dequant, RHT for H in {256,1024};
* RHT H in {512,2048}: <= 1 bf16 ulp (CUDA uses `rsqrtf`, ports use a correctly rounded 1/sqrt(H));
* fused GEMV / GEMM-WS: <= 1 bf16 ulp (achieved: 0 ulp everywhere);
* fused attention: max|err| <= 2e-4 * max|ref| + 1e-6 (CUDA `expf` is not correctly rounded).

## Results per backend: see results/kernels/*_test.json and README

## Upstream issues found while porting (all reproduced on the CUDA reference)
1. **Fused attention, GQA > 4**: `rice_fused_attention.cu` launches a fixed 128-thread block
   (= 4 q-heads x 32 lanes) but accepts `n_q_per_kv` up to 8, so q-heads 4..7 of every KV head are
   never computed (zeros).  Vector `attn_h4g7_t128` (GQA 7, i.e. Qwen2.5-7B): 12/28 rows wrong on
   CUDA, and identically wrong on every port (the ports are faithful; no silent fix).
2. **Stage-3 decoder with Rice k = 0**: 326/65536 symbols differ from the bitstream's true
   symbols (vector `codec_a13_k0_s256`), on CUDA and identically on the ports.  k >= 1 is fine;
   the integration never uses k = 0.
3. **`install_rice_attention_hooks` cannot work as shipped**: it binds `make_rice_attn_forward(...)`
   (a closure with no `self`) via `types.MethodType`, so every call raises "multiple values for
   argument 'hidden_states'".  It also reads `module.num_heads` (removed from LlamaAttention in
   current transformers) and returns 3 values where LlamaDecoderLayer unpacks 2.
4. **Weights + KV do not compose**: `build_rice_kv_cache` only finds attention layers whose
   `k_proj` is an `nn.Linear`; after `convert_linears` there are none -> IndexError.
5. **Decoders never terminate if sps does not divide total_symbols** (the last stream reads into
   the zero padding forever); all upstream callers satisfy it.
6. Symbols z > 255 (heavy-tailed weights) are written in full by the encoder but masked to 8 bits
   by every decoder (`& 0xFF`), i.e. they wrap, not clamp (vector `codec_heavy_a13_k2`: 34 such
   symbols; bit-exact behaviour reproduced by the ports).

## Pitfalls hit (and fixes)
* **fp8**: MI300X's native fp8 is E4M3-FNUZ, not the OCP E4M3 that `__nv_fp8_e4m3` emits.  Stored
  bytes must match CUDA, so both ports use a software RN+satfinite conversion (`hq_num.hpp`),
  verified bit-exact against CUDA on the vectors.
* **bf16**: carried as `uint16_t` with portable RN conversions (identical to `__float2bfloat16_rn`).
* **Intel SPIR-V division**: the first Arc B390 run failed 3 lattice_ext tests (`amax/(127*alpha)`,
  `amax/448`, `448/amax` off by 1 ulp): OpenCL/SPIR-V fp32 division is not correctly rounded by
  default.  Fixed with `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` -> bit-exact.
* **FMA contraction**: nvcc contracts `a*b+c` by default; HIP clang does too (fast-honor-pragmas);
  SYCL is built with `-ffp-contract=fast`, and the encoder's distance sums use explicit `fma` in
  the SYCL port and in the CPU reference.  Without this the E8 coset choice could flip on ties.
* **nvcc `-x`** applies to all inputs (the last one wins), so `-x cu a.cu -x c++ b.cpp` compiled the
  .cu files as C++; hipcc similarly applies its CUDA wrapper headers to plain C++.  Build per object.
* **wave64**: RHT and attention shuffles take `width = 32`; the coop decoder's `__syncwarp` becomes a
  wavefront-scope fence + wave barrier; the 64-thread warp-specialised GEMM is ONE wavefront on
  MI300X, so producer and consumer serialise (correct, no overlap).
* **SYCL sub-groups**: AMD has only sub-group 64, Intel 16/32, NVIDIA 32.  The RHT is templated on
  the sub-group size with the identical butterfly stage order (bit-identical for any size); the
  attention keeps 32-lane logical warps and reduces with `shift_group_left` inside a sub-group of
  >= 32 lanes; the coop decoder is restructured so no work-item returns before the barrier.
* **DPC++ nightly-2026-11-04** segfaults in `llvm-spirv` when one compile targets both
  `nvptx64-nvidia-cuda` and `spir64_x86_64` with the precise-division flags; separate binaries work.
* **AMD box lock**: `/root/gpu.lock` is a stale plain file used by other tenants with `flock`; the
  agreed convention is `mkdir /root/gpu-lock.d`.  `tools/amd_gpu_run.sh` takes both and refuses to
  start while the push-boxes `pb-vllm` container is up.
