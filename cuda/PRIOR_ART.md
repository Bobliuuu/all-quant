# Prior art: fused multi-token (K-query) FP4 KV-cache decode / verify attention on NVIDIA

Purpose: make our claim exactly true. Date of survey: 2026-11-11. Everything below is either (a) read from a
cited source, (b) measured by us (`results/priorart_B300.json`, `results/priorart_H100.json`, scripts
`cuda/bench_priorart.py` + `cuda/modal/modal_priorart.py`), or (c) explicitly marked **unverified**.

## 1. Bottom line (what we may and may not claim)

* **K-query KV sharing is not new and must never be claimed as new.** FlashInfer multi-query / append /
  `q_len_per_req` decode, tree attention, and speculative-decode verify kernels already read the KV once for several queries.
* **A fused K-query verify over an FP4 (NVFP4) paged KV cache already exists on NVIDIA Blackwell, publicly:**
  FlashInfer `trtllm_batch_decode_with_kv_cache(..., backend="trtllm-gen", kv_cache_sf=..., q_len_per_req=K)`
  (the TensorRT-LLM gen kernels shipped as cubins). We ran it on a B300 at our shapes, K=1,2,4,8: 0.066 / 0.077 / 0.097 / 0.094 ms
  (K=1 to K=8 costs 1.4x, while 8 independent decodes would cost 8x). So "first fused FP4 multi-token verify on NVIDIA" is **false**.
  It uses NVFP4 (E4M3 scale per 16, plus an fp32 global scale), an FP8 query (bf16 query had no kernel), and is SM100/SM103 only.
* **BitDecoding-FP4 (MIT) is single-query only** (its own vLLM guard rejects speculative decoding), NVIDIA SM100/103 only, CUDA >= 13.
  It supports NVFP4 and (in code) MXFP4/E8M0. Its "stacked / two-term query" splits ONE query into two FP4 residual terms; it is not K speculative tokens.
* **What is left that is true for us** (and is what the claim should say): the **UltraQuant KV layout** (E2M1 + UE8M0 per-32 scale, Hadamard-rotated K and Q, V unrotated,
  68 B per 128-element vector) with a **fused K-query verify**, running **cross-vendor** (CUDA + AMD gfx950/MI300-class + Intel SYCL). "No public CUDA port of UltraQuant with fused K-query verify
  found" is the accurate wording (upstream vLLM has a ROCm UltraQuant backend, section 1b), not "does not exist". On raw NVIDIA B300 speed, the existing NVFP4 kernels are currently **faster than our Triton fused kernel**
  (see section 4), so do not claim a speed win over them unless the CUDA port measures one.

## 1b. Upstream vLLM UltraQuant backend (read from public vLLM main source, 2026-11-12; we did not run it)

* **It exists and is merged**: vLLM PR #57057 "[Attention] Add UltraQuant 4-bit KV cache backend (FlyDSL D=256)" (merged 2026-10-06), `--kv-cache-dtype ultraquant_4bit`,
  `vllm/v1/attention/backends/ultraquant_attn.py`, ops in `vllm/v1/attention/ops/ultraquant/` and `flydsl_ultraquant_decode.py`. Format: FP4 E2M1 + UE8M0 per-32 scales, Hadamard-rotated keys, bf16 queries only, no attention-sink models.
* **Platform**: the backend is listed in the ROCm platform's backend candidates (`vllm/platforms/rocm.py`); `vllm/platforms/cuda.py` and `xpu.py` contain no UltraQuant reference. The FlyDSL decode kernel requires gfx950 and head_dim 256 (GQA 6/8/16);
  everything else goes through a Triton unified kernel whose QK path is written for native scaled FP4 MFMA on CDNA4. Whether the Triton fallback can be forced on CUDA we did **not** test.
* **K-query fusion**: the FlyDSL entry point takes one query per sequence (`query (B, Hq, D)`); the backend's decode path and its small-chunk continuation path call the Triton unified kernel with `max_query_len=1`
  (a chunk of q tokens becomes q single-query rows, so the KV is re-read per token). The unified kernel itself has query-block support, so we cannot say verify batches never share a read in any configuration;
  we only say the paths we read do not. AMD's own blog describes the same (verify expanded into B*K single-token rows).
* **What this changes for us**: "UltraQuant format on vLLM" is not new and is not ours. Our claim is limited to a standalone fused K-query verify kernel for that layout on AMD gfx942 (Triton), NVIDIA (CUDA) and Intel (SYCL).

## 2. What we searched (so "found no X" is auditable)

* arXiv 2611.09825 abstract page and the BitDecoding-FP4 repo (`github.com/OpenBitSys/BitDecoding-FP4`, commit 4e2907d, 2026-11-09): README, `benchmark/`, `src/nvfp4_decode_kernel/interface.py`, `_decode.py`, `src/nvfp4_vllm/guards.py`. Cloned and executed.
* FlashInfer docs (decode / `trtllm_batch_decode_with_kv_cache` / `xqa_batch_decode_with_kv_cache` / `xqa` / `nvfp4_quantize_paged_kv_cache`), and `flashinfer-python 0.7.4` executed on B300 and H100.
* vLLM FlashInfer attention backend API docs; SGLang `quantized_kv_cache` docs, LMSYS NVFP4-KV blog, SGLang PR #21601 (open at time of search); TensorRT-LLM docs (key features, feature-combination matrix);
  FlashAttention-3 README/paper and the vLLM FP8-KV blog; SageAttention3 paper (arXiv 2505.11594).
* Web search for the combination "UltraQuant CUDA port / MXFP4 UE8M0 KV K-query verify". Not exhaustively done by us: GitHub code search and the forks of the vLLM/SGLang UltraQuant PRs (including SGLang PR #41949)
  were checked by the coordinating agent, not re-run here. Upstream vLLM's UltraQuant backend was re-read from public source on 2026-11-12: see section 1b.
* Not searched / **unverified**: closed-source NVIDIA internals, TensorRT-LLM source (only docs were read), Mooncake/LMDeploy/other engines.

## 3. Per-project facts

Legend: Y = yes with source, N = no with source, ? = unverified.

| Project | FP4 KV | K-query verify sharing one KV read | Paged | Scale format | MI300X / Intel |
|---|---|---|---|---|---|
| **BitDecoding-FP4** ([arXiv](https://arxiv.org/abs/2611.09825), [repo](https://github.com/OpenBitSys/BitDecoding-FP4), MIT) | Y NVFP4; MXFP4 present in code (`_block_scale.py`, E8M0, 32-group), README only names NVFP4 | **N**: `fp4_decode` takes `[rows, heads_q, head_dim]`, one token per row; `guards.py` raises "speculative decoding is not supported by the NVFP4 KV cache ... the decode kernel attends one query token per sequence" | Y, page size fixed 128 (`PAGE_SIZE = 128`), BF16 sink + rolling BF16 tail | NVFP4: E4M3 per 16 (+ per-tensor handling); MXFP4: E8M0 per 32 | N: needs B200/B300 (SM100/103), CUDA >= 13.0, driver 580+ (README); built on CuTe-DSL FA4 code |
| **FlashInfer** ([attention API](https://docs.flashinfer.ai/api/attention.html)) | Y on SM100 via the trtllm-gen path: `kv_cache_sf` NVFP4, `nvfp4_quantize_paged_kv_cache` ([doc](https://docs.flashinfer.ai/generated/flashinfer.quantization.nvfp4_quantize_paged_kv_cache.html)). Quantizer errors "requires SM100 or newer" on H100 (measured). | **Y, measured** (see section 4): `q_len_per_req=K` on `trtllm_batch_decode_with_kv_cache` ([doc](https://docs.flashinfer.ai/generated/flashinfer.decode.trtllm_batch_decode_with_kv_cache.html)); also ragged `cum_seq_lens_q`; `xqa` path takes a draft-block `mask` ([xqa doc](https://docs.flashinfer.ai/generated/flashinfer.xqa.xqa.html)); `BatchPrefillWithPagedKVCacheWrapper` qo_len=K | Y (HND/NHD paged; trtllm-gen NVFP4 HND) | NVFP4 = E4M3 per 16 + fp32 global scale; **not UE8M0 / not 32-group** | N (CUDA only) |
| **TensorRT-LLM** ([docs](https://nvidia.github.io/TensorRT-LLM/latest/legacy/key-features.html)) | ? The NVFP4 KV cache shipped to FlashInfer as the "trtllm-gen" kernels (FlashInfer calls them that); TRT-LLM's own NVFP4-KV/spec-decode combination is **unverified** (docs fetched do not state it; the feature-combination matrix has no NVFP4-KV row) | Spec decode (MTP/Eagle, linear + dynamic trees) is supported in TRT-LLM generally (matrix), combination with NVFP4 KV ? | Y (paged KV is its core cache) | NVFP4 as above | N |
| **vLLM** ([FlashInfer backend docs](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/flashinfer/)) | Y `kv_cache_dtype=nvfp4` in the FlashInfer backend on SM100 (docs list `nvfp4`) | With the BitDecoding plugin spec decode is rejected (above). Stock-vLLM spec decode and NVFP4 KV + spec decode: ? (not read) | Y | NVFP4 (packed fp4 + fp8 block scales in last dim) | AMD: UltraQuant upstream path (per coordinator); Intel ? |
| **SGLang** ([quantized KV doc](https://docs.sglang.io/advanced_features/quantized_kv_cache.html), [LMSYS blog](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache)) | Y `--kv-cache-dtype fp4_e2m1` / `nvfp4`; SM120 and SM100/103 | ? Blog does not mention speculative decoding; PR #21601 (open when searched) proposes MTP with NVFP4/FP8 KV on SM120 | Y (paged KV system; decode-kernel paging detail unverified) | NVFP4 E4M3 per 16 + fp32 tensor scale (blog); the doc text calls it "MXFP4" loosely | N |
| **FlashAttention-3** ([paper](https://arxiv.org/abs/2407.08608), [repo](https://github.com/Dao-AILab/flash-attention)) | N (FP8 e4m3 forward; no FP4) | `flash_attn_with_kvcache` takes `seqlen_q > 1` (works as multi-query), fp8 KV with BF16 query ? | Y (`page_table`, page size multiple of 256) | per-head fp8 descale | N (Hopper only) |
| **SageAttention / SageAttention3** ([paper](https://arxiv.org/abs/2505.11594), [repo](https://github.com/thu-ml/SageAttention)) | FP4 *microscaling for Q,K,P,V inside attention* (RTX 5090/Blackwell), not a stored FP4 KV cache | N: prefill/diffusion-style attention; KV-cache decode support **unverified** (not read) | ? | NVFP4-style 1x16 groups | N |

Notes on the table:
* FlashInfer's trtllm-gen cubins are downloaded at runtime from NVIDIA's artifactory; with `flashinfer-python 0.7.4` on SM103 the NVFP4 kernel existed for
  `numTokensPerPage=64`, group ratio 8 (Hq=64/Hkv=8), tileSizeQ 8..64 (K=1..8) with an **FP8 query**. With a bf16 query the same call fails with
  `Missing TRTLLM-GEN kernel (decode) ... tileSizeQ=...` (all K).
* Correctness of any of these kernels' outputs at our shapes was **not** checked by this benchmark (timing only). K-query reads=1 for FlashInfer is inferred from timing (K=8 costs ~1.4x K=1), not from reading the kernel.

## 4. Head-to-head measurements (our shapes: B=8, ctx=32768, Hq=64, Hkv=8, D=128, page 64 unless noted)

Method: CUDA events, 30 timed iterations after 10 warmup, no CUDA graph, bf16 q (fp8 q where stated). `GB/s` = KV bytes of that format / ms.
KV bytes: bf16 1074 MB, fp8 537 MB, NVFP4 (E4M3 per 16) 302 MB (BitDecoding measured: 268 MB data + 33.5 MB scales), our UE8M0-per-32 format 285 MB (68 B/vec).
HBM peak = torch `copy_`/read-sum on the same GPU in the same job: **B300 6626 GB/s** (a conservative peak; bf16 decode exceeds it by up to 5%), **H100 80GB 3153 GB/s**.
Raw rows: `results/priorart_B300.json|.log`, `results/priorart_H100.json|.log`.
Caveat found while running: Modal's `gpu="H100"` can silently return an **H200** (it did; those partial runs are not used). The reported H100 numbers come from `gpu="H100!"` and report `NVIDIA H100 80GB HBM3`.

### B300 (SM103, 148 SMs), K queries per sequence

| Implementation (KV format) | K=1 ms | K=2 | K=4 | K=8 | GB/s / % of 6626 at K=1 | K-query verify |
|---|---|---|---|---|---|---|
| FlashInfer trtllm-gen, **NVFP4**, fp8 q, `q_len_per_req=K` | 0.0662 | 0.0772 | 0.0970 | 0.0943 | 4561 / 69% (K=8: 3201 / 48%) | **yes** (one pass, K=8 only 1.4x K=1) |
| BitDecoding-FP4 `fp4_decode`, NVFP4, K independent decodes (serial) | 0.0676 | 0.1314 | 0.2611 | 0.5207 | 4466 / 67% (re-reads KV K times: ~4650 GB/s effective) | **no** (single query/sequence) |
| BitDecoding-FP4, K*B rows in one launch (page table repeated) | - | 0.1226 | 0.2255 | 0.4246 | still K KV passes; L2 overlap helps (5690 GB/s effective at K=8) | no (rows are independent) |
| FlashInfer trtllm-gen fp8 KV, fp8 q | 0.0804 | 0.0813 | 0.0827 | 0.0868 | 6676 / 101% | yes |
| FlashInfer trtllm-gen fp8 KV, **bf16 q** | 0.1397 | 0.1491 | 0.1509 | 0.1393 | 3842 / 58% | yes |
| FlashInfer trtllm-gen bf16 KV | 0.1534 | 0.1543 | 0.1551 | 0.1593 | 7000 / 106% | yes |
| FlashInfer `BatchDecodeWithPagedKVCacheWrapper` bf16 / fp8 KV | 0.1627 / 0.1770 | - | - | - | 6601 / 99.6%, 3034 / 46% | single query |
| FlashInfer `BatchPrefillWithPagedKVCacheWrapper` (fa2) qo_len=K causal, bf16 | 0.1634 | 0.1642 | 0.2528 | 0.2580 | 6570 / 99% | yes (K-query) |
| same, fp8 KV | 0.1816 | 0.1824 | 0.3379 | 0.3433 | 2957 / 45% | yes |
| *Our existing Triton fused kernel (results/b300_b8_c32k_tuned.json, 68 B/vec)* | 0.288 | 0.421 | 0.532 | 0.421 | 990 / 15% | yes |

Reading: at these shapes the best public NVFP4 paths run at 0.066-0.097 ms for K=1..8. Our current Triton kernel is 3-4x slower at K=1 and 4-6x slower at K=8.
Whatever the CUDA port achieves has to be compared against **these** numbers, not against bf16. NVFP4 K=1 beats bf16 trtllm-gen by 2.3x (bytes ratio 3.6x), so even the vendor kernels are not at HBM peak in FP4.
FlashInfer's NVFP4 K>=4 is 47-59% of peak, i.e. there is headroom, but it is not zero-competition territory.

### H100 80GB (SM90, 132 SMs)

| Implementation | K=1 ms | K=2 | K=4 | K=8 | note |
|---|---|---|---|---|---|
| FlashInfer decode wrapper bf16 / fp8 KV | 0.3473 / 0.2051 | - | - | - | 3091 GB/s = 98% / 2618 = 83% |
| FlashInfer xqa `q_len_per_req=K` (draft-block mask), bf16 KV | 0.3497 | 0.3568 | 0.3576 | 0.4262 | 95-97% of peak |
| FlashInfer xqa, fp8 KV (bf16 q only; fp8 q rejected) | 0.2352 | 0.2260 | 0.2269 | 0.2124 | 72-80% |
| FlashInfer prefill-wrapper fa2 qo_len=K, bf16 / fp8 KV | 0.3502 / 0.2046 | 0.3517 / 0.2058 | 0.3614 / 0.3658 | 0.3689 / 0.3718 | fp8 K>=4 falls to 46% |
| FlashInfer prefill-wrapper **fa3** backend, bf16 | 1.7122 | 1.7751 | 1.7352 | (not reached) | slow at page 64; fp8 KV: JIT error recorded |
| NVFP4 KV on H100 | - | - | - | - | **not possible in FlashInfer**: `fp4_quantize requires SM100 or newer; got SM90` |
| BitDecoding-FP4 | - | - | - | - | **not run**: SM100/103 tcgen05 only (its own benchmark exits for non-SM10x) |
| FlashAttention-3 (`flash_attn_interface`) | - | - | - | - | **not run**: no pip wheel (`flash-attn-3`, `flash_attn_3`, `fa3-fwd` not found); source build (Hopper, ~hour) not attempted. FlashInfer's FA3-template backend is reported above as a proxy |

### What failed / did not run
* BitDecoding-FP4 first failed with `nvvm.mbarrier.arrive.expect_tx ... is not a valid CmpIPredicate` because `pip install flashinfer-python` pulled `nvidia-cutlass-dsl-libs-cu13 4.8.1` next to BitDecoding's pinned `nvidia-cutlass-dsl 4.6.0`. Pinning the libs to 4.6.0 fixes it, but then FlashInfer 0.7.4 (needs cutlass DSL >= 4.8, `FastDivmodDivisorV2`) cannot share the environment: BitDecoding and FlashInfer needed **separate images** (two B300 jobs).
* BitDecoding on H100: not applicable (above). BitDecoding MXFP4 path was not benchmarked (the benchmark only builds NVFP4 pages).
* FlashInfer NVFP4 with bf16 query on B300: no kernel (all K).
* FlashInfer fa3-backend fp8-KV prefill on H100: JIT error; K=8 fa3 rows not reached (JIT of the Hopper template exceeded the section time cap).
* FlashAttention-3 native: not installable (above). SageAttention: not benchmarked (not a KV-cache decode kernel as far as read; **unverified**).
* Run-time noise: single run, no CUDA graphs, no confidence intervals; differences below ~5% should not be read into.

## 5. Suggested wording for our claim

> Fused K-query (speculative-verify) paged decode attention over an **UltraQuant-format** FP4 KV cache (E2M1 + UE8M0 per-32 scales, Hadamard-rotated K/Q, 68 B per 128-vector), implemented on NVIDIA CUDA, AMD and Intel from one specification. In the public sources we checked (section 2), we found no CUDA implementation of this layout with fused K-query verify; upstream vLLM's UltraQuant backend is ROCm-oriented (section 1b). K-query KV sharing in general, and NVFP4 (E4M3-per-16) multi-token decode on Blackwell (FlashInfer trtllm-gen), already exist.

Do not say: "first FP4 K-query verify", "first on NVIDIA", "BitDecoding cannot do FP4 decode" (it can, K=1), or "faster than NVFP4 kernels" unless measured on the CUDA port.
