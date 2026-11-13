# Upstream UltraQuant (vLLM / SGLang) vs AllQuant: format and bit-compatibility

Sources (read 2026-11-11): vLLM `main` @ `3ea131ca2c` (PR #57057 "UltraQuant 4-bit KV cache backend", plus #61649, #61168 CPU),
files `vllm/v1/attention/ops/ultraquant/{format,reference,triton_store,triton_dequant,triton_unified_attention}.py` and
`vllm/v1/attention/backends/ultraquant_attn.py` (vendored verbatim under `cuda/compat/upstream/`, Apache-2.0, for reference);
pip `vllm==0.33.0` (ships the same package; this is what the Modal probe imported); SGLang PR #41949 (open, head `d1f16fc`).

## 1. Upstream format (exact)

| Item | vLLM (Triton store/decode, FlyDSL on gfx950) | SGLang PR #41949 |
|---|---|---|
| Element code | FP4 E2M1, 4-bit pattern `s eem` (0,.5,1,1.5,2,3,4,6; bit3 = sign) | same |
| Nibble order | low nibble = even element, high = odd (`lo | hi<<4`) | same ("low nibble first") |
| Group size | 32 along head_dim (fixed: matches scaled F8F6F4 MFMA) | 32 |
| Scale | UE8M0 byte, value `2^(b-127)`; **byte 0 = zero sentinel (decodes to 0.0)**; exponent clamped [-126,127] | same |
| Scale rule | `s_raw = c * absmax(group)`, **c = 0.156**; `exp = floor(log2(s_raw) + 0.5)` (round, half up); byte = `exp+127`; all-zero group -> byte 0 | same (`CONSTANT_C = 0.156`, floor(+0.5)) |
| Code rule | `x/2^exp` snapped to nearest of the 15 signed levels via `bucketize(midpoints)` (right=False); no explicit clamp (>6 -> 6); result +0 never -0 | same |
| Block (page) layout | AoS slot per (token, head): `[K codes D/2][K scales D/32][V codes D/2][V scales D/32]` = 136 B (D=128), 272 B (D=256); cache tensor `uint8 [num_blocks, block_size, Hk, slot_size(padded)]` | SoA pools (separate k/v code and scale buffers, `[slots, ..., D/2]` and `[slots, ..., D/32]`), 136 B/token/head/side at D=256 incl. scales |
| K rotation | Hadamard, **orthonormal Sylvester** (`H/sqrt(D)`, `H H^T = I`, natural/Sylvester order, no random signs), applied to K at store in fp32 (WHT butterfly in-kernel; `k @ H^T`), scales computed after rotation | same (store kernel fuses rotation) |
| Q | raw Q is rotated `q @ H` (H symmetric) in fp32 at decode **then cast to FP8 E4M3 (scale 1)** for the scaled MFMA (`tl.dot_scaled(Q e4m3, K e2m1+scale)`) | Q rotated into the K basis; bf16 queries in FlyDSL path |
| V | **not rotated**, same FP4/UE8M0 quantiser | same |
| Softmax scale | caller-supplied `scale` (= `head_dim**-0.5` normally) applied to the QK product in the rotated basis; valid because H is orthonormal; optional per-head sinks | same |
| Per-token norm fold | none | none |
| BF16 layers | handled outside the op by the generic `--kv-cache-dtype-skip-layers` (layer indices or attention-type names such as `sliding_window`); skipped layers keep an unquantised cache. No built-in "first/last 2" default in the UltraQuant op. | n/a (rejects sliding-window hybrid pools, MLA, spec-decode) |
| Paged addressing | `block_table[seq, blk]`, `slot = blk*block_size + off`; decode via varlen `query_start_loc` and `seq_lens` (causal: query `j` of `qlen` sees `seq_len-qlen+j+1` keys) | Triton-backend metadata |

Notes: upstream FP4 decode is arithmetic (bit-exact to the table). Upstream "decode" on non-gfx950 hardware is the Triton unified
kernel, which uses `tl.dot_scaled`; it ran fine on an H100 (sm_90, emulated upcast path).

## 2. Our format (uqmt/attn.py, cuda/uq_cuda.py)

`kc/vc uint8 [pages, Hkv, PAGE, D/2]`, `ks/vs uint8 [pages, Hkv, PAGE, D/32]` (SoA, head-major inside a page), low nibble = even,
UE8M0 `2^(s-127)` (no zero sentinel), `q bf16 [B,K,Hq,D]`, quantiser default `ceil(log2(amax/6))` (never clips); `uqmt/quant.py`
also has `rule="paper"` = `round(log2(c*amax))`.

## 3. Verdict

**Cache bytes are layout-compatible and semantically identical at the decode side; the *default quantiser* is not bit-identical
(different scale rule). With `rule="paper", c=0.156` plus a -0 canonicalisation the quantiser is bit-identical to upstream.**
Decode outputs agree to bf16 rounding (not bit-for-bit: different accumulation order / P rounding / Q precision).

Evidence (H100 sm_90 and B300 sm_103, torch 2.14, triton 3.8, vllm 0.33.0; `results/cuda_compat_H100.{json,log}`; CPU check
`results/cuda_compat_paper_rule_cpu.json`):

1. Upstream's own Triton store kernel vs upstream torch reference: V codes+scales bit-identical; K codes+scales 100% equal (D=128/256).
   So the torch reference in `cuda/compat/upstream/reference.py` is a valid oracle.
2. Layout converter round trip (upstream AoS `[blocks,bs,Hk,slot]` -> our SoA `[pages,Hkv,PAGE,*]` -> back) is byte-identical
   (`layout_roundtrip_identical: true` in every config). No requantisation is needed, only a byte permute (section 4).
3. Our default quantiser (`ceil`, "ours") vs upstream on the same fp32 inputs (unrotated, 20000 rows x D):
   scale bytes equal 40-41% (ours is always upstream or upstream+1: diff histogram {0: 32098, +1: 47902} at D=128);
   packed bytes equal 37%. Using upstream's scale with our code mapping gives 91.6% equal packed bytes (the rest is the -0 issue below).
   Relative MSE of dequantised values: upstream 1.70e-2 (4.6% of elements clip to +-6), ours 1.37e-2 (0% clip).
4. `uqmt.quant.quantize(x, "paper", 0.156)` vs upstream reference (CPU, 4 distributions x D in {128,256}):
   scale bytes 100% equal (except all-zero groups: upstream byte 0, ours a tiny exponent, both dequantise to 0).
   Packed bytes 91.5-94% equal before, **100% / bit-identical after mapping nibble 0x8 (-0) to 0x0** on Gaussian, log-normal-group and
   zero-group data. The only residual is exact-midpoint ties (bf16-rounded inputs: 98.8% equal): `bucketize` sends a negative tie to
   the larger magnitude, our `argmin` sends it to the smaller; harmless (1 code step, measure-zero on fp32 data).
5. Decode (D=128, Hq=8, Hkv=2, page 64, random block tables, K=1 and K=4, ragged seq_lens up to 2100), upstream Triton kernel
   (`ultraquant_unified_attention`, Q rotated + cast to e4m3 internally) on an upstream-quantised cache vs ours on the converted cache
   (Q fed as the same e4m3-rounded rotated Q, in bf16):

   | pair | rel L2 | max abs |
   |---|---|---|
   | upstream kernel vs fp32 reference on dequantised cache | 2.1e-3 | 2.9e-3 |
   | our Triton snapshot vs upstream kernel | 1.3e-3 to 2.2e-3 | 3.9e-3 (1 bf16 ulp) |
   | our CUDA kernel (`uq_decode_cuda`) vs upstream kernel | 2.3e-3 | 3.9e-3 |
   | our Triton vs fp32 reference | 2.1e-3 | 2.9e-3 |

   All are at bf16 output-rounding level (eps 2^-8 = 3.9e-3); none is bit-identical, including upstream vs its own fp32 oracle.
   Our quantiser's cache fed to the *upstream* kernel (after the reverse converter) vs our Triton kernel: rel L2 1.1e-3 to 2.1e-3.
   Attention-output error vs unquantised truth (random Gaussian K/V, 4-bit, so large in absolute terms): upstream quantiser 0.18, our
   `ceil` quantiser 0.16 (our rule is better here because it never clips).
6. Page size: our CUDA kernel rejects PAGE=16 (`invalid argument`; supports 64), our Triton path works at 16 and 64; upstream
   works for any block size. D=256: upstream OK; our Triton asserts head_dim in {64,128} and our CUDA launch is invalid for 256.
   (Not a format problem, a kernel-coverage gap.)
7. Same probe on B300 (sm_103, native Blackwell `dot_scaled`; `results/cuda_compat_B300.{json,log}`): store-kernel parity, converter round trip and all
   decode numbers are the same as H100 within the table's ranges (upstream vs ours Triton 1.3e-3 to 2.1e-3, vs ours CUDA 2.1e-3 to 2.3e-3, max abs <= 1 bf16 ulp).

## 4. Converter

```python
def upstream_to_ours(cache, D):          # cache: uint8 [blocks, bs, Hk, slot>=slot_size(D)]
    G = D // 32; c = cache.permute(0, 2, 1, 3)          # -> [blocks, Hk, bs, slot]
    return (c[..., :D//2].contiguous(), c[..., D//2:D//2+G].contiguous(),      # kc, ks
            c[..., D//2+G:D+G].contiguous(), c[..., D+G:D+2*G].contiguous())   # vc, vs
def ours_to_upstream(kc, ks, vc, vs):    # pad last dim to the allocator's slot size if larger
    return torch.cat([kc, ks, vc, vs], -1).permute(0, 2, 1, 3).contiguous()
```
Block table / `seq_lens` are interchangeable as-is (int32). Upstream stores K already Hadamard-rotated, so our kernels must be fed
the rotated Q (`q @ H`, orthonormal Sylvester, fp32) and keep `sm_scale = D**-0.5`; V is unrotated in both.

To make **our quantiser emit upstream-identical bytes**: `uqmt.quant.quantize(x, "paper", 0.156)` then map nibble `0x8 -> 0x0`
(`canon_negzero` in `cuda/compat/paper_rule_cpu_check.py`), and treat scale byte 0 as 0.0 in dequant (ours decodes it as 2^-127, which is
numerically zero-equivalent for finite data). Going the other way (upstream cache into our kernels) needs nothing beyond the permute.

## 5. Differences to keep in mind

* Scale rule: upstream `round(log2(0.156*amax))` (clips ~4.6% of elements to +-6, MSE-tuned) vs ours `ceil(log2(amax/6))` (no clipping,
  scale 1x or 2x larger). Mixed pipelines must pick one; caches from either decode correctly in either kernel since decode is
  `code * 2^(byte-127)` in both.
* Zero-group sentinel (byte 0) and -0 code: representation only.
* Q precision: upstream casts rotated Q to FP8 E4M3 (extra error); ours uses bf16 Q (or our fp8 path).
* BF16 first/last-2 layers: upstream expresses this via `--kv-cache-dtype-skip-layers`, not in the op.
* SGLang uses SoA pools (closer to ours) but the same per-element semantics; its format oracle is
  `quark/ultraquant_tensor.py` (same c, round-half-up, bucketize, orthonormal Hadamard).

## 6. Reproduce

`modal run --detach cuda/modal/modal_compat.py --gpu H100` (image installs pip vllm; overlays `main` ops if pip lacks UltraQuant);
`python -I cuda/compat/paper_rule_cpu_check.py` (CPU, no vllm needed). Test driver: `cuda/compat/compat_test.py`.
