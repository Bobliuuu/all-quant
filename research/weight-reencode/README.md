# Weight re-encoding for decode bandwidth (lane `wq-`)

The question: batch-1 decode is memory-bandwidth bound. Can we spend more ALU per weight to move fewer bytes?
That includes the idea of storing each weight's percentile rank against a fitted statistical distribution and
resampling it at runtime with the inverse CDF. If we can, at what compute cost per byte does it stop paying, on
each chip?

Everything below was measured on Modal (CPU for weight statistics, A100-80GB for quality, H100 / B200 / RTX PRO
6000 for kernels), except the literature numbers in section 1, which are quoted from their primary sources. Raw
outputs are in [`results/`](results), the full tables in [`results/TABLES.md`](results/TABLES.md), and the code is in this
folder.

## Verdict

1. **The literal percentile codec does not work.** Store `floor(F(z) * 2^k)` with per-group mean and std, then
   decode with `F^-1((c + 0.5) / 2^k)`.
   - Measured on Qwen3-8B: wikitext-2 PPL 1560 at 4.25 bpw and 18324 at 3.25 bpw, against bf16 9.355.
   - **Cause.** Equal-probability bins put the outermost level at only 1.86 sigma (4-bit). Qwen3-8B groups have
     weights up to 116 sigma (median group absmax/RMS 9 to 43, depending on the tensor), so the large weights
     get clipped.
   - **Two fixes, both measured.** Each brings the codec back:
     - Anchor the extreme level at the group absmax (`icdf4_gauss_g64_amax`): PPL 9.94.
     - Use the Bennett-optimal p^(1/3) compander, which is the same percentile idea against a √3-wider
       distribution (`cbrt4_t_g128`): PPL 10.48.
   - **The fixed version is NF4, and NF4 is still the best 4-bit RTN code** (9.606). The percentile codec with
     the absmax anchor is NF4 without the exact zero. Under GPTQ it ties NF4 (9.622 vs 9.693) and also works
     at 3 bits (10.66, against GPTQ NF3 at 10.39).
   - **Entropy.** The percentile code is maximum-entropy by construction (H0 = 3.98 of 4 bits), so entropy
     coding cannot shrink it any further.
2. **No codec beats a plain 4-bit LUT code on H100 or B200 at batch 1, because those chips have no ALU to spare.**
   - **Our kernels are competitive.** Our fused int4 mma GEMV matches or beats vLLM's Marlin, and both are
     already compute (issue) bound:
     - H100 gate_up: 27.2 µs ours vs 26.1 µs Marlin, about 2.0 TB/s of 3.35.
     - B200 gate_up: 22.3 µs ours vs 21.9 µs Marlin, about 2.4 TB/s of 8.
   - **The issue budget is small.** H100 can issue 5.3 lane-instructions per weight at 4.25 bpw before the
     memory time is hidden, and B200 only 2.5. The cheapest decode loop costs 7.3 to 8.
   - **So fewer bytes do not help:**
     - H100: 2-bit LUT 23.5 µs vs int4 27.2 µs (1.16×).
     - B200: 24.5 µs vs 22.3 µs (2-bit is slower).
     - Trellis/computed codes (QTIP 1MAD), entropy coding, and sparse bitmaps are 1.7 to 4.5× slower than int4
       on both chips.
3. **RTX PRO 6000 (GDDR7 1.8 TB/s, 188 SMs at 2.4 GHz) is the chip where re-encoding pays.** Its budget is
   18.7 instr/weight at 4.25 bpw.
   - **Speedups vs cuBLAS bf16** (sum of the four decoder-layer GEMVs of Qwen3-8B, batch 1):
     - 2-bit LUT: 6.09×.
     - QTIP-style 2-bit trellis: 4.58×.
     - 3-bit LUT: 4.19×.
     - int4: 3.63×.
   - **Huffman / rANS-class decode still loses everywhere.** It costs 27 to 38 instr/weight. For example, it
     makes the 4-bit codes 2.1× slower than fixed int4 on RTX PRO, for a 5 to 9% byte saving.
4. **Where the lossless floor is.** The real formats leave little to entropy coding:
   - Qwen3-8B int4 codes: H0 3.61 bits. NF4: 3.84. bf16: 10.64 bpw (DFloat11's floor).
   - gpt-oss-120b MXFP4 nibbles: 3.87 bits, E8M0 scales 1.66 bits. The lossless floor is 3.92 vs 4.25 bpw
     stored, so −7.8% at best.
   - **The ">50% zeros" claim applies only to ternary QAT models:**
     - It comes from BITCOS (arXiv 2609.16338), with 1.485 bpw at 51.48% zeros on CAT-Q Qwen3-1.7B.
     - We measured BitNet b1.58 2B4T at 42.19% zeros, which matches the paper.
     - Qwen3-8B has 0.00% exact zeros and 1.6% of weights below 0.01 RMS. Getting zeros from a dense model takes
       pruning, and 50% magnitude pruning costs PPL (section 7).
5. **Follow-ups** (section 11):
   - **B200:** decode-free cuBLASLt NVFP4 (W4A4) runs one Qwen3-8B layer in 34.2 µs at batch 1. That is 1.40×
     our int4, 1.84× Marlin and 2.14× bf16. MXFP4 through torch's API runs in 49.0 µs.
   - **H100:** a wgmma-fed int4 pipeline is estimated at about 1.5× at batch 1. Removing the loop overhead takes
     int4 from 7.9 to about 4 instr/weight, under the 5.3 budget.
   - **gpt-oss on RTX PRO:** a 3-bit LUT re-encode of a gpt-oss MoE layer is 1.20× faster than moe-mxfp4 at T=1,
     but RTN-regridding MXFP4 changes the layer output by 39%. It needs a calibrated re-encode, which is
     budgeted.
6. **Recommendation for magninference** (details in section 10):
   - Keep fixed-rate, register-decoded LUT/affine codes: MXFP4/NVFP4 or int4 on datacenter Hopper and Blackwell.
   - Spend bits on quality (NF4-style LUT, GPTQ), not on cleverer encodings.
   - On RTX PRO 6000-class parts, adopt 3-bit and 2-bit LUT codes and QTIP-style trellis codes. They pay by 1.2
     to 1.7× over int4, gated on quality.
   - Do not adopt entropy-coded, bitmap-sparse, or runtime inverse-CDF (erfinv) decoders on any of the three
     chips.

## 1. The "1.48 bpw, >50% zeros" paper

**BITCOS: "Breaking the 1.58-bit Barrier for Ternary LLMs"** (Georganas, Heinecke, Dubey, Intel), arXiv
2609.16338, Sep 2026. https://arxiv.org/abs/2609.16338

- **Encoding:** a dense presence bitmap (1 bit per weight) plus a compacted sign stream (1 bit per non-zero).
  At zero density z this costs B(z) = 2 − z bpw, plus one 16-bit scale per 128 weights. It is lossless.
- **Where the zeros were measured:** 29 ternary checkpoints, with up to 51.5% zeros.
  - CAT-Q Qwen3-1.7B: z = 51.48% → 1.485 bpw for the symbols, 1.610 bpw with scales.
  - BitNet b1.58 2B4T: z = 42.19% → 1.578 bpw.
  - Five-trit packing costs 1.625 bpw, and BITCOS loses to it below z ≈ 37.5%.
- **Kernel results (as reported; no NVIDIA results):**
  - GEMV is 1.01 to 1.28× faster than a 2-bit reference on Xeon, Core Ultra and Arc.
  - End-to-end decode is 1.02 to 1.27× faster.
  - It is slower at every density on the Lunar Lake CPU.
- **Our measurement, same encoding (prefix code 0 / 01 / 11 = bitmap + compact sign), on NVIDIA:**
  - **Zero fraction.** BitNet b1.58 2B4T bf16 master weights with absmean ternarization have 42.19% zeros
    (`results/analyze_bitnet_2b4t.json`), exactly the paper's number. The ternary entropy is 1.555 bits and
    bitmap+sign is 1.578.
  - **Kernel speed.** This kernel is our VLC decoder (section 8), which makes the bitmap code on-chip:
    - It is 2.3 to 2.9× **slower** than the same ternary weights stored at a fixed 2 bits (I2_S-style) on all
      three chips. Examples at gate_up, batch 1: H100 68 µs vs 24 µs, B200 57 vs 24, RTX PRO 56 vs 20.
    - The zero fraction barely matters (H100: 68.3 µs at z = 0.42, 67.4 µs at z = 0.90). Decode, not bytes,
      sets the time.
  - **Verdict on NVIDIA:** the bandwidth win is not realized. It needs ~21 to 29 instr/weight against a budget of
    ≤ 10.

bitnet.cpp's formats (https://arxiv.org/abs/2502.11880):

- I2_S (2 bpw), TL1 (2 bpw) and TL2 (1.67 bpw).
- Its GPU W2A8 kernel is 1.27 to 3.63× faster than BF16 GEMV on A100.

## 2. Compute-for-bandwidth codes (literature; quality and speed as reported)

| method | URL | encoding | bpw | quality (reported) | kernel speed (reported) |
|---|---|---|---|---|---|
| QTIP | arxiv.org/abs/2406.11235 | RHT + bitshift trellis (L=16), computed Gaussian code: 1MAD (LCG a=34038481, b=76625530, sum 4 bytes, (s-510)/147.8), 3INST (LCG, mask 0x8fff8fff, xor 0x3b603b60, sum two fp16), HYB (2 KiB LUT) | 2/3/4 | Llama-2-7B wiki2 5.17 / 5.38 / 7.05 (fp16 5.12, no FT); 70B 3.16 / 3.27 / 3.87 | RTX 6000 Ada, bs1, 2-bit 7B 188 tok/s vs fp16 55.9, QuIP# 186, AQLM 81.5 |
| SeedLM | arxiv.org/abs/2410.10714 | LFSR seed (16 bit) + P 4-bit coeffs per C-weight block | 3 / 4 | Llama-3-70B 4-bit 3.8 ppl, 3-bit 5.7 (fp16 2.9) | FPGA only (3.7-4.0x fewer cycles), no GPU |
| QuIP# | arxiv.org/abs/2402.04396 | RHT + E8P lattice codebook (2^16 in 8-D, ~1 KiB table) | 2 (+RVQ 3/4) | Llama-2-7B 5.19 / 5.41 / 6.19 | RTX 4090, 2-bit 7B 170 tok/s (30% of peak BW) |
| AQLM | arxiv.org/abs/2401.06118 | additive multi-codebook VQ, group 8 | ~2.0-2.3 | Llama-2-7B 6.59 (6.14 FT) at 2.02 | RTX 3090 GEMV 1.2-3.0x vs fp16 |
| VPTQ | arxiv.org/abs/2409.17066 | VQ vec 6-12, 4096-entry codebooks + residual + outliers | 2.02-2.26 | Llama-2-7B 6.13 at 2.02 | 7B 39.9 tok/s vs fp16 38.3 |
| HIGGS | arxiv.org/abs/2411.17525 | data-free: RHT + norm scale + MSE-optimal Gaussian grid (p=1,2) | 2-4 | Llama-3.1-8B 6.14 / 6.02 at ~4 bpw (fp16 5.607; NF 6.225) | RTX 4090 bs1 4-bit 139 tok/s vs fp16 57 (FLUTE) |
| GPTVQ | arxiv.org/abs/2402.15319 | Hessian-guided VQ, EM codebook init | 2-4 | not read in full | mobile CPU only |
| NF4 (QLoRA) | arxiv.org/abs/2305.14314 | quantile levels of N(0,1), asymmetric with exact 0, absmax per 64 | 4.127 (DQ) | Pile-CC mean ppl Int4 34.34, FP4 31.07, NF4+DQ 27.41 | none |
| SF4 | arxiv.org/abs/2405.03103 | Student-t (nu=5) quantile levels, absmax g128 | 4 | Llama-2-7B LAMBADA NF4 71.98 / SF4 72.42 (fp32 73.92) | none |
| AF4 / BOF4 | arxiv.org/abs/2306.06965, 2505.06653 | L1- / MSE-optimal codes for absmax-normalized blocks | 4 | Llama-3.1-8B wiki2 NF4 8.53, AF4 8.51, BOF4-S 8.46 | none |
| LUT-GEMM | arxiv.org/abs/2206.09557 | binary-coding quant + LUT partial sums | 3 | OPT-175B 3-bit | 2.1x latency vs OPTQ |
| FLUTE | arxiv.org/abs/2407.10960 | LUT kernel for NF/learned/int codes, bank-duplicated LUT, Stream-K | 3 / 4 | Llama-3-8B 6.2 (4b g128) vs 6.1 | 2-4x vs other LUT kernels at bs<32; e2e 1.3x on A100 |

**The user's idea mapped exactly.** "Percentile → distribution → resample" is quantile quantization: the NF4
construction applied with per-group (mu, sigma) normalization instead of absmax. Prior art, closest first:

- **DACQ** (arxiv.org/abs/2603.00364, Feb 2026). Fitted mu/sigma plus logistic inverse-CDF levels. It reports
  Llama-3-8B wikitext-2 15.7 (8.5 in its hybrid variant) vs AWQ 6.5. That is the same failure mode we measured.
- **Cholakov 2023** (MIT RSI, empirical mu/sigma normal quantiles). The sigma that minimizes reconstruction error
  gave worse perplexity.
- **NF4, SF4 and BOF4** use absmax normalization, which is the fix.
- **No fused inverse-CDF GEMV kernel exists.** We built one (`icdf4_erfinv`, section 8). It is the slowest codec
  we measured.

## 3. Lossless / entropy coding on GPU (literature + our decoder)

- **DFloat11** (arxiv.org/abs/2504.11651): Huffman-coded bf16 exponent, about 11 bpw, bit-exact. Its README says
  batch 1 is ~2× slower than bf16 when the model fits in memory; it wins only against CPU offload.
- **ECF8** (arxiv.org/abs/2510.02676): Huffman-coded FP8 exponent, 9.8 to 26.9% smaller. Throughput gains come
  only from a larger batch under a fixed memory budget.
- **ZipNN** (arxiv.org/abs/2411.05239): BF16 at ~66% size. CPU only: 1.65 GB/s per core, 80 GB/s on 16 workers.
- **Huff-LLM** (arxiv.org/abs/2502.00922): hardware design (Huffman decoders in the PE array), 15 to 32%
  bandwidth. **EntroLLM** (arxiv.org/abs/2505.02380): uint + Huffman on Jetson.
- **nvCOMP gANS:** ~265 GB/s decompression on Blackwell, per NVIDIA's blog. **Recoil** (parallel rANS,
  arxiv.org/abs/2306.12141): 90+ GB/s on Turing.

**Our measurements (Qwen3-8B, all 252 linear tensors, `results/analyze_qwen3_8b.json`).**

| quantity | value |
|---|---|
| bf16 lossless floor (exponent entropy + 8 raw bits, DFloat11-style) | exponent H = 2.64 bits, 10.64 bpw (66.5%) |
| H0 of int4 g128 RTN codes / NF4 g64 / percentile icdf4 / cbrt4 | 3.61 / 3.84 / 3.99 / 3.66 bits |
| H0 of int3 / NF3 / icdf3 / cbrt3 | 2.55 / 2.74 / 2.99 / 2.67 |
| H0 of int2 / NF2 / icdf2 / cbrt2 | 1.47 / 1.47 / 2.00 / 1.71 |
| gpt-oss-120b MXFP4 expert nibbles (2 of 14 shards, 14.9 B params) | H0 3.867 bits (byte-pair 3.860; context adds nothing); 12.1% zero codes; E8M0 scale H0 1.66 bits; lossless floor 3.92 vs 4.25 bpw (-7.8%) |

The `ten-percent` v2/entropy study ran context-mixing range coders on GPTQ codes and found the same thing: almost
all of the gain is the plain histogram, and context modelling adds only 0.5 to 1.8%.

**Our on-chip Huffman decoder** (`k_vlc`, section 8) decodes the warp-interleaved stream in shared memory. On
4-bit Huffman codes it reaches these L2-resident rates:

| GPU | symbols decoded | compressed data | per SM |
|---|---|---|---|
| H100 | 1141 G/s | 0.51 TB/s | 8.6 G/s/SM |
| B200 | 1374 G/s | 0.62 TB/s | — |
| RTX PRO 6000 | 1650 G/s | — | — |

That is about 2× nvCOMP's ANS rate, and still 6× (H100) to 11× (B200) too slow to hide behind HBM.

## 4. Sparsity in the encoding

- **Literature:**
  - SparseGPT: 50% + 4-bit on OPT-175B gives 8.29 vs dense 8.35. 2:4 gives 8.74.
  - Sparse-Marlin (2:4 + INT4): up to +65% over dense Marlin at the kernel level.
  - MACKO/DELTA4 (fp16 unstructured SpMV): break-even at ~20 to 25% sparsity, 1.3× at 50% vs cuBLAS on an
    RTX 4090.
  - SpInfer: beats cuBLAS from 30%.
  - All of these compare against fp16 dense, not against 4-bit.
- **Our kernels:**
  - `sparse4_p*` encodes '0' for a zero and '1' + 4 bits for a non-zero.
  - `ternvlc_p*` is the BITCOS bitmap+sign code.
  - Each is swept at p0 = 0.25 to 0.9 and compared against dense 4-bit and 2-bit codes at batch 1, 4 and 16.
- **Result: zeros never paid off in decode GEMV against dense int4 on any chip, at any sparsity up to 90%.**
  - The bitmap decode costs 26 to 38 instr/weight.
  - At 75% zeros, sparse4 (2.55 bpw) runs 85.5 µs vs int4 27.2 µs on H100, and 63.5 vs 35.5 µs on RTX PRO.
  - From the roofline (section 9), a bitmap code pays against int4 only if its decode costs under 8 instr/weight
    on H100. A ballot-and-shift prefix decoder cannot get there. The only route that works is structured: 2:4
    with a 2-bit index per kept value, as in Sparse-Marlin, where the decode is fixed-rate.
- **Quality of making zeros (Qwen3-8B, magnitude pruning, no retraining):** see section 7, table "pruning".

## 5. Runtime-resampled and generator weights

- **Low-rank + quantized residuals:**
  - LQER (arxiv.org/abs/2402.02446): W4A8 + rank-32 SVD of the scaled error, 4.3 bpw. Llama-2-7B 5.69 vs fp16
    5.48.
  - CALDERA (arxiv.org/abs/2405.18886): QuIP# 2-bit + 4-bit LR, rank 256, 2.2 to 2.4 bpw. Llama-2-7B 6.19.
  - Both add a skinny GEMM per layer. At batch 1 that is bandwidth for the factors: rank 256 at 4 bits on
    4096×4096 is +0.25 bpw, at no decode cost. They are compatible with any codec here.
- **Delta compression:** BitDelta (arxiv.org/abs/2402.10193) stores a 1-bit fine-tune delta. It only helps
  multi-tenant serving.
- **Generator weights:** SeedLM (PRNG seeds) and QTIP's computed codes are the GPU-relevant forms. We measured
  QTIP's 1MAD and 3INST decoders:
  - They cost 14 to 17 instr/weight, against 7.3 for a 2-bit LUT.
  - They pay only on RTX PRO 6000: 26.8 µs vs int4 35.5 µs at gate_up, batch 1.
  - SeedLM's LFSR-matrix product has no GPU kernel anywhere. It would decode a C=8 block as a 3-term
    combination of LFSR rows: ≥ 3 FMAs plus LFSR steps per weight, which is more than 1MAD.
- **Hypernetwork / procedural weights:** we found no GPU decode numbers in the literature.
- **The user's idea literally** (k-bit percentile rank + per-group distribution, inverse-CDF at runtime): sections
  7 and 8. Quality needs the absmax anchor. Speed: computing `erfinvf` costs 36 to 40 instr/weight, so it is the
  slowest decoder on every chip. A 16-entry LUT of the same levels costs 3 to 4 instr/weight more than int4.

## 6. Real-weight statistics (Modal CPU, `modal_analyze.py`)

**Qwen3-8B** (bf16, 6.95 B linear parameters). The fits are of group-standardized weights (g = 128, per-group
mean and std). KS is the median over tensors, and "best" is the lowest NLL.

| tensor | best fit | KS Gauss | KS Student-t | median nu (g128) | median nu (whole tensor) | median / max absmax:RMS |
|---|---|---|---|---|---|---|
| q_proj | t (36/36) | 0.0052 | 0.0014 | 36 | 8.7 | 18 / 57 |
| k_proj | t (36/36) | 0.0059 | 0.0013 | 32 | 9.2 | 16 / 36 |
| v_proj | t (36/36) | 0.0093 | 0.0041 | 28 | 14.3 | 9 / 15 |
| o_proj | t (35/36) | 0.0017 | 0.0009 | 563 | 13.9 | 23 / 66 |
| gate_proj | t (35/36) | 0.0034 | 0.0014 | 49 | 18.4 | 27 / 43 |
| up_proj | t (36/36) | 0.0043 | 0.0013 | 53 | 31.7 | 28 / 50 |
| down_proj | t (35/36) | 0.0053 | 0.0013 | 33 | 20.8 | 43 / 116 |

- **Shape:** within a group the weights are nearly Gaussian (nu 30 to 500). Whole tensors are Student-t with nu
  9 to 32. Laplace fits worst everywhere.
- **Outliers** reach 116× the RMS. They are what breaks std-normalized codes.
- **Zeros:** exact zeros are 0.00%, |w| < 0.01 RMS is 1.6%, and |w| < 0.1 RMS is 10.8%.
- **gpt-oss-120b:** see section 3. **BitNet 2B4T:** see section 1.

## 7. Quality: percentile / inverse-CDF codec vs NF4, MXFP4, QTIP-style, GPTQ, AWQ (Qwen3-8B)

**Protocol.**
- **What is quantized:** every 2-D weight inside the decoder layers. Embeddings, lm_head and norms stay bf16.
- **PPL:** wikitext-2-raw test, first 64 × 2048 tokens. This is identical to `traces/gap-lanes/compress`: bf16
  9.355 there and 9.355 here, so the AWQ/GPTQ rows from that lane are paired.
- **Tasks:** lm_eval 0-shot on arc_challenge (acc_norm), hellaswag (acc_norm, 1000), winogrande and piqa (1000).
  The stderr is about 1.4 to 1.5 points.
- **GPTQ (ours):** 128 × 2048 C4 calibration, block 128, damp 0.01.
- **bpw** includes the side information:
  - musig = fp16 mu + fp16 sigma per group; int = fp16 scale + fp16 zero.
  - NF and amax = one fp16 per group; MXFP4 = E8M0 per 32; TCQ = (L−k)/256 start-state bits + one fp16 per row.

**Config names.**
- `icdfK_<dist>_gG` is the user's codec: code = percentile bin, level = inverse CDF at the bin median.
  - The `_near` suffix assigns each weight to the nearest level instead of the percentile bin.
- `cbrt` uses the Bennett p^(1/3) compander law. `lloyd` uses the Lloyd-Max levels of the distribution.
- `kmeans` uses a per-tensor empirical LUT.
- `_amax` sets the extreme level to the group absmax, NF-style.
- `tcqK_L12` is QTIP-style: RHT + bitshift trellis with L = 12 and the 1MAD code, Viterbi encoding, no
  Hessian/LDLQ.
- `rhtlloydK` is RHT + scalar Lloyd-Max. It isolates how much of TCQ's gain comes from the trellis.

| config | bpw | PPL | rel. MSE | code H0 | arc_c | hellaswag | winogrande | piqa |
|---|---|---|---|---|---|---|---|---|
| bf16 | 16 | 9.355 | 0 | | 0.564 | 0.638 | 0.679 | 0.769 |
| **nf4_g64** | 4.25 | **9.606** | 0.0087 | 3.84 | 0.555 | 0.634 | 0.674 | 0.772 |
| tcq4_L12 (QTIP-style) | 4.04 | 9.571 | 0.0054 | | | | | |
| gptq:int4_g128 (ours) | 4.25 | 9.703 | 0.0152 | | 0.556 | 0.630 | 0.674 | 0.776 |
| GPTQ W4A16 g128 (llm-compressor, gap-lanes, wikitext calib) | 4.125 | 9.574 | | | | | | |
| AWQ W4A16 g128 (llm-compressor, gap-lanes) | 4.125 | 9.775 | | | | | | |
| icdf4_gauss_g64_amax (percentile + absmax anchor) | 4.25 | 9.944 | 0.0086 | 3.85 | | | | |
| rhtlloyd4 | 4.00 | 9.957 | 0.0095 | | | | | |
| icdf4_t_g64_amax | 4.25 | 9.969 | 0.0089 | 3.86 | 0.541 | 0.634 | 0.669 | 0.761 |
| int4_g128 (RTN) | 4.25 | 10.068 | 0.0107 | 3.61 | 0.545 | 0.608 | 0.676 | 0.766 |
| mxfp4 | 4.25 | 10.182 | 0.0138 | | 0.529 | 0.629 | 0.673 | 0.755 |
| cbrt4_t_g128 (percentile vs p^(1/3)) | 4.25 | 10.484 | 0.0107 | 3.61 | 0.560 | 0.634 | 0.676 | 0.757 |
| kmeans4_g64_amax | 4.25 | 10.612 | 0.0083 | 3.77 | 0.534 | 0.639 | 0.669 | 0.748 |
| lloyd4_t_g128 | 4.25 | 11.750 | 0.0100 | 3.70 | | | | |
| icdf4_gauss_g64_sig (sigma only) | 4.25 | 27.52 | 0.0195 | 3.98 | | | | |
| **icdf4_gauss_g128 (literal percentile)** | 4.25 | **1560** | 0.0209 | 3.98 | 0.218 | 0.313 | 0.526 | 0.520 |
| icdf4_t_g128 / laplace | 4.25 | 31226 / 1688 | 0.022 / 0.028 | 3.99 / 3.94 | | | | |
| gptq:int3_g128 | 3.25 | 10.921 | 0.0704 | | 0.502 | 0.603 | 0.642 | 0.764 |
| nf3_g64 | 3.25 | 11.197 | 0.0416 | 2.74 | 0.505 | 0.588 | 0.646 | 0.744 |
| tcq3_L12 (QTIP-style) | 3.04 | 11.322 | 0.0193 | | 0.500 | 0.607 | 0.666 | 0.736 |
| rhtlloyd3 | 3.00 | 12.023 | 0.0345 | | | | | |
| int3_g128 | 3.25 | 15.456 | 0.0493 | 2.55 | 0.387 | 0.547 | 0.586 | 0.722 |
| kmeans3_g64_amax | 3.25 | 21.17 | 0.0393 | 2.65 | 0.397 | 0.543 | 0.605 | 0.717 |
| cbrt3_t_g128 | 3.25 | 40.13 | 0.0407 | 2.64 | 0.299 | 0.425 | 0.537 | 0.633 |
| icdf3_t_g64_amax | 3.25 | 681.7 | 0.0384 | 2.71 | 0.275 | 0.345 | 0.515 | 0.612 |
| icdf3_gauss_g128 (literal percentile) | 3.25 | 18324 | 0.0495 | 2.98 | 0.259 | 0.282 | 0.497 | 0.534 |
| tcq2_L12 (QTIP-style) | 2.04 | 33.33 | 0.0724 | | 0.307 | 0.469 | 0.553 | 0.679 |
| rhtlloyd2 | 2.00 | 335.4 | 0.1174 | | 0.227 | 0.363 | 0.502 | 0.551 |
| every 2.25-bpw scalar RTN (int2, nf2, icdf2, cbrt2, lloyd2, kmeans2) | 2.25 | 3.6e4 to 2.3e6 | | | | | | |

**GPTQ and pruning rows** (same protocol):

| config | bpw | PPL | rel. MSE | arc_c | hellaswag | winogrande | piqa |
|---|---|---|---|---|---|---|---|
| gptq:nf4_g64 | 4.25 | 9.693 | 0.0121 | 0.560 | 0.632 | 0.687 | 0.768 |
| gptq:icdf4_t_g64_amax (percentile, Student-t, absmax anchor) | 4.25 | 9.622 | 0.0125 | 0.551 | 0.633 | 0.683 | 0.769 |
| gptq:int3_g128 | 3.25 | 10.921 | 0.0704 | 0.502 | 0.603 | 0.642 | 0.764 |
| **gptq:nf3_g64** | 3.25 | **10.391** | 0.0581 | 0.508 | 0.617 | 0.664 | 0.756 |
| gptq:icdf3_t_g64_amax | 3.25 | 10.663 | 0.0545 | 0.532 | 0.601 | 0.666 | 0.770 |
| prune50:bf16 (50% unstructured magnitude, 1-bit bitmap) | 9.0 | 49.61 | 0.068 | | | | |
| prune24:bf16 (2:4 magnitude, 2-bit index per kept value) | 9.0 | 265.7 | 0.127 | | | | |
| prune50:nf4_g64 (50% magnitude, then NF4 on the survivors) | 3.125 | 56.16 | 0.074 | | | | |

Magnitude pruning without retraining wrecks Qwen3-8B: 49.6 at 50% unstructured, 265.7 at 2:4. A dense model
has no free zeros. SparseGPT/Wanda-class retraining-free pruning is the published route, but it is
calibration-dependent and was not rerun here.

**GPTQ results.**
- **GPTQ rescues the anchored percentile code at 3 bits:** icdf3_t_g64_amax goes from 682 (RTN) to 10.66.
  GPTQ's error feedback absorbs whatever breaks it under RTN.
- **Best 3-bit result:** GPTQ NF3 at 10.39, ahead of GPTQ int3 at 10.92.
- **At 4 bits the percentile code is on par with NF4:** GPTQ icdf4_t_amax 9.622 vs GPTQ NF4 9.693 vs RTN NF4
  9.606. These are within about 0.1 PPL; the tasks are within stderr.
- **C4-calibrated GPTQ did not beat RTN NF4** on wikitext PPL at 4 bits.

**Findings.**
1. **Weight MSE does not predict PPL once outliers are clipped.**
   - The literal percentile codec has only 2× the MSE of int4, yet 155× the PPL.
   - kmeans (per-tensor MSE-optimal LUT) has the lowest MSE at 4 bits and is beaten by NF4 by 1.0 PPL. It sets
     its outer levels by population mass, not by the few weights that matter.
2. **Matching the distribution helps only after the range is anchored, and at 3 bits an exact-zero level is
   mandatory.**
   - Among the absmax-anchored 4-bit codes, Gaussian vs Student-t levels differ by 0.03 PPL.
   - At 3 bits the anchored percentile codes still fail: icdf3_gauss_g64_amax gives 359 and icdf3_t 682, while
     NF3 gives 11.20 at the same bpw. kmeans3_amax gives 21.2.
   - The failing grids all have lower weight MSE than NF3 (0.0386 vs 0.0416). The visible difference is that
     NF3 is asymmetric with an exact 0 level, while every failing grid is symmetric with no zero level. That is
     our working explanation, not a proven cause: int3 min-max (no exact zero either) degrades to 15.5, not 359.
   - Without the anchor, every std-normalized variant fails at 3 bits or below.
3. **Trellis codes are the real "more compute, fewer bits" win in quality.**
   - QTIP-style TCQ without a Hessian gives 3.04 bpw at 11.32, equal to NF3 at 3.25 bpw.
   - At 4.04 bpw it gives 9.571, better than NF4 at 4.25.
   - At 2.04 bpw it gives 33.3, against > 30,000 for every 2.25-bpw scalar code.
   - The trellis, not the RHT, carries most of the gain: rhtlloyd2 gives 335.
   - With LDLQ/Hessian feedback (QTIP's full recipe) these numbers would improve further. That was not run here.

## 8. Fused decode-GEMV kernels (`kernels/wq_gemv.cu`, `modal_kernels.py`)

**Design.** One code path for every codec:
- **Decode:** each lane decodes its 32 weights per 16×64 tile in registers, directly into `mma.sync m16n8k16`
  A-fragments. Weights are bf16 with fp32 accumulation. The FMAs run on tensor cores, so a codec's cost is only
  its decode, and it is amortized over the batch.
- **Tiling:** split-K over grid.y, and 1, 2 or 4 m-tiles per CTA (MT) so that activation fragments are reused.
- **Codecs:** `bf16`, `int4` (magic-number dequant, as in Marlin), `lut4` (16-entry smem LUT), `lut4p`
  (256-entry pair LUT, bank-replicated as in FLUTE), `icdf4_erfinv` (the percentile code decoded by computing
  sqrt(2)·erfinv((2c+1)/16 − 1) in registers), `lut3p`, `lut2p`, `tcq2_1mad` / `tcq2_3inst` (QTIP bitshift
  trellis, L = 16, k = 2, tail-biting per lane, computed code), and `k_vlc`.
- **`k_vlc`** is a prefix-code stream decoder with no per-lane offsets:
  - Lanes refill 32-bit words from a warp-interleaved stream by ballot + popc. The encoder replays that exact
    schedule.
  - Words are staged with `cp.async` into a per-warp shared ring. Decoding uses a 2^L table, and Q = 1 or 4
    independent sub-streams per lane give ILP.
  - It runs three codes: Huffman (length-limited to 8) over 4-bit codes, the BITCOS bitmap+sign code, and
    bitmap + 4-bit values.

**Benchmark setup.**
- **Correctness:** every codec is checked against `x @ W_hat^T`, with rel. L2 error ≤ 3e-3.
- **Weights:** real Qwen3-8B layer-18 weights, except where noted. TCQ uses random trellis words, and the
  ternary/sparse codecs are synthetic at the stated zero fraction.
- **Timing:**
  - Each config cycles ≥ 320 MB of distinct weight copies inside one CUDA graph, to defeat L2.
  - Time is the median of 9 replays. The min-to-max spread is under 2% for almost every row (see
    `results/*.log`).
  - (S, MT, Q) are tuned per codec, shape and batch, the same tuning for everyone.
- **Baselines:** cuBLAS bf16 (`F.linear`) and vLLM 0.33 Marlin (`uint4b8` g128) run on the same shapes with the
  same timing method.

**Sum over one Qwen3-8B decoder layer's GEMVs** (gate_up 24576×4096, down 4096×12288, qkv 6144×4096,
o 4096×4096), in µs. The speedup vs cuBLAS bf16 at the same batch is in parentheses.

| codec (bpw) | H100 B=1 | B=4 | B=16 | B200 B=1 | B=4 | B=16 | RTX PRO B=1 | B=4 | B=16 |
|---|---|---|---|---|---|---|---|---|---|
| cublas_bf16 (16.00) | 141.7 (1.00x) | 142.7 (1.00x) | 145.1 (1.00x) | 73.1 (1.00x) | 75.0 (1.00x) | 73.6 (1.00x) | 262.5 (1.00x) | 261.7 (1.00x) | 266.1 (1.00x) |
| bf16 (16.25) | 127.5 (1.11x) | 129.1 (1.11x) | 146.3 (0.99x) | 63.2 (1.16x) | 65.8 (1.14x) | 89.0 (0.83x) | 245.5 (1.07x) | 246.3 (1.06x) | 248.4 (1.07x) |
| marlin_vllm_int4_g128 (4.12) | 64.5 (2.20x) | 64.7 (2.21x) | 68.0 (2.13x) | 63.1 (1.16x) | 63.2 (1.19x) | 67.0 (1.10x) | 83.7 (3.14x) | 83.8 (3.12x) | 85.9 (3.10x) |
| int4 (4.25) | 56.6 (2.50x) | 59.1 (2.42x) | 85.0 (1.71x) | 48.0 (1.52x) | 50.5 (1.49x) | 84.9 (0.87x) | 72.2 (3.63x) | 74.3 (3.52x) | 85.1 (3.12x) |
| lut4_icdf (4.25) | 75.6 (1.88x) | 78.1 (1.83x) | 126.2 (1.15x) | 71.1 (1.03x) | 75.0 (1.00x) | 114.8 (0.64x) | 74.3 (3.53x) | 77.8 (3.36x) | 95.6 (2.78x) |
| lut4p_icdf (4.25) | 72.8 (1.95x) | 75.0 (1.90x) | 107.2 (1.35x) | 72.8 (1.00x) | 75.7 (0.99x) | 105.8 (0.70x) | 81.4 (3.23x) | 83.5 (3.14x) | 101.0 (2.63x) |
| icdf4_erfinv (4.25) | 242.4 (0.58x) | 243.5 (0.59x) | 250.4 (0.58x) | 199.6 (0.37x) | 201.4 (0.37x) | 259.0 (0.28x) | 133.4 (1.97x) | 133.9 (1.95x) | 171.5 (1.55x) |
| lut3p_cbrt (3.25) | 69.8 (2.03x) | 71.8 (1.99x) | 106.8 (1.36x) | 67.1 (1.09x) | 69.3 (1.08x) | 104.1 (0.71x) | 62.7 (4.19x) | 66.0 (3.96x) | 89.7 (2.97x) |
| lut2p_cbrt (2.25) | 51.9 (2.73x) | 54.9 (2.60x) | 91.1 (1.59x) | 52.3 (1.40x) | 55.0 (1.36x) | 89.4 (0.82x) | 43.1 (6.09x) | 47.4 (5.52x) | 74.0 (3.60x) |
| tern2b_p0.5 (2.25) | 51.8 (2.74x) | 54.7 (2.61x) | 91.1 (1.59x) | 52.3 (1.40x) | 54.9 (1.37x) | 89.4 (0.82x) | 43.0 (6.10x) | 47.4 (5.52x) | 74.9 (3.55x) |
| tcq2_1mad (2.01) | 92.9 (1.53x) | 94.5 (1.51x) | 110.7 (1.31x) | 86.2 (0.85x) | 88.8 (0.85x) | 112.8 (0.65x) | 57.3 (4.58x) | 60.9 (4.30x) | 83.5 (3.19x) |
| tcq2_3inst (2.01) | 100.7 (1.41x) | 102.6 (1.39x) | 120.9 (1.20x) | 98.3 (0.74x) | 100.6 (0.75x) | 124.5 (0.59x) | 59.7 (4.40x) | 62.4 (4.19x) | 88.7 (3.00x) |
| huff4_int4 (4.14) | 183.0 (0.77x) | 184.2 (0.77x) | 224.6 (0.65x) | 151.9 (0.48x) | 155.5 (0.48x) | 208.9 (0.35x) | 157.6 (1.67x) | 160.8 (1.63x) | 187.9 (1.42x) |
| huff4_cbrt (4.25) | 185.9 (0.76x) | 186.5 (0.77x) | 224.2 (0.65x) | 152.2 (0.48x) | 155.8 (0.48x) | 209.3 (0.35x) | 159.2 (1.65x) | 161.1 (1.62x) | 189.1 (1.41x) |
| ternvlc_p0.42 (2.01) | 139.2 (1.02x) | 140.1 (1.02x) | 165.5 (0.88x) | 118.8 (0.62x) | 120.5 (0.62x) | 149.9 (0.49x) | 121.1 (2.17x) | 125.7 (2.08x) | 156.3 (1.70x) |
| ternvlc_p0.515 (1.97) | 139.2 (1.02x) | 140.7 (1.01x) | 165.2 (0.88x) | 118.6 (0.62x) | 120.3 (0.62x) | 149.9 (0.49x) | 120.7 (2.18x) | 123.9 (2.11x) | 155.6 (1.71x) |
| ternvlc_p0.6 (1.82) | 138.6 (1.02x) | 139.4 (1.02x) | 164.6 (0.88x) | 118.6 (0.62x) | 120.3 (0.62x) | 149.6 (0.49x) | 119.2 (2.20x) | 121.9 (2.15x) | 153.7 (1.73x) |
| ternvlc_p0.75 (1.76) | 138.5 (1.02x) | 139.4 (1.02x) | 165.3 (0.88x) | 117.9 (0.62x) | 119.7 (0.63x) | 148.9 (0.49x) | 117.1 (2.24x) | 120.6 (2.17x) | 152.2 (1.75x) |
| ternvlc_p0.9 (1.52) | 136.7 (1.04x) | 137.6 (1.04x) | 163.1 (0.89x) | 117.5 (0.62x) | 119.4 (0.63x) | 148.2 (0.50x) | 117.1 (2.24x) | 120.7 (2.17x) | 152.5 (1.74x) |
| sparse4_p0.25 (4.49) | 183.4 (0.77x) | 184.2 (0.77x) | 207.0 (0.70x) | 151.1 (0.48x) | 152.0 (0.49x) | 186.9 (0.39x) | 157.6 (1.67x) | 161.5 (1.62x) | 181.9 (1.46x) |
| sparse4_p0.5 (3.52) | 177.9 (0.80x) | 178.8 (0.80x) | 201.2 (0.72x) | 147.7 (0.49x) | 148.6 (0.50x) | 184.2 (0.40x) | 150.8 (1.74x) | 152.5 (1.72x) | 173.1 (1.54x) |
| sparse4_p0.75 (2.55) | 172.7 (0.82x) | 173.5 (0.82x) | 194.0 (0.75x) | 143.9 (0.51x) | 144.8 (0.52x) | 180.5 (0.41x) | 135.4 (1.94x) | 139.2 (1.88x) | 162.6 (1.64x) |

**gate_up, batch 1: decode-bound ceiling and achieved bandwidth.** The ceiling is the L2-resident run, giving
the codec's compute-only rate. I_eff = issue rate / ceiling, in lane-instructions per weight.

| codec | H100 us / act. GB/s / ceiling Gw/s / I_eff | B200 | RTX PRO 6000 |
|---|---|---|---|
| cublas_bf16 | 67.5 / 2981 / - / - | 35.5 / 5674 / - / - | 133.0 / 1514 / - / - |
| bf16 | 63.5 / 3218 / 5316 / 6.3 | 30.0 / 6805 / 6292 / 5.9 | 126.3 / 1619 / 3030 / 20.8 |
| marlin_vllm_int4_g128 | 26.1 / 1988 / - / - | 21.9 / 2371 / - / - | 37.7 / 1378 / - / - |
| int4 | 27.2 / 1967 / 4241 / 7.9 | 22.3 / 2393 / 4900 / 7.6 | 35.5 / 1506 / 6270 / 10.0 |
| lut4_icdf | 36.8 / 1451 / 2966 / 11.3 | 34.1 / 1569 / 3103 / 12.0 | 35.6 / 1502 / 4180 / 15.1 |
| lut4p_icdf | 34.4 / 1557 / 3220 / 10.4 | 34.7 / 1541 / 3267 / 11.4 | 38.3 / 1397 / 4459 / 14.1 |
| icdf4_erfinv | 122.4 / 437 / 847 / 39.6 | 97.9 / 546 / 1030 / 36.1 | 65.7 / 813 / 1621 / 38.9 |
| lut3p_cbrt | 32.4 / 1262 / 3308 / 10.1 | 31.4 / 1303 / 3403 / 10.9 | 29.2 / 1401 / 4801 / 13.1 |
| lut2p_cbrt | 23.5 / 1203 / 4588 / 7.3 | 24.5 / 1156 / 4465 / 8.3 | 19.8 / 1430 / 6457 / 9.8 |
| tern2b_p0.5 | 23.8 / 1192 / 4580 / 7.3 | 24.5 / 1157 / 4459 / 8.3 | 19.7 / 1438 / 6352 / 9.9 |
| tcq2_1mad | 45.8 / 551 / 2324 / 14.4 | 41.9 / 603 / 2502 / 14.9 | 26.8 / 944 / 3945 / 16.0 |
| tcq2_3inst | 49.8 / 507 / 2102 / 15.9 | 48.2 / 524 / 2180 / 17.1 | 27.8 / 909 / 3699 / 17.0 |
| huff4_int4 | 90.2 / 578 / 1141 / 29.4 | 74.2 / 703 / 1374 / 27.1 | 75.5 / 852 / 1649 / 38.2 |
| huff4_cbrt | 92.1 / 580 / 1138 / 29.4 | 74.3 / 719 / 1374 / 27.1 | 75.6 / 871 / 1663 / 37.9 |
| ternvlc_p0.42 | 68.3 / 370 / 1528 / 21.9 | 57.1 / 442 / 1799 / 20.7 | 58.2 / 434 / 2160 / 29.2 |
| ternvlc_p0.515 | 68.4 / 363 / 1536 / 21.8 | 57.1 / 435 / 1799 / 20.7 | 58.1 / 428 / 2154 / 29.2 |
| ternvlc_p0.6 | 68.1 / 336 / 1543 / 21.7 | 57.1 / 401 / 1801 / 20.7 | 57.3 / 399 / 2193 / 28.7 |
| ternvlc_p0.75 | 68.5 / 323 / 1551 / 21.6 | 56.7 / 390 / 1815 / 20.5 | 56.7 / 390 / 2187 / 28.8 |
| ternvlc_p0.9 | 67.4 / 284 / 1559 / 21.5 | 56.5 / 339 / 1818 / 20.5 | 56.3 / 340 / 2204 / 28.6 |
| sparse4_p0.25 | 90.3 / 626 / 1136 / 29.5 | 73.4 / 770 / 1392 / 26.7 | 74.2 / 883 / 1650 / 38.2 |
| sparse4_p0.5 | 87.8 / 505 / 1165 / 28.7 | 71.8 / 618 / 1423 / 26.1 | 70.4 / 769 / 1699 / 37.1 |
| sparse4_p0.75 | 85.5 / 376 / 1201 / 27.9 | 70.1 / 459 / 1455 / 25.6 | 63.5 / 672 / 1750 / 36.0 |

**Other results.**
- **moe-mxfp4 (magninference, gpt-oss-120b MoE layer, synthetic MXFP4), RTX PRO 6000:**
  - T = 1: 47.1 µs, 1125 GB/s, 63% of 1.79 TB/s.
  - T = 4: 148.7 µs, 77%. T = 16: 448.7 µs, 84%.
  - Our dense int4 GEMV on the same chip reaches 1506 GB/s (84%) at batch 1. Our 2-bit LUT reaches 1430 GB/s.
  - The kernel does not build for sm_90a or sm_100a, because it uses `mma ... kind::mxf8f6f4 block_scale`. It
    has no H100 or B200 numbers.
- **What batch does:** time is almost flat from B=1 to B=4 (+2 to 10%). At B=16, int4 grows 1.5× on H100. The
  fixed-rate codes stay ahead at B=16 on RTX PRO (2-bit LUT still 3.6× vs cuBLAS) but not on B200.

## 9. Roofline: when does fewer bytes stop paying?

`t = max(W·b/8 / BW, W·I / R)`, where:
- W = number of weights, b = bpw including side info.
- I = issued lane-instructions per weight: the decode plus the per-weight share of loads, mma and addressing.
- R = SMs × 4 schedulers × 32 lanes × clock.

A codec stops paying when I > R·b / (8·BW), the **ALU budget per weight**:

| GPU | R (T lane-instr/s) | BW (TB/s) | budget at 16 / 4.25 / 3.25 / 2.25 / 1.6 bpw |
|---|---|---|---|
| H100 SXM | 33.5 | 3.35 | 20.0 / 5.3 / 4.1 / 2.8 / 2.0 |
| B200 | 37.2 | 8.0 | 9.3 / 2.5 / 1.9 / 1.3 / 0.9 |
| RTX PRO 6000 | 63.0 | 1.79 | 70.3 / 18.7 / 14.3 / 9.9 / 7.0 |

**Measured I_eff per codec** (gate_up, from the L2-resident runs; the same loop on each chip):

| codec | H100 | B200 | RTX PRO |
|---|---|---|---|
| 2-bit LUT (`lut2p`) | 7.3 | 8.3 | 9.8 |
| int4 | 7.9 | 7.6 | 10.0 |
| 3-bit LUT | 10.1 | 10.9 | 13.1 |
| 4-bit LUT | 10.4 | 11.4 | 14.1 |
| QTIP 1MAD | 14.4 | 14.9 | 16.0 |
| bitmap ternary | 21.5 | 20.5 | 28.6 |
| sparse bitmap + int4 | 28 | 26 | 36 |
| Huffman 4-bit | 29 | 27 | 38 |
| erfinv | 40 | 36 | 39 |

**The model predicts the measured times within 3 to 15%** (`roofline.py`):

| codec | GPU | predicted | measured |
|---|---|---|---|
| int4 | H100 | 23.7 µs | 27.2 µs |
| lut2p | RTX PRO | 15.8 µs | 19.8 µs |
| erfinv | B200 | 97.7 µs | 97.9 µs |

**Crossover per chip.** Against the int4 incumbent, a codec at b bpw pays when
`max(b / (8·BW), I / R) < max(4.25 / (8·BW), I_int4 / R)`:

- **H100.** int4 is already ALU bound (7.9 > 5.3). A new codec pays only if it is cheaper to decode than int4
  per weight, and only 2-bit LUT is (1.16×). Every extra instruction per weight costs about 3 µs on gate_up.
  The trade stops paying at about **8 instr/weight**, whatever the bpw.
- **B200.** 2.6× the bandwidth of H100 but only 1.1× the issue rate. Even bf16 is close to ALU bound in an mma
  loop: the budget at 16 bpw is 9.3 instr/weight. Nothing beats int4. Making decode cheaper (fewer instructions
  per weight: native FP4 MMA, wgmma/tcgen05 with smem-staged weights) matters more than making weights smaller.
  The trade stops paying at about **7.6 instr/weight**.
- **RTX PRO 6000.** int4 is bandwidth bound (10 < 18.7), so lower bpw pays until I reaches R·b/(8·BW):
  - **9.9 instr at 2.25 bpw:** 2-bit LUT sits exactly at the crossover.
  - **8.8 at 2 bpw:** QTIP-style TCQ is over it, yet still beats int4.
  - **14.3 at 3.25 bpw:** 3-bit LUT is inside it.
  - **18.7 at 4.25 bpw:** a 4-bit LUT is free.

**Where the numbers come from.**
- **R:** SMs × 4 × 32 × boost clock. H100 has 132 SMs at 1.98 GHz, B200 148 at 1.965, RTX PRO 188 at 2.617 (the
  Server Edition box reported 2.43 GHz).
- **BW:** spec values. Our d2d copies measured 3.03 / 6.50 / 1.46 TB/s, counting read + write.
- **I_eff caveat:** for bf16, the L2-resident run is L2-bandwidth bound, so its I_eff is an upper bound.

## 10. Recommendation for magninference

1. **H100 and B200 (Magni's datacenter targets): do not adopt new weight encodings for batch-1 decode.**
   - The W4A16 mma path is issue-bound at about 2 to 2.4 TB/s, and Marlin is too.
   - **Bigger lever on Blackwell:** use the native block-scaled FP4 MMA, so the decode leaves the ALU entirely.
     Measured with cuBLASLt NVFP4 W4A4: 1.40× our int4 loop at batch 1 and 2.5× at B=16 (section 11). Next, gate
     W4A4 (NVFP4 activations) on quality, or use W4A8 tcgen05.
   - **Bigger lever on Hopper:** a wgmma-fed dequant pipeline, with weights decoded once into smem per warpgroup.
     It cuts the per-weight overhead that dominates our I_eff (about 6.4 of int4's 7.9 instructions are
     loop overhead; the dequant itself is 1.5).
   - **Quality lever:** spend it on NF4-style LUT levels with GPTQ (H100: 4-bit LUT costs +2.5 instr/weight,
     1.26× slower than int4), or keep MXFP4/int4 and use GPTQ.
2. **RTX PRO 6000 (Blackwell workstation, GDDR7): adopt 3-bit and 2-bit fixed-rate LUT codes and QTIP-style
   trellis codes as the batch-1 to 16 weight path.**
   - **Layer speedups vs int4:** 3-bit LUT 1.15× (72.2 → 62.7 µs per layer), 2-bit LUT 1.68× (→ 43.1 µs),
     QTIP-style 2-bit 1.26× (→ 57.3 µs).
   - **Quality decides the bits.** At 3 bits we measured GPTQ NF3 10.39, NF3 11.20 and QTIP-style TCQ3 11.32 (RTN, no Hessian),
     vs bf16 9.36 and NF4 9.61. A 2-bit code needs QTIP's full recipe (Hessian + fine-tune) to be usable; ours
     reaches 33.3 without it.
   - **For gpt-oss-120b on this chip:** the moe-mxfp4 kernel is at 63% of bandwidth at T = 1. A 3-bit trellis or
     LUT re-encode of the experts would cut bytes 25%. Check quality first; MXFP4 → 3-bit is a further lossy
     step.
3. **Do not adopt on any chip:**
   - Entropy coding of 4-bit codes. The floor is −5 to −8% bytes, at 3 to 4× the decode cost.
   - Bitmap-sparse or ternary-bitmap codes.
   - Runtime inverse-CDF. The percentile idea is useful only as a level-placement rule, and with the absmax anchor
     it is NF4.
4. **One cheap win to file:** E8M0 scales have H0 1.66 bits. Storing per-tensor delta-coded scales is 0.2 bpw
   that a fixed 4-bit code could drop, but only at real decode cost.

## 11. Follow-ups: native FP4 on B200, Hopper loop overhead, 3-bit gpt-oss experts (`modal_extra.py`)

**B200, decode-free native block-scaled FP4.**
- **Setup:**
  - NVFP4 runs through `torch._scaled_mm`. MXFP4 runs through `torch.nn.functional.scaled_mm` with
    BlockWise1x32 and the 32_4_4 swizzle. Both are torch 2.15 + cuBLASLt with tcgen05 block-scaled MMA, W4A4.
  - The decoder-layer GEMVs and the timing method are the same as in section 8, with ≥ 400 MB of weight copies
    per graph. rel. L2 error vs the dequantized reference is 0.0017.
  - Scales are uniform for timing (the scale bytes are still read).
  - The activations are FP4, so this is W4A4, not W4A16. Activation quantization is not timed: one small kernel
    per GEMV, or fused into the previous op.

| path (B200, one Qwen3-8B layer) | bpw | B=1 µs | B=4 | B=16 | x cuBLAS bf16 (B=1) | x our int4 (B=1) | gate_up B=1 GB/s |
|---|---|---|---|---|---|---|---|
| cuBLAS bf16 | 16 | 73.1 | 75.0 | 73.6 | 1.00 | | 5674 |
| cuBLASLt FP8 rowwise W8A8 | 8 | 69.8 | 63.6 | 63.8 | 1.05 | | 3379 |
| our int4 W4A16 (mma.sync loop) | 4.25 | 48.0 | 50.5 | 84.9 | 1.52 | 1.00 | 2393 |
| vLLM Marlin W4A16 | 4.125 | 63.1 | 63.2 | 67.0 | 1.16 | 0.76 | 2371 |
| cuBLASLt MXFP4 W4A4 | 4.25 | 49.0 | 45.7 | 47.3 | 1.49 | 0.98 | 2849 |
| **cuBLASLt NVFP4 W4A4** | 4.5 | **34.2** | 34.7 | 34.6 | **2.14** | **1.40** | 4427 |

- **Taking the decode off the ALU pays on B200.** NVFP4 runs at 1.40× our int4 loop and 1.84× Marlin. It is flat
  from batch 1 to 16, while int4 falls to 0.87× cuBLAS at B=16.
- **It still reaches only about 4.4 TB/s of 8** on the largest GEMV, so B200 batch-1 GEMV stays latency/occupancy
  bound even without decode.
- **MXFP4 through torch's new API is no faster than our int4.** It is probably a different cuBLASLt heuristic.
  Use NVFP4 or a CUTLASS/FlashInfer MXFP4 kernel.
- **W4A4 quality was not measured here.** NVFP4 activations add error on top of the weights.

**H100: what the int4 loop's 7.9 instr/weight are made of** (`kernels/wq_micro.cu`, L2-resident weights, all
132 SMs):

| loop content | lane-instr / weight |
|---|---|
| 16-B weight load only | 1.09 |
| + int4 dequant (LOP3, HSUB2, HFMA2) | 3.62 |
| + 4 × mma.sync with activations in registers | 4.17 |
| full `k_fixed<int4>` (+ activation-fragment LDGs, param LDGs, addressing, reduction) | 7.9 |

- **What a wgmma pipeline removes:** wgmma takes A from registers and B from shared memory via a descriptor, and
  uses TMA for weights. That drops the activation LDGs and addressing (about 3.7 instr/weight) and most of the
  mma issue (about 0.55).
- **What it keeps:** the dequant, about 2.5. The micro-benchmark measures 2.5, not the 1.5 counted statically.
- **Estimate:** I ≈ 3.6 to 4.2, below the 5.3 budget. So H100 int4 would become bandwidth-bound: gate_up
  ≈ 17 µs vs 27.2 µs now, **about 1.5× at batch 1**.
- **Caveat:** this is an estimate from the micro-benchmark, not a measured wgmma kernel.
- **The same logic says the next byte saving only pays after that rewrite.** Once I is about 4, a 3-bit LUT
  (about +2.2 instr) becomes ALU-bound again at 4.1 budget, while 2-bit int-style codes might just fit.

**RTX PRO 6000: 3-bit LUT re-encode of one gpt-oss-120b MoE layer (layer 28, real weights).**
- **Setup:**
  - Every expert's MXFP4 weights are dequantized and re-encoded with RTN.
  - The layer output is compared with the MXFP4 layer on 256 random hidden states through the real router.
  - Timing cycles 8 (T=1) or 3 (T=16) routing variants so that nothing is L2-resident.

| encoding of the experts | bpw | layer-output rel. L2 vs MXFP4 | cosine |
|---|---|---|---|
| NF3 g64 (absmax) | 3.25 | 0.391 | 0.930 |
| int4 g64 | 4.5 | 0.243 | 0.971 |
| NF4 g64 | 4.25 | 0.149 | 0.989 |

| kernel (RTX PRO 6000) | T=1 µs (4 experts) | T=16 µs (46 experts) |
|---|---|---|
| magni moe-mxfp4 (fused, incl. swiglu + combine) | 39.8 | 395.9 |
| ours, NF3 lut3p, the active experts' gate_up + down as 2 stacked GEMVs | 33.1 (1.20×) | 451.5 (0.88×) |
| ours, int4 g64, same stacking | 38.4 | 485.0 |

- **The stacked GEMVs carry exactly the routed layer's bytes,** but they exclude swiglu, combine and routing
  (a few µs). The down projection uses a synthetic input.
- **Speed: the 3-bit re-encode pays at T=1, by about 1.2× of GEMV time.** At T=16 it loses, because the fused
  grouped kernel batches tokens per expert, while ours runs every expert at B=16.
- **Quality: the RTN re-encode is not acceptable.**
  - MXFP4 → NF3 changes the layer output by 39% (cosine 0.93).
  - Even MXFP4 → NF4 at the same bpw changes it by 15%. MXFP4 is gpt-oss's trained format, so any regridding is a
    lossy second quantization.
  - A usable 3-bit gpt-oss needs a calibrated re-encode (GPTQ/LDLQ from routed activations) and a GSM8K gate.
    That is budgeted below.

**Needs budget (not run):**

| item | what | estimate |
|---|---|---|
| Full QTIP 2-bit on Qwen3-8B | Hessians (C4, ~0.3 h), LDLQ + L=16 bitshift trellis (16× our L=12 Viterbi states: ~3.5 A100-h), QTIP fine-tune (2 to 4 H100-h), PPL + tasks (0.5 h) | ~$25-35 |
| gpt-oss-120b 3-bit experts, GSM8K-gated | per-expert Hessians from routed tokens + GPTQ/LDLQ re-encode of 36 x 128 experts (~2-3 H100-h), GSM8K-250 + MMLU subset on a fake-quant model (1 B200/H200, 1-2 h), our lut3p kernel in the moe path | ~$20-25 |
| wgmma int4 / NF4 GEMV on H100 | write and validate the kernel; ~1 H100-h of tuning | ~$5 |

## Files

- `codecs.py`: all codecs (fake-quant reference implementations), including the percentile codec, Bennett,
  Lloyd, kmeans, MXFP4, NF, RHT and the trellis Viterbi.
- `modal_analyze.py`: item 6 statistics (Modal CPU).
- `modal_quality.py`: item 7 (A100-80GB). It also contains the GPTQ with arbitrary scalar codecs and the pruning
  configs.
- `kernels/wq_gemv.cu`, `kernels/kpack.py`: item 8 kernels and packers, plus the VLC encoder that replays the
  decoder's refill schedule.
- `kernels/bench_marlin.py`, `modal_marlin.py`: Marlin baseline (the vLLM build). The standalone wheel faults
  with an illegal instruction under CUDA 13.
- `modal_kernels.py`, `roofline.py`, `summarize.py`: kernel benchmark, item 9 model, tables.
- `results/`: raw JSON and logs, and `TABLES.md` with every row.

Spend: about $33.5 of the $35 cap (Modal list prices, estimated from app wall clock).
