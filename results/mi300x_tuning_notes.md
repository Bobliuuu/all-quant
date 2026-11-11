# MI300X tuning notes (fused K-query FP4 decode, B8 ctx32k, Hq64/Hkv8/D128)

Best: UQ_WARPS=2, UQ_STAGES=2, nsplit=16, page=64 -> K=1 0.36 ms (~790 GB/s FP4 bytes), K=8 0.54 ms, 4.1x vs K independent queries.

Findings
- Bit-trick e2m1->bf16 dequant (placement + x2^126) replaced where-chain: ~1.5x with warps=2.
- 2 warps beat 4/8; num_stages=1 is 2x slower than 2/3; nsplit 16-64 flat for K=1, 16 best for K=8.
- Page size 128/256 flat or worse (256 much worse).
- Loading scales once per 32-group and broadcasting: no gain (K=1 same, K=8 slower), reverted.
- Diagnostic with trivial dequant (not correct, timing only): K=1 1.13 TB/s, K=8 0.44 ms. So dequant ALU is ~30%; the rest is kernel structure/occupancy.
- Peaks (peak.py): bf16 640 TFLOPS, fp8 1.12 PFLOPS, HBM copy 3.9 TB/s. Kernel is still well below HBM peak.

Not tried: fp8-MFMA path with per-group column scaling (MX-style scaled dot emulation), persistent kernel with software pipelining of block-table loads, bf16 partials.

## fp8-MFMA K-score path (UQ_FP8K=1, default off)
FP4->e4m3fnuz by bit placement (value*2^-7, folded into group scale), fp8 queries, per-32-group fp8 dots with column scaling.
Correct (max err 4e-3 vs 3e-4 for bf16 path; the error is fp8 query quantisation) but SLOWER on MI300X:
best K=1 0.38 ms (bf16 path 0.36), K=8 0.66 ms (bf16 path 0.54). Extra join/reshape shuffles + 4 small dots outweigh the cheaper dequant.
Kept behind the flag; not used by default.
