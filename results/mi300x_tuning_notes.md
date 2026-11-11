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
