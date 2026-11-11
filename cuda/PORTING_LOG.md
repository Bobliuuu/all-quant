# CUDA port effort log (UltraQuant K-query FP4 decode)

Started 2026-11-11T06:40Z (wall clock, UTC). Entries are appended per iteration.

| iter | time (UTC) | what | outcome |
|---|---|---|---|
| 0 | 06:40 | read Triton reference (422 LOC), wrote tools/gen_vectors.py (15 CPU reference cases, 4.8 MB), uq_decode.h | vectors generated |
| 1 | 06:41 | wrote uq_decode.cu v1 (~330 LOC): mma.sync bf16, cp.async ring, smem dequant (bit trick), ldmatrix, per-warp key split, split-KV + warp-per-row reduce; harness + torch ext + wrapper | untested (no local nvcc; Modal only) |
| 2 | 06:58 | first Modal run (H100 host actually reported an H200; B300). Correct first try: 528 harness configs ran; bench-shape err 1e-4..2e-4 vs fp32 reference. Pitfalls: macro CK(e) shadow bug; abs-1e-3 target is unreachable on short/high-magnitude outputs because bf16 output rounding alone is ~4e-3 (criterion changed to 1e-3 + 2^-7 rel); pip torch default index pulled cu13 wheels (use --index-url cu129); Modal image build ~10 min first time | B300 K=1/2/4/8 fused 0.218/0.219/0.253/0.350 ms (Triton tuned 0.308/0.310/0.425/0.494): 1.41x/1.41x/1.68x/1.41x; only 12-20% of HBM copy peak (6574 GB/s) |
