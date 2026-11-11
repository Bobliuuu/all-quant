# CUDA port effort log (UltraQuant K-query FP4 decode)

Started 2026-11-11T06:40Z (wall clock, UTC). Entries are appended per iteration.

| iter | time (UTC) | what | outcome |
|---|---|---|---|
| 0 | 06:40 | read Triton reference (422 LOC), wrote tools/gen_vectors.py (15 CPU reference cases, 4.8 MB), uq_decode.h | vectors generated |
| 1 | 06:41 | wrote uq_decode.cu v1 (~330 LOC): mma.sync bf16, cp.async ring, smem dequant (bit trick), ldmatrix, per-warp key split, split-KV + warp-per-row reduce; harness + torch ext + wrapper | untested (no local nvcc; Modal only) |
