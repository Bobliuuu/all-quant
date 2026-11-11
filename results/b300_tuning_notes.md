# B300 tuning notes (B=8, ctx=32768, Hq=64, Hkv=8, D=128)

Single B300 (sm_103), one Modal job; sweep of UQ_WARPS {1,2,4,8} x UQ_STAGES {1,2,3} x nsplit {8,16,32,64}, K in {1,8} (96 rows in `b300_sweep.json`/`.log`). Correctness at B=2 ctx=4096: max|err| 3e-4 (K=1,8), under 1e-3.

## Best config
- K=1: warps=4 stages=3 nsplit=64
- K=8 (also used for K=2,4): warps=8 stages=3 nsplit=16

## Tuned vs original-default (warps=4 stages=2 nsplit=16, same HEAD kernel)

| K | tuned fused_ms | base_ms | speedup | GB/s | default fused_ms | default speedup | default GB/s |
|---|---|---|---|---|---|---|---|
| 1 | 0.288 | 0.287 | 1.00 | 990 | 0.357 | 1.00 | 798 |
| 2 | 0.421 | 0.777 | 1.84 | 677 | 0.357 | 1.79 | 799 |
| 4 | 0.532 | 1.479 | 2.78 | 536 | 0.422 | 2.82 | 676 |
| 8 | 0.421 | 2.850 | 6.77 | 677 | 0.450 | 4.98 | 633 |

K=2 and K=4 were not swept; they reuse the K=8 config, which is worse for them than the default (0.357 / 0.422 ms). Best-of-both per K would take the default row for K=2,4. Files: `b300_b8_c32k_tuned.json/.log`, `b300_b8_c32k_origdefault_head.json`.

## Sweep sensitivity (best fused_ms)
- K=1: warps 1/2/4/8 = 0.424/0.327/0.288/0.400; stages 1/2/3 = 0.322/0.327/0.288; nsplit 8/16/32/64 = 0.328/0.296/0.290/0.288.
- K=8: warps 1/2/4/8 = 1.408/0.756/0.450/0.421; stages 1/2/3 = 0.445/0.423/0.421; nsplit 8/16/32/64 = 0.426/0.421/0.429/0.501.

## vs MI300X (`mi300x_b8_c32k_tuned.json`: warps=2 stages=2 nsplit=16, K=8 0.542 ms, 4.1x, 527 GB/s)
B300 prefers more warps (8 for K=8, 4 for K=1) and deeper pipelining (stages=3) where MI300X peaked at 2 warps / 2 stages; nsplit is similar (16 for K=8, large for K=1), and tuned K=8 is 6.8x over baseline vs 4.1x on MI300X. K=1 shows no fused gain on either GPU.
