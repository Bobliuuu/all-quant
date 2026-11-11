# Scale rule / Hadamard / NVFP4 on real K/V (gaps-agent)

Models: Mistral-7B-v0.1, Qwen3-8B (bf16, HF, A100 via Modal). Wikitext-2 test, 16 x 2048 tokens. Whole-sequence FP4 KV simulation (quantise K post-RoPE, V at v_proj output, per layer). Hadamard rotates K only (Q implicitly; exact score invariance), V unrotated. `rel MSE` = mean over layers of normalised squared error; `attn` = attention-output rel. MSE (last 64 queries, 1024-token seq). Code: `bench/gaps/scale_rule_modal.py`, raw: `results/gaps_scale_rule.json`.

## mistralai/Mistral-7B-v0.1  (bf16 ppl 5.240)

| config | ppl | dppl | K rel MSE | V rel MSE | attn-out rel MSE |
|---|---|---|---|---|---|
| ceil | 5.392 | +0.152 | 0.0195 | 0.0154 | 0.0558 |
| paper_c0.125 | 6.108 | +0.867 | 0.0406 | 0.0346 | 0.2486 |
| paper_c0.156 | 5.567 | +0.327 | 0.0212 | 0.0184 | 0.1152 |
| paper_c0.195 | 5.398 | +0.157 | 0.0162 | 0.0137 | 0.0557 |
| paper_c0.250 | 5.405 | +0.164 | 0.0210 | 0.0162 | 0.0569 |
| paper_c0.167 | 5.482 | +0.242 | 0.0184 | 0.0159 | 0.0868 |
| nvfp4 | 5.299 | +0.059 | 0.0086 | 0.0088 | 0.0207 |
| ceil+rot | 5.282 | +0.041 | 0.0133 | 0.0154 | 0.0187 |
| paper_c0.125+rot | 5.535 | +0.295 | 0.0294 | 0.0346 | 0.0640 |
| paper_c0.156+rot | 5.334 | +0.093 | 0.0176 | 0.0184 | 0.0287 |
| paper_c0.195+rot | 5.282 | +0.042 | 0.0124 | 0.0137 | 0.0173 |
| paper_c0.250+rot | 5.283 | +0.043 | 0.0136 | 0.0162 | 0.0193 |
| paper_c0.167+rot | 5.318 | +0.078 | 0.0151 | 0.0159 | 0.0231 |
| nvfp4+rot | 5.273 | +0.032 | 0.0091 | 0.0088 | 0.0112 |
| kpaper0.156+rot_vceil | 5.307 | +0.067 | 0.0176 | 0.0154 | 0.0258 |
| ceil_edge2 | 5.368 | +0.127 | 0.0198 | 0.0151 | 0.0581 |
| paper_c0.156+rot_edge2 | 5.318 | +0.078 | 0.0176 | 0.0181 | 0.0281 |
| ceil+rot_edge2 | 5.273 | +0.033 | 0.0132 | 0.0151 | 0.0178 |

## Qwen/Qwen3-8B  (bf16 ppl 9.504)

| config | ppl | dppl | K rel MSE | V rel MSE | attn-out rel MSE |
|---|---|---|---|---|---|
| ceil | 10.158 | +0.654 | 0.0216 | 0.0155 | 0.1042 |
| paper_c0.125 | 11.243 | +1.739 | 0.0466 | 0.0344 | 0.2181 |
| paper_c0.156 | 10.428 | +0.924 | 0.0258 | 0.0185 | 0.1596 |
| paper_c0.195 | 10.021 | +0.516 | 0.0182 | 0.0137 | 0.0910 |
| paper_c0.250 | 10.181 | +0.677 | 0.0230 | 0.0163 | 0.1110 |
| paper_c0.167 | 10.002 | +0.497 | 0.0219 | 0.0160 | 0.1332 |
| nvfp4 | 9.644 | +0.140 | 0.0083 | 0.0088 | 0.0357 |
| ceil+rot | 10.131 | +0.626 | 0.0129 | 0.0155 | 0.0568 |
| paper_c0.125+rot | 16.404 | +6.900 | 0.0337 | 0.0344 | 0.2054 |
| paper_c0.156+rot | 10.827 | +1.322 | 0.0170 | 0.0185 | 0.0964 |
| paper_c0.195+rot | 10.186 | +0.682 | 0.0122 | 0.0137 | 0.0547 |
| paper_c0.250+rot | 10.143 | +0.638 | 0.0133 | 0.0163 | 0.0583 |
| paper_c0.167+rot | 10.551 | +1.047 | 0.0144 | 0.0160 | 0.0714 |
| nvfp4+rot | 10.085 | +0.581 | 0.0093 | 0.0088 | 0.0419 |
| kpaper0.156+rot_vceil | 10.624 | +1.120 | 0.0170 | 0.0155 | 0.0988 |
| ceil_edge2 | 10.173 | +0.669 | 0.0213 | 0.0153 | 0.1008 |
| paper_c0.156+rot_edge2 | 9.826 | +0.322 | 0.0173 | 0.0183 | 0.0768 |
| ceil+rot_edge2 | 9.641 | +0.137 | 0.0130 | 0.0153 | 0.0441 |

## Findings

1. **Our current rule `ceil(log2(amax/6))` is already about as good as it gets; do NOT adopt the paper's c=0.156.** `ceil` is algebraically `round(log2(c*amax))` with c = sqrt(2)/6 = 0.2357, so it sits inside the best band (c 0.195-0.25). c=0.156 (paper) clips too often on these models: dppl +0.33 vs +0.15 (Mistral), +0.92 vs +0.65 (Qwen3); c=0.125 is much worse (+0.87 / +1.74). Best plain c: 0.195 (Mistral +0.157 ~ ceil +0.152; Qwen3 +0.516 vs +0.654) and 1/6 on Qwen3 (+0.497) -- but they are not consistently better (c=1/6 is worse on Mistral, +0.24), and with Hadamard ceil/0.195/0.25 are within noise of each other. Caveat: the paper's c was fit to other models/their scale convention; our data shows it does not transfer here.
2. **Hadamard rotation on K is the real win, model dependent.** Mistral-7B: dppl +0.152 -> +0.041 (-73%), attn-out error 0.056 -> 0.019. Qwen3-8B: only +0.654 -> +0.626 alone (K MSE -40%, attn-out -45%), because its loss is dominated by boundary layers (below).
3. **First/last-2 layers BF16 matters on Qwen3-8B** (rot + edge2: +0.626 -> +0.137, -78%) and is minor on Mistral (+0.041 -> +0.033). Without rotation edge2 alone barely helps (Qwen3 +0.654 -> +0.669), so on Qwen3 the pair (rotation + BF16 edges) is needed.
4. **NVFP4 (E4M3 scales, 16-group) is best unrotated**: dppl +0.059 (Mistral) / +0.140 (Qwen3) -- better than MXFP4+rotation on Qwen3 and the lowest K/V MSE overall (0.9%) -- for +0.25 bits/elt (4.5 vs 4.25 incl. scale). Rotation did not help NVFP4 on Qwen3 (+0.581, worse; per-tensor scale from rotated K amax is the suspected cause, not investigated) and helps slightly on Mistral (+0.032). NVFP4 needs an E4M3 scale path in the kernel (not UE8M0-readable by `uq_decode`).
5. K and V scale rule: using the paper rule on K only with V on ceil is a bit better than paper on both (Mistral +0.067 vs +0.093) but not better than ceil+rot (+0.041).

## Recommendation
- Keep `quantize_fp4` default (`ceil`). No change to attn.py's rule. Document that it equals the paper rule at c=0.236 and that c=0.156 is worse on Mistral-7B/Qwen3-8B.
- Add as opt-in (outside the kernel, no kernel change): Hadamard on K and Q (`uqmt.quant.fwht` / `rotate_kq`) -- recommended on by default for the verify path when the 1-2 extra WHT passes are affordable; and a per-layer policy with first/last 2 layers BF16 (`uqmt.quant.layer_policy`).
- NVFP4 is the accuracy-optimal format if a kernel with an E4M3 scale path is written (B300/native FP4); on MI300X keep MXFP4+rotation.
- Limits: perplexity on 32k tokens, one seed (deterministic data order), attention-error from one 1024-token sequence; no long-context tasks (gap 9).
