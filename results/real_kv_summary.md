# Real-KV accuracy, Mistral-7B-v0.1 (wikitext-2 test)

Model: `mistralai/Mistral-7B-v0.1` (the same model as amd-experiments; downloaded fine, no fallback needed), fp16, sliding window disabled so all context is attended.
Real Q/K/V from a prefill forward; Q for the last K tokens is captured pre-attention (post-RoPE) and rounded to bf16 (kernel input). Reference = fp32 attention over the **unquantised** fp16 K/V with the same q and the same causal rule over the K-token block.
Metric: cosine and relative L2 of the attention output (all 32 q heads x K queries) vs reference, averaged; table shows mean over the 32 layers and the worst layer.

## Attention-output accuracy, 4096 real tokens of context

| method | K | mean cos | worst-layer cos | mean rel-L2 | worst rel-L2 |
|--|--:|--:|--:|--:|--:|
| UltraQuant FP4/UE8M0 fused kernel | 1 | 0.9757 | 0.9461 | 0.242 | 0.354 |
| UltraQuant FP4/UE8M0 fused kernel | 2 | 0.9757 | 0.9478 | 0.244 | 0.334 |
| UltraQuant FP4/UE8M0 fused kernel | 4 | 0.9761 | 0.9578 | 0.239 | 0.345 |
| UltraQuant FP4/UE8M0 fused kernel | 8 | 0.9755 | 0.9486 | 0.241 | 0.349 |
| amd-experiments TQ3 fused kernel (K=1 only) | 1 | 0.9756 | 0.9358 | 0.235 | 0.405 |
| FP4 simulated (quant-dequant, fp32 attn) vs kernel: kernel arithmetic error | 1 | 1.0000 | 1.0000 | 0.002 | 0.002 |
| fp4_ue8m0_sim (quant-dequant KV, fp32 attention) | 1 | 0.9757 | 0.9461 | 0.242 | 0.354 |
| tq3_sim (quant-dequant KV, fp32 attention) | 1 | 0.9756 | 0.9358 | 0.235 | 0.405 |
| fp8_pertensor_fixed240 (quant-dequant KV, fp32 attention) | 1 | 0.9987 | 0.9968 | 0.053 | 0.079 |
| fp8_pertoken (quant-dequant KV, fp32 attention) | 1 | 0.9994 | 0.9987 | 0.037 | 0.054 |
| int4_pertensor (quant-dequant KV, fp32 attention) | 1 | 0.8186 | 0.5239 | 0.703 | 1.199 |
| kivi4_own (quant-dequant KV, fp32 attention) | 1 | 0.9967 | 0.9904 | 0.081 | 0.122 |
| planar3 (quant-dequant KV, fp32 attention) | 1 | 0.7795 | 0.5642 | 1.590 | 3.296 |
| iso3 (quant-dequant KV, fp32 attention) | 1 | 0.7907 | 0.5646 | 1.492 | 3.078 |
| rotor3 (quant-dequant KV, fp32 attention) | 1 | 0.7922 | 0.5595 | 1.526 | 3.199 |

Reconstruction cosine of the stored vectors themselves (mean over layers, K / V):

| method | K cos | V cos |
|--|--:|--:|
| fp4_ue8m0_sim | 0.9909 | 0.9926 |
| tq3_sim | 0.9831 | 0.9831 |
| fp8_pertensor_fixed240 | 0.9997 | 0.9997 |
| fp8_pertoken | 0.9997 | 0.9997 |
| int4_pertensor | 0.9225 | 0.8809 |
| kivi4_own | 0.9989 | 0.9966 |
| planar3 | 0.9516 | 0.9765 |
| iso3 | 0.9573 | 0.9783 |
| rotor3 | 0.9569 | 0.9778 |

Per-layer detail (K=1 and K=8 cosine): `results/real_kv_attn_4k.json`. Mean KV reconstruction rel-L2 of FP4 over layers: 0.138.

## Attention-output accuracy, 8192 real tokens of context

| method | K | mean cos | worst-layer cos | mean rel-L2 | worst rel-L2 |
|--|--:|--:|--:|--:|--:|
| UltraQuant FP4/UE8M0 fused kernel | 1 | 0.9748 | 0.9446 | 0.244 | 0.383 |
| UltraQuant FP4/UE8M0 fused kernel | 2 | 0.9760 | 0.9455 | 0.239 | 0.387 |
| UltraQuant FP4/UE8M0 fused kernel | 4 | 0.9780 | 0.9579 | 0.229 | 0.355 |
| UltraQuant FP4/UE8M0 fused kernel | 8 | 0.9789 | 0.9647 | 0.222 | 0.325 |
| amd-experiments TQ3 fused kernel (K=1 only) | 1 | 0.9739 | 0.9312 | 0.238 | 0.392 |
| FP4 simulated (quant-dequant, fp32 attn) vs kernel: kernel arithmetic error | 1 | 1.0000 | 1.0000 | 0.002 | 0.002 |
| fp4_ue8m0_sim (quant-dequant KV, fp32 attention) | 1 | 0.9748 | 0.9446 | 0.244 | 0.383 |
| tq3_sim (quant-dequant KV, fp32 attention) | 1 | 0.9739 | 0.9312 | 0.238 | 0.392 |
| fp8_pertensor_fixed240 (quant-dequant KV, fp32 attention) | 1 | 0.9988 | 0.9973 | 0.049 | 0.072 |
| fp8_pertoken (quant-dequant KV, fp32 attention) | 1 | 0.9994 | 0.9985 | 0.035 | 0.054 |
| int4_pertensor (quant-dequant KV, fp32 attention) | 1 | 0.7685 | 0.6035 | 0.785 | 1.195 |
| kivi4_own (quant-dequant KV, fp32 attention) | 1 | 0.9968 | 0.9928 | 0.078 | 0.116 |
| planar3 (quant-dequant KV, fp32 attention) | 1 | 0.6886 | 0.4668 | 1.804 | 4.333 |
| iso3 (quant-dequant KV, fp32 attention) | 1 | 0.7054 | 0.4616 | 1.717 | 4.096 |
| rotor3 (quant-dequant KV, fp32 attention) | 1 | 0.7026 | 0.4835 | 1.690 | 3.933 |

Reconstruction cosine of the stored vectors themselves (mean over layers, K / V):

| method | K cos | V cos |
|--|--:|--:|
| fp4_ue8m0_sim | 0.9911 | 0.9926 |
| tq3_sim | 0.9831 | 0.9831 |
| fp8_pertensor_fixed240 | 0.9997 | 0.9997 |
| fp8_pertoken | 0.9997 | 0.9997 |
| int4_pertensor | 0.9204 | 0.8709 |
| kivi4_own | 0.9989 | 0.9966 |
| planar3 | 0.9540 | 0.9764 |
| iso3 | 0.9595 | 0.9782 |
| rotor3 | 0.9590 | 0.9777 |

Per-layer detail (K=1 and K=8 cosine): `results/real_kv_attn_8k.json`. Mean KV reconstruction rel-L2 of FP4 over layers: 0.137.

## Perplexity with simulated (quantise-dequantise) KV: wikitext-2 test, 8 non-overlapping windows of 4096 tokens

K and V (post-RoPE, every layer, every position including the current token) are quantised then dequantised inside attention in a teacher-forced forward; this equals decode with that cache format. Rotation-based TQ3 uses amd-experiments' own compress/decompress.

| KV format | bytes/vec | ppl | delta ppl vs fp16 | mean NLL |
|--|--:|--:|--:|--:|
| fp16 | 256 | 4.880 | +0.00% | 1.5851 |
| fp4 | 68 | 5.026 | +3.00% | 1.6146 |
| tq3 | 52 | 5.013 | +2.73% | 1.6120 |

Their reported TQ3 cosine 0.983 is a KV/attention cosine on their data; on our real-KV attention-output metric TQ3 gives the number in the table above.
