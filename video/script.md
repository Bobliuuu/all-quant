# UltraQuant explainer: narration script
Each `##` is a scene; each line is one narrated sentence (drives timing, captions and the .srt).
Numbers shown on screen are read from results/*.json and results/*.md at build time; keep the text below consistent with them.

## title
UltraQuant: four-bit KV caching for context-heavy agents.
This is a tour of one fused decode kernel, and of what it takes to run it on three vendors.

## vendors
The same operation, a fused multi-token decode over a four-bit KV cache, now runs on AMD, NVIDIA and Intel GPUs.
Each port is checked against reference vectors from the same quantiser, and passes.
At K equals eight, fusing beats K independent passes everywhere: from about two point five times to six point eight times.

## memory
Why it matters: decode is memory-bound. Every new token reads the whole KV cache, about one operation per byte, while these GPUs need over one hundred sixty.
So for long-context agents, fewer KV bytes means faster tokens.

## pack
UltraQuant stores four-bit E2M1 floats, with one eight-bit power-of-two UE8M0 scale per thirty-two values.
That is sixty-eight bytes per head and token, against two hundred fifty-six for half precision and one hundred twenty-eight for FP8.
About three point eight times smaller than half precision.

## kquery
Speculative decoding verifies K candidate tokens per step.
A straightforward kernel runs K independent queries, so each re-reads every page of the cache: eight times the traffic at K equals eight.
Our fused kernel reads each page once, with the K tokens stacked along the M dimension of the matrix multiply.
One loaded tile serves every token.

## results
In Triton at batch eight and thirty-two thousand tokens, K equals eight takes 0.54 milliseconds on an MI300X against 2.22 for independent queries, and 0.42 on a B300 against 2.85.
At K equals one there is nothing to share, so the two match.

## position
Now, honest positioning. On a B300, NVIDIA's own FlashInfer kernels for NVFP4 are faster than ours, and already fuse K-query verify.
Our contribution is the UltraQuant format with fused verify, on three vendors. We found no public CUDA or SYCL port of it.
BitDecoding-FP4 handles one query at a time, so it re-reads the cache K times.

## quality
On Qwen3-8B, FP8 and int8 are lossless in perplexity, and KIVI-4 and NVFP4 are the best four-bit-class formats.
The UltraQuant MXFP4-style format costs about half a perplexity point, about the same as TurboQuant-4.
NVFP4 scales through our kernel cost only 0.06, against 0.55 for MXFP4, for four more bytes per vector.

## step
End to end, the whole decode step is one point three to one point six times faster than fp16 on MI300X at batch eight with K equals eight, and up to two point one times on an H100.
At batch one there is almost no gain, because attention is a small slice of the step.
The real value is a cache three point eight times smaller, and cheap verification at large batch and long context.

## realworld
The only serving-level data so far is a proxy: stock vLLM on an H100, comparing bf16 and fp8 KV, not FP4.
Fp8 doubles capacity and speeds up spec decode by about a quarter, but greedy output drifts within tens of tokens, so quality must be measured.

## caveats
Caveats. MI300X dequantises FP4 in software, and NVIDIA's vendor kernels are faster on B300.
These are kernel microbenchmarks and a projected step model, with no FP4 serving numbers yet, and K equals one gains nothing.
The paper is arXiv 2606.20474. Code and results are in the repository.
