# UltraQuant explainer: narration script
Each `##` is a scene; each line is one narrated sentence (drives timing, captions and the .srt).
Numbers are read from results/*.json at build time; the text below must match them (build.py asserts the headline ones).

## title
UltraQuant: four-bit KV caching for context-heavy agents.
This is a short tour of the idea, and of one fused decode kernel that makes speculative decoding cheap on a four-bit cache.

## vendors
<!-- Included only when results/cuda_*.json or results/sycl_*.json exist. -->
The same fused operation now runs on three vendors: AMD, NVIDIA and Intel.
Here is the fused speedup at the largest K measured on each backend, read straight from the results files.
Every bar uses the same op and the same four-bit cache format.

## memory
When a language model generates text, each new token attends to every earlier token, so the GPU must read the whole key-value cache from memory.
That is almost no arithmetic per byte: about one floating point operation for every byte read in half precision.
To keep its matrix units busy, an MI300X needs roughly one hundred sixty operations per byte, and a B300 about two hundred seventy. These ratios come from our measured peaks.
So decode waits on memory, and for long-context agents the KV cache is the bottleneck. Fewer bytes means faster tokens.

## pack
UltraQuant stores each value as a four-bit float, E2M1: one sign bit, two exponent bits, one mantissa bit.
Every group of thirty-two values shares one eight-bit power-of-two scale, called UE8M0.
For a head dimension of one hundred twenty-eight, that is sixty-four bytes of values plus four bytes of scales: sixty-eight bytes per head and token.
Half precision needs two hundred fifty-six, and FP8 needs one hundred twenty-eight. That is about three point eight times smaller than half precision.

## kquery
Speculative decoding and multi-token prediction verify K candidate tokens in a single step.
A straightforward kernel treats them as K independent queries, and each one re-reads every page of the cache.
At K equals eight, that is eight times the memory traffic for the very same data.
Our fused kernel reads each page once. The query rows for all K tokens are stacked along the M dimension of the matrix multiply, so one loaded tile serves every token.
Same pages, same four-bit dequantisation, one pass.

## results
Here are measurements at batch eight, thirty-two thousand tokens of context, sixty-four query heads and eight KV heads.
On an MI300X, at K equals eight, the fused kernel takes 0.54 milliseconds, versus 2.22 for independent queries: 4.1 times faster.
On a B300, after tuning, K equals eight takes 0.42 milliseconds, versus 2.85: 6.8 times faster.
At K equals one there is nothing to share, so the two match. The gain grows with K, because page reads are shared.
Tuning mattered: bit-trick dequantisation, a tuned warp count and pipeline depth, and sixteen-way KV splits.

## caveats
What does this mean in practice? A long-context agent keeps more history in the same memory, and speculative verification stops multiplying the cost of reading it.
Now the honest caveats. On MI300X, which is gfx942, FP4 is dequantised in software; the paper's native scaled matrix instructions live on CDNA4, gfx950.
Our kernels reach about half a terabyte to one terabyte per second of FP4 bytes, well below the HBM peaks of three point nine and six point four terabytes per second, so there is headroom.
Outputs match the baseline to about one in ten thousand. The paper is arXiv 2606.20474, and the code and results are in the repository.
