# Prior art: HyperQuant on non-NVIDIA hardware

Survey date: 2026-11-13.  The search was not exhaustive.

* **Upstream** (github.com/moonmath-ai/HyperQuant, MIT, 16 stars at survey time) is CUDA-only.  The README
  says "Hardware: H100 / A100 (SM 80+)", and the extensions are built with `-arch=sm_90a` through
  `torch.utils.cpp_extension`.  The paper reports H100 numbers only and names Blackwell as a target.
* **Forks**: the two public forks (kimhons/HyperQuant, splinter21/HyperQuant) have `pushed_at` equal to
  upstream's (2026-06-23), i.e. no commits of their own (GitHub API, 2026-11-13).
* **Web search** ("HyperQuant ROCm/HIP/SYCL port", "HyperQuant MI300X"): no port found.  MoonMath has
  published an MI300X HIP attention kernel ("A Fast Attention Kernel for MI300X, Written in HIP, Not
  Assembly", per their technology page), which is unrelated to HyperQuant's Rice/E8 kernels.  AMD Quark
  and other MI300 quantisation work cover fp8/int8/MXFP4, not E8 lattice + Rice coding.

So the accurate claim is: **we found no public ROCm/HIP or SYCL port of the HyperQuant (MetalRice) kernels**,
not "none exists".  Nothing here is algorithmically new.  The ports are faithful translations, the
algorithm is the paper's, and the six upstream issues in PORTING_LOG.md are observations about the
released code.

Sources: https://huggingface.co/papers/2606.23406, https://www.emergentmind.com/papers/2606.23406,
https://api.github.com/repos/moonmath-ai/HyperQuant/forks, https://moonmath.ai/hyperquant/
