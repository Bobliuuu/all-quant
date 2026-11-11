# SYCL port on Modal: CPU (OpenCL) and NVIDIA (DPC++ CUDA) -- notes to merge

Driver: `sycl/modal/uq_modal.py` (`modal run --detach sycl/modal/uq_modal.py --mode cpu` / `--mode gpu --gpu a100|h100|l4` / `--mode probe`).
Results: `results/sycl_cpu_opencl.{log,json}`, `results/sycl_nvidia_{a100,h100}.{log,json}`, `results/sycl_probe_gpu.log`.

## Result: SAME source, zero source changes
`sycl/uq_decode.hpp`, `test_vectors.cpp`, `bench.cpp` were not modified. Built with `-DUQ_NO_XMX` only.
`[[sycl::reqd_sub_group_size(SGS)]]` was a non-issue: sg 16 is accepted on the Intel OpenCL CPU device, sg 32 on NVIDIA (warp = 32). Use `--sg 32` on NVIDIA.

| target | toolchain flags | vectors (`--sweep`, 9 cases x nsplit 1,2,3,8) | worst max_abs_err |
|---|---|---|---|
| SYCL CPU (Intel OpenCL CPU RT, AMD EPYC 7V73X) | `-fsycl-targets=spir64_x86_64`, `--sg 16` | 36/36 PASS | 7.715e-03 (edge_short_k4; within the 1e-3 + 2^-7 rel bf16 criterion) |
| A100-SXM4-80GB (CUDA backend) | `-fsycl-targets=nvptx64-nvidia-cuda`, `--sg 32` | 36/36 PASS (also wg=256 run 36/36) | 7.715e-03 |
| H100 80GB HBM3 (CUDA backend) | same | 36/36 PASS (also wg=256 36/36) | 7.715e-03 |
Errors are bit-identical across CPU/A100/H100 (max_abs_err per case identical to 4 digits).

## Toolchain (what worked)
- Image: `nvidia/cuda:12.6.3-devel-ubuntu22.04` (needs CUDA toolkit for libdevice/ptxas) + Modal python.
- DPC++: prebuilt intel/llvm GitHub nightly `nightly-2026-11-04` `sycl_linux.tar.gz` (428 MB) unpacked to /opt/dpcpp. It ships the CUDA UR adapter (`libur_adapter_cuda.so`), no build from source needed. clang 24.0.0git, "DPC++ compiler 7.2.0".
  `export PATH=/opt/dpcpp/bin:$PATH LD_LIBRARY_PATH=/opt/dpcpp/lib:...`
- CPU device: the nightly does NOT include an OpenCL CPU runtime. `apt install intel-oneapi-runtime-opencl` from the Intel oneAPI apt repo (pulls intel-oneapi-runtime-dpcpp-sycl-opencl-cpu) gives `[opencl:cpu]` in `sycl-ls`; add `/opt/intel/oneapi/lib:/opt/intel/oneapi/redist/lib` to LD_LIBRARY_PATH. Select with `ONEAPI_DEVICE_SELECTOR=opencl:cpu`; CUDA with `cuda:gpu`.
- Modal's driver reports CUDA 13.3 while the image toolkit is 12.6; the nightly CUDA adapter and nvptx64 builds worked regardless (JIT from PTX not needed since A100 sm_80 / H100 sm_90 are compiled by the default sm_50-ish PTX path and JITed by the driver).
- Image build ~3 min (Intel apt repo is the slow part); runs themselves ~2-5 min.

## Pitfalls
- `modal run --detach ... --mode gpu --gpu a100`: passing the two flags in one quoted string breaks the entrypoint; also detach only keeps the LAST triggered function alive when the client disconnects, so one function per `modal run`.
- Bench default sg is 16; you must pass `--sg 32` on NVIDIA (the driver does).
- CPU bench uses `--iters 3` (the portable kernel is slow on CPU: 8-563 ms/call).
- Run `sycl-ls` first: both `opencl:cpu` and `cuda:gpu` showed up in the same container only because the GPU function image has both runtimes.

## Perf (portable path 0, `-DUQ_NO_XMX`, fused K-query, FP4 KV bytes/s vs copy-kernel read peak)
Peaks measured by bench copy kernels: A100-80GB copy(r+w) 1593 GB/s, read 1686 GB/s; H100 copy 2999 GB/s, read 3139 GB/s; CPU (EPYC, 17 CUs) copy 41.7 GB/s, read 23.4 GB/s.
Fused ms (K-indep baseline ms, speedup = K-indep/fused):

| B | ctx | K | A100 fused | A100 K-indep | H100 fused | H100 K-indep | CPU fused |
|---|---|---|---|---|---|---|---|
| 2 | 4096 | 1 | 1.200 | - | 0.451 | - | 8.73 |
| 2 | 4096 | 4 | 1.622 | 3.075 (x1.90) | 0.616 | 1.660 (x2.69) | 20.2 |
| 2 | 4096 | 8 | 1.820 | 4.975 (x2.73) | 0.796 | 3.210 (x4.03) | 36.0 |
| 2 | 16384 | 1 | 3.537 | - | 1.398 | - | 32.9 |
| 2 | 16384 | 4 | 4.908 | 9.028 (x1.84) | 2.036 | 5.283 (x2.59) | 80.3 |
| 2 | 16384 | 8 | 6.763 | 17.68 (x2.61) | 2.744 | 10.32 (x3.76) | 149.6 |
| 8 | 4096 | 1 | 3.541 | - | 1.402 | - | 36.5 |
| 8 | 4096 | 4 | 4.916 | 8.333 (x1.70) | 2.034 | 4.523 (x2.22) | 92.6 |
| 8 | 4096 | 8 | 6.767 | 16.08 (x2.38) | 2.740 | 8.830 (x3.22) | 141.7 |
| 8 | 16384 | 1 | 13.98 | - | 5.208 | - | 129.0 |
| 8 | 16384 | 4 | 19.29 | 32.45 (x1.68) | 7.706 | 16.90 (x2.19) | 319.8 |
| 8 | 16384 | 8 | 26.56 | 62.69 (x2.36) | 10.58 | 33.40 (x3.16) | 562.9 |

Bandwidth utilisation is LOW: unique FP4-KV bytes/s is 5-10 GB/s on A100 (0.3-0.6% of the 1686 GB/s read peak) and 11-27 GB/s on H100 (0.4-0.9% of 3139 GB/s); CPU 0.2-1.1 GB/s (1-5% of 23 GB/s). The portable kernel is compute/SLM-bound (fp32 dequant to SLM + scalar FMA QK/PV, one tile at a time with 4 barriers per 32-token tile), not bandwidth-bound; it was tuned for Intel Arc. A real NVIDIA path would need tensor-core MMA (e.g. DPC++ joint_matrix for CUDA / bf16 mma) and cp.async-style staging; not attempted. nsplit was "auto" (4*CUs/(B*Hkv), pow2), which gives 16-32 splits at B=2; no sweep done on NVIDIA.
Fused vs K-independent speedup grows with K (up to 4.0x on H100) because the KV page read and dequant is shared across the K query rows.
