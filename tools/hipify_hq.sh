#!/usr/bin/env bash
# Mechanical CUDA->HIP translation of the upstream HyperQuant kernels (step 1 of the
# HIP port).  Output goes to hip/gen/; the hand-written fixes for wave64, fp8 (OCP
# vs FNUZ), bf16 and cuBLASLt are applied on top by tools/hip_fixups.py and the
# result is committed as hip/*.  Re-run to see exactly what is mechanical.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UP=$ROOT/upstream
OUT=$ROOT/hip/gen
mkdir -p "$OUT"
declare -A MAP=(
  [cuda/stage2_cuda_encoder.h]=stage2_hip_encoder.h
  [cuda/stage2_cuda_encoder.cu]=stage2_hip_encoder.hip
  [cuda/stage2_cuda_decoder.h]=stage2_hip_decoder.h
  [cuda/stage2_cuda_decoder.cu]=stage2_hip_decoder.hip
  [cuda/stage3_cuda_decoder.h]=stage3_hip_decoder.h
  [cuda/stage3_cuda_decoder.cu]=stage3_hip_decoder.hip
  [cuda/rht_cuda.h]=rht_hip.h
  [cuda/rht_cuda.cu]=rht_hip.hip
  [integrations/llama/metalrice_device.cuh]=metalrice_device_hip.hpp
  [integrations/llama/fused_decode_gemv.cu]=fused_decode_gemv.hip
  [integrations/llama/rice_fused_attention.cu]=rice_fused_attention.hip
)
for src in "${!MAP[@]}"; do
  sed -E \
    -e 's#<cuda_runtime\.h>#"hq_hip_compat.hpp"#' \
    -e '/#include <cuda_bf16\.h>/d' -e '/#include <cuda_fp8\.h>/d' \
    -e 's#"stage2_cuda_encoder\.h"#"stage2_hip_encoder.h"#; s#"stage2_cuda_decoder\.h"#"stage2_hip_decoder.h"#' \
    -e 's#"stage3_cuda_decoder\.h"#"stage3_hip_decoder.h"#; s#"rht_cuda\.h"#"rht_hip.h"#' \
    -e 's#"metalrice_device\.cuh"#"metalrice_device_hip.hpp"#' \
    -e 's/Stage2CudaEncoder/Stage2HipEncoder/g; s/Stage2CudaDecoder/Stage2HipDecoder/g; s/Stage3CudaDecoder/Stage3HipDecoder/g; s/RhtCuda/RhtHip/g' \
    -e 's/\bcudaError_t\b/hipError_t/g; s/\bcudaSuccess\b/hipSuccess/g; s/\bcudaGetErrorString\b/hipGetErrorString/g' \
    -e 's/\bcudaMalloc\b/hipMalloc/g; s/\bcudaFree\b/hipFree/g; s/\bcudaMemset\b/hipMemset/g' \
    -e 's/\bcudaMemcpy2D\b/hipMemcpy2D/g; s/\bcudaMemcpy\b/hipMemcpy/g; s/\bcudaMemcpyAsync\b/hipMemcpyAsync/g' \
    -e 's/\bcudaMemcpyHostToDevice\b/hipMemcpyHostToDevice/g; s/\bcudaMemcpyDeviceToHost\b/hipMemcpyDeviceToHost/g; s/\bcudaMemcpyDeviceToDevice\b/hipMemcpyDeviceToDevice/g' \
    -e 's/\bcudaEvent_t\b/hipEvent_t/g; s/\bcudaEventCreate\b/hipEventCreate/g; s/\bcudaEventRecord\b/hipEventRecord/g; s/\bcudaEventDestroy\b/hipEventDestroy/g; s/\bcudaEventElapsedTime\b/hipEventElapsedTime/g; s/\bcudaEventSynchronize\b/hipEventSynchronize/g' \
    -e 's/\bcudaDeviceSynchronize\b/hipDeviceSynchronize/g; s/\bcudaGetLastError\b/hipGetLastError/g; s/\bcudaStream_t\b/hipStream_t/g' \
    -e 's/\bcheck_cuda\b/check_hip/g' \
    -e 's/__nv_bfloat16/hq_bf16_t/g; s/__float2bfloat16_rn/hq_float2bf16/g; s/__float2bfloat16/hq_float2bf16/g; s/__bfloat162float/hq_bf162float/g' \
    -e 's/__shfl_xor_sync\(0xFFFFFFFFu, ([^,]+), ([^)]+)\)/__shfl_xor(\1, \2, 32)/g' \
    -e 's/__shfl_down_sync\(0xFFFFFFFF, ([^,]+), ([^)]+)\)/__shfl_down(\1, \2, 32)/g' \
    -e 's/__syncwarp\(\)/hq_syncwarp32()/g' \
    "$UP/$src" > "$OUT/${MAP[$src]}"
done
echo "hipified ${#MAP[@]} files -> $OUT"
grep -n -E 'cuda|__nv_|_sync\(' "$OUT"/* | grep -v -E '^\S+:\s*//|cuda_stream|"cuda' || true
