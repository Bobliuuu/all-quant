// hq_hip_compat.hpp -- the few things CUDA has that HIP spells differently or lacks.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "../common/hq_num.hpp"

// bf16 is carried as raw uint16_t with the portable, bit-exact RN conversions from
// hq_num.hpp (identical to __float2bfloat16_rn / __bfloat162float), so the port does
// not depend on a particular ROCm bf16 header revision.
typedef uint16_t hq_bf16_t;
__host__ __device__ __forceinline__ float hq_bf162float(hq_bf16_t b) { return hqn::bf2f(b); }
__host__ __device__ __forceinline__ hq_bf16_t hq_float2bf16(float f) { return hqn::f2bf_rn(f); }

// __syncwarp() for a 32-lane logical warp.  On wave64 (CDNA3 / MI300X) a 32-lane
// "warp" is half a wavefront, whose lanes already execute in lockstep; what is needed
// is a wavefront-scope memory fence so that LDS writes by one lane are visible to the
// others, plus a scheduling barrier.  On wave32 (RDNA) the same sequence is correct.
__device__ __forceinline__ void hq_syncwarp32() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
