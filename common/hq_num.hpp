// hq_num.hpp -- portable, bit-exact numeric helpers shared by the CPU reference,
// the HIP port and the SYCL port (host + device). No vendor intrinsics.
//
//   f2bf_rn   : fp32 -> bf16 round-to-nearest-even (NaN -> quiet NaN), identical to
//               CUDA __float2bfloat16_rn / __float2bfloat16.
//   bf2f      : bf16 -> fp32.
//   f2e4m3_rn : fp32 -> OCP FP8 E4M3 (e4m3fn) round-to-nearest-even, saturate to
//               +-448 (NaN -> 0x7F), identical to CUDA __nv_fp8_e4m3(float)
//               (__NV_SATFINITE, cudaRoundNearest). MI300X's native fp8 is the FNUZ
//               variant, so the HIP port uses this software conversion to keep the
//               stored byte format identical to CUDA.
//   SplitMix64 / gaussian(): deterministic RNG (no std:: distributions, whose
//               algorithms differ between libstdc++, libc++ and MSVC STL).
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HQ_HD __host__ __device__ __forceinline__
#else
#define HQ_HD inline
#endif

namespace hqn {

HQ_HD uint32_t f2u(float f) { return __builtin_bit_cast(uint32_t, f); }
HQ_HD float u2f(uint32_t u) { return __builtin_bit_cast(float, u); }

HQ_HD uint16_t f2bf_rn(float f) {
  uint32_t u = f2u(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu)) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
HQ_HD float bf2f(uint16_t b) { return u2f((uint32_t)b << 16); }

// OCP E4M3: bias 7, max normal 448 = 0x7E, NaN = 0x7F/0xFF, no inf.
HQ_HD uint8_t f2e4m3_rn(float f) {
  uint32_t u = f2u(f);
  uint8_t sign = (uint8_t)((u >> 24) & 0x80u);
  uint32_t a = u & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return 0x7F;                // NaN (CUDA gives 0x7F for NaN)
  float af = u2f(a);
  if (af >= 448.0f) return (uint8_t)(sign | 0x7E);  // saturate (incl. +-inf)
  // smallest normal 2^-6; subnormal step 2^-9
  int e = (int)(a >> 23) - 127;
  uint32_t res;
  if (e < -6) {
    // subnormal: value = m * 2^-9, m in [0,7]; round-to-nearest-even of af*512
    float s = af * 512.0f;                          // exact scaling by power of 2
    uint32_t m = (uint32_t)s;                       // floor
    float rem = s - (float)m;
    if (rem > 0.5f || (rem == 0.5f && (m & 1u))) m++;
    res = m;                                        // m==8 -> exponent field 1, mantissa 0: correct
  } else {
    uint32_t mant = a & 0x007FFFFFu;
    uint32_t exp8 = (uint32_t)(e + 7);              // 1..15
    uint32_t m3 = mant >> 20;
    uint32_t rest = mant & 0xFFFFFu;
    uint32_t half = 0x80000u;
    if (rest > half || (rest == half && (m3 & 1u))) {
      m3++;
      if (m3 == 8) { m3 = 0; exp8++; }
    }
    res = (exp8 << 3) | m3;
    if (res > 0x7Eu) res = 0x7Eu;                  // rounding up past 448 -> saturate
  }
  return (uint8_t)(sign | res);
}
HQ_HD float e4m3_to_f(uint8_t v) {
  uint32_t s = (v & 0x80u) ? 0x80000000u : 0u;
  uint32_t e = (v >> 3) & 0xFu, m = v & 7u;
  if (e == 0xF && m == 7) return u2f(0x7FC00000u);
  float r;
  if (e == 0) r = (float)m * (1.0f / 512.0f);
  else r = u2f(((e + 120u) << 23) | (m << 20));
  return u2f(f2u(r) | s);
}

struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
  double gaussian() {  // Box-Muller, one value per call (deterministic, double precision)
    double u1 = uniform(), u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
};

}  // namespace hqn
