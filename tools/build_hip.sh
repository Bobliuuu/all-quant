#!/usr/bin/env bash
# Build the HIP port test + bench on the AMD box (run there, inside /root/hq-port/src).
# HIP sources -> objects with hipcc; the shared C++ drivers with g++; link with hipcc.
set -e
cd "$(dirname "$0")/.."
mkdir -p build/obj
ARCH=${HQ_HIP_ARCH:-gfx942}
F="-O3 -std=c++17 --offload-arch=$ARCH -Ihip -Icommon -w"
for s in hip/backend_hip.cpp hip/stage2_hip_encoder.hip hip/stage2_hip_decoder.hip hip/stage3_hip_decoder.hip hip/fused_decode_gemv.hip hip/rice_fused_attention.hip; do
  o=build/obj/$(basename $s).o
  hipcc $F -x hip -c $s -o $o 2>&1 | grep -E "error" -A2 | head -40 || true &
done
for t in test bench; do g++ -O2 -std=c++17 -Icommon -c common/${t}_main.cpp -o build/obj/${t}_main.o & done
wait
OBJS="build/obj/backend_hip.cpp.o build/obj/stage2_hip_encoder.hip.o build/obj/stage2_hip_decoder.hip.o build/obj/stage3_hip_decoder.hip.o build/obj/fused_decode_gemv.hip.o build/obj/rice_fused_attention.hip.o"
for t in test bench; do hipcc --offload-arch=$ARCH $OBJS build/obj/${t}_main.o -o build/${t}_hip -lhipblaslt 2>&1 | grep -E "error|undefined" | head -20 || true; done
ls -la build/ | grep _hip
