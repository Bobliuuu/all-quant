#!/usr/bin/env bash
# One lock hold: correctness + bench of the opt-in MI300X launch variants.
cd "$(dirname "$0")/.."
mkdir -p build/out
for bt in 64 128; do
  HQ_HIP_GEMV_BT=$bt HQ_HIP_WS_SPLIT=1 timeout 300 ./build/test_hip vectors --json build/out/test_hip_tuned_bt$bt.json > build/out/test_hip_tuned_bt$bt.log 2>&1
  tail -1 build/out/test_hip_tuned_bt$bt.log
  rm -f build/out/bench_hip_tuned_bt$bt.jsonl
  HQ_HIP_GEMV_BT=$bt HQ_HIP_WS_SPLIT=1 timeout 600 ./build/bench_hip build/out/bench_hip_tuned_bt$bt.jsonl > /dev/null 2>&1
  grep -E '"gemv|"gemm_ws' build/out/bench_hip_tuned_bt$bt.jsonl | sed 's/"backend":"[^"]*",//' | cut -c1-120
done
