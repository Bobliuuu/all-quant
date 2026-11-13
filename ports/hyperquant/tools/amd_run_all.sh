#!/usr/bin/env bash
# One lock hold on the AMD box: HIP correctness tests + microbenchmarks (~2-4 min).
cd "$(dirname "$0")/.."
mkdir -p build/out
rocm-smi --showproductname --showclocks 2>/dev/null | grep -E "Card Series|sclk|mclk" | head -4 > build/out/gpuinfo.txt
timeout 300 ./build/test_hip vectors --json build/out/test_hip.json > build/out/test_hip.log 2>&1
echo "test rc=$?"; tail -1 build/out/test_hip.log
rm -f build/out/bench_hip.jsonl
timeout 900 ./build/bench_hip build/out/bench_hip.jsonl ${HQ_BENCH_ARGS:-} > build/out/bench_hip.log 2>&1
echo "bench rc=$?"; wc -l build/out/bench_hip.jsonl
