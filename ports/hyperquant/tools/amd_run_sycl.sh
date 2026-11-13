#!/usr/bin/env bash
# One lock hold on the AMD box: SYCL port (AdaptiveCpp, hip:gfx942) tests + bench.
# Bundle (built on Modal by sycl/modal_acpp_amd.py) unpacked at /root/hq-port/acpp-amd/bundle.
B=/root/hq-port/acpp-amd/bundle
export LD_LIBRARY_PATH=$B/acpp/lib:/opt/rocm/lib:$LD_LIBRARY_PATH ACPP_VISIBILITY_MASK=hip
cd "$(dirname "$0")/.."
mkdir -p build/out
timeout 300 $B/test_sycl_amd vectors --json build/out/test_sycl_amd.json > build/out/test_sycl_amd.log 2>&1
echo "test rc=$?"; grep -E "backend|FAIL|SUMMARY" build/out/test_sycl_amd.log | head -12
rm -f build/out/bench_sycl_amd.jsonl
timeout 900 $B/bench_sycl_amd build/out/bench_sycl_amd.jsonl > build/out/bench_sycl_amd.log 2>&1
echo "bench rc=$?"; wc -l build/out/bench_sycl_amd.jsonl
