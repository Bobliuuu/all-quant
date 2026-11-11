#!/usr/bin/env bash
# MI300X (host alias: magni-amd) NVFP4 test + bench.  PENDING: host was unreachable when written.
# usage (from repo root, local):  bash bench/nvfp4/run_amd.sh
# Syncs to /root/uqmt-nvfp4/ and runs under the shared GPU lock (each hold <= 10 min).
set -euo pipefail
HOST=magni-amd; DEST=/root/uqmt-nvfp4
rsync -a --exclude __pycache__ uqmt tests bench/nvfp4 "$HOST:$DEST/" 2>/dev/null || {
  ssh "$HOST" "mkdir -p $DEST/bench"; rsync -a --exclude __pycache__ uqmt tests "$HOST:$DEST/"; rsync -a --exclude __pycache__ bench/nvfp4 "$HOST:$DEST/bench/"; }
ssh "$HOST" "cd $DEST && flock /root/gpu.lock timeout 600 python -m pytest tests/test_nvfp4.py -q 2>&1 | tail -15 | tee nvfp4_test_mi300x.log"
ssh "$HOST" "cd $DEST && flock /root/gpu.lock timeout 600 python bench/nvfp4/bench_nvfp4.py --nsplits 0 8 16 32 --out nvfp4_bench_mi300x.json 2>&1 | tee nvfp4_bench_mi300x.log"
rsync -a "$HOST:$DEST/nvfp4_bench_mi300x.json" "$HOST:$DEST/nvfp4_bench_mi300x.log" "$HOST:$DEST/nvfp4_test_mi300x.log" results/
