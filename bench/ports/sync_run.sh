#!/usr/bin/env bash
# usage: bash bench/ports/sync_run.sh <method> [speed|check|quality|all]   (alias-only ssh; shared GPU lock)
set -euo pipefail
M=$1; WHAT=${2:-all}
HOST=magni-amd; DEST=/root/uqports
ssh "$HOST" "mkdir -p $DEST/results"
rsync -aR --exclude __pycache__ uqmt bench/ports bench/quality results/mi300x_peak.json "$HOST:$DEST/"
RUN="cd $DEST && source /root/uq-venv/bin/activate && export TRITON_DEFAULT_FP_FUSION=1 AMD_KERNELS=/root/h2h/kernels"
if [[ $WHAT == speed || $WHAT == all ]]; then
  ssh "$HOST" "$RUN && flock /root/gpu.lock timeout 1500 python bench/ports/run_port.py --method $M --out results/ports_$M.json 2>&1 | grep -v -i warn | tee results/ports_${M}_speed.log"
fi
if [[ $WHAT == check || $WHAT == all ]]; then
  ssh "$HOST" "$RUN && flock /root/gpu.lock timeout 900 python bench/ports/mp_run.py --method $M --out results/ports_$M.json 2>&1 | grep -v -i warn | tee results/ports_${M}_check.log"
fi
if [[ $WHAT == quality ]]; then   # M may be a comma list; one lock hold per model
  for MK in mistral7b qwen3_8b; do
    ssh "$HOST" "$RUN && flock /root/gpu.lock timeout 3000 python bench/ports/mp_quality.py --method $M --models $MK 2>&1 | grep -v -i -E 'warn|Loading weights' | tee -a results/ports_quality_$MK.log"
  done
  M="*"
fi
mkdir -p results
rsync -a "$HOST:$DEST/results/ports_$M*" results/ || true
