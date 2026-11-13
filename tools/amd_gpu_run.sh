#!/usr/bin/env bash
# Run one short GPU command on the shared AMD box, respecting every tenant convention:
#  1. never while the push-boxes vLLM container (pb-vllm) is up  -> exit 75 (retry later)
#  2. mkdir /root/gpu-lock.d   (coordinator convention; /root/gpu.lock is a stale file)
#  3. flock /root/gpu.lock     (what the h2h/* and push-boxes scripts actually take)
# Both locks are released on exit.  usage (on the box): amd_gpu_run.sh <command...>
if docker ps --format '{{.Names}}' | grep -q '^pb-vllm$'; then echo "pb-vllm is up: not running GPU work"; exit 75; fi
mkdir /root/gpu-lock.d 2>/dev/null || { echo "gpu-lock.d held"; exit 75; }
trap 'rmdir /root/gpu-lock.d' EXIT
flock -n /root/gpu.lock -c "echo lock acquired \$(date -u +%H:%M:%S); $(printf '%q ' "$@")"
rc=$?; [ $rc -eq 1 ] && echo "gpu.lock flock held"
exit $rc
