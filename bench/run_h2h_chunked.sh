#!/bin/bash
# Runs h2h_microbench.py one (B,ctx) chunk at a time; each chunk takes /root/gpu.lock separately (shared box etiquette).
source /root/uq-venv/bin/activate; cd /root/h2h
R=/root/uqmt/results
for B in 1 8 32; do for C in 4096 16384 32768 65536; do
  [ -f $R/h2h_raw_B${B}_c${C}.json ] && continue
  flock /root/gpu.lock timeout 900 python h2h_microbench.py --uq-dir /root/uqpin --amd-dir /root/h2h/kernels \
     --batches $B --ctxs $C --out $R/h2h_raw_B${B}_c${C}.json > $R/h2h_run_B${B}_c${C}.log 2>&1
done; done; echo ALLDONE > $R/h2h_chunks_done
