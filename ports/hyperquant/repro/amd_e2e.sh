#!/usr/bin/env bash
# End-to-end PPL on the MI300X (run on the AMD box from /root/hq-port/src).
# Step 1 (no GPU): fetch + sha-verify the model.  Step 2 (GPU, under tools/amd_gpu_run.sh): PPL.
set -e
cd "$(dirname "$0")/.."
. /root/hq-port/venv/bin/activate
export HF_HOME=/root/hq-port/hf HQ_UPSTREAM=$PWD/upstream TORCH_EXTENSIONS_DIR=/root/hq-port/torch_ext
M=/root/hq-port/models/llama31-8b-instruct
if [ "$1" = "fetch" ]; then
  python - <<'PY'
import hashlib, json, os
from huggingface_hub import snapshot_download
M = "/root/hq-port/models/llama31-8b-instruct"
exp = json.load(open("repro/model_hashes.json"))["files"]
snapshot_download("NousResearch/Meta-Llama-3.1-8B-Instruct", local_dir=M, allow_patterns=list(exp))
def blob(p):
    d = open(p, "rb").read(); return hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
rep = {k: (sha(os.path.join(M, k)) if "sha256" in e else blob(os.path.join(M, k))) == (e.get("sha256") or e.get("git_blob")) for k, e in exp.items()}
print(json.dumps(rep))
PY
  exit 0
fi
mkdir -p build/out
cd upstream
HQ_JOIN=nonempty python ../repro/ppl_eval.py --model $M --configs "$2" --out ../build/out/ppl_rocm.jsonl --tok-cache /root/hq-port/wt2_tokens.pt
