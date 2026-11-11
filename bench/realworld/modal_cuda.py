"""CUDA-side comparison (H100): same serving harness, vLLM fp16(bf16) vs fp8 KV.  Run:
   modal run --detach bench/realworld/modal_cuda.py   (results land in results/realworld_cuda_*.json via local_entrypoint)"""
import json, os, subprocess, sys
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").pip_install("vllm", "aiohttp", "pyarrow", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).add_local_dir(HERE, "/rw", copy=True))
vol = modal.Volume.from_name("uq-rw-models", create_if_missing=True)
app = modal.App("uq-realworld-cuda")

JOBS = []
_AG = "--sessions 48 --turns 6 --prefix-tokens 24000 --gen 128 --distinct"
_SP = "--n 64 --gen 384 --conc {c} --k {k} --extra '--speculative-config {{\"method\":\"ngram\",\"num_speculative_tokens\":{k},\"prompt_lookup_max\":4}}'"
for kv, nm in (("auto", "bf16"), ("fp8", "fp8")):
    JOBS.append((f"agentdistinct_{nm}", "Mistral-7B-v0.1", kv, "agent", _AG))
    JOBS.append((f"agentshared_{nm}", "Mistral-7B-v0.1", kv, "agent", "--sessions 32 --turns 8 --prefix-tokens 24000 --gen 128"))
    JOBS.append((f"rag_{nm}", "Mistral-7B-v0.1", kv, "rag", "--n 300 --ctx-tokens 16000 --conc 16"))
    JOBS.append((f"needle_{nm}", "Mistral-7B-v0.1", kv, "needle", "--n 4 --ctx-tokens 30000 --conc 8"))
    JOBS.append((f"drift_{nm}", "Mistral-7B-v0.1", kv, "drift", "--n 24 --ctx-tokens 12000 --gen 512 --conc 8"))
    for c in (8, 32):
        for k in (0, 2, 4, 8):
            ex = _SP.format(c=c, k=k) if k else f"--n 64 --gen 384 --conc {c} --k 0"
            JOBS.append((f"spec_c{c}_k{k}_{nm}", "Mistral-7B-v0.1", kv, "spec", ex))
    JOBS.append((f"agentdistinct_q3_{nm}", "Qwen3-8B", kv, "agent", _AG))
    JOBS.append((f"rag_q3_{nm}", "Qwen3-8B", kv, "rag", "--n 300 --ctx-tokens 16000 --conc 16"))


@app.function(image=image, gpu="H100", timeout=14400, volumes={"/models": vol})
def run_all():
    from huggingface_hub import snapshot_download
    for m, repo in (("Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1"), ("Qwen3-8B", "Qwen/Qwen3-8B")):
        if not os.path.exists(f"/models/{m}/config.json"):
            snapshot_download(repo, local_dir=f"/models/{m}", allow_patterns=["*.json", "model-*.safetensors", "tokenizer*"])
    vol.commit()
    env = dict(os.environ, RW_MODELS="/models", RW_VLLM_BIN=subprocess.check_output(["which", "vllm"], text=True).strip(), RW_LOGS="/tmp/logs")
    out = {}
    for tag, model, kv, wl, extra in JOBS:
        cmd = f"python /rw/run_cfg.py --model {model} --kv {kv} --workload {wl} --tag cuda_h100_{tag} --outdir /tmp/out {extra}"
        r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, cwd="/rw")
        p = f"/tmp/out/realworld_cuda_h100_{tag}.json"
        out[tag] = json.load(open(p)) if os.path.exists(p) else {"error": (r.stdout + r.stderr)[-2000:]}
    return out

@app.local_entrypoint()
def main():
    res = run_all.remote()
    d = os.path.join(HERE, "..", "..", "results"); os.makedirs(d, exist_ok=True)
    for tag, v in res.items():
        json.dump(v, open(os.path.join(d, f"realworld_cuda_h100_{tag}.json"), "w"), indent=1)
    print({k: (v.get("result", {}).get("out_tok_per_s"), v.get("error")) for k, v in res.items()})
