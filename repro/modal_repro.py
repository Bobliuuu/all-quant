"""Modal driver for the H100 reproduction (all apps/volumes prefixed hq-).

  modal run repro/modal_repro.py::download            # CPU: fetch + sha-verify model
  modal run repro/modal_repro.py --configs "bf16,kv:2:qjl" --tag t6
  modal run repro/modal_repro.py::throughput --configs "bf16,w,kv,wkv" --tag t10

Model: NousResearch/Meta-Llama-3.1-8B-Instruct, whose safetensors, config.json and
tokenizer files are byte-identical to the gated meta-llama/Llama-3.1-8B-Instruct
(hashes in repro/model_hashes.json, verified in-container after download).
"""
import hashlib, json, os, subprocess, sys
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = "/models/llama31-8b-instruct"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ninja-build", "build-essential")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.57.1", "accelerate", "pandas", "pyarrow", "ninja",
                 "huggingface_hub", "numpy", "safetensors", "sentencepiece")
    .env({"TORCH_EXTENSIONS_DIR": "/models/torch_ext", "HF_HOME": "/models/hf",
          "HQ_UPSTREAM": "/work/upstream", "TORCH_CUDA_ARCH_LIST": "9.0a"})
    .add_local_dir(os.path.join(ROOT, "upstream"), "/work/upstream")
    .add_local_dir(HERE, "/work/repro")
)
app = modal.App("hq-repro", image=image)
models = modal.Volume.from_name("hq-models", create_if_missing=True)
results = modal.Volume.from_name("hq-results", create_if_missing=True)
VOLS = {"/models": models, "/results": results}


def _git_blob(p):
    data = open(p, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


@app.function(volumes=VOLS, timeout=3600, cpu=4)
def download():
    from huggingface_hub import snapshot_download
    exp = json.load(open("/work/repro/model_hashes.json"))["files"]
    snapshot_download("NousResearch/Meta-Llama-3.1-8B-Instruct", local_dir=MODEL_DIR,
                      allow_patterns=list(exp.keys()))
    rep = {}
    for fn, e in exp.items():
        p = os.path.join(MODEL_DIR, fn)
        got = _sha256(p) if "sha256" in e else _git_blob(p)
        want = e.get("sha256") or e.get("git_blob")
        rep[fn] = got == want
    json.dump(rep, open(os.path.join(MODEL_DIR, "VERIFY.json"), "w"), indent=1)
    models.commit()
    print(json.dumps(rep, indent=1))
    bad = [k for k, v in rep.items() if not v and k != "tokenizer_config.json"]
    assert not bad, f"hash mismatch {bad}"
    return rep


def _run(cmd, log):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log, "a") as f:
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
    return p.wait()


@app.function(gpu="H100!", volumes=VOLS, timeout=6 * 3600)
def ppl(configs: str, tag: str, nwin: int = 141, env: str = ""):
    os.makedirs("/results/ppl", exist_ok=True)
    os.makedirs("/root/.cache/hyperquant", exist_ok=True)
    # persist the SNR<->bps calibration cache across runs (seeded once, shared)
    cal = "/models/hq_cache/lattice_bps_calibration.json"
    os.makedirs(os.path.dirname(cal), exist_ok=True)
    if os.path.exists(cal):
        subprocess.run(f"cp {cal} /root/.cache/hyperquant/", shell=True)
    out = f"/results/ppl/{tag}.jsonl"
    log = f"/results/ppl/{tag}.log"
    _run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", log)
    for e in env.split(";"):  # several env settings -> sequential runs in one container
        rc = _run(f"cd /work/upstream && {e} python /work/repro/ppl_eval.py --model {MODEL_DIR} "
                  f"--configs '{configs}' --out {out} --nwin {nwin} --tok-cache /models/wt2_tokens.pt", log)
    if os.path.exists("/root/.cache/hyperquant/lattice_bps_calibration.json") and not os.path.exists(cal):
        subprocess.run(f"cp /root/.cache/hyperquant/lattice_bps_calibration.json {cal}", shell=True)
    models.commit(); results.commit()
    return open(out).read() if os.path.exists(out) else f"rc={rc}"


@app.function(gpu="H100!", volumes=VOLS, timeout=3 * 3600)
def throughput(configs: str, tag: str, extra: str = ""):
    os.makedirs("/results/tput", exist_ok=True)
    cal = "/models/hq_cache/lattice_bps_calibration.json"
    os.makedirs("/root/.cache/hyperquant", exist_ok=True)
    if os.path.exists(cal):
        subprocess.run(f"cp {cal} /root/.cache/hyperquant/", shell=True)
    out = f"/results/tput/{tag}.jsonl"
    log = f"/results/tput/{tag}.log"
    _run(f"cd /work/upstream && python /work/repro/throughput.py --model {MODEL_DIR} "
         f"--configs '{configs}' --out {out} {extra}", log)
    models.commit(); results.commit()
    return open(out).read() if os.path.exists(out) else "no output"


@app.local_entrypoint()
def main(configs: str = "bf16", tag: str = "test", nwin: int = 141, mode: str = "ppl", extra: str = "", env: str = ""):
    os.makedirs(os.path.join(ROOT, "results", "repro"), exist_ok=True)
    if mode == "ppl":
        res = ppl.remote(configs, tag, nwin, env)
    else:
        res = throughput.remote(configs, tag, extra)
    sub = "ppl" if mode == "ppl" else "tput"
    with open(os.path.join(ROOT, "results", "repro", f"{sub}_{tag}.jsonl"), "w") as f:
        f.write(res)
    print(res)
