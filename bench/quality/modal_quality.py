"""Modal launcher: one container per (model, method-group).  modal run --detach bench/quality/modal_quality.py --models mistral7b --groups all"""
import modal, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
app = modal.App("uq-quality")
img = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11").apt_install("git")
       .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate", "datasets<4", "scipy", "numpy==2.1.3", "sentencepiece", "protobuf", "huggingface_hub")
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"])
       .add_local_dir(os.path.join(HERE), "/work/q", ignore=["__pycache__"])
       .add_local_dir(os.path.expanduser("~/amd-experiments-ro/kernels"), "/work/amdk", ignore=["__pycache__", "hip", "ref"]))
vol = modal.Volume.from_name("uq-quality-results", create_if_missing=True)
hfv = modal.Volume.from_name("uq-quality-hf", create_if_missing=True)

@app.function(gpu="H100", image=img, timeout=6 * 3600, volumes={"/out": vol, "/hf": hfv},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})] if os.environ.get("HF_TOKEN") else [])
def job(model, methods, metrics, needle_lens="8192,16384,32768", nwin=8, tag="run"):
    import subprocess, sys
    env = dict(os.environ, HF_HOME="/hf", PYTHONPATH="/work", AMD_KERNELS="/work/amdk")
    out = f"/out/{tag}"
    cmd = [sys.executable, "/work/q/qeval.py", "--model", model, "--methods", methods, "--metrics", metrics, "--needle_lens", needle_lens, "--out", out, "--nwin", str(nwin)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd="/work")
    print(p.stdout[-6000:], p.stderr[-3000:]); vol.commit(); hfv.commit()
    return {f: json.load(open(f"{out}/{f}")) for f in os.listdir(out) if f.endswith(".json")}

@app.local_entrypoint()
def main(model="mistral7b", methods="fp16", metrics="ppl,attn,verify,needle", needle_lens="8192,16384,32768", nwin=8, tag="run"):
    res = job.remote(model, methods, metrics, needle_lens, nwin, tag)
    os.makedirs(os.path.join(ROOT, "results", "quality_raw"), exist_ok=True)
    for f, d in res.items(): json.dump(d, open(os.path.join(ROOT, "results", "quality_raw", f), "w"), indent=1)
