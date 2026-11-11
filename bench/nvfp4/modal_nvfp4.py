"""Modal launcher (NVIDIA): tests + kernel bench on A100 and H100.
  modal run --detach bench/nvfp4/modal_nvfp4.py::main
Results are written to the Modal volume `uq-nvfp4-results` and returned to the local entrypoint."""
import modal, os, json, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
app = modal.App("uq-nvfp4")
img = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
       .pip_install("torch==2.5.1", "numpy==2.1.3", "pytest")
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"])
       .add_local_dir(os.path.join(ROOT, "tests"), "/work/tests", ignore=["__pycache__"])
       .add_local_dir(HERE, "/work/bench/nvfp4", ignore=["__pycache__"]))
vol = modal.Volume.from_name("uq-nvfp4-results", create_if_missing=True)

def _sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, cwd="/work"); return r.stdout + r.stderr

def _job(tag):
    out = {"gpu": _sh("nvidia-smi --query-gpu=name --format=csv,noheader").strip()}
    out["pytest"] = _sh("python -m pytest tests/test_nvfp4.py -q 2>&1 | tail -15")
    _sh(f"python bench/nvfp4/bench_nvfp4.py --nsplits 0 8 16 32 --out /out/nvfp4_bench_{tag}.json > /out/nvfp4_bench_{tag}.log 2>&1")
    out["log"] = open(f"/out/nvfp4_bench_{tag}.log").read()
    out["json"] = json.load(open(f"/out/nvfp4_bench_{tag}.json"))
    vol.commit(); return out

@app.function(gpu="A100-80GB", image=img, timeout=3000, volumes={"/out": vol})
def a100(): return _job("a100")

@app.function(gpu="H100", image=img, timeout=3000, volumes={"/out": vol})
def h100(): return _job("h100")

@app.local_entrypoint()
def main():
    hs = [a100.spawn(), h100.spawn()]
    for tag, h in zip(("a100", "h100"), hs):
        o = h.get()
        json.dump(o["json"], open(os.path.join(ROOT, "results", f"nvfp4_bench_{tag}.json"), "w"), indent=1)
        open(os.path.join(ROOT, "results", f"nvfp4_bench_{tag}.log"), "w").write(f"{o['gpu']}\n{o['pytest']}\n{o['log']}")
        print(tag, o["pytest"], o["log"][-1500:])


# ------------------------------------------------------------- quality (H100)
qimg = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11").apt_install("git")
        .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate", "datasets<4", "scipy", "numpy==2.1.3", "sentencepiece", "protobuf", "huggingface_hub")
        .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"])
        .add_local_dir(os.path.join(ROOT, "bench", "quality"), "/work/bench/quality", ignore=["__pycache__"])
        .add_local_dir(HERE, "/work/bench/nvfp4", ignore=["__pycache__"])
        .add_local_dir(os.path.expanduser("~/amd-experiments-ro/kernels"), "/work/amdk", ignore=["__pycache__", "hip", "ref"]))
hfv = modal.Volume.from_name("uq-quality-hf", create_if_missing=True)

@app.function(gpu="H100", image=qimg, timeout=4 * 3600, volumes={"/out": vol, "/hf": hfv},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})] if os.environ.get("HF_TOKEN") else [])
def quality(model):
    env = dict(os.environ, HF_HOME="/hf", AMD_KERNELS="/work/amdk")
    p = subprocess.run([sys.executable, "bench/nvfp4/quality_nvfp4.py", "--model", model, "--out", f"/out/nvfp4_quality_{model}.json"],
                       env=env, capture_output=True, text=True, cwd="/work")
    vol.commit(); hfv.commit()
    try: j = json.load(open(f"/out/nvfp4_quality_{model}.json"))
    except Exception: j = None
    return dict(log=p.stdout[-6000:] + p.stderr[-4000:], json=j)

@app.local_entrypoint()
def qmain():
    hs = {m: quality.spawn(m) for m in ("qwen3_8b", "mistral7b")}
    for m, h in hs.items():
        o = h.get(); print(m, o["log"][-2500:])
        open(os.path.join(ROOT, "results", f"nvfp4_quality_{m}.log"), "w").write(o["log"])
        if o["json"]: json.dump(o["json"], open(os.path.join(ROOT, "results", f"nvfp4_quality_{m}.json"), "w"), indent=1)
