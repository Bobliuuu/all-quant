"""cuda2 real-model runs on Modal (H100 / B300): perplexity through Triton vs CUDA kernels, and whole decode-step time vs fp16 SDPA.
  modal run --detach bench/cuda2/modal_cuda2_model.py --gpu H100 --job quality --model qwen3_8b
  modal run --detach bench/cuda2/modal_cuda2_model.py --gpu H100 --job e2e
Results: Modal volume uq-cuda2-results (/out/<gpu>_<job>_*)."""
import modal, subprocess, os, json, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("uq-cuda2-model")
img = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11").apt_install("git", "ninja-build")
       .pip_install("torch", index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("transformers==4.51.3", "accelerate", "datasets<4", "scipy", "numpy", "ninja", "pybind11", "sentencepiece", "protobuf", "huggingface_hub")
       .add_local_dir(os.path.join(ROOT, "cuda"), "/work/cuda", ignore=["__pycache__", "*.pyc", "vectors", "vectors_nvfp4"])
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"])
       .add_local_dir(os.path.join(ROOT, "bench", "cuda2"), "/work/bench/cuda2", ignore=["__pycache__"])
       .add_local_file(os.path.join(ROOT, "bench", "e2e_decode.py"), "/work/bench/e2e_decode.py"))
img2 = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11").apt_install("git", "ninja-build")
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cu129")
        .pip_install("transformers>=4.57", "accelerate", "scipy", "numpy", "ninja", "pybind11", "sentencepiece", "protobuf", "huggingface_hub")
        .add_local_dir(os.path.join(ROOT, "cuda"), "/work/cuda", ignore=["__pycache__", "*.pyc", "vectors", "vectors_nvfp4"])
        .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"])
        .add_local_dir(os.path.join(ROOT, "bench", "cuda2"), "/work/bench/cuda2", ignore=["__pycache__"])
        .add_local_file(os.path.join(ROOT, "bench", "e2e_decode.py"), "/work/bench/e2e_decode.py"))   # e2e_decode.py needs the newer cache API (cache.layers)
vol = modal.Volume.from_name("uq-cuda2-results", create_if_missing=True)
hfv = modal.Volume.from_name("uq-quality-hf", create_if_missing=True)

def run(cmd, log):
    env = dict(os.environ, HF_HOME="/hf", CC="gcc", CXX="g++")
    p = subprocess.Popen(cmd, shell=True, cwd="/work", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log, "w") as f:
        for line in p.stdout:
            print(line, end="", flush=True); f.write(line); f.flush()
    p.wait(); vol.commit(); hfv.commit()

def body(gpu, job, model, extra):
    os.makedirs("/out", exist_ok=True)
    tag = f"/out/{gpu}_{job}_{model}"
    if job == "quality":
        run(f"python -I bench/cuda2/quality_cuda2.py --model {model} --out {tag}.json {extra}", tag + ".log")
    else:   # whole decode step, B=8 ctx 32768, fp16 SDPA vs our kernel (CUDA port via shim)
        run(f"sed -e 's#/root/uq-venv/lib/python3.12/site-packages#/usr/local/lib/python3.11/site-packages#' bench/e2e_decode.py > bench/e2e_decode_cu2.py && python bench/e2e_decode_cu2.py --mode sweep --uq-dir bench/cuda2/shim --ctxs 32768 --batches 8 --ks 1 8 --out {tag}.json {extra}", tag + ".log")

kw = dict(image=img, timeout=4 * 3600, volumes={"/out": vol, "/hf": hfv})
kw2 = dict(kw, image=img2)
@app.function(gpu="H100!", **kw)
def h100(job: str, model: str, extra: str = ""): body("H100", job, model, extra)
@app.function(gpu="B300", **kw)
def b300(job: str, model: str, extra: str = ""): body("B300", job, model, extra)
@app.function(gpu="H100!", **kw2)
def h100_e(job: str, model: str, extra: str = ""): body("H100", job, model, extra)
@app.function(gpu="B300", **kw2)
def b300_e(job: str, model: str, extra: str = ""): body("B300", job, model, extra)

@app.local_entrypoint()
def main(gpu: str = "H100", job: str = "quality", model: str = "qwen3_8b", extra: str = ""):
    ({"H100": h100, "B300": b300} if job == "quality" else {"H100": h100_e, "B300": b300_e})[gpu].remote(job, model, extra)
