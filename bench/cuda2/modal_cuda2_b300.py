"""cuda2 Modal runner.  B300 runner for variant 4. usage: modal run bench/cuda2/modal_cuda2_b300.py --mode vec|benchf|bench|sh [--extra "..."]  (suffix f = fast build without legacy kernels)"""
import modal, subprocess, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("uq-cuda2-b300")
img = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
       .apt_install("git", "curl", "ninja-build")
       .pip_install("torch", index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("numpy", "ninja", "pybind11")
       .add_local_dir(os.path.join(ROOT, "cuda"), "/work/cuda", ignore=["__pycache__", "*.pyc"])
       .add_local_dir(os.path.join(ROOT, "bench", "cuda2"), "/work/bench/cuda2", ignore=["__pycache__", "*.pyc"])
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__", "*.pyc"])
       .add_local_dir(os.path.join(ROOT, "tools"), "/work/tools", ignore=["__pycache__", "*.pyc"]))
vol = modal.Volume.from_name("uq-cuda2-b300-results", create_if_missing=True)

def sh(c, timeout=3300):
    p = subprocess.Popen(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in p.stdout:
        print(line, end="", flush=True); out.append(line)
    p.wait()
    return ""

def body(mode, extra, defs=""):
    d = f"/out/B300_{mode}"; os.makedirs(d, exist_ok=True)
    print(sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader") or "", end="")
    G = "-gencode arch=compute_103a,code=sm_103a"
    F = f"-O3 -std=c++17 {G} -I /work/cuda {defs}"
    fast = mode.endswith("f")
    if fast: F += " -DUQ_SM90_ONLY"; mode = mode[:-1]
    if mode == "vec":      # CPU-reference vectors
        sh(f"cd /work && nvcc {F} -Xptxas -v -o bench_sm100 bench/cuda2/bench_sm100.cu 2>&1 | grep -E 'error|registers|spill' | head -20")
        sh(f"cd /work && timeout -s KILL 300 ./bench_sm100 --vectors cuda/vectors {extra}")
    elif mode == "bench":
        sh(f"cd /work && nvcc {F} -o bench_sm100 bench/cuda2/bench_sm100.cu 2>&1 | grep -E 'error' | head -20")
        sh(f"cd /work && timeout -s KILL 900 ./bench_sm100 {extra}")
    elif mode == "sh":
        sh(extra)
    vol.commit()

@app.function(gpu="B300", image=img, timeout=3600, volumes={"/out": vol})
def run_b300(mode: str, extra: str = "", defs: str = ""): body(mode, extra, defs)

@app.local_entrypoint()
def main(mode: str = "vec", extra: str = "", defs: str = ""):
    run_b300.remote(mode, extra, defs)
