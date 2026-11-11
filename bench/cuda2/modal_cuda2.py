"""cuda2 Modal runner.  usage: modal run [--detach] bench/cuda2/modal_cuda2.py --gpu H100 --mode sm90 [--extra "..."]
modes: sm90 (build variant 3 + bench), check (test_vectors incl. variant 3), ptxas (register/spill report), sh (extra = shell command)"""
import modal, subprocess, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("uq-cuda2")
img = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
       .apt_install("git", "curl", "ninja-build")
       .pip_install("torch", index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("numpy", "ninja", "pybind11")
       .add_local_dir(os.path.join(ROOT, "cuda"), "/work/cuda", ignore=["__pycache__", "*.pyc"])
       .add_local_dir(os.path.join(ROOT, "bench", "cuda2"), "/work/bench/cuda2", ignore=["__pycache__", "*.pyc"])
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__", "*.pyc"])
       .add_local_dir(os.path.join(ROOT, "tools"), "/work/tools", ignore=["__pycache__", "*.pyc"]))
vol = modal.Volume.from_name("uq-cuda2-results", create_if_missing=True)

def sh(c, timeout=3300):
    p = subprocess.Popen(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in p.stdout:
        print(line, end="", flush=True); out.append(line)
    p.wait()
    return ""

def body(gpu, mode, extra, defs=""):
    d = f"/out/{gpu}_{mode}"; os.makedirs(d, exist_ok=True); log = open(f"{d}/log.txt", "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
    P(sh("nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader"))
    arch = "-gencode arch=compute_90a,code=sm_90a" if "H100" in gpu else "-arch=native"
    if "H100" not in gpu: defs = defs.replace("-DUQ_ENABLE_SM90", "")
    F = f"-O3 -std=c++17 {arch} " + ("-DUQ_ENABLE_SM90 " if "H100" in gpu else "") + f"-I /work/cuda {defs}"
    fast = mode.endswith("f") and mode not in ("sh",)
    if fast: F += " -DUQ_SM90_ONLY"; mode = mode[:-1]
    if mode == "sm90":
        P(sh(f"cd /work && nvcc {F} -o bench_sm90 bench/cuda2/bench_sm90.cu cuda/uq_decode.cu 2>&1 | grep -v Remark | tail -40"))
        P(sh(f"cd /work && timeout -s KILL 240 ./bench_sm90 {extra}" + (" --only3" if fast else "") + f""))
    elif mode == "check":
        P(sh(f"cd /work/cuda && nvcc {F} -o tv test_vectors.cu uq_decode.cu 2>&1 | tail -30"))
        P(sh("cd /work/cuda && ./tv vectors 1e-2 2>&1 | tail -40"))
    elif mode == "checknv":   # NVFP4 vectors (variant 1 everywhere, variant 3 on H100)
        P(sh(f"cd /work/cuda && nvcc {F} -o tvnv ../bench/cuda2/test_vectors_nv.cu uq_decode.cu 2>&1 | grep -E 'error|fatal' | head -20"))
        P(sh("cd /work/cuda && ./tvnv vectors_nvfp4 1e-2 2>&1 | grep -E 'first fail|RESULT|FAIL' | head -12"))
    elif mode == "bench":   # torch bench vs Triton snapshot (JIT-builds uq_cuda_ext); extra = args of cuda/bench_cuda.py
        P(sh(f"cd /work/cuda && python bench_cuda.py {extra} --out /out/{gpu}_bench/result.json 2>&1 | grep -vE 'Remark|warning|instantiation|detected during|^\\s*$' | tail -60"))
    elif mode == "nvbench":
        P(sh(f"cd /work && python bench/cuda2/bench_nvfp4_cuda.py --out /out/{gpu}_nvbench/result.json {extra} 2>&1 | grep -vE 'Remark|warning|instantiation|detected during|^\\s*$' | tail -30"))
    elif mode == "ptxas":
        P(sh(f"cd /work/cuda && nvcc {F} -Xptxas -v -c -o /tmp/x.o uq_decode.cu 2>&1 | grep -A2 -E 'uq_sm90' | grep -E 'registers|spill|Compiling' | head -40"))
    elif mode == "sh":
        P(sh(extra))
    log.close(); vol.commit()

def _fn(gpu): return dict(gpu=gpu, image=img, timeout=3600, volumes={"/out": vol})
@app.function(**_fn("H100!"))
def run_h100(mode: str, extra: str = "", defs: str = ""): body("H100", mode, extra, defs)
@app.function(**_fn("B300"))
def run_b300(mode: str, extra: str = "", defs: str = ""): body("B300", mode, extra, defs)
@app.function(**_fn("A100-80GB"))
def run_a100(mode: str, extra: str = "", defs: str = ""): body("A100", mode, extra, defs)
@app.function(**_fn("L40S"))
def run_l40s(mode: str, extra: str = "", defs: str = ""): body("L40S", mode, extra, defs)
@app.function(**_fn("RTX-PRO-6000"))
def run_rtx(mode: str, extra: str = "", defs: str = ""): body("RTXPRO6000", mode, extra, defs)
FUNCS = {"H100": run_h100, "B300": run_b300, "A100": run_a100, "L40S": run_l40s, "RTXPRO6000": run_rtx}

@app.local_entrypoint()
def main(gpu: str = "H100", mode: str = "sm90", extra: str = "", defs: str = ""):
    FUNCS[gpu].remote(mode, extra, defs)
