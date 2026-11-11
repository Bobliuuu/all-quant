"""Run the CUDA port on Modal GPUs.  usage:
  modal run --detach cuda/modal/modal_cuda.py --gpu H100 --mode check
Results land in the Modal volume `uq-cuda-results` under /out/<gpu>_<mode>/ and are printed.
"""
import modal, subprocess, os, json, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("uq-cuda")
img = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
       .apt_install("git", "curl", "ninja-build")
       .pip_install("torch", "triton", "numpy", "ninja", extra_index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("flashinfer-python", "pybind11")
       .add_local_dir(os.path.join(ROOT, "cuda"), "/work/cuda", ignore=["__pycache__", "*.pyc"]))
vol = modal.Volume.from_name("uq-cuda-results", create_if_missing=True)


def sh(c, env=None, timeout=3000):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(c, shell=True, capture_output=True, text=True, env=e, timeout=timeout)
    return r.stdout + r.stderr


def body(gpu, mode, extra):
    d = f"/out/{gpu}_{mode}"; os.makedirs(d, exist_ok=True)
    log = open(f"{d}/log.txt", "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
    P(sh("nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader"))
    P(sh("nvcc --version | tail -2"))
    P(sh("python -c 'import torch,triton;print(torch.__version__,triton.__version__,torch.cuda.get_device_capability())'"))
    if mode == "check":
        P(sh("cd /work/cuda && nvcc -O3 -std=c++17 -arch=native -lineinfo -o test_vectors test_vectors.cu uq_decode.cu 2>&1 | tail -40"))
        P(sh("cd /work/cuda && ./test_vectors vectors 1e-3 2>&1 | tail -60"))
    else:
        P(sh(f"cd /work/cuda && python {mode}.py {extra} --out {d}/result.json 2>&1 | tail -200", timeout=3400))
    log.close(); vol.commit()


def _fn(gpu):
    return dict(gpu=gpu, image=img, timeout=3600, volumes={"/out": vol})

@app.function(**_fn("B300"))
def run_b300(mode: str = "check", extra: str = ""): body("B300", mode, extra)
@app.function(**_fn("H100"))
def run_h100(mode: str = "check", extra: str = ""): body("H100", mode, extra)
@app.function(**_fn("A100-80GB"))
def run_a100(mode: str = "check", extra: str = ""): body("A100-80GB", mode, extra)
@app.function(**_fn("L40S"))
def run_l40s(mode: str = "check", extra: str = ""): body("L40S", mode, extra)
@app.function(**_fn("RTX-PRO-6000"))
def run_rtx6000(mode: str = "check", extra: str = ""): body("RTX-PRO-6000", mode, extra)

FUNCS = {"B300": run_b300, "H100": run_h100, "A100-80GB": run_a100, "L40S": run_l40s, "RTX-PRO-6000": run_rtx6000}


@app.local_entrypoint()
def main(gpu: str = "H100", mode: str = "check", extra: str = ""):
    FUNCS[gpu].remote(mode, extra)
