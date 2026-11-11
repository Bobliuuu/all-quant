"""Upstream-vLLM UltraQuant compat probe on Modal.
  modal run --detach cuda/modal/modal_compat.py --gpu H100
Results: Modal volume `uq-cuda-results` /out/compat_<gpu>/ (log.txt, result.json)."""
import modal, subprocess, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("uq-compat")
img = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
       .apt_install("git", "curl", "ninja-build")
       .pip_install("vllm", extra_options="--extra-index-url https://download.pytorch.org/whl/cu129")
       .pip_install("numpy", "ninja", "pybind11")
       .add_local_dir(os.path.join(ROOT, "cuda"), "/work/cuda", ignore=["__pycache__", "*.pyc"]))
vol = modal.Volume.from_name("uq-cuda-results", create_if_missing=True)


def sh(c, timeout=3000):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def body(gpu):
    d = f"/out/compat_{gpu}"; os.makedirs(d, exist_ok=True)
    log = open(f"{d}/log.txt", "w")
    def P(s):
        print(s, flush=True); log.write(str(s) + "\n"); log.flush()
    P(sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader"))
    P(sh("python -c 'import torch,triton,vllm;print(torch.__version__,triton.__version__,vllm.__version__)' 2>&1 | tail -3"))
    have = sh("python -c 'import vllm.v1.attention.ops.ultraquant.triton_store' 2>&1 | tail -2")
    P("pip vllm has ultraquant: " + ("NO " + have if "Error" in have else "yes"))
    if "Error" in have:  # overlay upstream main ops dir
        P(sh("cd /tmp && git clone --depth 1 https://github.com/vllm-project/vllm.git vllm_main 2>&1 | tail -1; "
             "git -C /tmp/vllm_main rev-parse HEAD"))
        P(sh("SP=$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))'); "
             "cp -r /tmp/vllm_main/vllm/v1/attention/ops/ultraquant $SP/v1/attention/ops/; "
             "cp -rn /tmp/vllm_main/vllm/v1/attention/ops/turboquant_soa $SP/v1/attention/ops/ 2>&1; "
             "cp /tmp/vllm_main/vllm/v1/attention/ops/triton_unified_attention.py $SP/v1/attention/ops/; echo overlaid"))
    P(sh(f"cd /work && python cuda/compat/compat_test.py {d}/result.json 2>&1 | tail -400", timeout=3400))
    log.close(); vol.commit()


def _fn(gpu): return dict(gpu=gpu, image=img, timeout=3600, volumes={"/out": vol})
@app.function(**_fn("H100"))
def run_h100(): body("H100")
@app.function(**_fn("B300"))
def run_b300(): body("B300")
@app.function(**_fn("A100-80GB"))
def run_a100(): body("A100-80GB")
FUNCS = {"H100": run_h100, "B300": run_b300, "A100-80GB": run_a100}

@app.local_entrypoint()
def main(gpu: str = "H100"): FUNCS[gpu].remote()
