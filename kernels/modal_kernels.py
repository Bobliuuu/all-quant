"""Build + check + bench the CUDA kernel (and the SYCL port via DPC++'s CUDA backend) on Modal.
  modal run kernels/modal_kernels.py --gpu H100 --what cuda
  modal run kernels/modal_kernels.py --gpu H100 --what sycl
Writes logs/JSONL to the `pq-results` volume under kern_<gpu>_<what>/ and prints them.
"""
import os
import subprocess

import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPCPP_TAG = "nightly-2026-11-04"
app = modal.App("pq-kernels")
cuda_img = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .add_local_dir(os.path.join(ROOT, "kernels"), "/work/kernels", ignore=["**/*.o", "**/pq_bench"])
)
sycl_img = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("wget", "ca-certificates", "libtbb12", "libhwloc15", "zlib1g-dev", "libnuma1", "gnupg")
    .run_commands(
        "wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main' > /etc/apt/sources.list.d/oneapi.list",
        "apt-get update && apt-get install -y intel-oneapi-runtime-opencl",
        f"mkdir -p /opt/dpcpp && cd /opt/dpcpp && wget -q https://github.com/intel/llvm/releases/download/{DPCPP_TAG}/sycl_linux.tar.gz && tar xzf sycl_linux.tar.gz && rm sycl_linux.tar.gz",
    )
    .add_local_dir(os.path.join(ROOT, "kernels"), "/work/kernels", ignore=["**/*.o", "**/pq_bench"])
)
vol = modal.Volume.from_name("pq-results", create_if_missing=True)

SHAPES = [  # B T Hq Hkv bK bV
    (1, 32768, 32, 8, 25, 24), (8, 32768, 32, 8, 25, 24), (8, 32768, 32, 8, 18, 17), (8, 32768, 32, 8, 16, 16),
    (32, 8192, 32, 8, 25, 24), (8, 32768, 32, 1, 25, 24), (8, 131072, 16, 1, 25, 24),
]
SYCL_ENV = ("export PATH=/opt/dpcpp/bin:$PATH; export LD_LIBRARY_PATH=/opt/dpcpp/lib:/opt/intel/oneapi/lib:"
            "/opt/intel/oneapi/redist/lib:/opt/intel/oneapi/lib/intel64:$LD_LIBRARY_PATH; ")


def sh(cmd, log, env=""):
    p = subprocess.run(["bash", "-lc", env + cmd], capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    log.write(f"$ {cmd}\n{out}\n")
    log.flush()
    print(f"$ {cmd}\n{out}", flush=True)
    return p.returncode, out


def body(gpu, what, shapes):
    d = f"/out/kern_{gpu}_{what}"
    os.makedirs(d, exist_ok=True)
    log = open(f"{d}/log.txt", "w")
    sh("nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,memory.total --format=csv", log)
    V = "/work/kernels/vectors"
    if what == "cuda":
        cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", log)[1].strip().replace(".", "")
        arch = {"90": "sm_90a", "100": "sm_100a", "103": "sm_103a"}.get(cc, "sm_" + cc)
        rc, _ = sh(f"cd /work/kernels/cuda && nvcc -O3 -std=c++17 -arch={arch} -lineinfo pq_bench.cu -o pq_bench", log)
        if rc:
            return
        sh(f"cd /work/kernels/cuda && ./pq_bench check {V} > {d}/check.jsonl; tail -1 {d}/check.jsonl", log)
        exe = "/work/kernels/cuda/pq_bench bench"
        env = ""
    else:
        env = SYCL_ENV
        sh("sycl-ls", log, env)
        rc, _ = sh("cd /work/kernels/sycl && clang++ -fsycl -O3 -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 "
                   "pq_sycl.cpp -o pq_sycl", log, env)
        if rc:
            return
        sh(f"cd /work/kernels/sycl && ONEAPI_DEVICE_SELECTOR=cuda:gpu ./pq_sycl check {V} > {d}/check_gpu.jsonl; tail -1 {d}/check_gpu.jsonl", log, env)
        sh(f"cd /work/kernels/sycl && ONEAPI_DEVICE_SELECTOR=opencl:cpu ./pq_sycl check {V} > {d}/check_cpu.jsonl; tail -1 {d}/check_cpu.jsonl", log, env)
        exe = "ONEAPI_DEVICE_SELECTOR=cuda:gpu /work/kernels/sycl/pq_sycl bench"
    for s in shapes:
        sh(f"{exe} {V}/tables {' '.join(map(str, s))} 0 30 >> {d}/bench.jsonl; tail -3 {d}/bench.jsonl", log, env)
    if what == "cuda":
        sh(f"cd /work/kernels/cuda && python3 torch_baseline.py >> {d}/torch_sdpa.jsonl; cat {d}/torch_sdpa.jsonl", log)
    log.close()
    vol.commit()


@app.function(gpu="H100", image=cuda_img, timeout=30 * 60, volumes={"/out": vol})
def cuda_h100():
    body("H100", "cuda", SHAPES)


@app.function(gpu="B200", image=cuda_img, timeout=20 * 60, volumes={"/out": vol})
def cuda_b200():
    body("B200", "cuda", SHAPES)


@app.function(gpu="H100", image=sycl_img, timeout=30 * 60, volumes={"/out": vol})
def sycl_h100():
    body("H100", "sycl", SHAPES)


@app.local_entrypoint()
def main(gpu: str = "H100", what: str = "cuda"):
    {("H100", "cuda"): cuda_h100, ("B200", "cuda"): cuda_b200, ("H100", "sycl"): sycl_h100}[(gpu, what)].remote()
