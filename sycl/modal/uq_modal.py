"""Modal driver: build + test + bench the UltraQuant SYCL port on a SYCL CPU device
(Intel OpenCL CPU runtime) and on NVIDIA GPUs (DPC++ CUDA backend).
  modal run --detach sycl/modal/uq_modal.py --mode probe|cpu|gpu --gpu A100
Results are written to results/ in the repo by the local entrypoint and also kept in a Modal Volume.
"""
import os, subprocess, json, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SYCL = os.path.dirname(HERE)
ROOT = os.path.dirname(SYCL)
DPCPP_TAG = os.environ.get("UQ_DPCPP_TAG", "nightly-2026-11-04")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("wget", "curl", "gnupg", "ca-certificates", "cmake", "ninja-build", "libtbb12", "libtbb-dev",
                 "ocl-icd-libopencl1", "clinfo", "libhwloc15", "numactl", "zlib1g-dev", "libnuma1")
    .run_commands(
        # Intel OpenCL CPU runtime (oneAPI apt repo)
        "wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main' > /etc/apt/sources.list.d/oneapi.list",
        "apt-get update && apt-get install -y intel-oneapi-runtime-opencl",
        # DPC++ nightly (intel/llvm) with CUDA plugin
        f"mkdir -p /opt/dpcpp && cd /opt/dpcpp && wget -q https://github.com/intel/llvm/releases/download/{DPCPP_TAG}/sycl_linux.tar.gz && tar xzf sycl_linux.tar.gz && rm sycl_linux.tar.gz",
    )
    .add_local_dir(SYCL, "/work/sycl", ignore=["modal/**", "dev/**", "*.exe", "*.obj"])
)

app = modal.App("uq-sycl-port", image=image)
vol = modal.Volume.from_name("uq-sycl-results", create_if_missing=True)

ENV = ("export PATH=/opt/dpcpp/bin:$PATH; export LD_LIBRARY_PATH=/opt/dpcpp/lib:/opt/intel/oneapi/lib:/opt/intel/oneapi/lib/intel64:$LD_LIBRARY_PATH; "
       "export CUDA_PATH=/usr/local/cuda; ")

def sh(cmd, log, check=False):
    log.append(f"$ {cmd}")
    p = subprocess.run(["bash", "-lc", ENV + cmd], capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    log.append(out)
    if check and p.returncode: raise RuntimeError("\n".join(log[-3:]))
    return p.returncode, out

def common(mode, selector, targets, sg, tag):
    log = []
    sh("clang++ --version | head -2; sycl-ls; nvidia-smi -L; ls /opt/dpcpp/lib | grep -i -E 'pi_|ur_adapter|cuda' ; "
       "cat /etc/OpenCL/vendors/* 2>/dev/null; ls /opt/intel/oneapi/lib 2>/dev/null | head", log)
    if mode == "probe":
        return "\n".join(log), {}
    os.makedirs("/work/b", exist_ok=True)
    src = "/work/sycl"
    flags = f"-fsycl -O3 -std=c++17 -DUQ_NO_XMX -fsycl-targets={targets} -I{src}"
    for t in ("test_vectors", "bench"):
        rc, out = sh(f"cd /work/b && clang++ {flags} {src}/{t}.cpp -o {t}", log)
        if rc: return "\n".join(log) + f"\nBUILD FAILED {t}", {}
    env = f"ONEAPI_DEVICE_SELECTOR={selector} "
    rc, out = sh(f"cd /work/b && {env} ./test_vectors {src}/vectors --sg {sg} --sweep", log)
    tv_rc = rc
    res = {"test_rc": tv_rc}
    if mode == "gpu":
        sh(f"cd /work/b && {env} ./test_vectors {src}/vectors --sg {sg} --wg 256 --tn 32 --sweep | tail -3", log)
    bench_args = "--B 2,8 --ctx 4096,16384 --K 1,4,8 " + f"--sg {sg} --path portable --json /work/b/bench.jsonl"
    if mode == "cpu": bench_args += " --iters 3"
    sh(f"cd /work/b && {env} ./bench {bench_args}", log)
    try: res["bench_jsonl"] = open("/work/b/bench.jsonl").read()
    except Exception: res["bench_jsonl"] = ""
    return "\n".join(log), res

@app.function(timeout=3600, volumes={"/out": vol})
def run_cpu(tag="cpu"):
    # SYCL CPU device via the Intel OpenCL CPU runtime
    return common("cpu", "opencl:cpu", "spir64_x86_64", 16, tag)

def _gpu(tag, sg=32, sel="cuda:gpu"):
    return common("gpu", sel, "nvptx64-nvidia-cuda", sg, tag)

@app.function(timeout=3600, gpu="A100")
def probe_gpu(): return common("probe", "cuda:gpu", "", 32, "probe")
@app.function(timeout=3600, gpu="A100-80GB")
def gpu_a100(): return _gpu("a100")
@app.function(timeout=3600, gpu="H100")
def gpu_h100(): return _gpu("h100")
@app.function(timeout=3600, gpu="L4")
def gpu_l4(): return _gpu("l4")
@app.function(timeout=3600)
def probe_cpu(): return common("probe", "opencl:cpu", "", 16, "probe")

@app.local_entrypoint()
def main(mode: str = "probe", gpu: str = "a100"):
    rd = os.path.join(ROOT, "results")
    fn = {"probe": probe_gpu, "cpu": run_cpu, "a100": gpu_a100, "h100": gpu_h100, "l4": gpu_l4, "probecpu": probe_cpu}[mode if mode != "gpu" else gpu]
    log, res = fn.remote()
    name = {"probe": "probe", "probecpu": "probecpu"}.get(mode, mode if mode == "cpu" else f"nvidia_{gpu}")
    pre = "cpu" if mode in ("cpu", "probecpu") else "nvidia"
    tagname = {"cpu": "sycl_cpu_opencl", "probe": "sycl_probe_gpu", "probecpu": "sycl_probe_cpu"}.get(mode, f"sycl_nvidia_{gpu}")
    open(os.path.join(rd, tagname + ".log"), "w").write(log)
    if res.get("bench_jsonl"):
        open(os.path.join(rd, tagname + ".json"), "w").write(res["bench_jsonl"])
    print(log[-6000:])
