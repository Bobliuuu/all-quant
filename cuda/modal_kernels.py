"""Build + test + bench the kernels on a Modal H100 (CUDA reference = upstream kernels).
  modal run cuda/modal_kernels.py --mode cuda            # upstream CUDA kernels
  modal run cuda/modal_kernels.py --mode sycl            # SYCL port via DPC++ CUDA backend
Results are written to results/kernels/ by the local entrypoint.
"""
import os, subprocess
import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPCPP_TAG = os.environ.get("HQ_DPCPP_TAG", "nightly-2026-11-04")

base = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install("build-essential", "wget", "ca-certificates"))
cuda_img = base
sycl_img = (base.apt_install("libtbb12", "libhwloc15", "zlib1g-dev", "libnuma1", "ocl-icd-libopencl1", "gnupg")
            .run_commands(
                "wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi.gpg",
                "echo 'deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main' > /etc/apt/sources.list.d/oneapi.list",
                "apt-get update && apt-get install -y intel-oneapi-runtime-opencl",
                f"mkdir -p /opt/dpcpp && cd /opt/dpcpp && wget -q https://github.com/intel/llvm/releases/download/{DPCPP_TAG}/sycl_linux.tar.gz && tar xzf sycl_linux.tar.gz && rm sycl_linux.tar.gz"))
IGN = ["build/**", "results/**", ".git/**", "**/__pycache__/**", "repro/**"]
cuda_img = cuda_img.add_local_dir(ROOT, "/work", ignore=IGN)
sycl_img = sycl_img.add_local_dir(ROOT, "/work", ignore=IGN)
app = modal.App("hq-kernels")


def sh(cmd, log):
    log.append(f"$ {cmd}")
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    log.append((p.stdout or "") + (p.stderr or ""))
    print(log[-1][-4000:], flush=True)
    return p.returncode


CUDA_SRCS = ("cuda/backend_cuda.cu upstream/cuda/stage2_cuda_encoder.cu upstream/cuda/stage2_cuda_decoder.cu "
             "upstream/cuda/stage3_cuda_decoder.cu upstream/integrations/llama/fused_decode_gemv.cu "
             "upstream/integrations/llama/rice_fused_attention.cu")
NVCC = "nvcc -O3 -std=c++17 -arch=sm_90a -Iupstream/cuda -Iupstream/integrations/llama -Icommon -Icuda"


@app.function(gpu="H100!", image=cuda_img, timeout=3600)
def run_cuda(bench: bool = True, quick: bool = False):
    log = []
    sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem --format=csv; nvcc --version | tail -2", log)
    build = ("cd /work && mkdir -p build/obj && for f in " + CUDA_SRCS + "; do "
             f"{NVCC} -c $f -o build/obj/$(basename $f).o & done; "
             "for t in test bench; do g++ -O2 -std=c++17 -Icommon -c common/${t}_main.cpp -o build/obj/${t}_main.o & done; wait; "
             "for t in test bench; do nvcc -arch=sm_90a build/obj/*.cu.o build/obj/${t}_main.o -o build/${t}_cuda -lcublasLt -lcublas; done")
    sh(build + " 2>&1 | grep -v -i warning | tail -40", log)
    sh("cd /work && ./build/test_cuda vectors --json build/test_cuda.json", log)
    out = {"log": "\n".join(log)}
    try: out["test_json"] = open("/work/build/test_cuda.json").read()
    except Exception: pass
    if bench:
        sh(f"cd /work && ./build/bench_cuda build/bench_cuda.jsonl {'--quick' if quick else ''}", log)
        try: out["bench_jsonl"] = open("/work/build/bench_cuda.jsonl").read()
        except Exception: pass
    out["log"] = "\n".join(log)
    return out


SYCL_ENV = ("export PATH=/opt/dpcpp/bin:$PATH; export LD_LIBRARY_PATH=/opt/dpcpp/lib:/opt/intel/oneapi/lib:"
            "/opt/intel/oneapi/lib/intel64:$LD_LIBRARY_PATH; ")


@app.function(gpu="H100!", image=sycl_img, timeout=3600)
def run_sycl(bench: bool = True, quick: bool = False):
    log = []
    sh(SYCL_ENV + "clang++ --version | head -2; sycl-ls; nvidia-smi -L", log)
    flags = ("-fsycl -O3 -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 "
             "-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_90 -Icommon -Isycl")
    for t in ("test", "bench"):
        sh(SYCL_ENV + f"cd /work && mkdir -p build && clang++ {flags} sycl/backend_sycl.cpp common/{t}_main.cpp -o build/{t}_sycl 2>&1 | grep -v -i warning | tail -30", log)
    out = {}
    for sel, tag in (("cuda:gpu", "nvidia"), ("opencl:cpu", "cpu")):
        sh(SYCL_ENV + f"cd /work && ONEAPI_DEVICE_SELECTOR={sel} ./build/test_sycl vectors --json build/test_sycl_{tag}.json", log)
        try: out[f"test_json_{tag}"] = open(f"/work/build/test_sycl_{tag}.json").read()
        except Exception: pass
    if bench:
        sh(SYCL_ENV + f"cd /work && ONEAPI_DEVICE_SELECTOR=cuda:gpu ./build/bench_sycl build/bench_sycl.jsonl {'--quick' if quick else ''}", log)
        try: out["bench_jsonl"] = open("/work/build/bench_sycl.jsonl").read()
        except Exception: pass
    out["log"] = "\n".join(log)
    return out


@app.local_entrypoint()
def main(mode: str = "cuda", bench: bool = True, quick: bool = False, tag: str = "h100"):
    rd = os.path.join(ROOT, "results", "kernels"); os.makedirs(rd, exist_ok=True)
    r = (run_cuda if mode == "cuda" else run_sycl).remote(bench, quick)
    pre = f"{mode}_{tag}"
    open(os.path.join(rd, pre + ".log"), "w").write(r.get("log", ""))
    for k, v in r.items():
        if k.startswith("test_json"):
            open(os.path.join(rd, f"{pre}_{k.replace('test_json', 'test')}.json".replace("__", "_")), "w").write(v)
        if k == "bench_jsonl":
            open(os.path.join(rd, pre + "_bench.jsonl"), "w").write(v)
    print(r.get("log", "")[-3000:])
