"""Build AdaptiveCpp (SYCL) with the ROCm/HIP backend for gfx942 in a CPU-only Modal container,
compile the HyperQuant SYCL port's test + bench for hip:gfx942, and return a tarball to copy to the
MI300X box (whose CPU is shared with serving benchmarks, so no heavy compile there).

  modal run sycl/modal_acpp_amd.py            # -> build/acpp_amd_bundle.tgz (+ log)

Toolchain: ubuntu 24.04 (box: 24.04, glibc 2.39), LLVM/clang 20 from apt.llvm.org (AdaptiveCpp
25.10 supports LLVM <= 20; ROCm's own LLVM 24 does not build it), ROCm 7.0.2 user space (HIP runtime
soname libamdhip64.so.7 = the box's; ROCm 7.0 device bitcode is LLVM-20 based, readable by clang 20).
"""
import os, subprocess
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
ACPP_TAG = "v25.10.0"
ROCM = "7.0.2"

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("wget", "gnupg", "ca-certificates", "cmake", "ninja-build", "git", "build-essential", "python3",
                 "libnuma-dev", "libelf-dev", "libdrm-dev", "pkg-config", "file")
    .run_commands(
        # LLVM 20
        "wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > /usr/share/keyrings/llvm.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main' > /etc/apt/sources.list.d/llvm.list",
        # ROCm 7.0.2 (headers, HIP runtime, device libs, comgr) -- no kernel driver needed
        "wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /usr/share/keyrings/rocm.gpg",
        f"echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/{ROCM} noble main' > /etc/apt/sources.list.d/rocm.list",
        "printf 'Package: *\\nPin: release o=repo.radeon.com\\nPin-Priority: 600\\n' > /etc/apt/preferences.d/rocm-pin-600",
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y clang-20 llvm-20-dev libclang-20-dev lld-20 libomp-20-dev "
        "hip-dev hip-runtime-amd rocm-device-libs comgr hsa-rocr-dev rocm-core",
        "ls /opt/ && ls /opt/rocm*/lib | grep -E 'amdhip64|hsa-runtime' | head",
    )
    .add_local_dir(PORT, "/work", ignore=["build/**", "results/**", "upstream/**", "repro/**", "**/__pycache__/**", ".git/**"])
)
app = modal.App("hq-acpp-amd", image=image)
vol = modal.Volume.from_name("hq-acpp", create_if_missing=True)


def sh(cmd, log, check=True):
    log.append(f"$ {cmd}")
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    log.append(out[-6000:])
    print(out[-3000:], flush=True)
    if check and p.returncode:
        raise RuntimeError(f"failed ({p.returncode}): {cmd}\n{out[-3000:]}")
    return p.returncode


@app.function(cpu=32, memory=65536, timeout=3600, volumes={"/cache": vol})
def build():
    log = []
    data = b""
    try:
        sh("nproc; clang-20 --version | head -1; cat /opt/rocm/.info/version", log)
        if os.path.exists("/cache/acpp/bin/acpp"):
            sh("cp -a /cache/acpp /opt/acpp", log)
        else:
            sh(f"git clone -q --depth 1 --branch {ACPP_TAG} https://github.com/AdaptiveCpp/AdaptiveCpp.git /src/acpp", log)
            sh("cd /src/acpp && mkdir -p build && cd build && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release "
               "-DCMAKE_INSTALL_PREFIX=/opt/acpp -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 "
               "-DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-20 "
               "-DWITH_ROCM_BACKEND=ON -DROCM_PATH=/opt/rocm -DWITH_CUDA_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF "
               "-DWITH_LEVEL_ZERO_BACKEND=OFF -DDEFAULT_TARGETS='hip:gfx942' 2>&1 | tail -40", log)
            sh("cd /src/acpp/build && ninja -j32 install 2>&1 | tail -15", log)
            sh("rm -rf /cache/acpp && cp -a /opt/acpp /cache/acpp", log)
            vol.commit()
        sh("ls /opt/acpp/lib /opt/acpp/lib/hipSYCL 2>&1 | head -30", log, check=False)
        F = "-O3 -std=c++17 -ffp-contract=fast --acpp-targets=hip:gfx942 -Icommon -Isycl -w"
        for t in ("test", "bench"):
            sh(f"cd /work && mkdir -p build && /opt/acpp/bin/acpp {F} sycl/backend_sycl.cpp common/{t}_main.cpp -o build/{t}_sycl_amd 2>&1 | tail -40", log)
        sh("cd /work/build && file test_sycl_amd && ldd test_sycl_amd | head -20", log, check=False)
        # bundle: acpp runtime (+ hip backend plugin) and the two binaries; the box provides ROCm 7.x libs
        # AdaptiveCpp always loads its OpenMP host backend, which needs LLVM 20's libomp.so.5 (not on the box)
        sh("mkdir -p /out/bundle && cp -a /opt/acpp /out/bundle/acpp && cp -L /usr/lib/llvm-20/lib/libomp.so.5 /out/bundle/acpp/lib/ && cp /work/build/test_sycl_amd /work/build/bench_sycl_amd /out/bundle/ "
           "&& tar czf /out/acpp_amd_bundle.tgz -C /out bundle && ls -la /out", log)
        data = open("/out/acpp_amd_bundle.tgz", "rb").read()
    except Exception as e:  # return the log either way
        log.append(f"ERROR: {e}")
    return "\n".join(log), data


@app.local_entrypoint()
def main():
    log, data = build.remote()
    os.makedirs(os.path.join(PORT, "build"), exist_ok=True)
    open(os.path.join(PORT, "build", "acpp_amd_build.log"), "w").write(log)
    if data:
        open(os.path.join(PORT, "build", "acpp_amd_bundle.tgz"), "wb").write(data)
        print(f"bundle: {len(data) / 1e6:.1f} MB")
    print(log[-4000:])
