"""Prior-art benchmarks (BitDecoding-FP4, FlashInfer, FA3) at the UltraQuant shapes.
  modal run --detach cuda/modal/modal_priorart.py --gpu B300
  modal run --detach cuda/modal/modal_priorart.py --gpu H100
Results go to Modal volume `uq-priorart-results` (/out/priorart_<gpu>.json|log) and are printed;
fetch:  modal volume get uq-priorart-results priorart_B300.json results/
B300 uses a CUDA 13.0 / torch cu130 image (BitDecoding-FP4 needs CUDA>=13, driver 580+, SM100/103 only).
H100 uses CUDA 12.9 / torch cu129 (no BitDecoding: it is tcgen05/SM100-only)."""
import modal, subprocess, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("uq-priorart")
BD_COMMIT = "4e2907d74d8b1cdb3bce2b4795fc86bf76385e7f"

img13 = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("git", "curl", "ninja-build")
         .pip_install("torch==2.11.0", extra_index_url="https://download.pytorch.org/whl/cu130")
         .pip_install("flashinfer-python", "numpy", "cuda-python", "typing_extensions", "torch-c-dlpack-ext")
         .pip_install("nvidia-cutlass-dsl==4.6.0", "quack-kernels==0.6.1")
         .pip_install("nvidia-cutlass-dsl-libs-cu13==4.6.0", "nvidia-cutlass-dsl-libs-base==4.6.0")  # flashinfer pulls libs 4.8.1 -> MLIR mismatch
         .pip_install("apache-tvm-ffi==0.1.10")  # BitDecoding pins this; flashinfer wants >=0.1.11 (conflict noted in PRIOR_ART.md)
         .run_commands("git clone https://github.com/OpenBitSys/BitDecoding-FP4.git /work/BitDecoding-FP4 && "
                       f"cd /work/BitDecoding-FP4 && git checkout {BD_COMMIT}")
         .add_local_file(os.path.join(ROOT, "cuda", "bench_priorart.py"), "/work/bench_priorart.py"))

# B300 FlashInfer image: unpinned cutlass-dsl (flashinfer 0.7.x needs >=4.8; BitDecoding pins 4.6.0 -> cannot share an env)
img13fi = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
           .apt_install("git", "curl", "ninja-build")
           .pip_install("torch==2.11.0", extra_index_url="https://download.pytorch.org/whl/cu130")
           .pip_install("flashinfer-python", "numpy")
           .add_local_file(os.path.join(ROOT, "cuda", "bench_priorart.py"), "/work/bench_priorart.py"))

img12 = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("git", "curl", "ninja-build")
         .pip_install("torch", "numpy", "ninja", extra_index_url="https://download.pytorch.org/whl/cu129")
         .pip_install("flashinfer-python")
         .add_local_file(os.path.join(ROOT, "cuda", "bench_priorart.py"), "/work/bench_priorart.py"))

vol = modal.Volume.from_name("uq-priorart-results", create_if_missing=True)


def sh(c, timeout=3000, stream=None):
    if stream is None:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    import time
    p = subprocess.Popen(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time(); buf = []
    import threading
    def rd():
        for line in p.stdout:
            buf.append(line); stream(line.rstrip())
    th = threading.Thread(target=rd, daemon=True); th.start()
    while p.poll() is None and time.time() - t0 < timeout: time.sleep(2)
    if p.poll() is None: p.kill(); stream("TIMEOUT: killed after %ds" % timeout)
    th.join(5); return "".join(buf)


def body(gpu, sections, tag=""):
    os.makedirs("/out", exist_ok=True)
    log = open(f"/out/priorart_{gpu}{tag}.log", "w")
    def P(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    P(sh("nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader"))
    P(sh("nvcc --version | tail -2"))
    P(sh("pip list 2>/dev/null | grep -iE '^(torch|flashinfer|nvidia-cutlass|quack|apache-tvm|flash)' "))
    fa3_note = ""
    if any(x[0] == "fa3" for x in sections):
        for cand in ("flash-attn-3", "flash_attn_3", "fa3-fwd"):
            o = sh(f"pip install {cand} 2>&1 | tail -3"); fa3_note += f"pip install {cand}: {o}\n"
            if "Successfully installed" in o: break
        P("FA3 install attempts:\n" + fa3_note)
    peak = ""
    for sec, tmo in sections:   # one process per section, each with its own wall-clock cap
        out = f"/out/priorart_{gpu}{tag}_{sec}.json"
        P(f"##### section {sec} (cap {tmo}s)")
        sh(f"cd /work && MAX_JOBS=6 python -u bench_priorart.py --sections {sec} {peak} --out {out} 2>&1", timeout=tmo, stream=P)
        vol.commit()
        if sec == "peak":
            import json
            try: peak = "--peak %s" % json.load(open(out))["hbm_peak_GBps"]
            except Exception as ex: P("peak parse failed: %s" % ex)
    log.close(); vol.commit()


@app.function(gpu="B300", image=img13, timeout=3600, cpu=16, memory=98304, volumes={"/out": vol})
def run_b300(only: str = ""):
    secs = [("peak", 300), ("bitdec", 1500), ("flashinfer", 1500), ("fa3", 120)]
    if only: secs = [x for x in secs if x[0] in only.split(",")]
    body("B300", secs, tag="_" + only.replace(",", "+") if only else "")

@app.function(gpu="B300", image=img13fi, timeout=3600, cpu=16, memory=98304, volumes={"/out": vol})
def run_b300_fi(): body("B300", [("peak", 300), ("fi_mq", 3000), ("fa3", 60)], tag="_fimq")

@app.function(gpu="B300", image=img13fi, timeout=3600, cpu=16, memory=98304, volumes={"/out": vol})
def run_b300_fi128(): body("B300", [("peak", 300), ("fi_mq128", 3000)], tag="_fimq128")

@app.function(gpu="H100", image=img12, timeout=3600, cpu=16, memory=98304, volumes={"/out": vol})
def run_h100(): body("H100", [("peak", 300), ("flashinfer", 1800), ("fa3", 120)])


# "H100" may be silently upgraded to an H200 by Modal (observed); "H100!" pins a true H100 80GB.
@app.function(gpu="H100!", image=img12, timeout=3600, cpu=16, memory=98304, volumes={"/out": vol})
def run_h100_strict(): body("H100strict", [("peak", 300), ("fi_mq", 3000), ("fa3", 60)], tag="_fimq")


@app.local_entrypoint()
def main(gpu: str = "H100", only: str = ""):
    if gpu == "B300FI128": run_b300_fi128.remote()
    elif gpu == "H100STRICT": run_h100_strict.remote()
    elif gpu == "B300FI": run_b300_fi.remote()
    elif gpu == "B300": run_b300.remote(only)
    else: run_h100.remote()
