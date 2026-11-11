"""Native FP4 block-scaled QK^T experiments on Modal.
  modal run --detach cuda/modal/modal_native.py --gpu RTX-PRO-6000   # sm_120a mma.sync kind::mxf8f6f4
  modal run --detach cuda/modal/modal_native.py --gpu B300           # sm_103a tcgen05 via CUTLASS + bf16 baseline
Results: Modal volume `uq-native-results` /out/<gpu>/ (log.txt + files).  CUTLASS headers are shipped from a local
checkout at ~/cutlass-ref/cutlass (commit recorded in the log).
"""
import modal, subprocess, os, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CUT = os.path.expanduser("~/cutlass-ref/cutlass")
app = modal.App("uq-native-fp4")
img = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
       .apt_install("git")
       .pip_install("torch", index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("numpy")
       .add_local_dir(os.path.join(ROOT, "cuda", "native_fp4"), "/work/native_fp4", ignore=["__pycache__", "*.md"])
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"])
       .add_local_dir(os.path.join(CUT, "include"), "/work/cutlass/include")
       .add_local_dir(os.path.join(CUT, "tools/util/include"), "/work/cutlass/tools/util/include")
       .add_local_dir(os.path.join(CUT, "examples/common"), "/work/cutlass/examples/common"))
vol = modal.Volume.from_name("uq-native-results", create_if_missing=True)


def sh(c, timeout=3000):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def body(gpu, do_cutlass=True):
    d = f"/out/{gpu}" + ("" if do_cutlass else "_qk"); os.makedirs(d, exist_ok=True)
    log = open(f"{d}/log.txt", "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
    def R(c, tail=60, timeout=3000):
        P("$", c); o = sh(c, timeout); P("\n".join(o.strip().split("\n")[-tail:])); return o
    R("nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader")
    R("nvcc --version | tail -2")
    P("cutlass commit b2f249a (local checkout)")
    R("python -c 'import torch;print(torch.__version__,torch.cuda.get_device_capability())'")
    cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader").strip()
    arch = {"12.0": "sm_120a", "10.3": "sm_103a", "10.0": "sm_100a"}[cc]
    GEN = f"-gencode arch=compute_{arch[3:]},code={arch}"   # explicit: plain -arch=sm_XXXa produced non-'a' PTX in 12.9
    W = "/work/native_fp4"
    # ---- data (torch reference) : validation set N=8192, M=64
    R(f"cd /work && UQ_CPU=1 UQ_ROOT=/work python -I {W}/gen_data.py /work/data64 8192 64 128 && UQ_CPU=1 UQ_ROOT=/work python -I {W}/gen_data.py /work/data32 8192 32 128")
    # ---- QK kernels
    nat = "-DUQ_NATIVE" if arch == "sm_120a" else ""
    o = R(f"cd {W} && nvcc -O3 -std=c++17 {GEN} {nat} -Xptxas -v -o /work/qk_mmasync qk_mmasync.cu 2>&1 | grep -E 'error|registers|spill' | head -40")
    if arch != "sm_120a":   # evidence: mma.sync block_scale is NOT assembled on sm_100a/103a
        R(f"cd {W} && nvcc -O3 -std=c++17 {GEN} -DUQ_NATIVE -o /work/qk_mmasync_nat qk_mmasync.cu 2>&1 | head -12")
    for dd in ("data64", "data32"):
        R(f"/work/qk_mmasync /work/{dd} 64 30", tail=80)
    if arch == "sm_120a":   # torch cross-check of the written scores
        R("""cd /work && python -I - <<'PY'
import numpy as np, torch, json
for dd in ('data64','data32'):
    m=json.load(open(f'/work/{dd}/meta.json')); M,N=m['M'],m['N']
    ref=np.fromfile(f'/work/{dd}/S_ref.bin',dtype=np.float32).reshape(M,N)
    for nm in ('native','bf16'):
        s=np.fromfile(f'/work/{dd}/S_{nm}.bin',dtype=np.float32).reshape(M,N)
        err=np.abs(s-ref); print(dd,nm,'maxabs',err.max(),'maxrel/absmax',err.max()/np.abs(ref).max(),'exact_frac',(err==0).mean())
PY""")
    for nm in ("native", "bf16"):
        for dd in ("data64",):
            sh(f"cp /work/{dd}/S_{nm}.bin /dev/null")
    # ---- CUTLASS tcgen05 block-scaled (datacenter only)
    if do_cutlass and arch in ("sm_100a", "sm_103a"):
        CI = "-I/work/cutlass/include -I/work/cutlass/tools/util/include -I/work/cutlass/examples/common"
        variants = [(128, 128, 1, 1), (128, 256, 1, 1), (256, 128, 2, 1), (256, 256, 2, 1), (128, 64, 1, 1)]
        for tm, tn, cm, cn in variants:
            exe = f"/work/qkc_{tm}x{tn}_c{cm}{cn}"
            t0 = time.time()
            R(f"cd {W} && nvcc -O3 -std=c++17 {GEN} --expt-relaxed-constexpr -DTILE_M={tm} -DTILE_N={tn} -DCL_M={cm} -DCL_N={cn} {CI} -o {exe} qk_cutlass_sm100.cu 2>&1 | grep -E 'error' | head -10", tail=12)
            P(f"compile {tm}x{tn} c{cm}x{cn}: {time.time()-t0:.0f}s exists={os.path.exists(exe)}")
            if not os.path.exists(exe): continue
            R(f"{exe} --m=128 --n=2048 --k=128 --iterations=5 --verify=1", tail=12)
            for (m, n, k) in [(128, 16384, 128), (128, 262144, 128), (128, 1048576, 128), (128, 16384, 4096), (128, 262144, 1024)]:
                R(f"{exe} --m={m} --n={n} --k={k} --iterations=30 --verify=0 | grep -E 'RESULT|Disposition'", tail=3)
    vol.commit(); log.close(); vol.commit()


@app.function(gpu="B300", image=img, timeout=3600 * 2, volumes={"/out": vol})
def run_b300(cutlass: bool = True): body("B300", cutlass)

@app.function(gpu="RTX-PRO-6000", image=img, timeout=3600, volumes={"/out": vol})
def run_rtx(): body("RTX-PRO-6000")

@app.local_entrypoint()
def main(gpu: str = "RTX-PRO-6000", cutlass: bool = True):
    if gpu == "B300": run_b300.remote(cutlass)
    else: run_rtx.remote()
