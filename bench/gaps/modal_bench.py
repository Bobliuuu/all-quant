"""Tree bench on an NVIDIA GPU via Modal (streams output). modal run bench/gaps/modal_bench.py"""
import os, modal
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("uq-gaps-bench")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch", "triton", "numpy")
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"])
       .add_local_dir(os.path.join(ROOT, "bench", "gaps"), "/work/bench/gaps", ignore=["__pycache__"]))

@app.function(gpu="A100", image=img, timeout=1200)
def go():
    import subprocess
    p = subprocess.Popen("cd /work && python -u bench/gaps/tree_bench.py --iters 20 --out /tmp/t.json 2>&1", shell=True, stdout=subprocess.PIPE, text=True)
    for l in p.stdout:
        print(l, end="", flush=True)
    p.wait()
    print(open("/tmp/t.json").read())

@app.local_entrypoint()
def main():
    go.remote()
