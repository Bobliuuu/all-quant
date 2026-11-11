"""Run tests/test_tree.py + tests/test_quant.py (+ tree bench) on an NVIDIA GPU via Modal. modal run bench/gaps/modal_tests.py"""
import os, modal
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("uq-gaps-tests")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch", "triton", "numpy", "pytest")
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"])
       .add_local_dir(os.path.join(ROOT, "tests"), "/work/tests", ignore=["__pycache__"])
       .add_local_dir(os.path.join(ROOT, "bench", "gaps"), "/work/bench/gaps", ignore=["__pycache__"]))

@app.function(gpu="A100", image=img, timeout=1800)
def go():
    import subprocess
    r = subprocess.run("cd /work && python -m pytest tests/test_tree.py tests/test_quant.py -q -x 2>&1 | tail -25", shell=True, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    r = subprocess.run("cd /work && python bench/gaps/tree_bench.py --out /tmp/t.json 2>&1 | grep -v Warn", shell=True, capture_output=True, text=True)
    print(r.stdout[-4000:])
    print(open("/tmp/t.json").read())

@app.local_entrypoint()
def main():
    go.remote()
