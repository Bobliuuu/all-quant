"""Model-quality runs on Modal (H100). Results land in volume `pq-results`.
  modal run --detach experiments/modal_quality.py --task kl --args "--nseq 8 --ctx 16384"
  modal run --detach experiments/modal_quality.py --task niah --args "..."
Fetch: modal volume get pq-results <file> results/
"""
import os
import subprocess

import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app = modal.App("pq-quality")
img = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.57.1", "accelerate", "pyarrow", "pandas", "huggingface_hub", "numpy")
    .env({"HF_HOME": "/models/hf"})
    .add_local_dir(os.path.join(ROOT, "pqrepro"), "/work/pqrepro", ignore=["__pycache__"])
    .add_local_dir(os.path.join(ROOT, "experiments"), "/work/experiments", ignore=["__pycache__"])
)
models = modal.Volume.from_name("pq-models", create_if_missing=True)
results = modal.Volume.from_name("pq-results", create_if_missing=True)


@app.function(gpu="H100", image=img, timeout=3 * 3600, volumes={"/models": models, "/out": results})
def run(task: str, args: str):
    from huggingface_hub import hf_hub_download
    pg = hf_hub_download("emozilla/pg19-test", "data/test-00000-of-00001-29a571947c0b5ccc.parquet",
                         repo_type="dataset", cache_dir="/models/hf")
    script = {"kl": "kl_pg19.py", "niah": "niah.py"}[task]
    cmd = f"cd /work && python experiments/{script} --pg19 {pg} {args}"
    print(cmd, flush=True)
    p = subprocess.run(cmd, shell=True)
    models.commit()
    results.commit()
    return p.returncode


@app.local_entrypoint()
def main(task: str = "kl", args: str = ""):
    print(run.remote(task, args))
