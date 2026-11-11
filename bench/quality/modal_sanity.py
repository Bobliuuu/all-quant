import modal, os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__)); import modal_quality as mq
app = mq.app
@app.function(gpu="A100-80GB", image=mq.img, timeout=1800, volumes={"/out": mq.vol, "/hf": mq.hfv})
def run():
    env = dict(os.environ, HF_HOME="/hf", PYTHONPATH="/work", AMD_KERNELS="/work/amdk")
    p = subprocess.run([sys.executable, "/work/q/sanity_blockquant.py"], env=env, capture_output=True, text=True); mq.vol.commit(); return p.stdout[-5000:] + p.stderr[-1500:]
@app.local_entrypoint()
def main2(): print(run.remote())
