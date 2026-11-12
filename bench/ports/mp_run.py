"""Own-cache correctness for a port: kernel output vs fp32 attention on the port's exact dequantisation (dequant_seq),
at full benchmark shape, K=1/2/4/8; merges {rel_err_vs_own_dequant} into results/ports_<method>.json.
usage: python mp_run.py --method fp8 --out results/ports_fp8.json   (under flock /root/gpu.lock)"""
import argparse, importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--B", type=int, default=8)
ap.add_argument("--ctx", type=int, default=32768); ap.add_argument("--out", default=None); ap.add_argument("--opts", default="{}")
a = ap.parse_args()
import torch
from common import DEV, make_kv, ref_attn, merge_json
sys.path.insert(0, os.path.join(HERE, a.method))
spec = importlib.util.spec_from_file_location("port_" + a.method, os.path.join(HERE, a.method, "port.py")); mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
port = mod.make_port(DEV, **json.loads(a.opts))
torch.manual_seed(0)
kf, vf, qs = make_kv(a.B, a.ctx, "llm")
port.write(kf, vf)
res = {}
for K in (1, 2, 4, 8):
    q = torch.stack([qs(64 * K).reshape(K, 64, 128) for _ in range(a.B)]).to(torch.bfloat16)
    o = port.decode(q).float(); errs = []
    for b in (0, a.B - 1):
        kd, vd = port.dequant_seq(b)
        r = ref_attn(q[b], kd, vd)
        errs.append(((o[b] - r).norm() / r.norm()).item())
    res[K] = max(errs)
    print("K", K, "rel_err_vs_own_dequant", res[K], flush=True)
out = a.out or os.path.join(HERE, "..", "..", "results", f"ports_{a.method}.json")
merge_json(out, dict(rel_err_vs_own_dequant=res))
