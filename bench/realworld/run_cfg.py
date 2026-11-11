"""Run one (model, kv dtype, extra server args) config with one workload; write results/realworld_<tag>.json.
Run under:  flock /root/gpu.lock timeout 590 python run_cfg.py ...   (each hold <= 10 min)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwlib import Server

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--kv", default="auto")
p.add_argument("--workload", required=True, choices=["agent", "rag", "spec", "needle", "drift", "capacity"])
p.add_argument("--tag", required=True); p.add_argument("--extra", default="")
p.add_argument("--maxlen", type=int, default=40960); p.add_argument("--gmu", type=float, default=0.90)
p.add_argument("--port", type=int, default=8100)
p.add_argument("--sessions", type=int, default=16); p.add_argument("--turns", type=int, default=8)
p.add_argument("--prefix-tokens", type=int, default=16000); p.add_argument("--chunk-tokens", type=int, default=1200)
p.add_argument("--gen", type=int, default=96); p.add_argument("--n", type=int, default=100)
p.add_argument("--ctx-tokens", type=int, default=16000); p.add_argument("--k", type=int, default=0)
p.add_argument("--conc", type=int, default=8); p.add_argument("--outdir", default="/root/rw/out")
a = p.parse_args()
os.makedirs(a.outdir, exist_ok=True)
extra = a.extra.split() if a.extra else []
res = {"tag": a.tag, "model": a.model, "kv": a.kv, "extra": a.extra, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    with Server(a.model, a.kv, a.port, extra, a.maxlen, a.gmu) as srv:
        res["server"] = srv.log_facts()
        mod = __import__("wl_" + a.workload)
        res["result"] = mod.run(srv, a)
        try: res["metrics_tail"] = {k: v for k, v in srv.metrics().items() if "spec_decode" in k or "prefix_cache" in k}
        except Exception: pass
except Exception as e:
    res["error"] = repr(e)
json.dump(res, open(f"{a.outdir}/realworld_{a.tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "result"}, indent=1)[:1500])
r = res.get("result", {}); print({k: v for k, v in r.items() if k not in ("outputs", "samples")})
