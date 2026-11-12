"""Speed + attention-accuracy runner for a port. Writes results/ports_<method>.json (merge-updating) in the shared format:
{method, bytes_per_vec, ms_per_call:{K:ms}, gbs:{K:GB/s}, ppl_delta:{mistral7b,qwen3_8b}, attn_cos:{K:cos}, ...}
Run:  python run_port.py --method spectralquant [--baselines] [--kv llm]   (always under flock /root/gpu.lock on the shared box)"""
import argparse, importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--B", type=int, default=8); ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--Hq", type=int, default=64); ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--kv", default="llm", choices=["llm", "iid"]); ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--baselines", action="store_true"); ap.add_argument("--uq-dir", default="/root/uqpin"); ap.add_argument("--amd-dir", default="/root/h2h/kernels")
ap.add_argument("--out", default=None); ap.add_argument("--opts", default="{}", help="json kwargs for make_port")
a = ap.parse_args()
import torch, torch.nn.functional as F
from common import *
spec = importlib.util.spec_from_file_location("port", os.path.join(HERE, a.method, "port.py")); mod = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(HERE, a.method)); spec.loader.exec_module(mod)
port = mod.make_port(DEV, **json.loads(a.opts))
B, S, Hq = a.B, a.ctx, a.Hq; G = Hq // HKV
out = a.out or os.path.join(HERE, "..", "..", "results", f"ports_{a.method}.json")
torch.manual_seed(0)
kf, vf, qs = make_kv(B, S, a.kv)
port.calibrate(kf[:1, :, :4096], vf[:1, :, :4096]) if hasattr(port, "calibrate") else None
port.write(kf, vf); torch.cuda.synchronize()
bv = port.bytes_per_vec(D); res = dict(method=a.method, kv=a.kv, B=B, ctx=S, Hq=Hq, Hkv=HKV, D=D, bytes_per_vec=bv, ms_per_call={}, ms_per_call_hot={}, gbs={}, attn_cos={}, attn_relerr={}, baselines={})
def mkq(K): return torch.stack([qs(Hq * K).reshape(K, Hq, D) for _ in range(B)]).to(torch.bfloat16)
for K in a.ks:
    q = mkq(K)
    o = port.decode(q)
    cs, rs = [], []
    for b in range(2):
        c, r = cos_rel(o[b], ref_attn(q[b], kf[b], vf[b])); cs.append(c); rs.append(r)
    res["attn_cos"][K] = sum(cs) / 2; res["attn_relerr"][K] = sum(rs) / 2
    res["ms_per_call"][K] = timeit(lambda: port.decode(q), a.reps, cold=True); res["ms_per_call_hot"][K] = timeit(lambda: port.decode(q), a.reps, cold=False)
    res["gbs"][K] = gbs(B, S, bv, res["ms_per_call"][K])
    print(json.dumps(dict(K=K, ms=res["ms_per_call"][K], hot=res["ms_per_call_hot"][K], gbs=res["gbs"][K], cos=res["attn_cos"][K], rel=res["attn_relerr"][K])), flush=True)
if hasattr(port, "time_write"):
    res["write_ms_decode_step"] = {K: timeit(lambda: port.time_write(kf[:, :, :K], vf[:, :, :K]), 10, cold=False) for K in a.ks}
res["fp16_bytes_per_vec"] = 256
if a.baselines:
    sys.path.insert(0, a.uq_dir); sys.path.insert(0, a.amd_dir)
    from attn import quantize_fp4, uq_decode
    from turboquant_mi300x import TurboQuantMI300X
    from tq_triton import turboquant_gqa_attention_fwd
    PAGE = 64; npp = S // PAGE; npg = B * npp
    pg = lambda x: x.reshape(B, HKV, npp, PAGE, D).permute(0, 2, 1, 3, 4).reshape(npg, HKV, PAGE, D)
    def ch(x):
        c, s = [], []
        for i in range(0, x.shape[0], 512):
            p, sc = quantize_fp4(x[i:i + 512].float()); c.append(p); s.append(sc)
        return torch.cat(c), torch.cat(s)
    kc, ks = ch(pg(kf)); vc, vs = ch(pg(vf)); bt = torch.arange(npg, device=DEV, dtype=torch.int32).reshape(B, npp); seq = torch.full((B,), S, device=DEV, dtype=torch.int32)
    tq = TurboQuantMI300X(bits=3, device=DEV)
    def ctq(x):
        flat = x.reshape(-1, D); comp = torch.cat([tq.compress_tensor(flat[i:i + (1 << 21)].float()) for i in range(0, flat.shape[0], 1 << 21)])
        return comp[:, 4:].contiguous().view(B, HKV, S, 48), comp[:, :4].contiguous().view(-1).view(torch.float32).view(B, HKV, S)
    kp, kn = ctq(kf); vp, vn = ctq(vf)
    for K in a.ks:
        q = mkq(K); r = [ref_attn(q[b], kf[b], vf[b]) for b in range(2)]
        o = uq_decode(q, kc, ks, vc, vs, bt, seq, nsplit=16)
        cu = sum(cos_rel(o[b], r[b])[0] for b in range(2)) / 2
        msu = timeit(lambda: uq_decode(q, kc, ks, vc, vs, bt, seq, nsplit=16), a.reps, cold=True)
        res["baselines"].setdefault("uq_fp4_fused", {})[K] = dict(ms=msu, gbs=gbs(B, S, 68, msu), cos=cu, bytes_per_vec=68)
        q1 = q[:, :1].permute(0, 2, 1, 3).half().contiguous()  # [B,Hq,1,D]
        t1 = lambda: turboquant_gqa_attention_fwd(tq.rotate_queries(q1.float()).half(), kp, kn, vp, vn, gqa_ratio=G, rotation=tq.rotation, sm_scale=SM)
        o1 = t1(); ct = sum(cos_rel(o1[b].permute(1, 0, 2), ref_attn(q[b, :1], kf[b], vf[b]))[0] for b in range(2)) / 2
        mst = timeit(lambda: [t1() for _ in range(K)], a.reps, cold=True)
        res["baselines"].setdefault("tq3_gqa_Kindep", {})[K] = dict(ms=mst, gbs=gbs(B, S, 52, mst) * 1, cos=ct, bytes_per_vec=52, note="K independent K=1 calls (their kernel has no K-query mode); cos is for query 0 only")
        qf = q.permute(0, 2, 1, 3).reshape(B, HKV, G * K, D).half().contiguous()
        msf = timeit(lambda: F.scaled_dot_product_attention(qf, kf, vf, scale=SM), a.reps, cold=True)
        res["baselines"].setdefault("fp16_sdpa", {})[K] = dict(ms=msf, gbs=gbs(B, S, 256, msf), cos=1.0, bytes_per_vec=256)
        print(json.dumps(dict(K=K, uq_ms=msu, tq3_ms=mst, fp16_ms=msf, uq_cos=cu, tq3_cos=ct)), flush=True)
os.makedirs(os.path.dirname(out), exist_ok=True)
merge_json(out, res); print("wrote", out)
