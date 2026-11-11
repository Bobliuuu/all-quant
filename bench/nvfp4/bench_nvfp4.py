"""Kernel benchmark: fused NVFP4 (E4M3/16 scales) vs fused MXFP4 (UE8M0/32) vs K-independent baselines.
Run from the repo root:  python bench/nvfp4/bench_nvfp4.py --out results/nvfp4_bench_<gpu>.json
Cache contents are random valid codes (timing is data independent); correctness is checked
against the torch reference on a real-quantised small cache first."""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from uqmt import attn as A, nvfp4 as N

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=8); p.add_argument("--ctx", type=int, default=32768)
p.add_argument("--hq", type=int, default=32); p.add_argument("--hkv", type=int, default=8)
p.add_argument("--d", type=int, default=128); p.add_argument("--page", type=int, default=64)
p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
p.add_argument("--nsplits", type=int, nargs="+", default=[0], help="0 = auto_nsplit; else sweep, best reported")
p.add_argument("--reps", type=int, default=7); p.add_argument("--iters", type=int, default=30)
p.add_argument("--out", default=None)
a = p.parse_args()
dev = "cuda"; torch.manual_seed(0)
B, HQ, HKV, D, PAGE = a.batch, a.hq, a.hkv, a.d, a.page
name = torch.cuda.get_device_name(0)

# ---- correctness on a real-quantised small cache
k = torch.randn(2, HKV, 1000, D, device=dev).bfloat16(); v = torch.randn(2, HKV, 1000, D, device=dev).bfloat16()
c = N.build_paged_cache_nvfp4(k, v, PAGE, shuffle=True)
chk = {}
for K in (1, 4, 8, 16):
    q = torch.randn(2, K, HQ, D, device=dev).bfloat16()
    ref = N.reference_nvfp4(q, c.kc, c.ks, c.vc, c.vs, c.block_table, c.seq_lens, c.k_ts, c.v_ts)
    o = c.decode(q).float()
    chk[K] = ((o - ref).norm() / ref.norm()).item()
print("check rel err vs torch ref", chk, flush=True)

# ---- big random cache (valid codes; e4m3 scale bytes in a plausible 0x28..0x48 range; ue8m0 ~ 2**-4..2**-2)
pages_per = (a.ctx + PAGE - 1) // PAGE; npages = B * pages_per
rb = lambda *s: torch.randint(0, 256, s, device=dev, dtype=torch.uint8)
kc, vc = rb(npages, HKV, PAGE, D // 2), rb(npages, HKV, PAGE, D // 2)
nks, nvs = (torch.randint(0x28, 0x48, (npages, HKV, PAGE, D // 16), device=dev, dtype=torch.uint8) for _ in range(2))
mks, mvs = (torch.randint(121, 125, (npages, HKV, PAGE, D // 32), device=dev, dtype=torch.uint8) for _ in range(2))
bt = torch.randperm(npages, device=dev, dtype=torch.int32).reshape(B, pages_per)
seq = torch.full((B,), a.ctx, device=dev, dtype=torch.int32)
bytes_nv = B * a.ctx * HKV * (D // 2 + D // 16) * 2
bytes_mx = B * a.ctx * HKV * (D // 2 + D // 32) * 2

def timeit(fn):
    for _ in range(3): fn()
    ts = []
    for _ in range(a.reps):
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(a.iters): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / a.iters)
    return statistics.median(ts)

rows = []
for K in a.ks:
    q = torch.randn(B, K, HQ, D, device=dev).bfloat16()
    r = dict(gpu=name, K=K, B=B, ctx=a.ctx, Hq=HQ, Hkv=HKV, D=D)
    for ns in a.nsplits:
        nsk = None if ns == 0 else ns
        cand = {
            "nvfp4_fused": lambda: N.uq_decode_nvfp4(q, kc, nks, vc, nvs, bt, seq, k_ts=0.01, v_ts=0.01, fused=True, nsplit=nsk),
            "mxfp4_fused": lambda: A.uq_decode(q, kc, mks, vc, mvs, bt, seq, fused=True, nsplit=nsk),
            "nvfp4_indep": lambda: N.uq_decode_nvfp4(q, kc, nks, vc, nvs, bt, seq, k_ts=0.01, v_ts=0.01, fused=False, nsplit=nsk),
            "mxfp4_indep": lambda: A.uq_decode(q, kc, mks, vc, mvs, bt, seq, fused=False, nsplit=nsk),
        }
        for kname, f in cand.items():
            t = timeit(f)
            if kname not in r or t < r[kname]: r[kname] = t; r[kname + "_nsplit"] = ns
    r["nvfp4_vs_mxfp4_fused"] = r["nvfp4_fused"] / r["mxfp4_fused"]
    r["fused_speedup_nvfp4"] = r["nvfp4_indep"] / r["nvfp4_fused"]
    r["fused_speedup_mxfp4"] = r["mxfp4_indep"] / r["mxfp4_fused"]
    r["nvfp4_fused_GBps"] = bytes_nv / r["nvfp4_fused"] / 1e6; r["mxfp4_fused_GBps"] = bytes_mx / r["mxfp4_fused"] / 1e6
    rows.append(r)
    print(json.dumps({k_: (round(x, 4) if isinstance(x, float) else x) for k_, x in r.items()}), flush=True)
if a.out:
    json.dump(dict(gpu=name, check_rel_err=chk, rows=rows), open(a.out, "w"), indent=1)
