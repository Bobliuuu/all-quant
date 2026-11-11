"""Correctness + benchmark: fused K-query vs K independent single-query FP4 decode."""
import argparse, json, sys, time
import torch
from attn_triton_snapshot import quantize_fp4, uq_decode, reference

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=8)
p.add_argument("--ctx", type=int, default=32768)
p.add_argument("--hq", type=int, default=64)
p.add_argument("--hkv", type=int, default=8)
p.add_argument("--d", type=int, default=128)
p.add_argument("--page", type=int, default=64)
p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
p.add_argument("--nsplit", type=int, default=16)
p.add_argument("--check", action="store_true")
p.add_argument("--out", default=None)
a = p.parse_args()

dev = "cuda"
torch.manual_seed(0)
B, H, HKV, D, PAGE = a.batch, a.hq, a.hkv, a.d, a.page
pages_per = (a.ctx + PAGE - 1) // PAGE
npages = B * pages_per
kf = torch.randn(npages, HKV, PAGE, D, device=dev)
vf = torch.randn(npages, HKV, PAGE, D, device=dev)
kc, ks = quantize_fp4(kf); vc, vs = quantize_fp4(vf)
del kf, vf
bt = torch.randperm(npages, device=dev, dtype=torch.int32).reshape(B, pages_per)
seq = torch.full((B,), a.ctx, device=dev, dtype=torch.int32)
kv_bytes = lambda: B * a.ctx * HKV * (D // 2 + D // 32) * 2

def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n

rows = []
for K in a.ks:
    q = torch.randn(B, K, H, D, device=dev, dtype=torch.bfloat16)
    f = lambda: uq_decode(q, kc, ks, vc, vs, bt, seq, fused=True, nsplit=a.nsplit)
    u = lambda: uq_decode(q, kc, ks, vc, vs, bt, seq, fused=False, nsplit=a.nsplit)
    err = None
    if a.check:
        sub = slice(0, 2)
        ref = reference(q[sub], kc, ks, vc, vs, bt[sub], seq[sub])
        e1 = (f()[sub].float() - ref).abs().max().item()
        e2 = (u()[sub].float() - ref).abs().max().item()
        err = {"fused": e1, "baseline": e2}
        print(f"K={K} max|err| fused={e1:.4f} baseline={e2:.4f}", flush=True)
    tf, tu = timeit(f), timeit(u)
    r = dict(K=K, fused_ms=tf * 1e3, base_ms=tu * 1e3, speedup=tu / tf,
             fused_GBps=kv_bytes() / tf / 1e9, err=err, **vars(a))
    rows.append(r)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in
                      ("K", "fused_ms", "base_ms", "speedup", "fused_GBps")}), flush=True)
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
