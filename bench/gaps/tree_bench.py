"""Tree-mask verify overhead.  python bench/gaps/tree_bench.py --out results/gaps_tree_<dev>.json
Run on a GPU under the shared lock.  Cache is random uint8 codes (timing only)."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from uqmt import attn as A, tree as T

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--hq", type=int, default=32); ap.add_argument("--hkv", type=int, default=8)
ap.add_argument("--d", type=int, default=128); ap.add_argument("--page", type=int, default=64)
ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
ap.add_argument("--iters", type=int, default=50); ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda"
B, P, D = a.batch, a.page, a.d
npg = (a.ctx + P - 1) // P
torch.manual_seed(0)
kc = torch.randint(0, 256, (B * npg, a.hkv, P, D // 2), dtype=torch.uint8, device=dev)
vc = torch.randint(0, 256, (B * npg, a.hkv, P, D // 2), dtype=torch.uint8, device=dev)
ks = torch.randint(118, 124, (B * npg, a.hkv, P, D // 32), dtype=torch.uint8, device=dev)
vs = torch.randint(118, 124, (B * npg, a.hkv, P, D // 32), dtype=torch.uint8, device=dev)
bt = torch.randperm(B * npg, device=dev).int().reshape(B, npg)
sl = torch.full((B,), a.ctx, dtype=torch.int32, device=dev)


def bench(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(a.iters):
            fn()
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / a.iters)
    return min(ts)


rows = []
for K in a.ks:
    q = torch.randn(B, K, a.hq, D, device=dev, dtype=torch.bfloat16)
    args = (q, kc, ks, vc, vs, bt, sl)
    binary = T.parents_to_mask([-1] + [(j - 1) // 2 for j in range(1, K)]).to(dev)
    star = T.star_mask(K).to(dev)
    r = dict(K=K, B=B, ctx=a.ctx)
    r["causal_ms"] = bench(lambda: A.uq_decode(*args))
    r["tree_nomask_ms"] = bench(lambda: T.uq_tree_decode(*args, None))
    r["tree_chainmask_ms"] = bench(lambda: T.uq_tree_decode(*args, T.chain_mask(K, device=dev)))
    r["tree_binary_ms"] = bench(lambda: T.uq_tree_decode(*args, binary))
    r["tree_star_ms"] = bench(lambda: T.uq_tree_decode(*args, star))
    r["independent_K_ms"] = bench(lambda: A.uq_decode(*args, fused=False))
    r["mask_overhead_pct_vs_causal"] = 100 * (r["tree_binary_ms"] / r["causal_ms"] - 1)
    r["mask_overhead_pct_vs_nomask"] = 100 * (r["tree_binary_ms"] / r["tree_nomask_ms"] - 1)
    r["speedup_vs_independent"] = r["independent_K_ms"] / r["tree_binary_ms"]
    print(json.dumps(r), flush=True)
    rows.append(r)
if a.out:
    json.dump(dict(device=torch.cuda.get_device_name(), rows=rows), open(a.out, "w"), indent=1)
