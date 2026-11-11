"""cuda2: NVFP4 (E4M3 scale per 16 + per-tensor fp32) CUDA kernels vs the Triton NVFP4 kernel (uqmt/nvfp4.py), B8 ctx32768 Hq64/Hkv8 D128 PAGE64.
  python bench/cuda2/bench_nvfp4_cuda.py --out results/cuda2_<gpu>_nvfp4_b8_c32k.json"""
import argparse, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cuda"))
import torch
from uq_cuda import uq_decode_cuda, ext
from uqmt import nvfp4 as N, quant as Q
p = argparse.ArgumentParser(); p.add_argument("--out", default=None); p.add_argument("--ctx", type=int, default=32768); p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
a = p.parse_args(); dev = "cuda"; torch.manual_seed(0)
B, HQ, HKV, D, PAGE = 8, 64, 8, 128, 64; ppr = a.ctx // PAGE; npg = B * ppr
prop = torch.cuda.get_device_properties(0); print("device", prop.name, flush=True)
def chunk(n, ts=None):
    cs, ss = [], []
    for s in range(0, n, 256):
        m = min(256, n - s); x = torch.randn(m, HKV, PAGE, D, device=dev)
        c, sc, t = Q.quantize_nvfp4(x, tensor_scale=ts if ts else 4.0 / (6 * 448)); cs.append(c); ss.append(sc.view(torch.uint8))
    return torch.cat(cs), torch.cat(ss), (ts if ts else 4.0 / (6 * 448))
kc, ks, k_ts = chunk(npg); vc, vs, v_ts = chunk(npg)
bt = torch.randperm(npg, device=dev, dtype=torch.int32).reshape(B, ppr); seq = torch.full((B,), a.ctx, device=dev, dtype=torch.int32)
kvb = B * a.ctx * HKV * (D // 2 + D // 16) * 2
def timeit(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
rows = []
for K in a.ks:
    q = torch.randn(B, K, HQ, D, device=dev, dtype=torch.bfloat16)
    sub = slice(0, 2); ref = N.reference_nvfp4(q[sub], kc, ks, vc, vs, bt[sub], seq[sub], k_ts, v_ts)
    cands = [dict(variant=1, nwarps=nw, stages=st, nsplit=ns) for nw in (2, 4, 8) for st in (2, 3) for ns in (8, 16, 24, 32)]
    if ext().has_sm90() and K >= 4:
        cands += [dict(variant=3, nwarps=2, stages=3, mb=2, nsplit=ns) for ns in (2, 3)] + [dict(variant=3, nwarps=1, stages=33, mb=3, nsplit=2)]
    best = None
    for c in cands:
        try: ms = timeit(lambda: uq_decode_cuda(q, kc, ks, vc, vs, bt, seq, fmt=1, k_ts=k_ts, v_ts=v_ts, **c), 8)
        except Exception: continue
        if best is None or ms < best[0]: best = (ms, c)
    c = best[1]; f = lambda: uq_decode_cuda(q, kc, ks, vc, vs, bt, seq, fmt=1, k_ts=k_ts, v_ts=v_ts, **c); ms = timeit(f)
    err = (f()[sub].float() - ref).abs().max().item()
    tb = None
    for ns in (16, 32, 64):
        for w in (2, 4):
            os.environ["UQ_WARPS"] = str(w)
            try: t = timeit(lambda: N.uq_decode_nvfp4(q, kc, ks, vc, vs, bt, seq, k_ts=k_ts, v_ts=v_ts, nsplit=ns), 8)
            except Exception: continue
            if tb is None or t < tb[0]: tb = (t, dict(w=w, n=ns))
    os.environ["UQ_WARPS"] = str(tb[1]["w"]); tms = timeit(lambda: N.uq_decode_nvfp4(q, kc, ks, vc, vs, bt, seq, k_ts=k_ts, v_ts=v_ts, nsplit=tb[1]["n"]))
    row = dict(K=K, device=prop.name, kv_bytes=kvb, cuda_nvfp4_ms=ms, cuda_cfg=c, cuda_err_vs_fp32_ref=err, triton_nvfp4_ms=tms, triton_cfg=tb[1], cuda_vs_triton=tms / ms, cuda_GBps=kvb / ms / 1e6)
    rows.append(row); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
