"""Benchmark: native CUDA vs Triton snapshot (and optionally FlashInfer FP8-KV decode) on the same shapes
as results/b300_b8_c32k_tuned.json.   python bench_cuda.py [--tune] [--check] [--flashinfer] --out r.json"""
import argparse, json, os, sys, time, itertools
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_triton_snapshot as T
from uq_cuda import uq_decode_cuda

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=8); p.add_argument("--ctx", type=int, default=32768)
p.add_argument("--hq", type=int, default=64); p.add_argument("--hkv", type=int, default=8)
p.add_argument("--d", type=int, default=128); p.add_argument("--page", type=int, default=64)
p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
p.add_argument("--tune", action="store_true"); p.add_argument("--check", action="store_true")
p.add_argument("--flashinfer", action="store_true"); p.add_argument("--no-triton", action="store_true")
p.add_argument("--iters", type=int, default=30); p.add_argument("--out", default=None)
a = p.parse_args()
dev = "cuda"; torch.manual_seed(0)
B, H, HKV, D, PAGE = a.batch, a.hq, a.hkv, a.d, a.page
ppr = (a.ctx + PAGE - 1) // PAGE; npg = B * ppr
prop = torch.cuda.get_device_properties(0)
print("device", prop.name, "SMs", prop.multi_processor_count, flush=True)


def qchunk(n):
    out = [[], [], [], []]
    for s in range(0, n, 256):
        m = min(256, n - s)
        for i in (0, 2):
            c, sc = T.quantize_fp4(torch.randn(m, HKV, PAGE, D, device=dev))
            out[i].append(c); out[i + 1].append(sc)
    return [torch.cat(x) for x in out]
kc, ks, vc, vs = qchunk(npg)
bt = torch.randperm(npg, device=dev, dtype=torch.int32).reshape(B, ppr)
seq = torch.full((B,), a.ctx, device=dev, dtype=torch.int32)
kvb = B * a.ctx * HKV * (D // 2 + D // 32) * 2


def timeit(fn, n=None):
    n = n or a.iters
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n


# HBM peak: copy_ of 2 GiB
x = torch.empty(1 << 30, dtype=torch.uint8, device=dev); y = torch.empty_like(x)
t = timeit(lambda: y.copy_(x), 20)
peak = 2 * x.numel() / (t * 1e-3) / 1e9
del x, y
print(f"HBM copy peak {peak:.0f} GB/s", flush=True)

rows = []
for K in a.ks:
    q = torch.randn(B, K, H, D, device=dev, dtype=torch.bfloat16)
    row = dict(K=K, device=prop.name, hbm_copy_GBps=peak, kv_bytes=kvb)
    sub = slice(0, 2)
    ref = None
    if a.check:
        ref = T.reference(q[sub].float().to(torch.bfloat16), kc, ks, vc, vs, bt[sub], seq[sub])
    cands = [dict(nwarps=nw, stages=st, nsplit=ns) for nw in (0,) for st in (3,) for ns in (32,)]
    if a.tune:
        bm = max(16, 1 << (K * 8 - 1).bit_length())
        nws = {16: (1, 2, 4), 32: (2, 4), 64: (4, 8), 128: (8,)}[bm]
        cands = [dict(nwarps=nw, stages=st, nsplit=ns, variant=1) for nw in nws for st in (2, 3, 4) for ns in (8, 16, 24, 32, 48, 64, 96)]
        cands += [dict(nwarps=nw, stages=st, nsplit=ns, variant=2, mb=mb) for mb in (1, 2) for nw in (2, 4, 8, 16) for st in (2, 3, 4) for ns in (4, 8, 12, 16, 24, 32, 48)]
    best = None
    for c in cands:
        try:
            f = lambda: uq_decode_cuda(q, kc, ks, vc, vs, bt, seq, fused=True, **c)
            ms = timeit(f, 10 if a.tune else a.iters)
        except Exception as ex:
            continue
        if best is None or ms < best[0]: best = (ms, c)
    ms_f, cfg = best
    f = lambda: uq_decode_cuda(q, kc, ks, vc, vs, bt, seq, fused=True, **cfg)
    ms_f = timeit(f)
    # K-independent baseline: K=1 problems, tuned over a small grid (the fused cfg is invalid for BLOCK_M=16)
    ms_u, ucfg = None, None
    ucands = [dict(variant=2, mb=1, nwarps=nw, stages=2, nsplit=ns) for nw in (2, 4) for ns in (8, 16, 24)] + [dict(variant=1, nwarps=4, stages=3, nsplit=ns) for ns in (16, 32)]
    for c in ucands:
        try:
            t_ = timeit(lambda: uq_decode_cuda(q, kc, ks, vc, vs, bt, seq, fused=False, **c), 5)
        except Exception:
            continue
        if ms_u is None or t_ < ms_u: ms_u, ucfg = t_, c
    ms_u = timeit(lambda: uq_decode_cuda(q, kc, ks, vc, vs, bt, seq, fused=False, **ucfg))
    row.update(cuda_fused_ms=ms_f, cuda_base_ms=ms_u, cuda_cfg=cfg, cuda_fused_GBps=kvb / ms_f / 1e6,
               cuda_pct_peak=100 * kvb / ms_f / 1e6 / peak, cuda_speedup_vs_base=ms_u / ms_f)
    if a.check:
        row["cuda_err"] = (f()[sub].float() - ref).abs().max().item()
    if not a.no_triton:
        tb = None
        tc = [dict(w=2, s=2, n=16)]
        if a.tune: tc = [dict(w=w, s=s, n=n) for w in (2, 4) for s in (2, 3) for n in (16, 32, 64)]
        for c in tc:
            os.environ["UQ_WARPS"], os.environ["UQ_STAGES"] = str(c["w"]), str(c["s"])
            try:
                tf = timeit(lambda: T.uq_decode(q, kc, ks, vc, vs, bt, seq, fused=True, nsplit=c["n"]), 10 if a.tune else a.iters)
            except Exception: continue
            if tb is None or tf < tb[0]: tb = (tf, c)
        c = tb[1]; os.environ["UQ_WARPS"], os.environ["UQ_STAGES"] = str(c["w"]), str(c["s"])
        tf = timeit(lambda: T.uq_decode(q, kc, ks, vc, vs, bt, seq, fused=True, nsplit=c["n"]))
        tu = timeit(lambda: T.uq_decode(q, kc, ks, vc, vs, bt, seq, fused=False, nsplit=c["n"]))
        row.update(triton_fused_ms=tf, triton_base_ms=tu, triton_cfg=c, triton_fused_GBps=kvb / tf / 1e6,
                   cuda_vs_triton=tf / ms_f)
    if a.flashinfer:
        try:
            import flashinfer
            for name, dt in (("fp8", torch.float8_e4m3fn), ("bf16", torch.bfloat16)):
                if name == "bf16" and a.ctx * B > 100000 and torch.cuda.get_device_properties(0).total_memory < 90e9:
                    continue
                kvd = torch.randn(npg, 2, HKV, PAGE, D, device=dev).to(dt) if name == "fp8" else torch.randn(npg, 2, HKV, PAGE, D, device=dev, dtype=dt)
                R = B * K  # K-independent requests (FlashInfer decode has q_len = 1)
                indptr = torch.arange(0, R + 1, device=dev, dtype=torch.int32) * ppr
                indices = bt.repeat_interleave(K, 0).reshape(-1).contiguous()
                last = torch.full((R,), (a.ctx - 1) % PAGE + 1, device=dev, dtype=torch.int32)
                ws = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
                w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "HND")
                w.plan(indptr, indices, last, H, HKV, D, PAGE, data_type=dt, q_data_type=torch.bfloat16)
                qq = q.reshape(R, H, D)
                ff = lambda: w.run(qq, kvd)
                row[f"flashinfer_{name}_indep_ms"] = timeit(ff)
                del kvd, ws, w
        except Exception as ex:
            row["flashinfer_error"] = repr(ex)[:300]
    rows.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    del q
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
