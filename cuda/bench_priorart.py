"""Prior-art head-to-head at the UltraQuant shapes: B=8 ctx=32768 Hq=64 Hkv=8 D=128, K queries/seq.
Runs inside the Modal image from cuda/modal/modal_priorart.py.   sections: peak flashinfer bitdec fa3
Every config is try/except'd; failures are recorded verbatim in the JSON ("error").
KV bytes (what one pass over the cache must read):  bf16 256 B/vec, fp8 128 B/vec,
NVFP4 (16-block E4M3 scale) 72 B/vec, ours (32-block UE8M0) 68 B/vec.
GB/s = kv_bytes * reads / ms, with reads = 1 for a kernel that shares one KV pass across the K queries and
reads = K for K independent decodes ("serial"/"rows" modes).  For FlashInfer multi-query paths reads=1 is an
ASSUMPTION (the kernel's own tile loop is not audited here)."""
import argparse, json, os, sys, time, traceback
import torch

p = argparse.ArgumentParser()
p.add_argument("--sections", nargs="+", default=["peak", "flashinfer", "bitdec", "fa3"])
p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
p.add_argument("--iters", type=int, default=30); p.add_argument("--warmup", type=int, default=10)
p.add_argument("--bitdec-src", default="/work/BitDecoding-FP4/src")
p.add_argument("--out", required=True)
p.add_argument("--peak", type=float, default=None)
a = p.parse_args()
B, CTX, HQ, HKV, D, PAGE = 8, 32768, 64, 8, 128, 64
dev = "cuda"; torch.manual_seed(0)
prop = torch.cuda.get_device_properties(0)
R = {"gpu": prop.name, "cc": f"{prop.major}.{prop.minor}", "sms": prop.multi_processor_count,
     "torch": torch.__version__, "shape": dict(B=B, ctx=CTX, Hq=HQ, Hkv=HKV, D=D, page=PAGE), "rows": []}
if a.peak: R["hbm_peak_GBps"] = a.peak
VEC = B * CTX * HKV * 2          # K+V vectors
BYTES = {"bf16": VEC * 256, "fp8": VEC * 128, "nvfp4": VEC * 72, "ours68": VEC * 68}


def save():
    json.dump(R, open(a.out, "w"), indent=1)


def P(*x):
    print(*x, flush=True)


def timeit(fn):
    for _ in range(a.warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(a.iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / a.iters


def row(name, K, fmt, ms=None, reads=1, note="", error=None, **kw):
    r = dict(impl=name, K=K, kv_format=fmt, note=note, **kw)
    if error: r["error"] = error
    else:
        r["ms"] = round(ms, 4); r["kv_bytes_MB"] = round(BYTES[fmt] / 1e6, 1); r["reads"] = reads
        r["GBps"] = round(BYTES[fmt] * reads / (ms * 1e-3) / 1e9, 1)
        if R.get("hbm_peak_GBps"): r["pct_hbm_peak"] = round(100 * r["GBps"] / R["hbm_peak_GBps"], 1)
    R["rows"].append(r); P(json.dumps(r)); save()


def attempt(name, K, fmt, fn, **kw):
    try:
        fn()                       # compile / first call
        torch.cuda.synchronize()
        ms = timeit(fn)
        row(name, K, fmt, ms, **kw)
    except Exception as ex:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        row(name, K, fmt, error=(type(ex).__name__ + ": " + str(ex))[-600:], **kw)
        traceback.print_exc()


# ---------------------------------------------------------------- HBM peak
def sec_peak():
    n = 4 * 2**30 // 2
    x = torch.empty(n, dtype=torch.bfloat16, device=dev).normal_(); y = torch.empty_like(x)
    ms = timeit(lambda: y.copy_(x))
    gbs = 2 * x.numel() * 2 / (ms * 1e-3) / 1e9
    # read-only reduction too (closer to attention's read-dominated traffic)
    ms2 = timeit(lambda: x.view(-1, 2**20).sum(0))
    gbs2 = x.numel() * 2 / (ms2 * 1e-3) / 1e9
    R["hbm_copy_GBps"] = round(gbs, 1); R["hbm_readsum_GBps"] = round(gbs2, 1)
    R["hbm_peak_GBps"] = round(max(gbs, gbs2), 1)
    P("HBM copy", gbs, "readsum", gbs2); save(); del x, y


# ---------------------------------------------------------------- FlashInfer
def sec_flashinfer(parts=("base", "mq")):
    import flashinfer
    R["flashinfer"] = flashinfer.__version__
    ppr = CTX // PAGE; npg = B * ppr
    kv_bf = torch.randn(npg, 2, HKV, PAGE, D, device=dev, dtype=torch.bfloat16) * 0.3   # HND
    kv_f8 = kv_bf.to(torch.float8_e4m3fn)
    idx = torch.arange(npg, device=dev, dtype=torch.int32)
    indptr = torch.arange(0, (B + 1) * ppr, ppr, device=dev, dtype=torch.int32)
    last = torch.full((B,), PAGE, device=dev, dtype=torch.int32)
    seq = torch.full((B,), CTX, device=dev, dtype=torch.int32)
    bt = idx.reshape(B, ppr)
    ws = torch.zeros(256 * 2**20, dtype=torch.uint8, device=dev)

    # (1) BatchDecodeWithPagedKVCacheWrapper (single query), bf16 & fp8 KV
    for fmt, kv, dt in ((("bf16", kv_bf, torch.bfloat16), ("fp8", kv_f8, torch.float8_e4m3fn)) if "base" in parts else ()):
        def mk(dt=dt, kv=kv):
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "HND", use_tensor_cores=True)
            w.plan(indptr, idx, last, HQ, HKV, D, PAGE, data_type=dt, q_data_type=torch.bfloat16)
            q = torch.randn(B, HQ, D, device=dev, dtype=torch.bfloat16)
            return lambda: w.run(q, kv)
        try:
            attempt("flashinfer.BatchDecodeWithPagedKVCacheWrapper", 1, fmt, mk(), note="single-query decode, tensor-core path")
        except Exception as ex:
            row("flashinfer.BatchDecodeWithPagedKVCacheWrapper", 1, fmt, error=str(ex)[-500:])

    # (2) BatchPrefillWithPagedKVCacheWrapper qo_len=K causal (multi-query path), bf16 & fp8 KV
    for K in (a.ks if "base" in parts else []):
        qo_indptr = torch.arange(0, (B + 1) * K, K, device=dev, dtype=torch.int32)
        for fmt, kv, dt in (("bf16", kv_bf, torch.bfloat16), ("fp8", kv_f8, torch.float8_e4m3fn)):
            for backend in ("fa2",):
                def mk(dt=dt, kv=kv, backend=backend, K=K, qo_indptr=qo_indptr):
                    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend=backend)
                    w.plan(qo_indptr, indptr, idx, last, HQ, HKV, D, PAGE, causal=True,
                           q_data_type=torch.bfloat16, kv_data_type=dt)
                    q = torch.randn(B * K, HQ, D, device=dev, dtype=torch.bfloat16)
                    return lambda: w.run(q, kv)
                try:
                    attempt(f"flashinfer.BatchPrefillWithPagedKVCacheWrapper[{backend}]", K, fmt, mk(),
                            note="qo_len=K causal; reads=1 assumed")
                except Exception as ex:
                    row(f"flashinfer.BatchPrefillWithPagedKVCacheWrapper[{backend}]", K, fmt, error=str(ex)[-500:])

    # (3) trtllm_batch_decode_with_kv_cache / xqa with q_len_per_req=K (speculative-decode multi-token decode)
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    kinds = []
    cap = prop.major
    # NVFP4 pages (SM100 trtllm-gen)
    nv = None
    if cap in (9, 10) and "mq" in parts:
        try:
            from flashinfer.quantization import nvfp4_quantize_paged_kv_cache
            kbf, vbf = kv_bf[:, 0].contiguous(), kv_bf[:, 1].contiguous()
            (kf, vf), (ksf, vsf), kg, vg = nvfp4_quantize_paged_kv_cache(kbf, vbf, "HND")
            nv = dict(kv=(kf, vf), sf=(ksf, vsf), kg=kg, vg=vg)
            del kbf, vbf
            R["nvfp4_scale_layout"] = f"data {tuple(kf.shape)} {kf.dtype}, sf {tuple(ksf.shape)} {ksf.dtype}"
        except Exception as ex:
            row("flashinfer.nvfp4_quantize_paged_kv_cache", 0, "nvfp4", error=str(ex)[-500:])
    sm_scale = D ** -0.5
    for K in (a.ks if "mq" in parts else []):
        for fmt in (["nvfp4"] if nv else []) + ["fp8", "bf16"]:
            for backend in (["trtllm-gen"] if cap == 10 else ["xqa"]):
                for qdt in ((torch.bfloat16, torch.float8_e4m3fn) if fmt != "bf16" else (torch.bfloat16,)):
                    def mk(fmt=fmt, backend=backend, qdt=qdt, K=K):
                        q = (torch.randn(B * K, HQ, D, device=dev) * 0.3).to(qdt)
                        out = torch.empty(B * K, HQ, D, device=dev, dtype=qdt)  # trtllm/xqa require out dtype == q dtype
                        kw = {}
                        if fmt == "nvfp4":
                            kvc = nv["kv"]; kw["kv_cache_sf"] = nv["sf"]
                            b1, b2 = sm_scale * nv["kg"], nv["vg"]
                        elif fmt == "fp8":
                            kvc = (kv_f8[:, 0], kv_f8[:, 1]); b1, b2 = sm_scale, 1.0
                        else:
                            kvc = (kv_bf[:, 0], kv_bf[:, 1]); b1, b2 = sm_scale, 1.0
                        if backend == "xqa" and K > 1:   # draft-block mask: uint16 [B,K,2]; row i sees prefix + draft tokens <= i
                            m = torch.zeros(B, K, 2, dtype=torch.uint16, device=dev)
                            for i in range(K): m[:, i, 0] = (1 << (i + 1)) - 1
                            kw["mask"] = m
                        wsb = torch.zeros(512 * 2**20, dtype=torch.uint8, device=dev)
                        def run():
                            return trtllm_batch_decode_with_kv_cache(
                                query=q, kv_cache=kvc, workspace_buffer=wsb, block_tables=bt, seq_lens=seq,
                                max_seq_len=CTX, bmm1_scale=b1, bmm2_scale=b2, out=out, kv_layout="HND",
                                backend=backend, q_len_per_req=K, **kw)
                        return run
                    try:
                        attempt(f"flashinfer.trtllm_batch_decode_with_kv_cache[{backend}]", K, fmt, mk(),
                                note=f"q_len_per_req={K}, q dtype {str(qdt).split('.')[-1]}; reads=1 assumed; "
                                     "NVFP4 = 16-block E4M3 scale + fp32 global scale (not UE8M0)")
                    except Exception as ex:
                        row(f"flashinfer.trtllm_batch_decode_with_kv_cache[{backend}]", K, fmt, error=str(ex)[-500:])
    # (4) last, because JIT-compiling the FA3 (Hopper) prefill template is slow: FlashInfer fa3 backend, qo_len=K
    if cap == 9 and "base" in parts:
        for K in a.ks:
            qo_indptr = torch.arange(0, (B + 1) * K, K, device=dev, dtype=torch.int32)
            for fmt, kv, dt in (("bf16", kv_bf, torch.bfloat16), ("fp8", kv_f8, torch.float8_e4m3fn)):
                def mk(dt=dt, kv=kv, K=K, qo_indptr=qo_indptr):
                    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa3")
                    w.plan(qo_indptr, indptr, idx, last, HQ, HKV, D, PAGE, causal=True,
                           q_data_type=torch.bfloat16, kv_data_type=dt)
                    q = torch.randn(B * K, HQ, D, device=dev, dtype=torch.bfloat16)
                    return lambda: w.run(q, kv)
                try:
                    attempt("flashinfer.BatchPrefillWithPagedKVCacheWrapper[fa3]", K, fmt, mk(), note="qo_len=K causal; reads=1 assumed")
                except Exception as ex:
                    row("flashinfer.BatchPrefillWithPagedKVCacheWrapper[fa3]", K, fmt, error=str(ex)[-500:])
    del kv_bf, kv_f8, nv


# ---------------------------------------------------------------- BitDecoding-FP4
def sec_bitdec():
    sys.path.insert(0, a.bitdec_src)
    from nvfp4_decode_kernel import _quantize, fp4_decode, split_partial_count
    P_ = 128                                          # kernel hard-codes PAGE_SIZE=128
    ppr = CTX // P_; npg = B * ppr
    k = (torch.randn(npg, P_, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3)
    v = torch.randn_like(k) * 0.3

    def qpages(fn, x):
        data = sc = None
        for s in range(0, x.shape[0], 32768):
            d_, s_ = fn(x[s:s + 32768])
            if data is None:
                data = torch.empty((x.shape[0], *d_.shape[1:]), dtype=d_.dtype, device=dev)
                st = torch.empty(x.shape[0] * s_.stride(0), dtype=torch.uint8, device=dev)
                sc = st.as_strided((x.shape[0], *s_.shape[1:]), s_.stride()).view(s_.dtype)
            data[s:s + 32768].copy_(d_); sc[s:s + 32768].copy_(s_)
        return data, sc
    kf, ks = qpages(_quantize.quantize_key_pages, k)
    vf, vs = qpages(_quantize.quantize_value_pages, v)
    R["bitdec_bytes"] = dict(k_data=kf.numel(), k_scales=ks.numel() * ks.element_size(),
                             v_data=vf.numel(), v_scales=vs.numel() * vs.element_size())
    actual = kf.numel() + ks.numel() * ks.element_size() + vf.numel() + vs.numel() * vs.element_size()
    BYTES["nvfp4"] = actual                               # measured, includes the kernel's scale padding
    del k, v
    pt = torch.arange(npg, dtype=torch.int32, device=dev).reshape(B, ppr)
    sl = torch.full((B,), CTX, dtype=torch.int32, device=dev)
    sms = prop.multi_processor_count
    sm_scale = D ** -0.5

    def make(rows, pt_, sl_, nsplit, terms):
        q = torch.randn(rows, HQ, D, dtype=torch.bfloat16, device=dev) * 0.3
        qs = _quantize.quantize_query(q, heads_kv=HKV, query_terms=terms)
        q0, s0 = qs[0], qs[1]
        q1, s1 = (qs[2], qs[3]) if terms == 2 else (None, None)
        out = torch.empty(rows, HQ, D, dtype=torch.bfloat16, device=dev)
        scr = {}
        if nsplit > 1:
            n = split_partial_count(rows, HKV, nsplit, residual=False, sms=sms)
            scr = dict(split_partial_scratch=torch.empty(n * rows * HQ * D, dtype=torch.float32, device=dev),
                       split_lse_scratch=torch.empty(n * rows * HQ, dtype=torch.float32, device=dev))
        def run():
            return fp4_decode(key_pages_fp4=kf, key_scales=ks, value_pages_fp4=vf, value_scales=vs,
                              fp4_page_table=pt_, seqused_fp4=sl_, query_fp4=q0, query_scales=s0,
                              query_fp4_second=q1, query_scales_second=s1, query_terms=terms,
                              softmax_scale=sm_scale, trusted_metadata=True, num_splits=nsplit, out=out, **scr)
        return run

    def best(K, mode, rows, pt_, sl_, terms, reads):
        res = {}
        for ns in (1, 2, 3, 4, 6, 8, 12, 16):
            try:
                run = make(rows, pt_, sl_, ns, terms)
                if mode == "serial":
                    runs = [make(B, pt_, sl_, ns, terms) for _ in range(K)]
                    fn = lambda: [r() for r in runs]
                else:
                    fn = run
                fn(); torch.cuda.synchronize()
                res[ns] = timeit(fn)
            except Exception as ex:
                P("bitdec fail", mode, K, ns, str(ex)[-200:])
                if not res and ns == 1: row("BitDecoding-FP4.fp4_decode", K, "nvfp4", error=str(ex)[-500:], mode=mode); return
        if not res: return
        ns = min(res, key=res.get)
        row("BitDecoding-FP4.fp4_decode", K, "nvfp4", res[ns], reads=reads, mode=mode, query_terms=terms,
            num_splits=ns, ms_by_splits={str(s): round(t, 4) for s, t in res.items()},
            note={"serial": f"K independent single-query decodes (kernel attends 1 query token per sequence; "
                            "its vLLM guard rejects speculative decoding); KV re-read K times",
                  "rows": "K*B rows in ONE launch with the page table repeated per row; still K KV passes"}[mode]
            + "; NVFP4 16-block E4M3 scales, page size 128")
    for K in a.ks:
        for terms in (1, 2):
            best(K, "serial", B, pt, sl, terms, reads=K)
            if K > 1:
                best(K, "rows", B * K, pt.repeat_interleave(K, 0), sl.repeat_interleave(K, 0), terms, reads=K)


# ---------------------------------------------------------------- FA3
def sec_fa3():
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except Exception as ex:
        row("FlashAttention-3.flash_attn_with_kvcache", 0, "fp8", error="not importable: " + str(ex)[-300:]); return
    ppr = CTX // 256 if False else CTX // 256
    PG = 256                                           # FA3 paged KV requires page size multiple of 256
    npg = B * (CTX // PG)
    for fmt, dt in (("bf16", torch.bfloat16), ("fp8", torch.float8_e4m3fn)):
        kc = (torch.randn(npg, PG, HKV, D, device=dev) * 0.3).to(dt); vc = kc.clone()
        bt = torch.arange(npg, dtype=torch.int32, device=dev).reshape(B, CTX // PG)
        cs = torch.full((B,), CTX, dtype=torch.int32, device=dev)
        for K in a.ks:
            q = (torch.randn(B, K, HQ, D, device=dev) * 0.3).to(dt if fmt == "fp8" else torch.bfloat16)
            attempt("FlashAttention-3.flash_attn_with_kvcache", K, fmt,
                    lambda q=q, kc=kc, vc=vc, bt=bt, cs=cs: flash_attn_with_kvcache(
                        q, kc, vc, cache_seqlens=cs, page_table=bt, causal=True),
                    note="seqlen_q=K causal, page 256; reads=1 assumed")
        del kc, vc


for s in a.sections:
    P("=== section", s)
    try:
        {"peak": sec_peak, "flashinfer": sec_flashinfer, "fi_mq": lambda: sec_flashinfer(("mq",)),
         "fi_base": lambda: sec_flashinfer(("base",)), "fi_mq128": lambda: (globals().__setitem__("PAGE", 128), sec_flashinfer(("mq",))), "bitdec": sec_bitdec, "fa3": sec_fa3}[s]()
    except Exception as ex:
        R["rows"].append(dict(impl=s, error="section failed: " + (type(ex).__name__ + ": " + str(ex))[-800:]))
        traceback.print_exc(); save()
    torch.cuda.empty_cache()
save()
P("DONE", a.out)
