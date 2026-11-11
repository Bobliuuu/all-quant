"""Other KV-quant methods, attention step (K=1 and K=8-as-8-calls vs fused), same shapes/timer as h2h_microbench.
Methods without a fused kernel in amd-experiments (fp8/int4 per-tensor baselines, Planar/Iso/Rotor, TQ3 decompress path) run as
'dequantise to fp16 then SDPA' exactly like their own bench_all_methods_decode.py; KIVI-style int4 is OUR implementation (no public
fused kernel used) with the same dequant-then-SDPA path. SemKV/StarKV: no public code found/used, not benchmarked."""
import argparse, json, sys, torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("--uq-dir"); ap.add_argument("--amd-dir"); ap.add_argument("--out")
ap.add_argument("--B", type=int, default=8); ap.add_argument("--ctxs", type=int, nargs="+", default=[16384, 32768]); a = ap.parse_args()
sys.path.insert(0, a.uq_dir); sys.path.insert(0, a.amd_dir)
from attn import quantize_fp4, uq_decode
from turboquant_mi300x import TurboQuantMI300X
from tq_triton import turboquant_gqa_attention_fwd
from block_quant_rocm import make_quantizer
dev, D, HKV, HQ, PAGE = "cuda", 128, 8, 32, 64; G = HQ // HKV; SM = D ** -0.5; B = a.B
def tm(fn, reps=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps
rows = []
for ctx in a.ctxs:
    torch.manual_seed(0)
    k = torch.randn(B, HKV, ctx, D, device=dev, dtype=torch.float16); v = torch.randn_like(k)
    q = torch.randn(B, HKV, G, D, device=dev, dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q, k, v, scale=SM).float()
    def cos(o): return F.cosine_similarity(o.float().reshape(-1, D), ref.reshape(-1, D), dim=-1).mean().item()
    res = {}
    def add(name, fn, bytes_vec, outfn=None):
        ms = tm(fn); o = (outfn or fn)()
        res[name] = dict(ms=ms, bytes_per_vec=bytes_vec, cos=cos(o) if o is not None else None)
        rows.append(dict(B=B, ctx=ctx, Hq=HQ, method=name, K=1, **res[name])); print(json.dumps(rows[-1]), flush=True)
    add("fp16_sdpa", lambda: F.scaled_dot_product_attention(q, k, v, scale=SM), 256)
    # fp8 per-tensor (amd-experiments baseline format)
    def mk8(x):
        sc = 240.0 / x.float().abs().amax(); return (x.float() * sc).clamp(-240, 240).to(torch.float8_e4m3fnuz), sc
    k8, ks8 = mk8(k); v8, vs8 = mk8(v)
    add("fp8_pertensor_dequant_sdpa", lambda: F.scaled_dot_product_attention(q, (k8.to(torch.float16) / ks8.half()), (v8.to(torch.float16) / vs8.half()), scale=SM), 128)
    # int4 per-tensor (their baseline format), lean unpack
    def mk4(x):
        sc = 7.0 / x.float().abs().amax(); t = ((x.float() * sc).round().clamp(-7, 7) + 7).to(torch.uint8)
        return (t[..., 0::2] | (t[..., 1::2] << 4)).contiguous(), sc
    k4, ks4 = mk4(k); v4, vs4 = mk4(v)
    def un4(p, sc): return (torch.stack([(p & 15), (p >> 4)], -1).reshape(*p.shape[:-1], D).to(torch.float16) - 7) / sc.half()
    add("int4_pertensor_dequant_sdpa", lambda: F.scaled_dot_product_attention(q, un4(k4, ks4), un4(v4, vs4), scale=SM), 64)
    # KIVI-style (own): asym 4-bit group-32 with fp16 scale+min; K grouped along tokens per channel, V along channels per token
    def mkkivi(x, is_key):
        xf = x.float()
        xg = xf.reshape(B, HKV, ctx // 32, 32, D) if is_key else xf.reshape(B, HKV, ctx, D // 32, 32)
        dim = 3 if is_key else 4
        mn, mx = xg.amin(dim, keepdim=True), xg.amax(dim, keepdim=True); sc = ((mx - mn) / 15).clamp_min(1e-8)
        t = ((xg - mn) / sc).round().clamp(0, 15).to(torch.uint8).reshape(B, HKV, ctx, D)
        return (t[..., 0::2] | (t[..., 1::2] << 4)).contiguous(), sc.half(), mn.half()
    def unkivi(p, sc, mn, is_key):
        t = torch.stack([(p & 15), (p >> 4)], -1).reshape(B, HKV, ctx, D).to(torch.float16)
        tg = t.reshape(B, HKV, ctx // 32, 32, D) if is_key else t.reshape(B, HKV, ctx, D // 32, 32)
        return (tg * sc + mn).reshape(B, HKV, ctx, D)
    kk, ksc, kmn = mkkivi(k, True); vk, vsc, vmn = mkkivi(v, False)
    add("kivi4_own_dequant_sdpa", lambda: F.scaled_dot_product_attention(q, unkivi(kk, ksc, kmn, True), unkivi(vk, vsc, vmn, False), scale=SM), 64 + 8)
    del k8, v8, k4, v4, kk, vk
    # TQ3 (fused + decompress path) & block rotation methods (decompress path)
    tq = TurboQuantMI300X(bits=3, device=dev)
    def comp(x, fn): return fn(x.reshape(-1, D).float())
    kt, vt = comp(k, tq.compress_tensor), comp(v, tq.compress_tensor)
    kp = kt[:, 4:].contiguous().view(B, HKV, ctx, 48); kn = kt[:, :4].contiguous().view(-1).view(torch.float32).view(B, HKV, ctx)
    vp = vt[:, 4:].contiguous().view(B, HKV, ctx, 48); vn = vt[:, :4].contiguous().view(-1).view(torch.float32).view(B, HKV, ctx)
    qh = q.reshape(B, HQ, 1, D)
    add("tq3_fused", lambda: turboquant_gqa_attention_fwd(tq.rotate_queries(qh.float()).half(), kp, kn, vp, vn, gqa_ratio=G, rotation=tq.rotation, sm_scale=SM).reshape(B, HKV, G, D), 52)
    add("tq3_decompress_sdpa", lambda: F.scaled_dot_product_attention(q, tq.decompress_tensor(kt, (B, HKV, ctx, D)).half(), tq.decompress_tensor(vt, (B, HKV, ctx, D)).half(), scale=SM), 52)
    del kt, vt, kp, vp
    for m in ("planar", "iso", "rotor"):
        bq = make_quantizer(m, bits=3, head_dim=D, device=dev)
        kc_, vc_ = bq.compress(k.reshape(-1, D).float()), bq.compress(v.reshape(-1, D).float())
        add(f"{m}3_decompress_sdpa", lambda: F.scaled_dot_product_attention(q, bq.decompress(kc_, (B * HKV * ctx, D)).half().reshape(B, HKV, ctx, D), bq.decompress(vc_, (B * HKV * ctx, D)).half().reshape(B, HKV, ctx, D), scale=SM), 52)
        del kc_, vc_
    # ours
    npp = ctx // PAGE
    pg = lambda x: x.reshape(B, HKV, npp, PAGE, D).permute(0, 2, 1, 3, 4).reshape(B * npp, HKV, PAGE, D)
    c = [quantize_fp4(pg(x[:, :, :ctx])[i:i + 512].float()) for x in (k, v) for i in range(0, B * npp, 512)]
    n = (len(c) // 2); kc = torch.cat([x[0] for x in c[:n]]); ks = torch.cat([x[1] for x in c[:n]]); vc = torch.cat([x[0] for x in c[n:]]); vs = torch.cat([x[1] for x in c[n:]])
    bt = torch.arange(B * npp, device=dev, dtype=torch.int32).reshape(B, npp); seq = torch.full((B,), ctx, device=dev, dtype=torch.int32)
    qb = q.reshape(B, 1, HQ, D).bfloat16()
    add("uq_fp4_fused_K1", lambda: uq_decode(qb, kc, ks, vc, vs, bt, seq, nsplit=16).float().reshape(B, HKV, G, D), 68)
    del k, v, kc, ks, vc, vs; torch.cuda.empty_cache()
    json.dump(dict(device=torch.cuda.get_device_name(0), rows=rows, note="cos = attention-output cosine vs fp16 SDPA on the same iid-Gaussian KV; unfused methods are dequantise-then-SDPA"), open(a.out, "w"), indent=1)
