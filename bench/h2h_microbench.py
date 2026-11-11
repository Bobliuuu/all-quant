"""Head-to-head microbench: UltraQuant FP4 fused K-query paged decode vs amd-experiments fused TQ3 GQA attention
vs fp16 SDPA, same box, same timing method (CUDA events, warmup + mean of reps).

Needs on sys.path: pinned uqmt/attn.py (as attn) and amd-experiments kernels/ (tq_triton, turboquant_mi300x).
Run: python h2h_microbench.py --uq-dir DIR --amd-dir DIR --out results/h2h_raw.json
"""
import argparse, json, sys, time, gc
ap = argparse.ArgumentParser()
ap.add_argument("--uq-dir", required=True); ap.add_argument("--amd-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
ap.add_argument("--ctxs", type=int, nargs="+", default=[4096, 16384, 32768, 65536])
ap.add_argument("--hqs", type=int, nargs="+", default=[32, 64])
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
if a.out.endswith('h2h_raw.json'): sys.exit('superseded by chunked runs (run_h2h_chunked.sh)')
sys.path.insert(0, a.uq_dir); sys.path.insert(0, a.amd_dir)
import torch, torch.nn.functional as F
from attn import quantize_fp4, dequantize_fp4, uq_decode
from turboquant_mi300x import TurboQuantMI300X
from tq_triton import turboquant_gqa_attention_fwd

dev, D, HKV, PAGE = "cuda", 128, 8, 64
SM = D ** -0.5
tq = TurboQuantMI300X(bits=3, device=dev)
B_FP16, B_TQ3, B_UQ = D * 2, 52, D // 2 + D // 32   # bytes per K-or-V vector


def tmeter(fn, reps):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def chunked_fp4(x, chunk=512):  # x [npages,Hkv,PAGE,D] fp16
    c, s = [], []
    for i in range(0, x.shape[0], chunk):
        p, sc = quantize_fp4(x[i:i + chunk].float()); c.append(p); s.append(sc)
    return torch.cat(c), torch.cat(s)


def chunked_tq3(x, chunk=1 << 21):  # x [B,H,S,D] fp16 -> planes [B,H,S,48], norms [B,H,S]
    B, H, S, _ = x.shape
    flat = x.reshape(-1, D)
    outs = []
    for i in range(0, flat.shape[0], chunk):
        outs.append(tq.compress_tensor(flat[i:i + chunk].float()))
    comp = torch.cat(outs)
    norms = comp[:, :4].contiguous().view(-1).view(torch.float32).view(B, H, S)
    planes = comp[:, 4:].contiguous().view(B, H, S, 48)
    return planes, norms


def ref_attn(q, k, v, causalK):
    """fp32 reference. q [Hq,K,D], k/v [Hkv,S,D] (fp16, unquantised). query j sees S-K+j+1 positions if causalK."""
    Hq, K, _ = q.shape; G = Hq // HKV; S = k.shape[1]
    qq = q.float().reshape(HKV, G, K, D)
    sc = torch.einsum("hgkd,hsd->hgks", qq, k.float()) * SM
    if causalK:
        lim = (S - K + 1 + torch.arange(K, device=q.device))
        mask = torch.arange(S, device=q.device)[None, :] >= lim[:, None]
        sc = sc.masked_fill(mask[None, None], float("-inf"))
    o = torch.einsum("hgks,hsd->hgkd", sc.softmax(-1), v.float())
    return o.reshape(Hq, K, D)


def cos_rel(o, r):  # o,r [Hq,K,D] -> mean cosine and mean rel-L2 over (head,query)
    o, r = o.float().reshape(-1, D), r.float().reshape(-1, D)
    cs = F.cosine_similarity(o, r, dim=-1).mean().item()
    rel = ((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-12)).mean().item()
    return cs, rel


rows, acc = [], []
def save():
    json.dump({"device": torch.cuda.get_device_name(0), "bytes_per_vec": dict(fp16=B_FP16, tq3=B_TQ3, uq_fp4=B_UQ),
               "rows": rows, "accuracy": acc}, open(a.out, "w"), indent=1)

for B in a.batches:
    for ctx in a.ctxs:
        torch.manual_seed(0)
        npp = ctx // PAGE; npages = B * npp
        print(f"== B={B} ctx={ctx}: build", flush=True); t0 = time.time()
        kf = torch.randn(B, HKV, ctx, D, device=dev, dtype=torch.float16)
        vf = torch.randn(B, HKV, ctx, D, device=dev, dtype=torch.float16)
        pg = lambda x: x.reshape(B, HKV, npp, PAGE, D).permute(0, 2, 1, 3, 4).reshape(npages, HKV, PAGE, D)
        kc, ks = chunked_fp4(pg(kf)); vc, vs = chunked_fp4(pg(vf))
        bt = torch.randperm(npages, device=dev, dtype=torch.int32)
        # permute physical pages: logical page i lives at bt-permuted slot; build tables consistently
        inv = torch.empty_like(bt); inv[bt.long()] = torch.arange(npages, device=dev, dtype=torch.int32)
        kc, ks, vc, vs = kc[bt.long()], ks[bt.long()], vc[bt.long()], vs[bt.long()]  # physical slot s holds logical bt[s]
        btab = inv.reshape(B, npp)                      # logical (b,i) -> physical slot
        seq = torch.full((B,), ctx, device=dev, dtype=torch.int32)
        kp, kn = chunked_tq3(kf); vp, vn = chunked_tq3(vf)
        print(f"   built in {time.time()-t0:.0f}s  mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

        # ---------------- accuracy (b=0 slice) ----------------
        if True:
            for regime in ("iid", "peaked"):
                for Hq in (32,):
                    G = Hq // HKV
                    for K in (1, 8):
                        g = torch.Generator(device=dev).manual_seed(1)
                        q0 = torch.randn(Hq, K, D, device=dev, generator=g, dtype=torch.float32)
                        if regime == "peaked":  # plant: each query = k of a random earlier token + noise (softmax concentrates)
                            idx = torch.randint(0, ctx - K, (HKV, G, K), device=dev, generator=g)
                            kk = kf[0][torch.arange(HKV, device=dev)[:, None, None], idx].float()  # [Hkv,G,K,D]
                            q0 = (kk + 0.3 * q0.reshape(HKV, G, K, D)).reshape(Hq, K, D)
                        q0 = q0.to(torch.bfloat16)
                        r = ref_attn(q0, kf[0], vf[0], causalK=True)
                        o_uq = uq_decode(q0.permute(1, 0, 2).unsqueeze(0).contiguous(), kc, ks, vc, vs, btab[:1], seq[:1], nsplit=16)[0]
                        o_uq = o_uq.permute(1, 0, 2)
                        c, rl = cos_rel(o_uq, r)
                        acc.append(dict(B=B, ctx=ctx, regime=regime, Hq=Hq, K=K, method="uq_fp4", cos=c, rel_l2=rl))
                        o_f = F.scaled_dot_product_attention(q0.reshape(HKV, G * K, D).unsqueeze(0).half(), kf[:1], vf[:1], scale=SM)
                        c, rl = cos_rel(o_f.reshape(Hq, K, D), r)
                        acc.append(dict(B=B, ctx=ctx, regime=regime, Hq=Hq, K=K, method="fp16_sdpa", cos=c, rel_l2=rl)) if K == 1 else None
                        if K == 1:   # theirs: K=1 only (non-causal == causal at K=1)
                            qh = q0.unsqueeze(0).half()  # [1,Hq,1,D]
                            qr = tq.rotate_queries(qh.float()).half()
                            o_t = turboquant_gqa_attention_fwd(qr, kp[:1], kn[:1], vp[:1], vn[:1], gqa_ratio=G, rotation=tq.rotation, sm_scale=SM)
                            c, rl = cos_rel(o_t[0], r)
                            acc.append(dict(B=B, ctx=ctx, regime=regime, Hq=Hq, K=1, method="tq3_gqa", cos=c, rel_l2=rl))
            save()
        # ---------------- timing ----------------
        for Hq in a.hqs:
            G = Hq // HKV
            for K in a.ks:
                q = torch.randn(B, K, Hq, D, device=dev, dtype=torch.bfloat16)
                q1 = torch.randn(B, 1, Hq, D, device=dev, dtype=torch.bfloat16)
                reps = a.reps if B * ctx <= 8 * 32768 else max(8, a.reps // 3)
                base = dict(B=B, ctx=ctx, Hq=Hq, Hkv=HKV, K=K)
                kv_tokens = B * ctx * HKV
                def row(method, ms, bytes_vec, calls=1, **kw):
                    gb = kv_tokens * 2 * bytes_vec / 1e9  # one pass over K+V
                    r = dict(base, method=method, ms=ms, kv_bytes_per_vec=bytes_vec, GBps_single_pass=gb / (ms / 1e3 / calls), **kw)
                    rows.append(r); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
                # fp16 SDPA: one call with M=G*K rows per kv head (non-causal; timing-equivalent), and K independent K=1 calls
                qs = lambda KK: torch.randn(B, HKV, G * KK, D, device=dev, dtype=torch.float16)
                qf, qf1 = qs(K), qs(1)
                row("fp16_sdpa_fused", tmeter(lambda: F.scaled_dot_product_attention(qf, kf, vf, scale=SM), reps), B_FP16)
                if K > 1:
                    def f_ind():
                        for _ in range(K): F.scaled_dot_product_attention(qf1, kf, vf, scale=SM)
                    row("fp16_sdpa_Kindep", tmeter(f_ind, reps), B_FP16, calls=K)
                # theirs (TQ3)
                qh1 = torch.randn(B, Hq, 1, D, device=dev, dtype=torch.float16)
                qr1 = tq.rotate_queries(qh1.float()).half()
                t_k = lambda: turboquant_gqa_attention_fwd(qr1, kp, kn, vp, vn, gqa_ratio=G, rotation=tq.rotation, sm_scale=SM)
                t_full = lambda: turboquant_gqa_attention_fwd(tq.rotate_queries(qh1.float()).half(), kp, kn, vp, vn, gqa_ratio=G, rotation=tq.rotation, sm_scale=SM)
                try:
                    if K == 1:
                        row("tq3_gqa_K1_incl_outrot", tmeter(t_k, reps), B_TQ3)
                        row("tq3_gqa_K1_incl_qrot_outrot", tmeter(t_full, reps), B_TQ3)
                    else:
                        def t_ind():
                            for _ in range(K): t_full()
                        row("tq3_gqa_Kindep", tmeter(t_ind, reps), B_TQ3, calls=K)
                        qhK = torch.randn(B, Hq, K, D, device=dev, dtype=torch.float16)
                        tK = lambda: turboquant_gqa_attention_fwd(qhK, kp, kn, vp, vn, gqa_ratio=G, rotation=tq.rotation, sm_scale=SM)
                        try: row("tq3_gqa_SqK_noncausal", tmeter(tK, reps), B_TQ3, note="their kernel with S_q=K, no causal mask, q not rotated: timing only")
                        except Exception as ex: print("tq3 SqK err", str(ex)[:100], flush=True)
                except Exception as ex:
                    print("tq3 err", B, ctx, Hq, K, str(ex)[:200], flush=True)
                # ours
                best = None
                for ns in (8, 16, 32, 64):
                    if ns > npp: continue
                    try: ms = tmeter(lambda: uq_decode(q, kc, ks, vc, vs, btab, seq, nsplit=ns), 10)
                    except Exception as ex: print("uq err", ns, str(ex)[:100]); continue
                    if best is None or ms < best[0]: best = (ms, ns)
                ns = best[1]
                row("uq_fp4_fused", tmeter(lambda: uq_decode(q, kc, ks, vc, vs, btab, seq, nsplit=ns), reps), B_UQ, nsplit=ns)
                if K > 1:
                    row("uq_fp4_Kindep", tmeter(lambda: uq_decode(q, kc, ks, vc, vs, btab, seq, fused=False, nsplit=ns), reps), B_UQ, calls=K, nsplit=ns)
                save()
        del kf, vf, kc, ks, vc, vs, kp, kn, vp, vn; gc.collect(); torch.cuda.empty_cache()
save(); print("DONE", flush=True)
