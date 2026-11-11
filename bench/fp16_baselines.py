"""Stronger fp16 decode-attention baselines on MI300X (same shapes as h2h_microbench): torch SDPA (flash path) vs bmm+softmax+bmm
(the formulation amd-experiments' own decode benchmark uses) vs split-KV via bmm. Rows: B, ctx, K, method, ms."""
import json, sys, torch, torch.nn.functional as F
out = sys.argv[1]
dev, D, HKV, G = "cuda", 128, 8, 4; SM = D ** -0.5
def tm(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps
def bmm_attn(q, k, v):   # q [B,Hkv,M,D], k/v [B,Hkv,S,D]
    s = torch.matmul(q, k.transpose(-1, -2)) * SM
    return torch.matmul(torch.softmax(s, -1, dtype=torch.float32).to(q.dtype), v)
def split_attn(q, k, v, ns):  # split-KV: fold splits into batch, merge with lse
    B, H, S, _ = k.shape; M = q.shape[2]
    kk = k.reshape(B, H, ns, S // ns, D); vv = v.reshape(B, H, ns, S // ns, D)
    s = torch.einsum("bhmd,bhnsd->bhnms", q, kk).float() * SM
    m = s.amax(-1, keepdim=True); p = torch.exp(s - m); l = p.sum(-1, keepdim=True)
    o = torch.einsum("bhnms,bhnsd->bhnmd", p.to(q.dtype), vv).float()
    gm = m.amax(2, keepdim=True); w = torch.exp(m - gm)
    return ((o * w).sum(2) / (l * w).sum(2)).to(q.dtype)
rows = []
for B in (1, 8, 32):
    for ctx in (4096, 16384, 32768, 65536):
        k = torch.randn(B, HKV, ctx, D, device=dev, dtype=torch.float16); v = torch.randn_like(k)
        for K in (1, 8):
            q = torch.randn(B, HKV, G * K, D, device=dev, dtype=torch.float16)
            r = dict(B=B, ctx=ctx, K=K, sdpa_ms=tm(lambda: F.scaled_dot_product_attention(q, k, v, scale=SM)), bmm_ms=tm(lambda: bmm_attn(q, k, v)))
            best = None
            for ns in (4, 8, 16, 32):
                if ctx % ns: continue
                t = tm(lambda: split_attn(q, k, v, ns)); best = min(best or (t, ns), (t, ns))
            r["splitkv_bmm_ms"], r["splitkv_ns"] = best
            r["best_fp16_ms"] = min(r["sdpa_ms"], r["bmm_ms"], r["splitkv_bmm_ms"])
            rows.append(r); print(json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}), flush=True)
        del k, v; torch.cuda.empty_cache()
json.dump(dict(device=torch.cuda.get_device_name(0), rows=rows), open(out, "w"), indent=1)
