"""KVQuant-4 real-model quality through the existing harness (bench/quality/qeval.py ppl_run protocol; same attention-cosine metric).
Cache path: HF cache update() receives post-RoPE K; we un-rotate (fp32, exact positions), quantise PRE-RoPE keys per channel with the offline-calibrated
(sc,zp) and per-layer nuq codebook, dequantise and re-rotate: numerically identical to storing pre-RoPE codes and rotating in the kernel.
Calibration: WikiText-2 train, 16 x 2048 tokens. --outlier PCT (diagnostic, quality only): K range from the PCT/2..1-PCT/2 per-channel quantiles and V per-token
range from per-token quantiles with the clipped elements kept in fp16 (sparse part of dense-and-sparse); 0 = dense only (the ported path).
Usage: python quality_kvq.py --model mistral7b|qwen3_8b --out results/kvquant_quality_<model>.json [--outlier 1.0] [--sink 1]"""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "quality")); sys.path.insert(0, os.path.join(HERE, ".."))
import torch, torch.nn.functional as F
from transformers import DynamicCache
import core, qeval

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mistral7b"); ap.add_argument("--out", required=True)
ap.add_argument("--nwin", type=int, default=8); ap.add_argument("--ncal", type=int, default=16); ap.add_argument("--skip-fp16", action="store_true")
ap.add_argument("--outlier", type=float, default=0.0); ap.add_argument("--sink", type=int, default=0, help="keep first N tokens fp16 (paper's attention-sink trick; not in the kernel)")
a = ap.parse_args()
model, tok = qeval.load(a.model); cfg = model.config
theta = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
print("rope theta", theta, flush=True)
def wiki_ids(split):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
W = wiki_ids("test"); Wtr = wiki_ids("train")
NL = cfg.num_hidden_layers; H = cfg.num_key_value_heads; D = 128


def kv_of(cache, li):
    if hasattr(cache, "layers"): return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


class RecCache(DynamicCache):
    def __init__(self): super().__init__(); self.rk = {}; self.rv = {}
    def update(self, k, v, li, kw=None):
        self.rk[li] = k.detach(); self.rv[li] = v.detach(); return super().update(k, v, li, kw)


t0 = time.time(); xs = Wtr[: a.ncal * 2048].view(a.ncal, 2048).cuda(); P = []
cs2, sn2 = core.rope_tables(2048, D, theta, "cuda")
rk, rv = {}, {}
with torch.no_grad():
    for i in range(0, a.ncal, 4):
        c = RecCache(); model(xs[i:i + 4], past_key_values=c, use_cache=True)
        for li in range(NL):
            rk.setdefault(li, []).append(core.rope(c.rk[li].float(), cs2, sn2, inverse=True)); rv.setdefault(li, []).append(c.rv[li].float())
        del c
for li in range(NL):
    kp = torch.cat(rk.pop(li), 0).permute(0, 2, 1, 3).reshape(-1, H, D)       # [N,H,D] pre-RoPE
    v = torch.cat(rv.pop(li), 0).permute(0, 2, 1, 3).reshape(-1, H, D)
    if a.outlier > 0: zp, sc = core.k_range_outlier(kp, a.outlier)
    else: zp, sc = core.fit_k(kp)
    cbk = core.nuq_fit(((kp - zp) / sc).clamp(-1, 1))
    mn, mx = v.amin(-1, keepdim=True), v.amax(-1, keepdim=True)
    cbv = core.nuq_fit(((v - (mx + mn) / 2) / ((mx - mn) / 2).clamp_min(1e-6)).clamp(-1, 1))
    P.append(dict(zp=zp, sc=sc, cbk=cbk, cbv=cbv)); del kp, v
print(f"calibrated in {time.time() - t0:.0f}s", flush=True)


def fq_k(k, li, pos0):    # k [B,H,T,D] post-RoPE fp16 -> post-RoPE fp16 after nuq4 round trip
    p = P[li]; T = k.shape[2]; cs, sn = core.rope_tables(T, D, theta, k.device, pos0)
    kp = core.rope(k.float(), cs, sn, inverse=True)
    if a.outlier > 0:   # sparse part: values outside the calibrated per-channel range are kept exactly
        n = (kp - p["zp"][None, :, None]) / p["sc"][None, :, None]; out = n.abs() > 1
        kq = core.dequant_k(core.quant_k(kp, p["zp"], p["sc"], p["cbk"]), p["zp"], p["sc"], p["cbk"]); kq = torch.where(out, kp, kq)
    else: kq = core.dequant_k(core.quant_k(kp, p["zp"], p["sc"], p["cbk"]), p["zp"], p["sc"], p["cbk"])
    return core.rope(kq, cs, sn).to(k.dtype)


def fq_v(v, li):
    p = P[li]; vf = v.float()
    if a.outlier > 0:   # per-token range from quantiles; outliers kept fp16
        lo = torch.quantile(vf, a.outlier / 200, dim=-1, keepdim=True); hi = torch.quantile(vf, 1 - a.outlier / 200, dim=-1, keepdim=True)
        zp, sc = ((hi + lo) / 2).half().float(), ((hi - lo) / 2).clamp_min(1e-6).half().float()
        n = (vf - zp) / sc; out = n.abs() > 1
        vq = torch.where(out, vf, core.cb_gather(p["cbv"], n) * sc + zp)
    else:
        c, sc, zp = core.quant_v(vf, p["cbv"]); vq = core.dequant_v(c, sc, zp, p["cbv"])
    return vq.to(v.dtype)


class KVQCache(DynamicCache):
    def update(self, k, v, li, kw=None):
        pos0 = self.get_seq_length(li); kq, vq = fq_k(k, li, pos0), fq_v(v, li)
        if a.sink and pos0 == 0: kq[:, :, :a.sink] = k[:, :, :a.sink]; vq[:, :, :a.sink] = v[:, :, :a.sink]
        return super().update(kq, vq, li, kw)


class Meth:
    def __init__(self, c): self.c = c
    def make_cache(self): return self.c()
    def seq_fq(self, x, is_key, layer=-1):
        if self.c is not KVQCache: return x
        y = fq_k(x, layer, 0) if is_key else fq_v(x, layer)
        if a.sink: y[:, :, :a.sink] = x[:, :, :a.sink]
        return y


@torch.no_grad()
def attn_error(method, ids, T=2048, nq=128):
    from transformers.models.llama import modeling_llama as ml
    caps = {}; hooks = []
    def mk(li):
        def hook(mod, args, kwargs, out):
            h = args[0] if args else kwargs["hidden_states"]; B, L, _ = h.shape; hd = getattr(mod, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            q = mod.q_proj(h).view(B, L, -1, hd)
            if hasattr(mod, "q_norm"): q = mod.q_norm(q)
            q = q.transpose(1, 2); cos, sin = kwargs["position_embeddings"]
            caps[li] = ((q * cos.unsqueeze(1)) + (ml.rotate_half(q) * sin.unsqueeze(1)))[:, :, -nq:].float()
        return hook
    for li, layer in enumerate(model.model.layers): hooks.append(layer.self_attn.register_forward_hook(mk(li), with_kwargs=True))
    cache = DynamicCache(); model(ids[:T].unsqueeze(0).cuda(), past_key_values=cache, use_cache=True)
    for h in hooks: h.remove()
    rows = []
    for li in range(NL):
        q = caps[li]; Kc, Vc = kv_of(cache, li); K, V = Kc.float(), Vc.float()
        Kq, Vq = method.seq_fq(Kc, True, li).float(), method.seq_fq(Vc, False, li).float()
        g = q.shape[1] // K.shape[1]; sc = 1 / math.sqrt(q.shape[-1]); mask = torch.ones(nq, T, device="cuda", dtype=torch.bool).tril(T - nq)
        def att(Kx, Vx):
            s = (q @ Kx.repeat_interleave(g, 1).transpose(-1, -2) * sc).masked_fill(~mask, -1e30); return s.softmax(-1) @ Vx.repeat_interleave(g, 1)
        o, oq = att(K, V), att(Kq, Vq)
        rows.append(dict(layer=li, cos=F.cosine_similarity(o.flatten(0, 2), oq.flatten(0, 2), dim=-1).mean().item(), rel_err=((o - oq).norm() / o.norm()).item(),
                         k_rel=((K - Kq).norm() / K.norm()).item(), v_rel=((V - Vq).norm() / V.norm()).item()))
    return dict(per_layer=rows, mean_cos=sum(r["cos"] for r in rows) / NL, mean_rel_err=sum(r["rel_err"] for r in rows) / NL,
                mean_k_rel=sum(r["k_rel"] for r in rows) / NL, mean_v_rel=sum(r["v_rel"] for r in rows) / NL)


res = dict(model=a.model, method="kvquant_nuq4", outlier_pct=a.outlier, sink=a.sink, theta=theta)
km, fpm = Meth(KVQCache), Meth(DynamicCache)
if not a.skip_fp16:
    res["ppl_fp16"] = qeval.ppl_run(model, W, fpm, nwin=a.nwin); print("fp16", res["ppl_fp16"]["ppl"], flush=True)
res["ppl_kvq"] = qeval.ppl_run(model, W, km, nwin=a.nwin); print("kvq", res["ppl_kvq"]["ppl"], flush=True)
if "ppl_fp16" in res: res["ppl_delta"] = res["ppl_kvq"]["ppl"] - res["ppl_fp16"]["ppl"]
res["attn"] = attn_error(km, W); print("attn cos", res["attn"]["mean_cos"], "k_rel", res["attn"]["mean_k_rel"], "v_rel", res["attn"]["mean_v_rel"], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
