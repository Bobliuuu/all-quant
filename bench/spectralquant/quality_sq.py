"""SpectralQuant (our Triton port) real-model quality: wikitext-2 ppl (same protocol as bench/quality/qeval.py: chunked prefill 256 +
token-by-token teacher-forced decode through the quantised cache) and per-layer attention-output cosine on real KV.
Calibration: WikiText-2 *train* (16 x 2048 tokens), post-RoPE K and V of every (layer, kv head), as in the package's calibrate path.
Usage: python quality_sq.py --model mistral7b|qwen3_8b --out results/spectralquant_quality_<model>.json [--nwin 8]"""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quality"))
import torch, torch.nn.functional as F
from transformers import DynamicCache
import sq_triton as sq
import qeval

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mistral7b"); ap.add_argument("--out", required=True)
ap.add_argument("--nwin", type=int, default=8); ap.add_argument("--ncal", type=int, default=16); ap.add_argument("--skip-fp16", action="store_true")
ap.add_argument("--opts", default="{}"); a = ap.parse_args(); opts = json.loads(a.opts)
model, tok = qeval.load(a.model); cfg = model.config
W = qeval.wiki_ids(tok); Wtr = qeval.wiki_ids(tok, "train")
NL = cfg.num_hidden_layers; H = cfg.num_key_value_heads


def kv_of(cache, li):
    if hasattr(cache, "layers"): return cache.layers[li].keys, cache.layers[li].values
    return cache.key_cache[li], cache.value_cache[li]


class RecCache(DynamicCache):
    def __init__(self): super().__init__(); self.rk = {}; self.rv = {}
    def update(self, k, v, li, kw=None):
        self.rk.setdefault(li, []).append(k.detach()); self.rv.setdefault(li, []).append(v.detach()); return super().update(k, v, li, kw)


t0 = time.time(); xs = Wtr[: a.ncal * 2048].view(a.ncal, 2048).cuda(); rk, rv = {}, {}
with torch.no_grad():
    for i in range(0, a.ncal, 4):
        c = RecCache(); model(xs[i:i + 4], past_key_values=c, use_cache=True)
        for li in range(NL):
            rk.setdefault(li, []).append(torch.cat(c.rk[li], 2)); rv.setdefault(li, []).append(torch.cat(c.rv[li], 2))
LP = []
for li in range(NL):
    kk = torch.cat(rk.pop(li), 0); vv = torch.cat(rv.pop(li), 0)   # [ncal,H,T,D]
    hk = [sq.fit_head(kk[:, h].reshape(-1, 128).float(), False, **opts) for h in range(H)]
    hv = [sq.fit_head(vv[:, h].reshape(-1, 128).float(), True, **opts) for h in range(H)]
    LP.append(sq.LayerParams(hk, hv)); del kk, vv
sq.unify_planes(LP)
dk = [sum(l.d_eff_k) / H for l in LP]; dv = [sum(l.d_eff_v) / H for l in LP]
bk = LP[0].bytes_per_vec(LP[0].planes_k); bvv = LP[0].bytes_per_vec(LP[0].planes_v)
ideal_k = sum(l.ideal_bytes(l.nb_k) for l in LP) / NL; ideal_v = sum(l.ideal_bytes(l.nb_v) for l in LP) / NL
print(f"calibrated in {time.time() - t0:.0f}s; mean d_eff K {sum(dk) / NL:.1f} V {sum(dv) / NL:.1f}; planes K {LP[0].planes_k} V {LP[0].planes_v}; bytes/vec K {bk} V {bvv} (ideal {ideal_k:.1f}/{ideal_v:.1f})", flush=True)


class SQCacheHF(DynamicCache):
    def update(self, k, v, li, kw=None):
        return super().update(sq.roundtrip(LP[li], k, True), sq.roundtrip(LP[li], v, False), li, kw)


class Meth:
    def __init__(self, c): self.c = c
    def make_cache(self): return self.c()
    def seq_fq(self, x, is_key, layer=-1): return sq.roundtrip(LP[layer], x, is_key) if self.c is SQCacheHF else x


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
    return dict(per_layer=rows, mean_cos=sum(r["cos"] for r in rows) / NL, mean_rel_err=sum(r["rel_err"] for r in rows) / NL)


res = dict(model=a.model, method="spectralquant_triton", opts=opts, d_eff_k_mean=sum(dk) / NL, d_eff_v_mean=sum(dv) / NL, planes_k=LP[0].planes_k, planes_v=LP[0].planes_v,
           bytes_per_vec=(bk + bvv) / 2, bytes_per_vec_k=bk, bytes_per_vec_v=bvv, ideal_bytes_k=ideal_k, ideal_bytes_v=ideal_v)
sqm, fpm = Meth(SQCacheHF), Meth(DynamicCache)
if not a.skip_fp16:
    res["ppl_fp16"] = qeval.ppl_run(model, W, fpm, nwin=a.nwin); print("fp16", res["ppl_fp16"]["ppl"], flush=True)
res["ppl_sq"] = qeval.ppl_run(model, W, sqm, nwin=a.nwin); print("sq", res["ppl_sq"]["ppl"], flush=True)
if "ppl_fp16" in res: res["ppl_delta"] = res["ppl_sq"]["ppl"] - res["ppl_fp16"]["ppl"]
res["attn"] = attn_error(sqm, W); print("attn cos", res["attn"]["mean_cos"], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
