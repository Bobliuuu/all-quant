"""Task B: real-KV accuracy on Mistral-7B-v0.1.
  mode=attn : prefill N wikitext tokens, per-layer attention-output cosine/rel-L2 of FP4 fused kernel (K=1,2,4,8 last tokens
              as the speculative block) and TQ3 (K=1) vs fp16 reference attention on the same real K/V and real Q.
  mode=ppl  : perplexity with simulated (quantise-dequantise) KV: fp16 vs FP4/UE8M0 vs TQ3 (rotation 3-bit).
"""
import argparse, json, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["attn", "ppl"], required=True)
ap.add_argument("--uq-dir", required=True); ap.add_argument("--amd-dir", required=True)
ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
ap.add_argument("--ntok", type=int, default=8192)
ap.add_argument("--windows", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.uq_dir); sys.path.insert(0, a.amd_dir)
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral import modeling_mistral as mm
from attn import quantize_fp4, dequantize_fp4, uq_decode
from turboquant_mi300x import TurboQuantMI300X
from datasets import load_dataset

dev, D, PAGE = "cuda", 128, 64
tok = AutoTokenizer.from_pretrained(a.model)
from transformers import AutoConfig
_cfg = AutoConfig.from_pretrained(a.model); _cfg.sliding_window = None  # Mistral-v0.1 SWA(4096) off: full-attention KV at long ctx, same as the decode loops here
model = AutoModelForCausalLM.from_pretrained(a.model, config=_cfg, dtype=torch.float16, device_map={"": 0}); model.eval()
cfg = model.config; Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads; G = Hq // Hkv; SM = D ** -0.5
tq = TurboQuantMI300X(bits=3, device=dev)
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids_all = tok(text, return_tensors="pt").input_ids[0]
print("tokens in wikitext-2 test:", ids_all.numel(), flush=True)


def fq_fp4(x):  # [..., D] fp16 -> fp16 dequantised
    sh = x.shape; xf = x.reshape(-1, D)
    out = torch.empty_like(xf)
    for i in range(0, xf.shape[0], 1 << 20):
        p, s = quantize_fp4(xf[i:i + (1 << 20)]); out[i:i + (1 << 20)] = dequantize_fp4(p, s).to(x.dtype)
    return out.reshape(sh)


def fq_tq3(x):
    sh = x.shape; xf = x.reshape(-1, D)
    out = torch.empty_like(xf)
    for i in range(0, xf.shape[0], 1 << 20):
        c = tq.compress_tensor(xf[i:i + (1 << 20)].float()); out[i:i + (1 << 20)] = tq.decompress_tensor(c, (c.shape[0], D)).to(x.dtype)
    return out.reshape(sh)


from block_quant_rocm import make_quantizer
_bq = {m: make_quantizer(m, bits=3, head_dim=D, device=dev) for m in ("planar", "iso", "rotor")}
def fq_block(x, m):
    sh = x.shape; xf = x.reshape(-1, D); out = torch.empty_like(xf)
    for i in range(0, xf.shape[0], 1 << 20):
        c = xf[i:i + (1 << 20)].float(); out[i:i + (1 << 20)] = _bq[m].decompress(_bq[m].compress(c), c.shape).to(x.dtype)
    return out.reshape(sh)
def fq_fp8_pt(x, mx=240.0):  # amd-experiments fp8 baseline format (per-tensor absmax, e4m3fnuz). NOTE their script scales to 448, but e4m3fnuz max is 240 -> NaN; fixed here.
    amax = x.float().abs().amax().clamp(min=1e-12); sc = mx / amax
    return ((x.float() * sc).clamp(-mx, mx).to(torch.float8_e4m3fnuz).float() / sc).to(x.dtype)
def fq_fp8_tok(x):  # fp8 e4m3fnuz with a per-vector (token,head) absmax scale: the fair fp8 KV baseline
    amax = x.float().abs().amax(-1, keepdim=True).clamp(min=1e-12); sc = 240.0 / amax
    return ((x.float() * sc).to(torch.float8_e4m3fnuz).float() / sc).to(x.dtype)
def fq_int4_pt(x):  # amd-experiments baseline: per-tensor symmetric int4
    amax = x.float().abs().amax().clamp(min=1e-12); sc = 7.0 / amax
    return ((x.float() * sc).round().clamp(-7, 7) / sc).to(x.dtype)
def _asym4(xg, dim):  # asymmetric 4-bit over `dim` groups
    mn, mx = xg.amin(dim, keepdim=True), xg.amax(dim, keepdim=True); sc = ((mx - mn) / 15).clamp_min(1e-8)
    return ((xg - mn) / sc).round().clamp(0, 15) * sc + mn
def fq_kivi4(x, is_key):  # KIVI-style: K per-channel over groups of 32 tokens, V per-token over groups of 32 channels (own impl.)
    xf = x.float(); B_, H_, S_, D_ = xf.shape
    if is_key: return _asym4(xf.reshape(B_, H_, S_ // 32, 32, D_), 3).reshape(B_, H_, S_, D_).to(x.dtype)
    return _asym4(xf.reshape(B_, H_, S_, D_ // 32, 32), 4).reshape(B_, H_, S_, D_).to(x.dtype)
OTHERS = {"fp4_ue8m0_sim": lambda x, k: fq_fp4(x), "tq3_sim": lambda x, k: fq_tq3(x), "fp8_pertensor_fixed240": lambda x, k: fq_fp8_pt(x), "fp8_pertoken": lambda x, k: fq_fp8_tok(x),
          "int4_pertensor": lambda x, k: fq_int4_pt(x), "kivi4_own": fq_kivi4,
          **{m + "3": (lambda x, k, m=m: fq_block(x, m)) for m in ("planar", "iso", "rotor")}}

_g = torch.randn(1, 1, 4096, D, device=dev, dtype=torch.float16)
print("sanity: reconstruction cosine on iid Gaussian vectors (their reported 0.983 regime):",
      {nm: round(F.cosine_similarity(fn(_g, True).float().reshape(-1, D), _g.float().reshape(-1, D), dim=-1).mean().item(), 4) for nm, fn in OTHERS.items()}, flush=True)

if a.mode == "attn":
    ids = ids_all[: a.ntok][None].to(dev); S = ids.shape[1]
    qcap = {}
    def mk(l):
        def hook(mod, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            hs = h[:, -8:]
            q = mod.q_proj(hs).view(1, 8, Hq, D).transpose(1, 2)
            k = mod.k_proj(hs).view(1, 8, Hkv, D).transpose(1, 2)
            q, _ = mm.apply_rotary_pos_emb(q, k, cos[:, -8:], sin[:, -8:])
            qcap[l] = q[0].transpose(0, 1).contiguous()  # [8,Hq,D]
        return hook
    hs_ = [layer.self_attn.register_forward_pre_hook(mk(l), with_kwargs=True) for l, layer in enumerate(model.model.layers)]
    with torch.no_grad(): out = model(ids, use_cache=True)
    for h in hs_: h.remove()
    cache = out.past_key_values
    getkv = (lambda l: (cache.layers[l].keys, cache.layers[l].values)) if hasattr(cache, "layers") else (lambda l: cache[l])
    npp = S // PAGE; Sp = npp * PAGE
    bt = torch.arange(npp, device=dev, dtype=torch.int32)[None]
    seq = torch.tensor([Sp], device=dev, dtype=torch.int32)

    def cos_rel(o, r):
        o, r = o.float().reshape(-1, D), r.float().reshape(-1, D)
        return F.cosine_similarity(o, r, dim=-1).mean().item(), ((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-12)).mean().item()

    layers = []
    for l in range(cfg.num_hidden_layers):
        k, v = getkv(l); k, v = k[:, :, :Sp].contiguous(), v[:, :, :Sp].contiguous()   # [1,Hkv,Sp,D]
        pg = lambda x: x.reshape(1, Hkv, npp, PAGE, D).permute(0, 2, 1, 3, 4).reshape(npp, Hkv, PAGE, D).contiguous()
        kc, ks = quantize_fp4(pg(k).float()); vc, vs = quantize_fp4(pg(v).float())
        kq, vq = fq_fp4(k), fq_fp4(v)
        res = {"layer": l, "k_absmax": k.float().abs().max().item(), "kv_rel_err_fp4": ((kq.float() - k.float()).norm() / k.float().norm()).item()}
        for K in (1, 2, 4, 8):
            qb = qcap[l][-K:].to(torch.bfloat16)                       # [K,Hq,D]
            o = uq_decode(qb[None].contiguous(), kc, ks, vc, vs, bt, seq, nsplit=16)[0]   # [K,Hq,D]
            # fp32 reference with unquantised fp16 K/V, same bf16-rounded q, causal over the K block
            qq = qb.float().permute(1, 0, 2).reshape(Hkv, G, K, D)
            sc = torch.einsum("hgkd,hsd->hgks", qq, k[0].float()) * SM
            lim = Sp - K + 1 + torch.arange(K, device=dev)
            sc = sc.masked_fill((torch.arange(Sp, device=dev)[None] >= lim[:, None])[None, None], float("-inf"))
            r = torch.einsum("hgks,hsd->hgkd", sc.softmax(-1), v[0].float()).reshape(Hq, K, D).permute(1, 0, 2)
            res[f"uq_K{K}"] = dict(zip(("cos", "rel_l2"), cos_rel(o, r)))
            if K == 1:
                kp = tq.compress_tensor(k.float().reshape(-1, D)); vp = tq.compress_tensor(v.float().reshape(-1, D))
                from tq_triton import turboquant_gqa_attention_fwd, compress_kv_for_triton
                kP, kN, vP, vN = compress_kv_for_triton(k, v, tq)
                qh = qb.half().permute(1, 0, 2)[None]                   # [1,Hq,1,D]
                ot = turboquant_gqa_attention_fwd(tq.rotate_queries(qh.float()).half(), kP, kN, vP, vN, gqa_ratio=G, rotation=tq.rotation, sm_scale=SM)
                res["tq3_K1"] = dict(zip(("cos", "rel_l2"), cos_rel(ot[0].permute(1, 0, 2), r)))
                # also: fake-quant K/V then fp32 attention (isolates kernel arithmetic from quantisation) for FP4
                sc2 = torch.einsum("hgkd,hsd->hgks", qq, kq[0].float()) * SM
                r2 = torch.einsum("hgks,hsd->hgkd", sc2.softmax(-1), vq[0].float()).reshape(Hq, K, D).permute(1, 0, 2)
                res["uq_kernel_vs_fp4_simulated_K1"] = dict(zip(("cos", "rel_l2"), cos_rel(o, r2)))
            if K == 1:
                res["others_K1"] = {}
                for nm, fn in OTHERS.items():
                    kk, vv = fn(k, True), fn(v, False)
                    sc3 = torch.einsum("hgkd,hsd->hgks", qq, kk[0].float()) * SM
                    r3 = torch.einsum("hgks,hsd->hgkd", sc3.softmax(-1), vv[0].float()).reshape(Hq, K, D).permute(1, 0, 2)
                    res["others_K1"][nm] = dict(zip(("cos", "rel_l2"), cos_rel(r3, r)))
                    res.setdefault("recon_cos", {})[nm] = dict(K=F.cosine_similarity(kk.float().reshape(-1, D), k.float().reshape(-1, D), dim=-1).mean().item(),
                                                                V=F.cosine_similarity(vv.float().reshape(-1, D), v.float().reshape(-1, D), dim=-1).mean().item())
        layers.append(res); print(json.dumps({k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in res.items() if k_ in ("layer", "uq_K1", "uq_K8", "tq3_K1")}), flush=True)
    json.dump(dict(model=a.model, ntok=Sp, device=torch.cuda.get_device_name(0), layers=layers), open(a.out, "w"), indent=1)

else:  # ppl
    mode_holder = {"m": "fp16"}
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers import AttentionInterface
    def fq_attn(module, query, key, value, attention_mask, **kw):
        m = mode_holder["m"]
        if m == "fp4": key, value = fq_fp4(key), fq_fp4(value)
        elif m == "tq3": key, value = fq_tq3(key), fq_tq3(value)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    AttentionInterface.register("fq", fq_attn)
    model.config._attn_implementation = "fq"
    W = 4096; res = {m: [] for m in ("fp16", "fp4", "tq3")}
    for w in range(a.windows):
        ids = ids_all[w * W:(w + 1) * W][None].to(dev)
        if ids.shape[1] < W: break
        for m in res:
            mode_holder["m"] = m
            with torch.no_grad():
                lg = model(ids).logits[0, :-1].float()
            res[m].append(F.cross_entropy(lg, ids[0, 1:]).item())
        print(w, {m: round(v[-1], 4) for m, v in res.items()}, flush=True)
    summ = {m: dict(mean_nll=sum(v) / len(v), ppl=2.718281828 ** (sum(v) / len(v))) for m, v in res.items()}
    print(summ)
    json.dump(dict(model=a.model, window=W, windows=len(res["fp16"]), per_window_nll=res, summary=summ, device=torch.cuda.get_device_name(0)), open(a.out, "w"), indent=1)
