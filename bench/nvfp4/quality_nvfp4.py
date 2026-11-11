"""Real-model quality THROUGH the fused kernels: wikitext-2 perplexity (chunked prefill + token-by-token decode;
decode attention for every layer runs uq_decode_nvfp4 / attn.uq_decode on the quantised paged cache) and per-layer
attention-output error of K=8 verify queries on real KV.  Run from repo root:
  python bench/nvfp4/quality_nvfp4.py --model qwen3_8b --out results/nvfp4_quality_qwen3_8b.json
Prefill chunks (L>16 queries) read the dequantised cache with SDPA (kernel supports K<=16 queries);
every cached token, prefill or decode, is stored quantised in the paged FP4 cache.
Per-layer per-tensor scales (NVFP4): ts = 2*amax/(6*448) from the first prefill chunk (static, 2x headroom)."""
import argparse, json, math, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench", "quality"))
import torch, torch.nn.functional as F
from transformers import DynamicCache
from transformers.modeling_utils import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from uqmt import attn as A, nvfp4 as N
import qeval, kvq

PAGE = 64
ACTIVE = [None]


class KCache(DynamicCache):
    def __init__(self, mode):
        super().__init__(); self.mode = mode; self.st = {}; ACTIVE[0] = self

    def get_seq_length(self, layer_idx=0):
        return self.st[layer_idx]["len"] if layer_idx in self.st else 0

    def _alloc(self, B, H, D, T=2048):
        npg = T // PAGE; z = lambda x: torch.zeros(B, npg, H, PAGE, x, dtype=torch.uint8, device="cuda")
        sg = D // 16 if self.mode == "nvfp4" else D // 32
        return dict(kc=z(D // 2), vc=z(D // 2), ks=z(sg), vs=z(sg), len=0, npg=npg, ts_k=1.0, ts_v=1.0, B=B, H=H, D=D,
                    bt=torch.arange(B * npg, dtype=torch.int32, device="cuda").reshape(B, npg))

    def _q(self, x, s, which):
        if self.mode == "nvfp4":
            if "ts_" + which + "_set" not in s:
                s["ts_" + which] = 2 * float(x.float().abs().max()) / (6 * 448); s["ts_" + which + "_set"] = 1
            c, sc, _ = N.quantize_nvfp4_bytes(x.float(), s["ts_" + which]); return c, sc
        return A.quantize_fp4(x.float())

    def update(self, k, v, layer_idx, cache_kwargs=None):
        B, H, L, D = k.shape
        if layer_idx not in self.st: self.st[layer_idx] = self._alloc(B, H, D)
        s = self.st[layer_idx]; t0 = s["len"]; pos = torch.arange(t0, t0 + L, device="cuda")
        for x, cn, sn, w in ((k, "kc", "ks", "k"), (v, "vc", "vs", "v")):
            c, sc = self._q(x, s, w)
            s[cn][:, pos // PAGE, :, pos % PAGE] = c.permute(2, 0, 1, 3)
            s[sn][:, pos // PAGE, :, pos % PAGE] = sc.permute(2, 0, 1, 3)
        s["len"] = t0 + L; T = s["len"]
        if L <= 16: return k, v          # decode: kernel reads the paged cache, these are unused
        deq = (lambda p, sb, ts: N.dequantize_nvfp4_bytes(p, sb, ts)) if self.mode == "nvfp4" else (lambda p, sb, ts: A.dequantize_fp4(p, sb))
        def full(c, sc, ts):
            x = deq(c, sc, ts)  # [B,npg,H,PAGE,D]
            return x.permute(0, 2, 1, 3, 4).reshape(B, H, -1, D)[:, :, :T].to(k.dtype)
        return full(s["kc"], s["ks"], s["ts_k"]), full(s["vc"], s["vs"], s["ts_v"])


def kernel_attn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    c = ACTIVE[0]
    if c is None or query.shape[2] > 16 or module.layer_idx not in c.st:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kw)
    s = c.st[module.layer_idx]; B, Hq, L, D = query.shape; npg = s["npg"]
    q = query.transpose(1, 2).contiguous().to(torch.bfloat16)
    sl = torch.full((B,), s["len"], dtype=torch.int32, device="cuda")
    v4 = lambda x: x.view(B * npg, s["H"], PAGE, x.shape[-1])
    if c.mode == "nvfp4":
        o = N.uq_decode_nvfp4(q, v4(s["kc"]), v4(s["ks"]), v4(s["vc"]), v4(s["vs"]), s["bt"], sl, k_ts=s["ts_k"], v_ts=s["ts_v"], sm_scale=scaling)
    else:
        o = A.uq_decode(q, v4(s["kc"]), v4(s["ks"]), v4(s["vc"]), v4(s["vs"]), s["bt"], sl, sm_scale=scaling)
    return o.to(query.dtype), None


class KMethod:
    def __init__(self, mode): self.mode = mode
    def make_cache(self): return KCache(self.mode)


@torch.no_grad()
def attn_err(model, ids, T=2048, K=8):
    from transformers.models.llama import modeling_llama as ml
    cfg = model.config; caps = {}; hooks = []
    def mk(li):
        def hook(mod, args, kwargs, out):
            h = args[0] if args else kwargs["hidden_states"]; B, L, _ = h.shape; hd = getattr(mod, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            q = mod.q_proj(h).view(B, L, -1, hd)
            if hasattr(mod, "q_norm"): q = mod.q_norm(q)
            q = q.transpose(1, 2); cos, sin = kwargs["position_embeddings"]
            caps[li] = ((q * cos.unsqueeze(1)) + (ml.rotate_half(q) * sin.unsqueeze(1)))[:, :, -K:]
        return hook
    for li, layer in enumerate(model.model.layers): hooks.append(layer.self_attn.register_forward_hook(mk(li), with_kwargs=True))
    ACTIVE[0] = None; cache = DynamicCache(); model(ids[:T].unsqueeze(0).cuda(), past_key_values=cache, use_cache=True)
    for h in hooks: h.remove()
    rows = []
    for li in range(len(model.model.layers)):
        q = caps[li].transpose(1, 2).contiguous()          # [1,K,Hq,D]
        k = cache.key_cache[li]; v = cache.value_cache[li]  # [1,Hkv,T,D]
        Hq = q.shape[2]; Hkv = k.shape[1]; G = Hq // Hkv; D = q.shape[-1]
        qf = q.float().reshape(1, K, Hkv, G, D); kf = k.float(); vf = v.float()
        sc = torch.einsum("bjhgd,bhnd->bjhgn", qf, kf) / math.sqrt(D)
        mask = (torch.arange(T, device="cuda")[None, :] <= (T - K + torch.arange(K, device="cuda"))[:, None])
        sc = sc.masked_fill(~mask[None, :, None, None, :], -1e30)
        ex = torch.einsum("bjhgn,bhnd->bjhgd", sc.softmax(-1), vf).reshape(1, K, Hq, D)
        qb = q.to(torch.bfloat16)
        cn = N.build_paged_cache_nvfp4(k.float(), v.float(), PAGE); on = cn.decode(qb).float()
        rn = N.reference_nvfp4(qb, cn.kc, cn.ks, cn.vc, cn.vs, cn.block_table, cn.seq_lens, cn.k_ts, cn.v_ts)
        cm = A.build_paged_cache(k.float(), v.float(), PAGE); om = cm.decode(qb).float()
        rm = A.reference(qb, cm.kc, cm.ks, cm.vc, cm.vs, cm.block_table, cm.seq_lens)
        cos = lambda a, b: F.cosine_similarity(a.reshape(-1, D), b.reshape(-1, D), dim=-1).mean().item()
        rel = lambda a, b: ((a - b).norm() / b.norm()).item()
        rows.append(dict(layer=li, nv_cos=cos(on, ex), nv_rel=rel(on, ex), mx_cos=cos(om, ex), mx_rel=rel(om, ex),
                         nv_kernel_vs_ref=rel(on, rn), mx_kernel_vs_ref=rel(om, rm)))
    m = lambda key: sum(r[key] for r in rows) / len(rows)
    return dict(per_layer=rows, **{k_: m(k_) for k_ in rows[0] if k_ != "layer"}, T=T, K=K)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3_8b"); ap.add_argument("--out", default=None)
    ap.add_argument("--nwin", type=int, default=8); ap.add_argument("--modes", default="fp16,nvfp4,mxfp4")
    a = ap.parse_args()
    AttentionInterface.register("sdpa", kernel_attn)
    model, tok = qeval.load(a.model); kvq.N_LAYERS[0] = model.config.num_hidden_layers
    W = qeval.wiki_ids(tok); res = dict(model=a.model, ppl={})
    for mode in a.modes.split(","):
        t0 = time.time()
        meth = kvq.build_registry()["fp16"] if mode == "fp16" else KMethod(mode)
        if mode == "fp16": ACTIVE[0] = None
        res["ppl"][mode] = qeval.ppl_run(model, W, meth, nwin=a.nwin); ACTIVE[0] = None
        print(a.model, mode, res["ppl"][mode]["ppl"], f"{time.time() - t0:.0f}s", flush=True)
    res["attn"] = attn_err(model, W)
    print({k: v for k, v in res["attn"].items() if k != "per_layer"}, flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
