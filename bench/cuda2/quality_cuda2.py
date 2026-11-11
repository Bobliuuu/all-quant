"""cuda2 real-model check: wikitext-2 perplexity THROUGH the fused decode kernels, Triton (uqmt) vs the CUDA port (cuda/uq_cuda.py).
Chunked prefill (256 tokens, read via SDPA on the dequantised paged cache) then teacher-forced decode of the remaining tokens in steps of
K query tokens (K=1 decode, K=8 verify-style), every decode step's attention for every layer runs the kernel on the paged quantised cache.
  python bench/cuda2/quality_cuda2.py --model qwen3_8b --ks 1 8 --fmts ue8m0 nvfp4 --out results/cuda2_quality_qwen3_8b.json
(self-contained: HF model + datasets only; quantisers from uqmt)"""
import argparse, json, math, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cuda"))
import torch, torch.nn.functional as F
from transformers import DynamicCache, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from uqmt import attn as A, nvfp4 as N

MODELS = {"mistral7b": "mistralai/Mistral-7B-v0.1", "qwen3_8b": "Qwen/Qwen3-8B"}
PAGE = 64
ACTIVE = [None]


class KCache(DynamicCache):
    def __init__(self, fmt, backend):
        super().__init__(); self.fmt = fmt; self.backend = backend; self.st = {}; ACTIVE[0] = self

    def get_seq_length(self, layer_idx=0):
        return self.st[layer_idx]["len"] if layer_idx in self.st else 0

    def _alloc(self, B, H, D, T=2048):
        npg = T // PAGE; z = lambda x: torch.zeros(B, npg, H, PAGE, x, dtype=torch.uint8, device="cuda")
        sg = D // 16 if self.fmt == "nvfp4" else D // 32
        return dict(kc=z(D // 2), vc=z(D // 2), ks=z(sg), vs=z(sg), len=0, npg=npg, ts_k=1.0, ts_v=1.0, B=B, H=H, D=D,
                    bt=torch.arange(B * npg, dtype=torch.int32, device="cuda").reshape(B, npg))

    def _q(self, x, s, which):
        if self.fmt == "nvfp4":
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
        if L <= 16: return k, v
        deq = (lambda p, sb, ts: N.dequantize_nvfp4_bytes(p, sb, ts)) if self.fmt == "nvfp4" else (lambda p, sb, ts: A.dequantize_fp4(p, sb))
        def full(c, sc, ts):
            x = deq(c, sc, ts)
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
    if c.backend == "triton":
        if c.fmt == "nvfp4":
            o = N.uq_decode_nvfp4(q, v4(s["kc"]), v4(s["ks"]), v4(s["vc"]), v4(s["vs"]), s["bt"], sl, k_ts=s["ts_k"], v_ts=s["ts_v"], sm_scale=scaling)
        else:
            o = A.uq_decode(q, v4(s["kc"]), v4(s["ks"]), v4(s["vc"]), v4(s["vs"]), s["bt"], sl, sm_scale=scaling)
    else:
        from uq_cuda import uq_decode_cuda
        if c.fmt == "nvfp4":
            o = uq_decode_cuda(q, v4(s["kc"]), v4(s["ks"]), v4(s["vc"]), v4(s["vs"]), s["bt"], sl, sm_scale=scaling, variant=c.variant, nwarps=c.nw, stages=c.kst,
                               mb=c.mb, fmt=1, k_ts=s["ts_k"], v_ts=s["ts_v"])
        else:
            o = uq_decode_cuda(q, v4(s["kc"]), v4(s["ks"]), v4(s["vc"]), v4(s["vs"]), s["bt"], sl, sm_scale=scaling, variant=c.variant, nwarps=c.nw, stages=c.kst, mb=c.mb)
    return o.to(query.dtype), None


@torch.no_grad()
def ppl(model, ids, fmt, backend, K, win=2048, prefill=1024, chunk=256, nwin=8, bs=8, cuda_cfg=None):
    n = min(nwin, ids.numel() // win); W = ids[: n * win].view(n, win).cuda()
    nll = torch.zeros((), device="cuda", dtype=torch.float64); cnt = 0; nll_dec = nll.clone(); cdec = 0
    for b0 in range(0, n, bs):
        x = W[b0:b0 + bs]
        if fmt == "fp16":
            ACTIVE[0] = None; cache = DynamicCache()
        else:
            cache = KCache(fmt, backend); cfg = cuda_cfg or {}
            cache.variant, cache.nw, cache.kst, cache.mb = cfg.get("variant", 0), cfg.get("nwarps", 0), cfg.get("stages", 0), cfg.get("mb", 0)
        for s in range(0, prefill, chunk):
            e = min(s + chunk, prefill)
            out = model(x[:, s:e], past_key_values=cache, use_cache=True, cache_position=torch.arange(s, e, device="cuda"))
            lg = out.logits.float(); tgt = x[:, s + 1:e + 1]; k = tgt.shape[1]
            nll += F.cross_entropy(lg[:, :k].reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="sum").double(); cnt += tgt.numel()
        t = prefill
        while t + K <= win - 1:
            out = model(x[:, t:t + K], past_key_values=cache, use_cache=True, cache_position=torch.arange(t, t + K, device="cuda"))
            l = F.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]), x[:, t + 1:t + K + 1].reshape(-1), reduction="sum").double()
            nll += l; nll_dec += l; cnt += x.shape[0] * K; cdec += x.shape[0] * K; t += K
        del cache
    ACTIVE[0] = None
    return dict(ppl=math.exp((nll / cnt).item()), ppl_decode_part=math.exp((nll_dec / cdec).item()), tokens=cnt, K=K, fmt=fmt, backend=backend)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3_8b"); ap.add_argument("--out", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8]); ap.add_argument("--fmts", nargs="+", default=["ue8m0", "nvfp4"])
    ap.add_argument("--nwin", type=int, default=8)
    a = ap.parse_args()
    AttentionInterface.register("sdpa", kernel_attn)
    name = MODELS[a.model]; tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test"); W = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    res = dict(model=a.model, device=torch.cuda.get_device_name(), rows=[])
    def run(**kw):
        t0 = time.time(); r = ppl(model, W, nwin=a.nwin, **kw); r["sec"] = round(time.time() - t0, 1); res["rows"].append(r); print(json.dumps(r), flush=True)
    run(fmt="fp16", backend="sdpa", K=1)
    for fmt in a.fmts:
        for K in a.ks:
            for be in ("triton", "cuda"):
                run(fmt=fmt, backend=be, K=K)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
