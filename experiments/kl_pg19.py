"""KL divergence from an FP8 KV cache on PG-19 (Pearl's Layer-1 metric), on an
open model (default Qwen3-8B; NOT Pearl's GLM-5.3).

The KV cache is pseudo-quantized at write time: a DynamicCache subclass
quantizes every new K (post-RoPE, as it is stored in a real cache) and V
before storing it, so every query attends to dequantized keys/values, exactly
what a decode step over a quantized cache sees. Queries stay bf16.

usage: python experiments/kl_pg19.py --out results/kl.json --nseq 8 --ctx 16384 \
          --configs bf16,fp8,pq-3.2,...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from kv_configs import CONFIGS  # noqa: E402


def make_cache_cls():
    from transformers import DynamicCache

    class QuantCache(DynamicCache):
        def __init__(self, qk, qv, opts=None, *a, **kw):
            super().__init__(*a, **kw)
            self.qk, self.qv = qk, qv
            self.opts = opts or {}
            self.mu = {}
            self.bits = []
            self.relerr = []  # (layer, relMSE_K, relMSE_V)

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            kin, vin = key_states, value_states
            for nm, x in (("k", kin), ("v", vin)):
                n = self.opts.get(nm + "center")
                if n and (nm, layer_idx) not in self.mu:
                    self.mu[(nm, layer_idx)] = x[:, :, :n].float().mean(2, keepdim=True).to(x.dtype)
            muk = self.mu.get(("k", layer_idx))
            muv = self.mu.get(("v", layer_idx))
            k, bk = self.qk(kin - muk if muk is not None else kin)
            v, bv = self.qv(vin - muv if muv is not None else vin)
            if muk is not None:
                k = k + muk
            if muv is not None:
                v = v + muv
            self.bits.append((bk + bv) / 2)
            ek = ((k.float() - key_states.float()) ** 2).sum() / (key_states.float() ** 2).sum()
            ev = ((v.float() - value_states.float()) ** 2).sum() / (value_states.float() ** 2).sum()
            self.relerr.append((layer_idx, ek.item(), ev.item()))
            return super().update(k, v, layer_idx, cache_kwargs)

    return QuantCache


def load_pg19(tok, nseq, ctx, path):
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pandas()
    seqs = []
    for txt in t["text"]:
        ids = tok(txt[: ctx * 8], return_tensors="pt").input_ids[0]
        if len(ids) >= ctx:
            seqs.append(ids[:ctx])
        if len(seqs) == nseq:
            break
    return seqs


@torch.no_grad()
def hidden(model, ids, Cache, cfg):
    cache = Cache(*CONFIGS[cfg])
    h = model.model(input_ids=ids[None].cuda(), past_key_values=cache, use_cache=True).last_hidden_state[0]
    bits = sum(cache.bits) / len(cache.bits)
    rel = cache.relerr
    del cache
    hidden.last_relerr = rel
    return h, bits


@torch.no_grad()
def kl_and_nll(model, h_ref, h, ids, chunk=1024):
    kls, nll, nll_ref = [], 0.0, 0.0
    tgt = ids[1:].cuda()
    T = h.shape[0] - 1
    for i in range(0, T, chunk):
        j = min(i + chunk, T)
        lr = torch.log_softmax(model.lm_head(h_ref[i:j]).float(), -1)
        lq = torch.log_softmax(model.lm_head(h[i:j]).float(), -1)
        kls.append((lr.exp() * (lr - lq)).sum(-1))
        nll += -lq.gather(1, tgt[i:j, None]).sum().item()
        nll_ref += -lr.gather(1, tgt[i:j, None]).sum().item()
    kl = torch.cat(kls)
    return kl, nll / T, nll_ref / T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--pg19", required=True)
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--ref", default="fp8")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    Cache = make_cache_cls()
    seqs = load_pg19(tok, a.nseq, a.ctx, a.pg19)
    cfgs = [c for c in a.configs.split(",") if c != a.ref]
    res = {"model": a.model, "ctx": a.ctx, "nseq": len(seqs), "ref": a.ref, "configs": {}}
    acc = {c: {"kl": [], "nll": [], "bits": None, "kl_by_pos": None} for c in cfgs}
    nll_ref_all = []
    t0 = time.time()
    for si, ids in enumerate(seqs):
        h_ref, bref = hidden(model, ids, Cache, a.ref)
        res["ref_bits"] = bref
        for c in cfgs:
            h, b = hidden(model, ids, Cache, c)
            kl, nll, nll_ref = kl_and_nll(model, h_ref, h, ids)
            acc[c]["kl"].append(kl.mean().item())
            acc[c]["nll"].append(nll)
            acc[c]["bits"] = b
            re = hidden.last_relerr
            acc[c]["relmse_k"] = sum(r[1] for r in re) / len(re)
            acc[c]["relmse_v"] = sum(r[2] for r in re) / len(re)
            acc[c]["relmse_by_layer"] = re
            # KL by position bucket (does error accumulate with context?)
            nb = 8
            bk = kl[: (len(kl) // nb) * nb].reshape(nb, -1).mean(-1).tolist()
            acc[c]["kl_by_pos"] = bk if acc[c]["kl_by_pos"] is None else [x + y for x, y in zip(acc[c]["kl_by_pos"], bk)]
            del h
            print(f"seq {si} {c:12s} relK={acc[c]['relmse_k']:.4f} relV={acc[c]['relmse_v']:.4f} bits={b:.4f} KL={kl.mean().item():.4f} nll={nll:.4f} ref_nll={nll_ref:.4f} t={time.time()-t0:.0f}s", flush=True)
        nll_ref_all.append(nll_ref)
        del h_ref
        torch.cuda.empty_cache()
    for c in cfgs:
        n = len(acc[c]["kl"])
        res["configs"][c] = {
            "bits_per_value": acc[c]["bits"],
            "kl_mean": sum(acc[c]["kl"]) / n,
            "kl_per_seq": acc[c]["kl"],
            "ppl": math.exp(sum(acc[c]["nll"]) / n),
            "kl_by_pos_bucket": [x / n for x in acc[c]["kl_by_pos"]],
            "relmse_k_last_seq": acc[c]["relmse_k"], "relmse_v_last_seq": acc[c]["relmse_v"],
            "relmse_by_layer_last_seq": acc[c]["relmse_by_layer"],
        }
    res["ref_ppl"] = math.exp(sum(nll_ref_all) / len(nll_ref_all))
    res["elapsed_s"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({c: (round(v["bits_per_value"], 4), round(v["kl_mean"], 4), round(v["ppl"], 3)) for c, v in res["configs"].items()}, indent=0))


if __name__ == "__main__":
    main()
