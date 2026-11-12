"""Why do planar/iso/rotor have bad ppl in quality_summary.md?  Per-layer K/V quantise->dequantise cosine on REAL KV (2048 wikitext tokens),
vs TurboQuant (WHT, amd-experiments) at 3 bit, plus a labelled diagnostic variant (not the published scheme): Walsh-Hadamard pre-mix before the block rotation."""
import sys, os, json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/root/h2h/kernels")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
import blockrot_common as c
from turboquant_mi300x import TurboQuantMI300X
dev = "cuda"; out = {}
def wht(x):
    n = x.shape[-1]; h = 1
    y = x.clone()
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h); a, b = y[..., 0, :], y[..., 1, :]; y = torch.stack([a + b, a - b], -2).reshape(*x.shape); h *= 2
    return y / math.sqrt(n)
for key, name in (("mistral7b", "/root/models/Mistral-7B-v0.1"), ("qwen3_8b", "/root/models/Qwen3-8B")):
    tok = AutoTokenizer.from_pretrained(name); m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda").eval()
    ids = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]), return_tensors="pt").input_ids[:, :2048].cuda()
    cache = DynamicCache()
    with torch.no_grad(): m(ids, past_key_values=cache, use_cache=True)
    KV = [(l.keys, l.values) for l in cache.layers] if hasattr(cache, "layers") else list(zip(cache.key_cache, cache.value_cache))
    meths = {f"{mm}3": c.make(mm, 3, dev) for mm in ("planar", "iso", "rotor")}; tq = TurboQuantMI300X(bits=3, device=dev)
    fq = {k: v.fake_quant for k, v in meths.items()}
    fq["tq3"] = lambda x: tq.decompress_tensor(tq.compress_tensor(x.reshape(-1, 128).float()), (x.numel() // 128, 128)).reshape(x.shape).to(x.dtype)
    # diagnostic: x -> WHT -> block-rot quant -> inverse WHT
    for k, v in meths.items(): fq[k + "+wht(diag)"] = (lambda x, f=v.fake_quant: wht(f(wht(x.float()).half()).float()).to(x.dtype))
    res = {k: dict(K=[], V=[]) for k in fq}; top = []
    for li, (K, V) in enumerate(KV):
        kk = K[0].float(); top.append(((kk ** 2).sum((0, 1)).max() / (kk ** 2).sum()).item())
        for k, f in fq.items():
            for w, X in (("K", K), ("V", V)):
                y = f(X); res[k][w].append(F.cosine_similarity(y.float().reshape(-1, 128), X.float().reshape(-1, 128), dim=-1).mean().item())
    out[key] = dict(layers=len(KV), top_channel_energy_frac_K_mean=sum(top) / len(top), cos={k: dict(K=sum(r["K"]) / len(r["K"]), V=sum(r["V"]) / len(r["V"]), K_min_layer=min(r["K"]), V_min_layer=min(r["V"])) for k, r in res.items()})
    print(key, json.dumps(out[key], indent=1), flush=True); del m; torch.cuda.empty_cache()
json.dump(out, open("/root/uqm-br/blockrot_diag.json", "w"), indent=1)
