"""Sanity: block-rotation quantisers (amd-experiments) on synthetic gaussian vs real Mistral K; Triton path vs torch fallback."""
import os, sys, json, torch
sys.path.insert(0, os.environ.get("AMD_KERNELS", "/work/amdk")); sys.path.insert(0, "/work/q")
import block_quant_rocm as bq, torch.nn.functional as F
from qeval import load, wiki_ids
import kvq
def stats(x, y): return dict(cos=F.cosine_similarity(x, y, dim=-1).mean().item(), rel=((x - y).norm() / x.norm()).item())
out = {}
model, tok = load("mistral7b"); ids = wiki_ids(tok)[:2048].unsqueeze(0).cuda()
c = kvq.DynamicCache(); model(ids, past_key_values=c, use_cache=True)
real = {"K_layer10": c.key_cache[10][0].transpose(0, 1).reshape(-1, 128).float(), "V_layer10": c.value_cache[10][0].transpose(0, 1).reshape(-1, 128).float(),
        "gauss": torch.randn(4096, 128, device="cuda")}
out["real_K_absmax_over_rms"] = (real["K_layer10"].abs().amax(-1) / real["K_layer10"].pow(2).mean(-1).sqrt()).mean().item()
for name, x in real.items():
    for m in ("planar", "iso", "rotor"):
        for b in (3, 4):
            for tri in (True, False):
                bq._TRITON_AVAILABLE = tri
                q = bq.make_quantizer(m, bits=b, device="cuda")
                try: y = q.decompress(q.compress(x), x.shape).float(); out[f"{name}/{m}{b}/{'triton' if tri else 'torch'}"] = stats(x, y)
                except Exception as e: out[f"{name}/{m}{b}/{'triton' if tri else 'torch'}"] = repr(e)[:150]
    tq = __import__("turboquant_mi300x").TurboQuantMI300X(bits=4, device="cuda")
    out[f"{name}/turbo4"] = stats(x, tq.decompress_tensor(tq.compress_tensor(x), x.shape))
print(json.dumps(out, indent=1)); os.makedirs("/out/sanity", exist_ok=True); json.dump(out, open("/out/sanity/blockquant_sanity.json", "w"), indent=1)
