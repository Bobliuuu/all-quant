"""Dump post-RoPE Q, K, V of a few layers on one PG-19 sequence (for offline codec analysis)."""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from kl_pg19 import load_pg19

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--pg19", required=True)
ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--layers", default="0,1,2,10,20,35")
ap.add_argument("--out", required=True)
a = ap.parse_args()
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as mq
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = load_pg19(tok, 1, a.ctx, a.pg19)[0]
keep = set(int(x) for x in a.layers.split(","))
store = {}
orig = mq.ALL_ATTENTION_FUNCTIONS["sdpa"] if hasattr(mq, "ALL_ATTENTION_FUNCTIONS") else None
from transformers.integrations.sdpa_attention import sdpa_attention_forward
def hook(module, query, key, value, *args, **kw):
    if module.layer_idx in keep:
        store[module.layer_idx] = (query[0].cpu(), key[0].cpu(), value[0].cpu())
    return sdpa_attention_forward(module, query, key, value, *args, **kw)
from transformers import AttentionInterface
AttentionInterface.register("dump", hook)
model.config._attn_implementation = "dump"
for l in model.model.layers: l.self_attn.config._attn_implementation = "dump"
with torch.no_grad():
    model(input_ids=ids[None].cuda())
torch.save(store, a.out)
print({k: [t.shape for t in v] for k, v in store.items()})
