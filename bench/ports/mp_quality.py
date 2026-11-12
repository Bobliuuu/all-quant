"""Quality of a port through the existing harness (bench/quality/qeval.py): wikitext-2 ppl (chunked prefill + token-by-token decode through
the quantised cache) and per-layer attention-output cosine on real KV, Mistral-7B-v0.1 and Qwen3-8B.  The cache format is the port's own
quantiser (port.quality_methods()).  Merges {ppl_delta, quality} into results/ports_<method>.json.
usage: python mp_quality.py --method fp8 --out results/ports_fp8.json   (under flock /root/gpu.lock)"""
import argparse, importlib.util, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quality"))
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--out", default=None, help="optional template containing {m}")
ap.add_argument("--models", default="mistral7b,qwen3_8b"); ap.add_argument("--nwin", type=int, default=8)
a = ap.parse_args()
import torch, kvq, qeval
from common import merge_json
import qeval as _q
mods = {}
reg = None
def load_mod(m):
    sys.path.insert(0, os.path.join(HERE, m))
    spec = importlib.util.spec_from_file_location("port_" + m, os.path.join(HERE, m, "port.py")); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod
def wiki_ids(tok, split="test"):   # box is offline: use the cached Salesforce/wikitext copy (same text as the harness' "wikitext")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
qeval.wiki_ids = wiki_ids
if not hasattr(kvq.DynamicCache, "key_cache"):   # transformers >= 4.56 keeps per-layer objects
    kvq.DynamicCache.key_cache = property(lambda s: [l.keys for l in s.layers])
    kvq.DynamicCache.value_cache = property(lambda s: [l.values for l in s.layers])
reg = kvq.build_registry()
methods = a.method.split(",")
owner = {}
for m in methods:
    mods[m] = load_mod(m)
    for name, d in mods[m].quality_methods().items():
        reg[name] = kvq.Method(name, d["family"], d["bits"], "real-port", d["bpv"], d.get("fk"), d.get("fv"), cache=d.get("cache"), seq=d.get("seq"), note=d["note"])
        owner[name] = m
outp = lambda m: os.path.join(HERE, "..", "..", "results", f"ports_{m}.json")
for mk in a.models.split(","):
    model, tok = qeval.load(mk); kvq.N_LAYERS[0] = model.config.num_hidden_layers
    W = qeval.wiki_ids(tok)
    base = None
    for mn in ["fp16"] + list(owner):
        m = reg[mn]; t0 = time.time()
        p = qeval.ppl_run(model, W, m, nwin=a.nwin)["ppl"]
        at = qeval.attn_error(model, W, m)
        if mn == "fp16": base = p
        r = dict(ppl=p, attn_cos=at["mean_cos"], attn_rel_err=at["mean_rel_err"], k_rel=at["mean_k_rel"], v_rel=at["mean_v_rel"])
        print(mk, mn, json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
        if mn == "fp16": continue
        r["dppl"] = p - base; r["ppl_fp16"] = base
        merge_json(a.out.replace("{m}", owner[mn]) if a.out else outp(owner[mn]), dict(ppl_delta={mk: {mn: r["dppl"]}}, quality={mk: {mn: r}}))
    del model; torch.cuda.empty_cache()
