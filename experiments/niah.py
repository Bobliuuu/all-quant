"""Small RULER-style needle-in-a-haystack check (our own generator, NOT the RULER suite).

Tasks (in the spirit of RULER's niah_single_2 and niah_multikey_1):
  single   : one "special magic number" sentence hidden in PG-19 text, asked back
  multikey : four needles with different keys, one key asked back
The haystack is PG-19 test text; the KV cache is pseudo-quantized with the same
QuantCache as kl_pg19.py (prefill and every decoded token). Greedy decoding,
Qwen3 chat template with thinking disabled. Score = exact 7-digit number present.

usage: python experiments/niah.py --pg19 <parquet> --ctx 8192 --n 12 --configs bf16,pq-3.2-kc --out x.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from kl_pg19 import make_cache_cls  # noqa: E402
from kv_configs import CONFIGS  # noqa: E402

WORDS = ["apple", "river", "lantern", "falcon", "marble", "cedar", "violet", "harbor", "ember", "quartz", "meadow", "copper"]


def build(tok, text_ids, ctx, task, rng):
    nneedle = 1 if task == "single" else 4
    keys = rng.sample(WORDS, nneedle)
    vals = [str(rng.randint(1000000, 9999999)) for _ in keys]
    needles = [f" One of the special magic numbers for {k} is: {v}. " for k, v in zip(keys, vals)]
    qi = rng.randrange(nneedle)
    question = (f"\n\nWhat is the special magic number for {keys[qi]} mentioned in the provided text? "
                f"Answer with the number only.")
    budget = ctx - 200
    start = rng.randrange(0, max(1, len(text_ids) - budget))
    hay = text_ids[start:start + budget].tolist()
    pos = sorted(rng.sample(range(0, len(hay)), nneedle))
    pieces, last = [], 0
    for p, nd in zip(pos, needles):
        pieces.append(tok.decode(hay[last:p]))
        pieces.append(nd)
        last = p
    pieces.append(tok.decode(hay[last:]))
    ctx_text = "".join(pieces)
    msgs = [{"role": "user", "content": "Read the text below and answer the question after it.\n\n" + ctx_text + question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    depth = pos[qi] / len(hay)
    return prompt, vals[qi], depth


@torch.no_grad()
def answer(model, tok, prompt, Cache, cfg, max_new=16):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    cache = Cache(*CONFIGS[cfg])
    out = model.generate(ids, past_key_values=cache, max_new_tokens=max_new, do_sample=False, use_cache=True)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--pg19", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow.parquet as pq
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    Cache = make_cache_cls()
    books = pq.read_table(a.pg19).to_pandas()["text"].tolist()
    text_ids = tok(" ".join(books[10:14])[: a.ctx * 40], return_tensors="pt").input_ids[0]
    rng = random.Random(1234)
    items = [(task, *build(tok, text_ids, a.ctx, task, rng)) for task in ("single", "multikey") for _ in range(a.n)]
    res = {"model": a.model, "ctx": a.ctx, "n_per_task": a.n, "configs": {}}
    t0 = time.time()
    for cfg in a.configs.split(","):
        rows = []
        for task, prompt, val, depth in items:
            txt, L = answer(model, tok, prompt, Cache, cfg)
            rows.append({"task": task, "ok": val in txt, "depth": round(depth, 3), "len": L, "out": txt.strip()[:40]})
        acc = {t: sum(r["ok"] for r in rows if r["task"] == t) / sum(1 for r in rows if r["task"] == t) for t in ("single", "multikey")}
        res["configs"][cfg] = {"acc": acc, "rows": rows}
        print(f"{cfg:12s} single={acc['single']:.3f} multikey={acc['multikey']:.3f} t={time.time()-t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
