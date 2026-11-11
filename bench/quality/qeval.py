"""Quality evaluation of KV-cache quantisation methods (see kvq.py).  One call = one (model, method).
Metrics: ppl (wikitext2/c4; chunked prefill + token-by-token decode through the quantised cache), attn (per-layer
attention-output error on real KV), needle (passkey 8k/16k/32k), ruler (synthetic RULER-style subset), verify (K=1..8), bytes.
"""
import argparse, json, math, os, random, sys, time
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvq

MODELS = {"mistral7b": "mistralai/Mistral-7B-v0.1", "qwen3_8b": "Qwen/Qwen3-8B"}


def load(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()
    return m, tok


def wiki_ids(tok, split="test", n_tok=None):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return ids if n_tok is None else ids[:n_tok]


def c4_ids(tok, n_tok=60000):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
    txt = "\n\n".join(ds[i]["text"] for i in range(400))
    return tok(txt, return_tensors="pt").input_ids[0][:n_tok]


# ------------------------------------------------------------------ (a) perplexity
@torch.no_grad()
def ppl_run(model, ids, method, win=2048, prefill=1024, chunk=256, nwin=8, bs=8):
    """teacher-forced: chunked prefill of first `prefill` tokens, then 1 token/step decode of the rest, all via quantised cache."""
    n = min(nwin, ids.numel() // win); W = ids[: n * win].view(n, win).cuda()
    nll_pre = torch.zeros((), device="cuda", dtype=torch.float64); nll_dec = nll_pre.clone(); c_pre = c_dec = 0
    for b0 in range(0, n, bs):
        x = W[b0:b0 + bs]; cache = method.make_cache(); pos = 0
        for s in range(0, prefill, chunk):
            e = min(s + chunk, prefill)
            out = model(x[:, s:e], past_key_values=cache, use_cache=True, cache_position=torch.arange(s, e, device="cuda"))
            lg = out.logits.float()
            tgt = x[:, s + 1:e + 1]; k = tgt.shape[1]
            nll_pre += F.cross_entropy(lg[:, :k].reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="sum").double(); c_pre += tgt.numel()
        for t in range(prefill, win - 1):
            out = model(x[:, t:t + 1], past_key_values=cache, use_cache=True, cache_position=torch.tensor([t], device="cuda"))
            nll_dec += F.cross_entropy(out.logits[:, -1].float(), x[:, t + 1], reduction="sum").double(); c_dec += x.shape[0]
        del cache
    # first prefill chunks: last token of prefill predicts token prefill (already counted by tgt slice above)
    tot = (nll_pre + nll_dec) / (c_pre + c_dec)
    return dict(ppl=math.exp(tot.item()), ppl_prefill_part=math.exp((nll_pre / c_pre).item()), ppl_decode_part=math.exp((nll_dec / c_dec).item()),
                tokens=c_pre + c_dec, windows=n, win=win, prefill=prefill, chunk=chunk)


# ------------------------------------------------------------------ (b) per-layer attention error on real KV
@torch.no_grad()
def attn_error(model, ids, method, T=2048, nq=128):
    from transformers.models.llama import modeling_llama as ml
    cfg = model.config; caps = {}; hooks = []
    def mk(li):
        def hook(mod, args, kwargs, out):
            h = args[0] if args else kwargs["hidden_states"]
            B, L, _ = h.shape; hd = getattr(mod, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            q = mod.q_proj(h).view(B, L, -1, hd)
            if hasattr(mod, "q_norm"): q = mod.q_norm(q)
            q = q.transpose(1, 2); cos, sin = kwargs["position_embeddings"]
            q = (q * cos.unsqueeze(1)) + (ml.rotate_half(q) * sin.unsqueeze(1))
            caps[li] = q[:, :, -nq:].float()
        return hook
    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(mk(li), with_kwargs=True))
    cache = kvq.DynamicCache(); x = ids[:T].unsqueeze(0).cuda()
    model(x, past_key_values=cache, use_cache=True)
    for h in hooks: h.remove()
    rows = []
    for li in range(len(model.model.layers)):
        q = caps[li]; K = cache.key_cache[li].float(); V = cache.value_cache[li].float()
        Kq = method.seq_fq(cache.key_cache[li], True, li).float(); Vq = method.seq_fq(cache.value_cache[li], False, li).float()
        g = q.shape[1] // K.shape[1]; sc = 1 / math.sqrt(q.shape[-1])
        mask = torch.ones(nq, T, device="cuda", dtype=torch.bool).tril(T - nq)
        def att(Kx, Vx):
            kk = Kx.repeat_interleave(g, 1); vv = Vx.repeat_interleave(g, 1)
            s = (q @ kk.transpose(-1, -2) * sc).masked_fill(~mask, -1e30)
            return s.softmax(-1) @ vv
        o = att(K, V); oq = att(Kq, Vq)
        cos = F.cosine_similarity(o.flatten(0, 2), oq.flatten(0, 2), dim=-1).mean().item()
        rel = ((o - oq).norm() / o.norm()).item()
        kk_rel = ((K - Kq).norm() / K.norm()).item(); vv_rel = ((V - Vq).norm() / V.norm()).item()
        rows.append(dict(layer=li, cos=cos, rel_err=rel, k_rel=kk_rel, v_rel=vv_rel))
    return dict(per_layer=rows, mean_cos=sum(r["cos"] for r in rows) / len(rows), mean_rel_err=sum(r["rel_err"] for r in rows) / len(rows),
                mean_k_rel=sum(r["k_rel"] for r in rows) / len(rows), mean_v_rel=sum(r["v_rel"] for r in rows) / len(rows), T=T, nq=nq)


# ------------------------------------------------------------------ (c) long-context retrieval
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
WORDS = ["apple", "river", "stone", "cloud", "lamp", "tiger", "ocean", "maple", "copper", "violet", "harbor", "ember"]

@torch.no_grad()
def gen_greedy(model, tok, ids, method, max_new=14, chunk=2048):
    cache = method.make_cache(); x = ids.unsqueeze(0).cuda(); L = x.shape[1]
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        out = model(x[:, s:e], past_key_values=cache, use_cache=True, cache_position=torch.arange(s, e, device="cuda"), logits_to_keep=1) \
            if "logits_to_keep" in model.forward.__code__.co_varnames else model(x[:, s:e], past_key_values=cache, use_cache=True, cache_position=torch.arange(s, e, device="cuda"))
    toks = []; nxt = out.logits[:, -1].argmax(-1)
    for i in range(max_new):
        toks.append(nxt.item()); pos = L + i
        out = model(nxt.view(1, 1), past_key_values=cache, use_cache=True, cache_position=torch.tensor([pos], device="cuda"))
        nxt = out.logits[:, -1].argmax(-1)
    del cache
    return tok.decode(toks)


def build_prompt(tok, n_tok, kind, seed, depth):
    rng = random.Random(seed)
    unit = len(tok(FILLER).input_ids)
    nrep = max(2, (n_tok - 200) // unit)
    def hay(n): return FILLER * n
    if kind == "passkey":
        key = str(rng.randint(10000, 99999))
        pre = int(nrep * depth)
        text = hay(pre) + f" The pass key is {key}. Remember it. {key} is the pass key. " + hay(nrep - pre) + "\nWhat is the pass key? The pass key is"
        return text, [key]
    if kind == "multikey":  # several distinct keys; ask for one
        names = rng.sample(WORDS, 4); keys = {n: str(rng.randint(10000, 99999)) for n in names}
        tgt = names[seed % 4]; pre = [int(nrep * d) for d in (0.15, 0.4, 0.65, 0.9)]
        parts = []; last = 0
        for n, p in zip(names, pre):
            parts.append(hay(p - last)); parts.append(f" The special magic number for {n} is {keys[n]}. "); last = p
        parts.append(hay(nrep - last)); text = "".join(parts) + f"\nWhat is the special magic number for {tgt}? The special magic number for {tgt} is"
        return text, [keys[tgt]]
    if kind == "kv":  # key-value retrieval with uuid-ish keys
        import uuid
        n_kv = max(8, n_tok // 70); pairs = [(str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128)))) for _ in range(n_kv)]
        qi = int(len(pairs) * depth); k, v = pairs[min(qi, len(pairs) - 1)]
        body = "\n".join(f'"{a}": "{b}"' for a, b in pairs)
        return "Extract the value corresponding to the specified key from the JSON object below.\n{" + body + "}\n" + f'Key: "{k}"\nValue: "', [v]
    raise ValueError(kind)


def retrieval(model, tok, method, kind, lengths, depths, trials):
    res = {}
    for L in lengths:
        ok = tot = 0
        for d in depths:
            for t in range(trials):
                text, ans = build_prompt(tok, L, kind, 1000 * L + 10 * t + int(d * 10), d)
                ids = tok(text, return_tensors="pt").input_ids[0]
                if ids.numel() > L + 300 and kind != "kv": ids = ids[-(L + 300):]
                out = gen_greedy(model, tok, ids, method, max_new=(40 if kind == "kv" else 12))
                ok += int(ans[0] in out); tot += 1
        res[str(L)] = dict(acc=ok / tot, n=tot)
    return res


# ------------------------------------------------------------------ (d) multi-token verify consistency
@torch.no_grad()
def verify(model, tok, ids_src, method, nprompts=200, plen=512, Kmax=8, bs=10, ref_cache={}):
    P = torch.stack([ids_src[i * plen:(i + 1) * plen] for i in range(nprompts)]).cuda()  # [N,plen] non-overlapping spans
    pm = method; match = torch.zeros(Kmax, device="cuda"); full = torch.zeros(Kmax, device="cuda"); n = 0
    fp = kvq.build_registry()["fp16"]
    for b0 in range(0, nprompts, bs):
        x = P[b0:b0 + bs]; B = x.shape[0]
        key = (b0, plen)
        if key not in ref_cache:  # fp16 greedy continuation g1..gK (reference)
            c = fp.make_cache(); o = model(x, past_key_values=c, use_cache=True, logits_to_keep=1) if True else None
            g = [o.logits[:, -1].argmax(-1)]
            for i in range(Kmax - 1):
                o = model(g[-1].view(B, 1), past_key_values=c, use_cache=True, cache_position=torch.tensor([plen + i], device="cuda")); g.append(o.logits[:, -1].argmax(-1))
            ref_cache[key] = torch.stack(g, 1)
        G = ref_cache[key]
        c = pm.make_cache(); cs = 256
        for s in range(0, plen - 1, cs):
            e = min(s + cs, plen - 1); model(x[:, s:e], past_key_values=c, use_cache=True, cache_position=torch.arange(s, e, device="cuda"), logits_to_keep=1)
        blk = torch.cat([x[:, plen - 1:plen], G[:, :Kmax - 1]], 1)  # [p_last, g1..g_{K-1}] verified in ONE multi-query step
        o = model(blk, past_key_values=c, use_cache=True, cache_position=torch.arange(plen - 1, plen - 1 + Kmax, device="cuda"))
        pred = o.logits.argmax(-1); eq = (pred == G)  # [B,Kmax]
        match += eq.float().sum(0); full += torch.cumprod(eq.int(), 1).float().sum(0); n += B
    mm = (match.cumsum(0) / (torch.arange(1, Kmax + 1, device="cuda") * n)).tolist()   # mean token match over first K positions
    acc = (full / n).tolist()                                                           # P(all first K tokens accepted)
    return dict(token_match_at_K=mm, all_accept_at_K=acc, per_pos_match=(match / n).tolist(), n_prompts=n, plen=plen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral7b"); ap.add_argument("--methods", default="fp16"); ap.add_argument("--metrics", default="ppl,attn,verify,needle")
    ap.add_argument("--needle_lens", default="8192,16384,32768"); ap.add_argument("--out", default="out"); ap.add_argument("--nwin", type=int, default=8)
    ap.add_argument("--nverify", type=int, default=200)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model, tok = load(a.model); reg = kvq.build_registry(); kvq.N_LAYERS[0] = model.config.num_hidden_layers
    W = wiki_ids(tok); metrics = a.metrics.split(",")
    for mn in a.methods.split(","):
        m = reg[mn]; path = f"{a.out}/{a.model}__{mn}.json"
        r = json.load(open(path)) if os.path.exists(path) else {}
        r.update(model=a.model, method=mn, family=m.family, kind=m.kind, bits=m.bits, note=m.note,
                 bytes_per_vec=m.bytes_per_vec(128), compression_vs_fp16=256 / m.bytes_per_vec(128))
        cfg = model.config; hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        r["kv_bytes_per_token_model"] = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * m.bytes_per_vec(hd)
        for met in metrics:
            t0 = time.time()
            try:
                if met == "ppl": r["ppl_wikitext2"] = ppl_run(model, W, m, nwin=a.nwin)
                elif met == "ppl_c4": r["ppl_c4"] = ppl_run(model, c4_ids(tok), m, nwin=4)
                elif met == "attn": r["attn"] = attn_error(model, W, m)
                elif met == "verify": r["verify"] = verify(model, tok, W, m, nprompts=a.nverify)
                elif met == "needle": r["needle"] = retrieval(model, tok, m, "passkey", [int(x) for x in a.needle_lens.split(",")], [0.1, 0.5, 0.9], 2)
                elif met == "ruler":
                    ls = [l for l in (4096, 8192, 16384) if l <= int(a.needle_lens.split(",")[-1])]
                    r["ruler"] = {k: retrieval(model, tok, m, k, ls, [0.2, 0.5, 0.8], 2) for k in ("multikey", "kv")}
            except Exception as e:
                import traceback; r[f"{met}_error"] = traceback.format_exc()[-800:]
            print(a.model, mn, met, f"{time.time() - t0:.0f}s", json.dumps({k: v for k, v in r.items() if k == {"ppl": "ppl_wikitext2"}.get(met, met)})[:300], flush=True)
            json.dump(r, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
