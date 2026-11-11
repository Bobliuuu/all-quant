"""Workload (b): long-document RAG QA with accuracy score (HotpotQA distractor, multi-hop).

Each question's 10 provided paragraphs are mixed with filler paragraphs from other questions until the context is
~ctx_tokens long; supporting paragraphs land at random depths.  Greedy answer, EM and token-F1 (SQuAD-normalised).
"""
import asyncio, random, re, string, collections, aiohttp
from rwlib import *

def norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())

def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    c = collections.Counter(p) & collections.Counter(g); n = sum(c.values())
    if not p or not g: return float(p == g)
    if n == 0: return 0.0
    pr, rc = n / len(p), n / len(g); return 2 * pr * rc / (pr + rc)

def load_hotpot(n):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet", repo_type="dataset",
                        revision="refs/convert/parquet") if False else None
    try:
        f = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet", repo_type="dataset",
                            revision="refs/convert/parquet")
    except Exception:
        f = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(f).to_pylist()

FEW = ("Answer each question using the documents. Reply with only a short answer (a few words).\n\n")

def make_prompts(rows, n, ctx_tokens, cpt, seed=0):
    rng = random.Random(seed); out = []
    paras = []
    for r in rows:
        for t, ss in zip(r["context"]["title"], r["context"]["sentences"]):
            paras.append(f"[{t}] " + "".join(ss))
    for i in range(n):
        r = rows[i]
        own = [f"[{t}] " + "".join(ss) for t, ss in zip(r["context"]["title"], r["context"]["sentences"])]
        budget = int(ctx_tokens * cpt) - sum(len(x) for x in own)
        fill = []
        while budget > 0:
            x = paras[rng.randrange(len(paras))]; fill.append(x); budget -= len(x)
        docs = own + fill; rng.shuffle(docs)
        ctx = "\n\n".join(f"Document {j+1}: {d}" for j, d in enumerate(docs))
        out.append((FEW + ctx + f"\n\nQuestion: {r['question']}\nAnswer:", r["answer"]))
    return out

async def one(srv, prompt, sem):
    async with sem:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
            return await stream_completion(s, srv.url, prompt, 24, extra={"stop": ["\n"]})

def run(srv, a):
    rows = load_hotpot(a.n + 5)
    cpt = 4.0
    ntok = count_tokens(srv.url, "\n\n".join("".join(r["context"]["sentences"][0]) for r in rows[:50]))
    cpt = len("\n\n".join("".join(r["context"]["sentences"][0]) for r in rows[:50])) / ntok
    pr = make_prompts(rows, a.n, a.ctx_tokens, cpt)
    async def go():
        sem = asyncio.Semaphore(a.conc); t0 = time.perf_counter()
        res = await asyncio.gather(*[one(srv, p, sem) for p, _ in pr]); return res, time.perf_counter() - t0
    res, wall = asyncio.run(go())
    em = []; f = []; samples = []
    for (p, g), r in zip(pr, res):
        pred = r.get("text", "").strip(); em.append(float(norm(pred) == norm(g))); f.append(f1(pred, g))
        samples.append({"gold": g, "pred": pred})
    ttft = [r["ttft"] for r in res if r.get("ttft")]
    return {"workload": "rag_hotpotqa", "n": a.n, "ctx_tokens": a.ctx_tokens, "EM": sum(em) / len(em), "F1": sum(f) / len(f),
            "wall_s": wall, "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99), "samples": samples}
