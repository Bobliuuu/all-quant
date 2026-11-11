"""Workload (c): speculative decoding on a code-editing task (n-gram / prompt-lookup drafter, K = num_speculative_tokens).

Server is started with --speculative-config (see run_cfg --extra).  Each request gives a real source file and asks for
an edited copy (so the output largely copies the prompt, the regime where multi-token verify matters)."""
import asyncio, aiohttp, os
from rwlib import *

def prompts(n, file_tokens_chars):
    files = []
    for dp, dn, fn in sorted(os.walk(os.path.join(vllm_src_root(), "v1"))):
        for f in sorted(fn):
            if f.endswith(".py"):
                s = open(os.path.join(dp, f), errors="ignore").read()
                if 6000 < len(s) < 14000: files.append(s)
    out = []
    for i in range(n):
        s = files[(i * 7) % len(files)][:file_tokens_chars]
        out.append("### ORIGINAL (python)\n" + s + "\n### EDITED (python): identical to ORIGINAL except every occurrence of `self.` is renamed to `this.`\n")
    return out

def scrape(srv):
    m = srv.metrics(); g = lambda k: sum(v for kk, v in m.items() if kk.startswith(k))
    return {"drafts": g("vllm:spec_decode_num_drafts"), "draft_tokens": g("vllm:spec_decode_num_draft_tokens"),
            "accepted": g("vllm:spec_decode_num_accepted_tokens")}

def run(srv, a):
    ps = prompts(a.n, 9000)
    async def one(p, sem):
        async with sem:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
                return await stream_completion(s, srv.url, p, a.gen)
    async def go():
        sem = asyncio.Semaphore(a.conc); t0 = time.perf_counter()
        async with aiohttp.ClientSession() as s: await stream_completion(s, srv.url, ps[0], 8)
        b = scrape(srv); t0 = time.perf_counter()
        res = await asyncio.gather(*[one(p, sem) for p in ps]); return res, time.perf_counter() - t0, b
    res, wall, b0 = asyncio.run(go()); b1 = scrape(srv)
    ok = [r for r in res if "error" not in r]
    itl = [x for r in ok for x in r["itls"]]; n = sum(r["chunks"] for r in ok)
    d = {k: b1[k] - b0[k] for k in b1}
    return {"workload": "spec_codeedit", "K": a.k, "conc": a.conc, "n": a.n, "gen": a.gen, "wall_s": wall,
            "out_tok_per_s": n / wall, "itl_p50": pct(itl, 50), "itl_p99": pct(itl, 99),
            "ttft_p50": pct([r["ttft"] for r in ok], 50), "errors": len(res) - len(ok), **d,
            "mean_accepted_per_draft": (d["accepted"] / d["drafts"]) if d["drafts"] else None,
            "tokens_per_step": (1 + d["accepted"] / d["drafts"]) if d["drafts"] else 1.0,
            "texts": [r["text"] for r in ok]}
