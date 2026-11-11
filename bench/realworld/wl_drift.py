"""Workload (d): long greedy generations for numerical-drift analysis (divergence point vs bf16-KV greedy)."""
import asyncio, aiohttp
from rwlib import *

def run(srv, a):
    text = codebase_text(); ntok = count_tokens(srv.url, text[:200000]); cpt = 200000 / ntok
    prompts = []
    for i in range(a.n):
        off = (i * 53321) % (len(text) - 100000)
        prompts.append("Repository file excerpts:\n" + text[off: off + int(a.ctx_tokens * cpt)] + "\n\n### REVIEW\nWrite a detailed code review of the code above, covering design, performance and bugs:\n1.")
    async def one(p, sem):
        async with sem:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
                return await stream_completion(s, srv.url, p, a.gen)
    async def go():
        sem = asyncio.Semaphore(a.conc); return await asyncio.gather(*[one(p, sem) for p in prompts])
    res = asyncio.run(go())
    toks = [tokenize(srv.url, r.get("text", "")) for r in res]
    return {"workload": "drift", "n": a.n, "ctx_tokens": a.ctx_tokens, "gen": a.gen, "texts": [r.get("text", "") for r in res], "token_ids": toks}
