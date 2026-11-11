"""Workload (b2): needle-in-a-haystack (passkey) retrieval over real code text, several depths/lengths."""
import asyncio, random, aiohttp
from rwlib import *

def run(srv, a):
    text = codebase_text(); ntok = count_tokens(srv.url, text[:200000]); cpt = 200000 / ntok
    rng = random.Random(1); jobs = []
    for L in (a.ctx_tokens // 2, a.ctx_tokens):
        for d in (0.05, 0.25, 0.5, 0.75, 0.95):
            for rep in range(a.n):
                key = str(rng.randrange(10**6, 10**7))
                off = rng.randrange(0, 100000); hay = (text + text)[off: off + int(L * cpt)]
                pos = int(len(hay) * d); pos = hay.rfind("\n", 0, pos)
                hay = hay[:pos] + f"\n# NOTE: the secret access code is {key}. Remember it.\n" + hay[pos:]
                jobs.append((L, d, key, hay + "\n\n# QUESTION: What is the secret access code mentioned in the NOTE above?\n# ANSWER: The secret access code is"))
    async def one(j, sem):
        async with sem:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
                return await stream_completion(s, srv.url, j[3], 12)
    async def go():
        sem = asyncio.Semaphore(a.conc); return await asyncio.gather(*[one(j, sem) for j in jobs])
    res = asyncio.run(go()); by = {}
    for (L, d, key, _), r in zip(jobs, res):
        by.setdefault(f"L{L}_d{d}", []).append(float(key in r.get("text", "")))
    allv = [v for x in by.values() for v in x]
    return {"workload": "needle", "acc": sum(allv) / len(allv), "n": len(allv), "by_cell": {k: sum(v) / len(v) for k, v in by.items()}}
