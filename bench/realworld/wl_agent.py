"""Workload (a): long-context multi-turn coding-agent trace.

N concurrent sessions share a long prefix (system prompt + a real code base: vLLM v1 python sources);
each session runs T turns; every turn appends a "tool output" (a different real source file chunk) and a
question, then generates `gen` tokens greedily; the model's answer is appended to the history (as a real
agent would).  Measures throughput, TTFT, ITL and records outputs for drift comparison.
"""
import asyncio, time, aiohttp
from rwlib import *

Q = ["Summarise what the code above does in two sentences.",
     "Which function in the latest tool output is most performance critical? Explain briefly.",
     "List two possible bugs in the latest tool output.",
     "Write a one-line docstring for the main class in the latest tool output.",
     "What does the latest tool output import from other modules?",
     "Suggest one refactor for the latest tool output.",
     "How would you unit-test the latest tool output? One sentence.",
     "Give the name of the file above that is most related to the latest tool output."]


def build(srv, prefix_tokens, turns, chunk_tokens):
    text = codebase_text()
    ntok = count_tokens(srv.url, text[:200000]); cpt = 200000 / ntok  # chars per token
    prefix = "You are a coding agent. Repository contents follow.\n" + text[: int(prefix_tokens * cpt)]
    rest = text[int(prefix_tokens * cpt):]
    return prefix, rest, cpt


async def session(sid, srv, prefix, rest, cpt, turns, chunk_tokens, gen, out, distinct=False):
    hist = prefix
    if distinct:  # per-session private repo snapshot: no prefix sharing (stresses KV capacity)
        k = (sid * 1009 + 11) % max(1, len(rest) // 2)
        hist = f"You are a coding agent for project #{sid}. Repository contents follow.\n" + rest[k:k + len(prefix)]
    step = int(chunk_tokens * cpt)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        for t in range(turns):
            off = ((sid * 7 + t) * step) % max(1, len(rest) - step)
            hist += f"\n### TOOL OUTPUT (turn {t})\n{rest[off:off+step]}\n### QUESTION\n{Q[(sid+t) % len(Q)]}\n### ANSWER\n"
            r = await stream_completion(s, srv.url, hist, gen)
            if "error" in r: out.append({"sid": sid, "turn": t, "error": r["error"]}); return
            out.append({"sid": sid, "turn": t, "ttft": r["ttft"], "itls": r["itls"], "e2e": r["e2e"],
                        "n": r["chunks"], "text": r["text"]})
            hist += r["text"]


def run(srv, a):
    prefix, rest, cpt = build(srv, a.prefix_tokens, a.turns, a.chunk_tokens)
    # warm shared prefix once (a real deployment has it cached already)
    async def go():
        out = []
        async with aiohttp.ClientSession() as s:
            await stream_completion(s, srv.url, prefix + "\n### QUESTION\nHi\n### ANSWER\n", 4)
        t0 = time.perf_counter()
        await asyncio.gather(*[session(i, srv, prefix, rest, cpt, a.turns, a.chunk_tokens, a.gen, out, a.distinct)
                               for i in range(a.sessions)])
        return out, time.perf_counter() - t0
    out, wall = asyncio.run(go())
    ok = [o for o in out if "error" not in o]
    itl = [x for o in ok for x in o["itls"]]
    ttft = [o["ttft"] for o in ok]
    ntok = sum(o["n"] for o in ok)
    return {"workload": "agent", "sessions": a.sessions, "turns": a.turns, "prefix_tokens": a.prefix_tokens,
            "chunk_tokens": a.chunk_tokens, "gen": a.gen, "wall_s": wall, "errors": len(out) - len(ok),
            "out_tok_per_s": ntok / wall, "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
            "itl_p50": pct(itl, 50), "itl_p99": pct(itl, 99), "gpu_mem_gib": gpu_mem_used_gib(),
            "outputs": {f"{o['sid']}_{o['turn']}": o["text"] for o in ok}}
