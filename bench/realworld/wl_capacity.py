"""Capacity probe: server start-up facts only (KV cache tokens, GiB, weights) + a 1-request sanity completion."""
import asyncio, aiohttp
from rwlib import *
def run(srv, a):
    async def go():
        async with aiohttp.ClientSession() as s:
            return await stream_completion(s, srv.url, "The capital of France is", 16)
    r = asyncio.run(go()); return {"workload": "capacity", "sanity_text": r.get("text"), "gpu_mem_gib": gpu_mem_used_gib()}
