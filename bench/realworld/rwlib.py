"""Shared helpers for UltraQuant real-world (serving-level) evaluation.

A `Server` context manager launches `vllm serve` for one (model, kv dtype, extra args)
config, waits for readiness, exposes scrape helpers, and always tears the server down
(process-group kill) so nothing is left running on the GPU box.
"""
import asyncio, json, os, re, signal, subprocess, sys, time
import aiohttp

VENV = os.environ.get("RW_VENV", "/root/vllm-venv")
MODELS = os.environ.get("RW_MODELS", "/root/models")


def vllm_src_root():
    import importlib.util
    return os.path.dirname(importlib.util.find_spec("vllm").origin)


class Server:
    def __init__(self, model, kv, port=8100, extra=(), maxlen=32768, gmu=0.90, log=None, env=None):
        self.model, self.kv, self.port = model, kv, port
        self.extra, self.maxlen, self.gmu = list(extra), maxlen, gmu
        self.log = log or os.path.join(os.environ.get("RW_LOGS", "/root/rw/logs"), f"server_{model}_{kv}_{port}.log")
        self.env = env or {}
        self.proc = None
        self.url = f"http://127.0.0.1:{port}"

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log), exist_ok=True)
        cmd = [os.environ.get("RW_VLLM_BIN", f"{VENV}/bin/vllm"), "serve", f"{MODELS}/{self.model}", "--served-model-name", "m",
               "--dtype", "bfloat16", "--max-model-len", str(self.maxlen),
               "--gpu-memory-utilization", str(self.gmu), "--port", str(self.port)]
        if self.kv != "auto":
            cmd += ["--kv-cache-dtype", self.kv]
        cmd += self.extra
        env = dict(os.environ, HF_HUB_OFFLINE="1", **self.env)
        self.lf = open(self.log, "w")
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self.lf, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        import urllib.request
        while True:
            if self.proc.poll() is not None:
                L = open(self.log, errors="ignore").read(); errs = [l for l in L.splitlines() if "Error" in l or "error" in l][:12]; raise RuntimeError("server died: " + " || ".join(x[:300] for x in errs) + " TAIL " + L[-600:])
            try:
                urllib.request.urlopen(self.url + "/v1/models", timeout=2).read()
                break
            except Exception:
                time.sleep(3)
            if time.time() - self.t0 > 900:
                raise RuntimeError("server start timeout")
        self.startup_s = time.time() - self.t0
        return self

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        time.sleep(3)

    def log_facts(self):
        """Parse KV capacity / memory facts out of the server log."""
        txt = open(self.log, errors="ignore").read()
        f = {}
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", txt)
        if m: f["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", txt)
        if m: f["max_concurrency_at_maxlen"] = float(m.group(2))
        m = re.search(r"Available KV cache memory: ([\d.]+) GiB", txt)
        if m: f["kv_cache_gib"] = float(m.group(1))
        m = re.search(r"Model loading took ([\d.]+) GiB", txt)
        if m: f["weights_gib"] = float(m.group(1))
        f["attn_backend"] = (re.findall(r"(?:Using|using) ([A-Za-z_0-9 ]+?) (?:attention )?backend", txt) or [None])[-1]
        f["startup_s"] = round(self.startup_s, 1)
        return f

    def metrics(self):
        import urllib.request
        txt = urllib.request.urlopen(self.url + "/metrics", timeout=10).read().decode()
        d = {}
        for line in txt.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            k, _, v = line.rpartition(" ")
            d[k] = float(v)
        return d


def gpu_mem_used_gib():
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", "--json"], text=True)
        j = json.loads(out)
        card = next(iter(j.values()))
        return int(card["VRAM Total Used Memory (B)"]) / 2**30
    except Exception:
        return None


async def stream_completion(session, url, prompt, max_tokens, temperature=0.0, extra=None):
    """One streamed /v1/completions request. Returns dict with text, ttft, itls, ntok."""
    body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "ignore_eos": False, "stream_options": {"include_usage": True}}
    if extra: body.update(extra)
    t0 = time.perf_counter(); ttft = None; last = t0; itls = []; text = []; n = 0; ntok = None
    async with session.post(url + "/v1/completions", json=body) as r:
        if r.status != 200:
            return {"error": f"{r.status} {(await r.text())[:300]}"}
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            if j.get("usage"): ntok = j["usage"].get("completion_tokens")
            if not j.get("choices"): continue
            ch = j["choices"][0]
            if ch.get("text"):
                now = time.perf_counter()
                if ttft is None: ttft = now - t0
                else: itls.append(now - last)
                last = now; text.append(ch["text"]); n += 1
            if j.get("usage"): pass
    # With speculative decoding one SSE chunk can carry several tokens: use usage.completion_tokens as the token
    # count and spread each inter-chunk gap evenly over the tokens it carried (per-token ITL approximation).
    ntok = ntok or n
    tpc = max(1.0, (ntok - 1) / max(1, len(itls))) if itls else 1.0
    itls_tok = [x / tpc for x in itls for _ in range(int(round(tpc)))] if tpc > 1.0 else itls
    return {"text": "".join(text), "ttft": ttft, "itls": itls_tok, "chunks": n, "ntok": ntok, "tok_per_chunk": tpc, "e2e": time.perf_counter() - t0}


def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); i = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return xs[i]


def count_tokens(server_url, text):
    import urllib.request
    req = urllib.request.Request(server_url + "/tokenize", data=json.dumps({"model": "m", "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return len(json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"])


def tokenize(server_url, text):
    import urllib.request
    req = urllib.request.Request(server_url + "/tokenize", data=json.dumps({"model": "m", "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]


def codebase_text(root=None, max_chars=2_000_000):
    root = root or os.path.join(vllm_src_root(), "v1")
    """Deterministic pseudo-'repo' text: real python sources, sorted, with file headers."""
    parts = []; tot = 0
    for dp, dn, fn in sorted(os.walk(root)):
        dn.sort()
        for f in sorted(fn):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                s = open(p, errors="ignore").read()
                if 2000 < len(s) < 60000:
                    parts.append(f"### FILE {os.path.relpath(p, root)}\n{s}\n")
                    tot += len(s)
                    if tot > max_chars: return "\n".join(parts)
    return "\n".join(parts)
