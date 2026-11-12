"""Shared harness for AMD Triton ports of KV-quant decode paths (B8 / ctx32768 / Hq64 / Hkv8 / D128, K=1/2/4/8 query tokens).

Port API (bench/ports/<method>/port.py exposes make_port(dev) -> object with):
  name: str
  bytes_per_vec(D) -> float          average stored bytes of one K-or-V vector (all metadata that scales with tokens)
  calibrate(kf, vf)                  optional offline calibration; kf, vf [B,Hkv,S,D] fp16 sample
  write(kf, vf)                      quantise-on-write (Triton) of [B,Hkv,S,D] fp16 into the port's internal cache
  decode(q) -> [B,K,Hq,D]            fused dequant + attention, q [B,K,Hq,D] bf16; query j sees S-K+j+1 tokens (causal verify block)
"""
import json, math, os, time
import torch, torch.nn.functional as F

DEV, D, HKV = "cuda", 128, 8
SM = D ** -0.5


def make_spectrum(kind, g):
    """per-head eigen-spectrum and random rotation, so that x = z*sqrt(lam) @ R^T has LLM-like anisotropy."""
    if kind == "iid":
        return None
    i = torch.arange(1, D + 1, dtype=torch.float32, device=DEV)
    return i


def make_kv(B, S, kind="llm", seed=0, chunk=4096):
    """kind=iid: N(0,1).  kind=llm: per-kv-head random rotation of a power-law spectrum (K steeper than V) plus 4 outlier channels in K,
    mimicking the anisotropy SpectralQuant/KVQuant exploit.  Returns kf, vf [B,Hkv,S,D] fp16 and a sampler for queries."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    kf = torch.empty(B, HKV, S, D, device=DEV, dtype=torch.float16); vf = torch.empty_like(kf)
    if kind == "iid":
        kf.normal_(generator=g); vf.normal_(generator=g)
        return kf, vf, (lambda n: torch.randn(n, D, device=DEV, generator=g))
    i = torch.arange(1, D + 1, dtype=torch.float32, device=DEV)
    lk, lv = i ** -1.3, i ** -0.45
    lk = lk / lk.sum() * D; lv = lv / lv.sum() * D
    Rk = torch.linalg.qr(torch.randn(HKV, D, D, device=DEV, generator=g))[0]
    Rv = torch.linalg.qr(torch.randn(HKV, D, D, device=DEV, generator=g))[0]
    for s0 in range(0, S, chunk):
        n = min(chunk, S - s0)
        zk = torch.randn(B, HKV, n, D, device=DEV, generator=g) * lk.sqrt(); zv = torch.randn(B, HKV, n, D, device=DEV, generator=g) * lv.sqrt()
        kf[:, :, s0:s0 + n] = torch.einsum("bhnd,hed->bhne", zk, Rk).half(); vf[:, :, s0:s0 + n] = torch.einsum("bhnd,hed->bhne", zv, Rv).half()
    # channel outliers (per-channel scale) on K, KVQuant/KIVI-style
    ch = torch.randperm(D, device=DEV, generator=g)[:4]; kf[..., ch] *= 4
    def qs(n):  # queries correlated with the key distribution
        z = torch.randn(n, D, device=DEV, generator=g) * lk.sqrt()
        return z @ Rk[0].T
    return kf, vf, qs


def ref_attn(q, kf, vf):
    """fp32 reference. q [K,Hq,D]; kf,vf [Hkv,S,D]. query j sees S-K+j+1 positions."""
    K, Hq, _ = q.shape; G = Hq // HKV; S = kf.shape[1]
    qq = q.float().reshape(K, HKV, G, D)
    sc = torch.einsum("khgd,hsd->hgks", qq, kf.float()) * SM
    lim = S - K + 1 + torch.arange(K, device=q.device)
    sc = sc.masked_fill((torch.arange(S, device=q.device)[None, :] >= lim[:, None])[None, None], float("-inf"))
    o = torch.einsum("hgks,hsd->khgd", sc.softmax(-1), vf.float())
    return o.reshape(K, Hq, D)


def cos_rel(o, r):
    o, r = o.float().reshape(-1, D), r.float().reshape(-1, D)
    return F.cosine_similarity(o, r, dim=-1).mean().item(), ((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-12)).mean().item()


_FLUSH = None
def timeit(fn, reps=20, cold=True):
    """mean ms/call. cold=True flushes the 256 MB MI300X Infinity Cache (1 GB write) before every timed call (per-call events)."""
    global _FLUSH
    if cold:
        if _FLUSH is None: _FLUSH = torch.empty(1 << 30, dtype=torch.uint8, device=DEV)
        for _ in range(2): fn()
        tot = 0.0; n = max(10, reps // 2)
        for _ in range(n):
            _FLUSH.zero_(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); e.synchronize(); tot += s.elapsed_time(e)
        return tot / n
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def gbs(B, S, bytes_vec, ms):
    return B * HKV * S * 2 * bytes_vec / 1e9 / (ms / 1e3)


def merge_json(path, update):
    d = {}
    if os.path.exists(path):
        d = json.load(open(path))
    def rec(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict): rec(a[k], v)
            else: a[k] = v
    rec(d, update); json.dump(d, open(path, "w"), indent=1); return d
