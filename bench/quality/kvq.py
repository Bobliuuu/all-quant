"""KV-cache quantisation methods as quantise->dequantise ("fake quant") in the HF cache path.

Every method exposes:  name, bits_desc, kind ('official'|'simulated'), bytes_per_vec(D) (K or V vector of D elems,
incl. scales) and a cache class factory.  Quantise happens on cache write, dequantised values are what attention reads,
for both chunked prefill and token-by-token decode.
"""
import math, os, sys
import torch
from transformers import DynamicCache

E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])


def _fp4_round(y):  # y already scaled to [-6,6]; nearest e2m1 value
    g = E2M1.to(y.device, y.dtype)
    idx = (y.abs().unsqueeze(-1) - g).abs().argmin(-1)
    return torch.sign(y) * g[idx]


def _grp(x, G):
    return x.reshape(*x.shape[:-1], x.shape[-1] // G, G)


# ---------------------------------------------------------------- per-vector (per-token) fake quantisers [B,H,T,D]
def fq_fp16(x): return x

def fq_fp8(x):  # plain e4m3 cast, scale 1.0 (what vLLM does without calibrated scales)
    return x.float().clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype)

def fq_fp8_scaled(x):  # per-token-head dynamic scale
    xf = x.float(); s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 448
    return ((xf / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)

def fq_int8(x):  # per-token-head symmetric
    xf = x.float(); s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return ((xf / s).round().clamp(-127, 127) * s).to(x.dtype)

def fq_int4_qserve(x):  # QoQ-style per-token-head asymmetric 4-bit (no SmoothAttention), fp16 scale+zero
    xf = x.float(); mn = xf.amin(-1, keepdim=True); mx = xf.amax(-1, keepdim=True)
    s = ((mx - mn) / 15).clamp_min(1e-8).half().float(); z = (-mn / s).round().clamp(0, 15)
    return (((xf / s).round() + z).clamp(0, 15) - z).mul(s).to(x.dtype)

def fq_uq_fp4(x):  # repo quantiser as-is (uqmt/attn.py): E2M1, UE8M0 per 32, ceil(log2(amax/6))
    from uqmt.attn import quantize_fp4, dequantize_fp4
    p, s = quantize_fp4(x); return dequantize_fp4(p, s).to(x.dtype)

def fq_mxfp4(x):  # OCP MX: shared exponent = floor(log2 amax) - emax_elem(2), E8M0, 32 group
    g = _grp(x.float(), 32); amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    sc = torch.exp2(torch.floor(torch.log2(amax)) - 2)
    return (_fp4_round((g / sc).clamp(-6, 6)) * sc).reshape(x.shape).to(x.dtype)

def fq_nvfp4(x):  # E4M3 scale per 16 + fp32 per-tensor(-per-write) scale
    g = _grp(x.float(), 16); gamax = g.abs().amax().clamp_min(1e-30)
    gs = gamax / (6 * 448)
    amax = g.abs().amax(-1, keepdim=True)
    sc = (amax / 6 / gs).clamp(max=448).to(torch.float8_e4m3fn).float().clamp_min(2 ** -9) * gs
    return (_fp4_round((g / sc).clamp(-6, 6)) * sc).reshape(x.shape).to(x.dtype)

def _wht(x):  # normalised Walsh-Hadamard along last dim (power of 2)
    D = x.shape[-1]; y = x.float().reshape(-1, D); h = 1
    while h < D:
        y = y.reshape(-1, D // (2 * h), 2, h); a, b = y[:, :, 0], y[:, :, 1]
        y = torch.stack([a + b, a - b], 2); h *= 2
    return (y.reshape(x.shape) / math.sqrt(D))

def _uq_c(x, c):  # paper-style scale: E = round(log2(c*amax)) (clips top of block)
    g = _grp(x.float(), 32); amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    sc = torch.exp2(torch.round(torch.log2(c * amax)))
    return (_fp4_round((g / sc).clamp(-6, 6)) * sc).reshape(x.shape)

def fq_uq_fp4_paper_k(x):  # key path: WHT rotation + c=0.156 round scale (paper recipe), inverse-rotated for fake-quant
    return _wht(_uq_c(_wht(x), 0.156)).to(x.dtype)  # WHT is involutive when normalised
def fq_uq_fp4_paper_v(x):  # values: no rotation, c=0.156
    return _uq_c(x, 0.156).to(x.dtype)
def fq_uq_fp4_whtk(x):  # repo ceil-scale + WHT on keys (isolates the rotation)
    return _wht(fq_uq_fp4(_wht(x))).to(x.dtype)


# ---------------------------------------------------------------- amd-experiments rotation methods (official code)
_AMD = None
def _amd():
    global _AMD
    if _AMD is None:
        for p in (os.environ.get("AMD_KERNELS", ""), os.path.expanduser("~/amd-experiments-ro/kernels"), "/work/amdk"):
            if p and os.path.isdir(p): sys.path.insert(0, p); break
        import block_quant_rocm as bq
        import turboquant_mi300x as tq
        _AMD = (bq, tq)
    return _AMD

_QCACHE = {}
def make_amd_fq(method, bits):
    bq, tq = _amd()
    def fq(x):
        B, H, T, D = x.shape; dev = x.device
        key = (method, bits, str(dev))
        if key not in _QCACHE:
            _QCACHE[key] = (tq.TurboQuantMI300X(head_dim=D, bits=bits, device=str(dev)) if method == "turbo"
                            else bq.make_quantizer(method, bits=bits, head_dim=D, device=str(dev)))
        q = _QCACHE[key]; x2 = x.reshape(-1, D).float().contiguous()
        if method == "turbo":
            out = q.decompress_tensor(q.compress_tensor(x2), x2.shape)
        else:
            out = q.decompress(q.compress(x2), x2.shape)
        return out.reshape(x.shape).to(x.dtype)
    return fq


# ---------------------------------------------------------------- KIVI (simulated reimplementation)
def _asym(x, bits, dim):  # asymmetric uniform quant with min/max along `dim`
    xf = x.float(); mn = xf.amin(dim, keepdim=True); mx = xf.amax(dim, keepdim=True)
    s = ((mx - mn) / (2 ** bits - 1)).clamp_min(1e-8).half().float(); z = mn.half().float()
    return (((xf - z) / s).round().clamp(0, 2 ** bits - 1) * s + z)

def kivi_k_block(k, bits, G=32):  # k [B,H,T,D], T % G == 0 ; per-channel: stats over G tokens
    B, H, T, D = k.shape
    return _asym(k.reshape(B, H, T // G, G, D), bits, 3).reshape(B, H, T, D).to(k.dtype)

def kivi_v_block(v, bits, G=32):  # per-token, groups of G along channels
    return _asym(_grp(v, G), bits, -1).reshape(v.shape).to(v.dtype)


class KiviCache(DynamicCache):
    """KIVI: recent R tokens stay fp16; older tokens in blocks of G are quantised in place (K per-channel, V per-token)."""
    def __init__(self, bits, G=32, R=128):
        super().__init__(); self.bits, self.G, self.R = bits, G, R; self.flushed = {}
    def update(self, k, v, layer_idx, cache_kwargs=None):
        K, V = super().update(k, v, layer_idx, cache_kwargs)
        n = K.shape[2]; f = self.flushed.get(layer_idx, 0)
        t = ((n - self.R) // self.G) * self.G if n > self.R else 0
        if t > f:
            K[:, :, f:t] = kivi_k_block(K[:, :, f:t], self.bits, self.G)
            V[:, :, f:t] = kivi_v_block(V[:, :, f:t], self.bits, self.G)
            self.flushed[layer_idx] = t
        return K, V


def kivi_seq(x, is_key, bits, G=32, R=128):  # whole-sequence equivalent, for the attention-error metric
    T = x.shape[2]; t = ((T - R) // G) * G if T > R else 0
    if t == 0: return x
    y = x.clone(); y[:, :, :t] = (kivi_k_block if is_key else kivi_v_block)(x[:, :, :t], bits, G); return y


# ---------------------------------------------------------------- KVQuant-style (simulated): per-channel K + non-uniform codebook + 1% outliers
def _nuq_fit(x, bits, iters=15):  # 1-D k-means (Lloyd) on normalised values, x flat sample
    xs = x.float().flatten(); q = torch.linspace(0.0, 1.0, 2 ** bits + 2, device=x.device)[1:-1]
    c = torch.quantile(xs[torch.randperm(xs.numel(), device=xs.device)[:200000]], q)
    for _ in range(iters):
        a = (xs.unsqueeze(-1) - c).abs().argmin(-1)
        for j in range(len(c)):
            m = a == j
            if m.any(): c[j] = xs[m].mean()
    return c


N_LAYERS = [32]  # set by qeval after model load


def is_edge(layer, n=2):
    return layer < n or layer >= N_LAYERS[0] - n


class GenericCache(DynamicCache):
    def __init__(self, fq_k, fq_v, edge_bf16=False):
        super().__init__(); self.fq_k, self.fq_v, self.edge = fq_k, fq_v, edge_bf16
    def update(self, k, v, layer_idx, cache_kwargs=None):
        if self.edge and is_edge(layer_idx): return super().update(k, v, layer_idx, cache_kwargs)
        return super().update(self.fq_k(k), self.fq_v(v), layer_idx, cache_kwargs)


# ---------------------------------------------------------------- registry
class Method:
    def __init__(self, name, family, bits, kind, bpv, fq_k=None, fq_v=None, cache=None, seq=None, note="", edge_bf16=False):
        self.edge = edge_bf16
        self.name, self.family, self.bits, self.kind, self.bpv = name, family, bits, kind, bpv
        self.fq_k, self.fq_v, self._cache, self.seq, self.note = fq_k, fq_v, cache, seq, note
    def make_cache(self):
        return self._cache() if self._cache else GenericCache(self.fq_k, self.fq_v, self.edge)
    def seq_fq(self, x, is_key, layer=-1):  # whole-sequence fake quant (for attention-error metric)
        if self.edge and layer >= 0 and is_edge(layer): return x
        if self.seq: return self.seq(x, is_key)
        return (self.fq_k if is_key else self.fq_v)(x)
    def bytes_per_vec(self, D=128): return self.bpv(D)


def build_registry():
    R = {}
    def add(m): R[m.name] = m
    add(Method("fp16", "fp16", 16, "reference", lambda D: 2 * D, fq_fp16, fq_fp16))
    add(Method("fp8_e4m3", "FP8", 8, "simulated", lambda D: D, fq_fp8, fq_fp8, note="plain e4m3 cast, scale 1.0"))
    add(Method("fp8_e4m3_scaled", "FP8", 8, "simulated", lambda D: D + 2, fq_fp8_scaled, fq_fp8_scaled, note="per-token-head dynamic scale"))
    add(Method("int8", "INT8", 8, "simulated", lambda D: D + 2, fq_int8, fq_int8, note="per-token-head symmetric"))
    add(Method("uq_fp4", "UltraQuant", 4.25, "official", lambda D: D // 2 + D // 32, fq_uq_fp4, fq_uq_fp4,
               note="repo quantize_fp4 (E2M1 + UE8M0/32, ceil scale, no rotation), exactly what uqmt kernel reads"))
    add(Method("uq_fp4_whtk", "UltraQuant", 4.25, "simulated", lambda D: D // 2 + D // 32, fq_uq_fp4_whtk, fq_uq_fp4,
               note="repo quantiser + WHT on K only (paper's rotation)"))
    add(Method("uq_fp4_c156", "UltraQuant", 4.25, "simulated", lambda D: D // 2 + D // 32, fq_uq_fp4_paper_v, fq_uq_fp4_paper_v,
               note="paper scale rule: E=round(log2(0.156*amax)) with clip, no rotation"))
    add(Method("uq_fp4_paper", "UltraQuant", 4.25, "simulated", lambda D: D // 2 + D // 32, fq_uq_fp4_paper_k, fq_uq_fp4_paper_v, edge_bf16=True,
               note="paper recipe: c=0.156 clip scale + WHT on K (Q rotated, q.k invariant) + BF16 first/last 2 layers; Q kept bf16 (paper: FP8); bytes exclude BF16 edge layers"))
    add(Method("mxfp4", "MXFP4", 4.25, "simulated", lambda D: D // 2 + D // 32, fq_mxfp4, fq_mxfp4, note="OCP MX floor-exponent scale / 32"))
    add(Method("nvfp4", "NVFP4", 4.5, "simulated", lambda D: D // 2 + D // 16, fq_nvfp4, fq_nvfp4, note="E4M3 scale / 16 + fp32 global per write"))
    add(Method("int4_qserve", "QServe-style", 4.5, "simulated", lambda D: D // 2 + 4, fq_int4_qserve, fq_int4_qserve,
               note="per-token-head asym int4, fp16 scale+zero; no SmoothAttention"))
    for b in (2, 4):
        add(Method(f"kivi{b}", "KIVI", b + 1.0, "simulated", lambda D, b=b: D * b // 8 + (D // 32) * 4,
                   cache=lambda b=b: KiviCache(b), seq=lambda x, k, b=b: kivi_seq(x, k, b),
                   note="K per-channel (32-token blocks), V per-token (group 32), fp16 residual 128 tokens, asym fp16 scale/zero; bytes exclude fp16 residual"))
    for meth, fam in (("turbo", "TurboQuant"), ("planar", "PlanarQuant"), ("iso", "IsoQuant"), ("rotor", "RotorQuant")):
        for b in ((2, 3, 4) if meth == "turbo" else (3, 4)):
            f = make_amd_fq(meth, b)
            add(Method(f"{meth}{b}", fam, b + 1.0, "official-port", lambda D, b=b: {2: 36, 3: 52, 4: 68}[b] * D // 128, f, f,
                       note="amd-experiments kernels (torch/Triton) run unchanged on CUDA; no QJL"))
    return R
