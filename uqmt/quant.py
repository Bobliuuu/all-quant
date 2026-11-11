"""Configurable FP4 quantiser variants for the UltraQuant KV cache.

Everything here produces tensors in the *existing* cache layout (packed uint8
codes [..., D/2], UE8M0 scale bytes [..., D/32], value = 2**(s-127)), so
``uqmt.uq_decode`` works unchanged on the output.  Only the scale *choice* and
an optional pre-rotation differ from ``uqmt.attn.quantize_fp4``.

Scale rules (``rule=``)
  'ceil'   : E = ceil(log2(amax/6))      (our default; never clips)
  'paper'  : E = round(log2(c*amax)), c=0.156 default, then clip to +-6
             (arXiv 2606.20474; c < 1/6 means scale is sometimes below amax/6,
             so the largest element can clip but the grid is finer elsewhere)

Hadamard rotation (arXiv 2606.20474): K and Q are multiplied by the same
orthonormal H (fwht below) *before* quantising K / before the kernel, so
q.k is exactly invariant in exact arithmetic; V is NOT rotated.  Fused-friendly
layout: rotate along head_dim in blocks of 2**k <= D (block-diagonal H; D=128
uses a single 128-point WHT, 7 butterfly stages, which maps to one wavefront
doing log2(D) add/sub passes in registers).  Fuse it into the K write kernel
(quantise-on-write) and into the Q load of the decode kernel; nothing else in
the kernel changes because K scores use rotated-q . dequant(rotated-k).

NVFP4 (``quantize_nvfp4``): 16-element groups with E4M3 scales (+ optional
per-tensor fp32 scale).  The E4M3 scale byte is NOT UE8M0, so this output is
not readable by uq_decode (needs a scale-path change in the kernel); use
``dequantize_nvfp4`` for studies.
"""
import math
from typing import Optional, Sequence

import torch

from .attn import FP4_VALUES, BLOCK, dequantize_fp4

PAPER_C = 0.156
_THRESH = None


def _grid(dev):
    return torch.tensor(FP4_VALUES, device=dev)


def _encode(y):
    """y [..., D] already scaled and clamped to [-6, 6] -> packed uint8 [..., D/2].
    Nearest-grid rounding, identical to attn.quantize_fp4."""
    D = y.shape[-1]
    code = (y.abs().unsqueeze(-1) - _grid(y.device)).abs().argmin(-1)
    code = code.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def scale_exponent(amax, rule="ceil", c=PAPER_C):
    amax = amax.clamp_min(1e-30)
    if rule == "ceil":
        e = torch.ceil(torch.log2(amax / 6.0))
    elif rule == "paper":
        e = torch.round(torch.log2(c * amax))
    else:
        raise ValueError(rule)
    return e.clamp(-127, 127)


def quantize(x: torch.Tensor, rule: str = "ceil", c: float = PAPER_C):
    """x [..., D] -> (packed uint8 [..., D/2], UE8M0 scales uint8 [..., D/32])."""
    D = x.shape[-1]
    assert D % BLOCK == 0
    xg = x.float().reshape(*x.shape[:-1], D // BLOCK, BLOCK)
    e = scale_exponent(xg.abs().amax(-1, keepdim=True), rule, c)
    y = (xg / torch.exp2(e)).clamp(-6, 6).reshape(*x.shape[:-1], D)
    return _encode(y), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()


def fake_quant(x, rule="ceil", c=PAPER_C, rotate=False, dtype=None):
    """quantise->dequantise (simulation).  rotate=True rotates before and un-rotates after."""
    xr = fwht(x.float()) if rotate else x.float()
    p, s = quantize(xr, rule, c)
    y = dequantize_fp4(p, s)
    if rotate:
        y = fwht(y)  # orthonormal H is its own inverse
    return y.to(dtype or x.dtype)


# ------------------------------------------------------------------ Hadamard
def fwht(x: torch.Tensor, block: Optional[int] = None) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform along the last dim, in blocks of
    `block` (default: whole last dim; must be a power of two).  H is symmetric and
    H@H = I so fwht is its own inverse.  O(D log D), log2(block) butterfly passes."""
    D = x.shape[-1]
    n = block or D
    assert n & (n - 1) == 0 and D % n == 0
    y = (x if x.dtype in (torch.float32, torch.float64) else x.float()).reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0], y[:, :, 1]
        y = torch.stack([a + b, a - b], 2).reshape(-1, n)
        h *= 2
    return (y / math.sqrt(n)).reshape(x.shape).to(x.dtype)


def hadamard_matrix(n, device="cpu", dtype=torch.float32):
    return fwht(torch.eye(n, device=device, dtype=dtype))


def rotate_kq(k, q, block=None):
    """Rotate K [...,D] and Q [...,D] with the same H (scores invariant). V untouched."""
    return fwht(k, block), fwht(q, block)


# ----------------------------------------------------------------- per-layer
def bf16_layers(num_layers: int, n: int = 2) -> set:
    """Layers kept in BF16: first n and last n (paper default n=2)."""
    return set(range(min(n, num_layers))) | set(range(max(0, num_layers - n), num_layers))


def layer_policy(num_layers: int, n: int = 2, attention_layers: Optional[Sequence[int]] = None):
    """layer -> 'bf16' | 'fp4'.  For hybrid models pass attention_layers (indices that own a
    KV cache); boundary counting is then over those layers only."""
    ids = list(attention_layers) if attention_layers is not None else list(range(num_layers))
    keep = bf16_layers(len(ids), n)
    return {l: ('bf16' if i in keep else 'fp4') for i, l in enumerate(ids)}


# --------------------------------------------------------------------- NVFP4
def quantize_nvfp4(x: torch.Tensor, tensor_scale: Optional[float] = None):
    """16-group E4M3-scale FP4.  Returns (packed uint8 [..., D/2], scale e4m3 [..., D/16],
    tensor_scale float).  scale_g = e4m3(amax_g / 6 / ts); ts = amax_all / (6*448) by default."""
    D = x.shape[-1]
    assert D % 16 == 0
    xf = x.float()
    ts = tensor_scale if tensor_scale is not None else max(float(xf.abs().max()) / (6 * 448), 1e-30)
    xg = xf.reshape(*x.shape[:-1], D // 16, 16)
    s = (xg.abs().amax(-1, keepdim=True) / 6.0 / ts).clamp(2 ** -9, 448).to(torch.float8_e4m3fn)
    sf = s.float() * ts
    y = (xg / sf).clamp(-6, 6).reshape(*x.shape[:-1], D)
    return _encode(y), s.squeeze(-1).contiguous(), ts


def dequantize_nvfp4(packed, scales, ts):
    g = _grid(packed.device)
    lo, hi = packed & 0xF, packed >> 4
    def dec(c):
        v = g[(c & 7).long()]
        return torch.where((c & 8) != 0, -v, v)
    D = packed.shape[-1] * 2
    out = torch.stack([dec(lo), dec(hi)], -1).reshape(*packed.shape[:-1], D)
    return out * (scales.float() * ts).repeat_interleave(16, -1)


# ------------------------------------------------------- quantise-on-write kernel
try:
    from .attn import triton, tl, HAVE_TRITON
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


@triton.jit
def _qow_kernel(X, CODE, SC, SLOT, N_TOK, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                RULE: tl.constexpr, C: tl.constexpr):
    """One program per (token, kv head, 32-group).  X [N, Hkv, D]; SLOT [N] int32 =
    page*PAGE + offset in the paged cache; CODE [pages,Hkv,PAGE,D/2]; SC [pages,Hkv,PAGE,D/32]."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.program_id(2)
    NG: tl.constexpr = D // 32
    o = tl.arange(0, 32)
    x = tl.load(X + (n * HKV + h) * D + g * 32 + o).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1e-30)
    if RULE == 0:
        e = tl.ceil(tl.log2(amax / 6.0))
    else:
        # round-half-away like torch.round only differs at exact .5 ties of log2; torch.round is
        # half-to-even, reproduced below
        t = tl.log2(C * amax)
        f = tl.floor(t)
        d = t - f
        e = tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, tl.where((f % 2.0) == 0.0, f, f + 1.0)))
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    y = x / tl.exp2(e)
    y = tl.minimum(tl.maximum(y, -6.0), 6.0)
    a = tl.abs(y)
    # nearest grid {0,.5,1,1.5,2,3,4,6}; argmin takes the first (lower) index on ties
    c = (a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a > 1.75).to(tl.int32) \
        + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32) + (a > 5.0).to(tl.int32)
    c = c | ((y < 0).to(tl.int32) << 3)
    c2 = tl.reshape(c, [16, 2])
    lo, hi = tl.split(c2)
    byte = (lo | (hi << 4)).to(tl.uint8)
    slot = tl.load(SLOT + n).to(tl.int64)
    page = slot // PAGE
    off = slot % PAGE
    row = (page * HKV + h) * PAGE + off
    tl.store(CODE + row * (D // 2) + g * 16 + tl.arange(0, 16), byte)
    tl.store(SC + row * NG + g, (e + 127.0).to(tl.uint8))


def quantize_on_write(x, code, scale, slots, rule="ceil", c=PAPER_C):
    """Quantise x [N, Hkv, D] (N new tokens) and scatter into the paged FP4 cache tensors
    `code` [pages,Hkv,PAGE,D/2] and `scale` [pages,Hkv,PAGE,D/32] at flat slots [N]
    (slot = page*PAGE + in-page offset).  Bit-exact with quant.quantize (rule 'ceil'/'paper')."""
    N, HKV, D = x.shape
    PAGE = code.shape[2]
    x = x.contiguous()
    slots = slots.to(torch.int32)
    _qow_kernel[(N, HKV, D // 32)](x, code, scale, slots, N, HKV=HKV, D=D, PAGE=PAGE,
                                   RULE=0 if rule == "ceil" else 1, C=float(c), num_warps=1)
