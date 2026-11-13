"""Scalar baselines with honest rate accounting (bits per value incl. scales)."""
from __future__ import annotations

from functools import lru_cache

import torch

from .rotation import RHT

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)


def fp8_e4m3(x, block=128, scale_bits=32):
    """FP8 e4m3 with one fp32 amax scale per `block` values (block=None: no scale, 8.0 bpv)."""
    if block is None:
        return x.to(torch.float8_e4m3fn).to(x.dtype), 8.0
    xs = x.reshape(-1, block).float()
    s = (xs.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-30)
    q = (xs / s).to(torch.float8_e4m3fn).float() * s
    return q.reshape(x.shape).to(x.dtype), 8 + scale_bits / block


def _round_e2m1(v):
    a = v.abs().double()
    grid = E2M1.to(v.device)
    i = (a[..., None] - grid).abs().argmin(-1)
    return (grid[i] * torch.sign(v.double())).to(v.dtype)


def nvfp4(x, block=16):
    """NVFP4: e2m1 values, e4m3 scale per 16, one fp32 scale per tensor -> 4.5 bpv."""
    xf = x.float()
    g = (xf.abs().amax() / (448 * 6)).clamp(min=1e-30)
    xs = xf.reshape(-1, block)
    s = (xs.abs().amax(-1, keepdim=True) / 6 / g).to(torch.float8_e4m3fn).float() * g
    s = s.clamp(min=1e-30)
    q = _round_e2m1((xs / s).clamp(-6, 6)) * s
    return q.reshape(x.shape).to(x.dtype), 4 + 8 / block


def int_grid(x, bits=4, block=32, scale_bits=16):
    """Symmetric uniform integer grid with one fp16 absmax scale per block."""
    L = 2 ** (bits - 1) - 1
    xs = x.reshape(-1, block).float()
    s = (xs.abs().amax(-1, keepdim=True) / L).half().float().clamp(min=1e-30)
    q = torch.round(xs / s).clamp(-L - 1, L) * s
    return q.reshape(x.shape).to(x.dtype), bits + scale_bits / block


@lru_cache(maxsize=None)
def lloyd_max(bits: int, iters: int = 300):
    """Lloyd-Max centroids for N(0,1) (numerical, fine grid)."""
    t = torch.linspace(-8, 8, 400001, dtype=torch.float64)
    p = torch.exp(-t * t / 2)
    n = 2 ** bits
    c = torch.linspace(-2.5, 2.5, n, dtype=torch.float64) * (1 + 0.3 * (bits - 2))
    for _ in range(iters):
        b = (c[1:] + c[:-1]) / 2
        idx = torch.bucketize(t, b)
        num = torch.zeros(n, dtype=torch.float64).scatter_add_(0, idx, p * t)
        den = torch.zeros(n, dtype=torch.float64).scatter_add_(0, idx, p)
        c = num / den
    return c


def turboquant_mse(x, bits=3, D=128, norm_bits=16):
    """TurboQuant (MSE variant, our re-implementation): random rotation, per-vector
    norm (fp16), Lloyd-Max scalar codebook for N(0,1) on each rotated coordinate."""
    rot = RHT(D, seed=0x7A7A)
    z = rot.fwd(x.reshape(-1, D).double())
    r = z.pow(2).mean(-1, keepdim=True).sqrt().half().double().clamp(min=1e-30)
    c = lloyd_max(bits).to(x.device)
    b = (c[1:] + c[:-1]) / 2
    q = c[torch.bucketize(z / r, b)] * r
    return rot.inv(q).to(x.dtype).reshape(x.shape), bits + norm_bits / D
