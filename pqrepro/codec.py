"""pq-repro: a PearlQuant-style E8 KV codec (our reproduction, not Pearl's code).

Per cached vector x (length D, e.g. one head of K or V, D = 128):
  1. rotate:      z = H diag(signs) x / sqrt(D)          (random Hadamard)
  2. scale:       s = fp16(rms(z) / alpha_b * m*)          (one fp16 per vector)
                  m* = best of n_search multipliers (free: s absorbs it)
  3. split z/s into D/8 groups of 8 and round each to the E8 codebook:
       kind="ball":    2^b most probable (smallest-norm) E8 points, leader index
       kind="voronoi": E8 / qE8 nested-lattice code (NestQuant), b = ceil(8 log2 q)
  4. store D/8 indices of b bits + 16-bit scale.
Rate = b/8 + 16/D bits per value, exactly (no entropy coding).
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import torch

from .e8 import _w, _wf

from . import ball as B
from . import voronoi as Vo
from .rotation import RHT


def _quant_groups(u, kind, param, return_idx=False):
    if kind == "ball":
        return B.quantize(u, B.ball_tables(param), return_idx)
    return Vo.quantize(u, param, return_idx)


_ALPHA_FILE = os.path.join(os.path.dirname(__file__), "alphas.json")


def calibrate_alpha(kind: str, param: int, D: int = 128) -> float:
    """Cached in pqrepro/alphas.json (deterministic: fixed seed)."""
    key = f"{kind}:{param}:{D}"
    tab = json.load(open(_ALPHA_FILE)) if os.path.exists(_ALPHA_FILE) else {}
    if key not in tab:
        tab[key] = _calibrate_alpha(kind, param, D)
        try:
            json.dump(tab, open(_ALPHA_FILE, "w"), indent=1, sort_keys=True)
        except OSError:
            pass
    return tab[key]


@lru_cache(maxsize=None)
def _calibrate_alpha(kind: str, param: int, D: int = 128, n: int = 3000, seed: int = 1) -> float:
    """alpha minimizing MSE for per-vector-normalized iid Gaussian vectors (golden search)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D, generator=g, dtype=torch.float64)
    x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
    u0 = x.reshape(-1, 8)

    def mse(a):
        return ((_quant_groups(u0 * a, kind, param) / a - u0) ** 2).mean().item()

    lo, hi = 0.5, 64.0
    # coarse log grid then golden refine
    grid = [lo * (hi / lo) ** (i / 24) for i in range(25)]
    vals = [mse(a) for a in grid]
    i = min(range(len(vals)), key=vals.__getitem__)
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    phi = (5 ** 0.5 - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = mse(c), mse(d)
    for _ in range(12):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = mse(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = mse(d)
    return (a + b) / 2


class E8KVCodec:
    def __init__(self, kind="ball", param=24, D=128, n_search=8, search_span=0.25, seed=0x5051,
                 scale_dtype=torch.float16):
        assert D % 8 == 0
        self.kind, self.param, self.D = kind, param, D
        self.bits_group = param if kind == "ball" else Vo.bits_for_q(param)
        self.rot = RHT(D, seed)
        self.alpha = calibrate_alpha(kind, param, D)
        self.mults = torch.linspace(1 - search_span, 1 + search_span, n_search) if n_search > 1 else torch.ones(1)
        self.scale_dtype = scale_dtype

    @property
    def bpv(self):
        return self.bits_group / 8 + torch.finfo(self.scale_dtype).bits / self.D

    @property
    def name(self):
        k = f"ball{self.param}" if self.kind == "ball" else f"vor-q{self.param}"
        return f"pq-{k}"

    def encode(self, x: torch.Tensor):
        """x (..., D) -> (idx (..., D/8) int64, scale (...,) fp16)."""
        shp = x.shape
        z = self.rot.fwd(_w(x.reshape(-1, self.D)))
        rms = z.pow(2).mean(-1).sqrt().clamp(min=1e-30)
        best_err = None
        for m in self.mults.tolist():
            s = _w((rms / (self.alpha * m)).to(self.scale_dtype)).clamp(min=1e-30)
            u = (z / s[:, None]).reshape(-1, 8)
            q, idx = _quant_groups(u, self.kind, self.param, True)
            err = ((q.reshape(z.shape) * s[:, None] - z) ** 2).sum(-1)
            idx = idx.reshape(z.shape[0], -1)
            if best_err is None:
                best_err, best_idx, best_s = err, idx, s
            else:
                w = err < best_err
                best_err = torch.where(w, err, best_err)
                best_idx = torch.where(w[:, None], idx, best_idx)
                best_s = torch.where(w, s, best_s)
        return best_idx.reshape(*shp[:-1], self.D // 8), best_s.to(self.scale_dtype).reshape(shp[:-1])

    def decode(self, idx: torch.Tensor, scale: torch.Tensor, dtype=torch.float32):
        shp = idx.shape
        flat = idx.reshape(-1)
        if self.kind == "ball":
            y = _w(B.unrank(flat, B.ball_tables(self.param))) / 2
        else:
            y = Vo.decode_coords(Vo.unpack(flat, self.param), self.param)
        z = y.reshape(-1, self.D) * _w(scale.reshape(-1, 1))
        return self.rot.inv(z).to(dtype).reshape(*shp[:-1], self.D)

    def quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        idx, s = self.encode(x)
        return self.decode(idx, s, x.dtype)
