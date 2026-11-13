"""Voronoi (nested-lattice) E8 code, as in NestQuant (Savkin, Porat, Ordentlich,
Polyanskiy, arXiv:2502.09720): codebook = E8 / qE8, i.e. the q^8 E8 points in the
Voronoi region of qE8. Index = integer coordinates of the point mod q
(rate = log2 q bits per value, packed in base q into ceil(8 log2 q) bits).

encode: y = Q_E8(u); c = coords(y) mod q
decode: y' = c @ G;   y_hat = y' - q * Q_E8(y' / q)
Overload (y_hat != y): shrink the input radially until it decodes correctly
(NestQuant instead tries a small set of scales beta and signals the choice;
our per-vector scale search plays the same role, see codec.py).
"""
from __future__ import annotations

import math

import torch

from .e8 import _w, _wf, GEN, GEN_INV, nearest_e8


def bits_for_q(q: int) -> int:
    return math.ceil(8 * math.log2(q) - 1e-9)


def encode_coords(y: torch.Tensor, q: int) -> torch.Tensor:
    c = torch.round(_w(y) @ GEN_INV.to(_w(y))).long()
    return torch.remainder(c, q)


def decode_coords(c: torch.Tensor, q: int) -> torch.Tensor:
    yp = _w(c) @ GEN.to(_w(c) if c.is_floating_point() else _wf(c.device))
    return yp - q * nearest_e8(yp / q)


def pack(c: torch.Tensor, q: int) -> torch.Tensor:
    idx = torch.zeros(c.shape[:-1], dtype=torch.long, device=c.device)
    for i in range(7, -1, -1):
        idx = idx * q + c[..., i]
    return idx


def unpack(idx: torch.Tensor, q: int) -> torch.Tensor:
    out = []
    for i in range(8):
        out.append(idx % q)
        idx = idx // q
    return torch.stack(out, -1)


def quantize(u: torch.Tensor, q: int, return_idx=False):
    y = nearest_e8(_w(u))
    c = encode_coords(y, q)
    yh = decode_coords(c, q)
    bad = (yh - y).abs().amax(-1) > 1e-6
    # start at radius q (covering radius of qE8), then step down 3%
    g0 = (q / _w(u).norm(dim=-1).clamp(min=1e-30)).clamp(max=1.0)
    k = 0
    while bad.any():
        sel = torch.nonzero(bad).squeeze(1)
        yb = nearest_e8(_w(u[sel]) * (g0[sel][:, None] * 0.97 ** k))
        k += 1
        cb = encode_coords(yb, q)
        ok = (decode_coords(cb, q) - yb).abs().amax(-1) <= 1e-6
        y[sel[ok]] = yb[ok]
        c[sel[ok]] = cb[ok]
        bad[sel[ok]] = False
    out = y.to(u.dtype)
    return (out, pack(c, q)) if return_idx else out
